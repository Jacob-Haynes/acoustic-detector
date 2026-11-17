"""Entry point: python -m acoustic [--simulate | --file clip.wav | --device N]"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from acoustic.config import DEMO_LAT, DEMO_LON, Settings


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Acoustic drone detector")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--simulate", action="store_true", help="synthetic soundscape, no hardware")
    src.add_argument("--file", help="replay a WAV/MP3/FLAC file instead of the microphone")
    p.add_argument("--loop", action="store_true", help="loop --file replay")
    p.add_argument("--device", help="input device index or name (default: AUDIO_DEVICE or system default)")
    p.add_argument("--list-devices", action="store_true", help="list audio input devices and exit")
    p.add_argument("--enroll", nargs="+", metavar="CLIP", help="enrol clips of your drone and exit")
    p.add_argument("--enroll-background", nargs="+", metavar="CLIP", help="enrol background clips and exit")
    p.add_argument("--web-port", type=int, help="web UI port (default 8100)")
    p.add_argument("--tak-host", help="TAK server host (env TAK_HOST, default 127.0.0.1)")
    p.add_argument("--tak-port", type=int, help="TAK CoT TCP port (env TAK_PORT, default 8087)")
    p.add_argument("--mqtt-host", help="MQTT broker on the Pi hub (env MQTT_HOST, default 127.0.0.1)")
    p.add_argument("--mqtt-port", type=int, help="env MQTT_PORT, default 1883")
    p.add_argument("--mqtt-user", help="env MQTT_USER (guest on the Pi, isrkit from a laptop)")
    p.add_argument("--mqtt-pass", help="env MQTT_PASS (prefer the env var: CLI args show in ps)")
    p.add_argument("--public-url", help="UI URL advertised in the heartbeat (env PUBLIC_URL)")
    p.add_argument("--no-cot", action="store_true", help="disable CoT output")
    p.add_argument("--no-mesh", action="store_true", help="disable meshbus alerts")
    p.add_argument("--no-yamnet", action="store_true", help="spectral features only")
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("acoustic")
    args = parse_args(argv)
    s = Settings.from_env()
    if args.web_port:
        s.web_port = args.web_port
    for arg, attr in (("tak_host", "tak_host"), ("tak_port", "tak_port"), ("mqtt_host", "mqtt_host"),
                      ("mqtt_port", "mqtt_port"), ("mqtt_user", "mqtt_user"), ("mqtt_pass", "mqtt_pass"),
                      ("public_url", "public_url")):
        if getattr(args, arg) is not None:
            setattr(s, attr, getattr(args, arg))
    persisted = s.data_dir / "audio_device.txt"
    if args.device:
        s.audio_device = args.device
    elif persisted.exists() and persisted.read_text().strip():
        s.audio_device = persisted.read_text().strip()    # last choice made in the web UI
    if args.no_yamnet:
        s.use_yamnet = False
    if args.no_mesh:
        s.mesh_enabled = False
    if args.simulate and not s.has_position:
        s.sensor_lat, s.sensor_lon = DEMO_LAT, DEMO_LON
    s.data_dir.mkdir(parents=True, exist_ok=True)

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    from acoustic.yamnet import Yamnet

    yamnet = None
    if s.use_yamnet:
        try:
            yamnet = Yamnet(s.yamnet_model, s.yamnet_class_map)    # degrades itself, never raises
        except Exception as e:  # noqa: BLE001 - belt and braces: a classifier must not stop the sensor
            log.warning("YAMNet unavailable (%s); prop-comb detection only", e)
            yamnet = None

    if args.enroll or args.enroll_background:
        from acoustic.calibration import Calibration
        from acoustic.sources import load_audio_file

        cal = Calibration(s.data_dir, yamnet=yamnet)
        for label, clips in (("drone", args.enroll or []), ("background", args.enroll_background or [])):
            for clip in clips:
                path = cal.add_clip(load_audio_file(clip), label)
                log.info("Enrolled %s as %s -> %s", clip, label, path)
        print(cal.status())
        return 0

    from acoustic.sources import FileSource, MicSource, SimSource

    if args.simulate:
        source = SimSource(quiet=s.demo_mode == "on_demand")
    elif args.file:
        source = FileSource(args.file, loop=args.loop)
    else:
        try:
            source = MicSource(s.audio_device, prefer=s.audio_prefer, rate=s.audio_rate)
        except Exception as e:  # noqa: BLE001
            if s.audio_device and not args.device:
                log.warning("Saved/configured device %r unusable (%s); falling back to auto-select",
                            s.audio_device, e)
                try:
                    source = MicSource(None, prefer=s.audio_prefer, rate=s.audio_rate)
                except Exception as e2:  # noqa: BLE001
                    log.error("No usable microphone (%s). Use --simulate or --file.", e2)
                    return 2
            else:
                log.error("No usable microphone (%s). Use --simulate or --file.", e)
                return 2

    from acoustic import __version__
    from acoustic.engine import Engine
    from acoustic.gemini import GeminiChecker
    from acoustic.meshbus import MeshBus, station_host
    from acoustic.tak_client import TakClient

    # Contract v5: one long-lived connection, registered as an EUD before any marker is sent.
    cot_client = None if args.no_cot else TakClient(
        s.tak_host, s.tak_port, tool="acoustic-detector", station=station_host(), label="ACOUSTIC",
        lat=s.sensor_lat if s.has_position else None, lon=s.sensor_lon if s.has_position else None,
        hae=s.sensor_hae, version=__version__).start()
    # One MQTT client for both mesh alerts and the heartbeat. --no-mesh only stops alert texts.
    mesh = MeshBus(s.mqtt_host, s.mqtt_port, s.mqtt_user, s.mqtt_pass, s.mesh_to, s.mesh_channel).start()
    gemini = GeminiChecker(s.gemini_api_key, s.gemini_model, s.gemini_min_interval_s)
    if not s.has_position and cot_client is not None:
        log.warning("SENSOR_LAT/SENSOR_LON not set: ATAK markers will not be sent")

    engine = Engine(s, yamnet=yamnet, cot_client=cot_client, mesh=mesh, gemini=gemini, source=source).start()
    mesh.start_heartbeat(engine.heartbeat)

    import uvicorn

    from acoustic.web import app as web

    web.engine = engine
    log.info("Acoustic detector %s: source=%s, yamnet=%s, gemini=%s, demo=%s, UI http://0.0.0.0:%d",
             s.sensor_name, source.description, bool(yamnet and yamnet.available),
             s.gemini_model if gemini.enabled else "off", s.demo_mode, s.web_port)

    async def serve():
        server = uvicorn.Server(uvicorn.Config(web.app, host="0.0.0.0", port=s.web_port,
                                               log_level="warning", loop="asyncio"))
        task = asyncio.create_task(web.periodic_broadcast())
        try:
            await server.serve()
        finally:
            task.cancel()

    # uvicorn handles SIGTERM/SIGINT, then restores the previous handlers and re-raises the
    # signal. With the default SIGTERM handler that kills us before cleanup, so install a
    # no-op one: the re-raise lands here and we fall through to publish "stopped" and exit 0.
    import signal

    signal.signal(signal.SIGTERM, lambda signum, frame: None)
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down")
        engine.stop()
        if mesh:
            mesh.stop()
        if cot_client:
            cot_client.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
