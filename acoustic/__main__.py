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
    if args.device:
        s.audio_device = args.device
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

    yamnet = Yamnet(s.yamnet_model) if s.use_yamnet else None

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
        source = SimSource()
    elif args.file:
        source = FileSource(args.file, loop=args.loop)
    else:
        try:
            source = MicSource(s.audio_device)
        except Exception as e:  # noqa: BLE001
            log.error("No usable microphone (%s). Use --simulate or --file.", e)
            return 2

    from acoustic.cot import CotClient
    from acoustic.engine import Engine
    from acoustic.gemini import GeminiChecker
    from acoustic.meshbus import MeshBus

    cot_client = None if args.no_cot else CotClient(s.tak_host, s.tak_port).start()
    mesh = None if not s.mesh_enabled else MeshBus(s.mqtt_host, s.mqtt_port, s.mqtt_user, s.mqtt_pass,
                                                   s.mesh_to, s.mesh_channel).start()
    gemini = GeminiChecker(s.gemini_api_key, s.gemini_model, s.gemini_min_interval_s)
    if not s.has_position and cot_client is not None:
        log.warning("SENSOR_LAT/SENSOR_LON not set: ATAK markers will not be sent")

    engine = Engine(s, yamnet=yamnet, cot_client=cot_client, mesh=mesh, gemini=gemini, source=source).start()

    import uvicorn

    from acoustic.web import app as web

    web.engine = engine
    log.info("Acoustic detector %s: source=%s, yamnet=%s, gemini=%s, UI http://0.0.0.0:%d",
             s.sensor_name, source.description, bool(yamnet and yamnet.available),
             s.gemini_model if gemini.enabled else "off", s.web_port)

    async def serve():
        server = uvicorn.Server(uvicorn.Config(web.app, host="0.0.0.0", port=s.web_port,
                                               log_level="warning", loop="asyncio"))
        task = asyncio.create_task(web.periodic_broadcast())
        try:
            await server.serve()
        finally:
            task.cancel()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        if mesh:
            mesh.stop()
        if cot_client:
            cot_client.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
