"""The sensor loop: buffer audio, build waterfall columns, analyse a window every HOP, fuse scores,
run hysteresis, fire CoT + mesh alerts, handle sample recording and Gemini requests."""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from acoustic import cot, demo, dsp, meshbus
from acoustic.calibration import Calibration
from acoustic.config import SAMPLE_RATE, WINDOW_SAMPLES, Settings
from acoustic.detector import Hysteresis, fuse
from acoustic.yamnet import UNAVAILABLE

logger = logging.getLogger(__name__)

HOP_SAMPLES = SAMPLE_RATE // 2         # analyse twice a second
RING_SECONDS = 30
GEMINI_CLIP_S = 4.0
SENSOR_MARKER_EVERY_S = 60.0
ACTIVE_RESEND_S = 15.0


class Engine:
    def __init__(self, settings: Settings, yamnet=None, cot_client=None, mesh=None, gemini=None,
                 source=None, speaker=None):
        self.s = settings
        self.yamnet = yamnet
        self.cot = cot_client
        self.mesh = mesh
        self.gemini = gemini
        self.source = source
        self.calibration = Calibration(settings.data_dir, yamnet=yamnet)
        if (yamnet is not None and yamnet.available and self.calibration.enrolled
                and "yamnet" not in self.calibration.ref.get("drone", {})):
            # profile was built without the model (e.g. enrolled before YAMNet was installed)
            threading.Thread(target=self.calibration.rebuild, daemon=True).start()
        self.hyst = Hysteresis(settings.detect_on, settings.detect_off, settings.on_frames,
                               settings.off_frames, settings.alert_cooldown_s)

        self._lock = threading.Lock()
        self._ring = np.zeros(RING_SECONDS * SAMPLE_RATE, dtype=np.float32)
        self._ring_pos = 0
        self._total = 0
        self._wf_pending = 0
        self._wf_cols: collections.deque[bytes] = collections.deque(maxlen=400)
        self._since_analysis = 0
        self._analysis_due = threading.Event()
        self._stop = threading.Event()

        self._recording: dict | None = None
        self.last_record: dict | None = None

        self.state: dict = {"confidence": 0.0, "raw": 0.0, "components": {}, "active": False}
        self.history: collections.deque[tuple[float, float]] = collections.deque(maxlen=240)
        self.log: collections.deque[dict] = collections.deque(maxlen=200)
        self._last_sensor_marker = 0.0
        self._started = time.time()
        self._peak = 0.0
        self._last_active_send = 0.0
        self._log_path = Path(settings.data_dir) / "detections.jsonl"

        # step-by-step demo (DEMO-TRIGGER-CONTRACT.md)
        self.speaker = speaker if speaker is not None else demo.Speaker(settings.demo_speaker)
        self._signature: np.ndarray | None = None
        self._inject: dict | None = None          # drone-sound: {"sig", "pos"} mixed into feed()
        self._demo_until: dict[str, float] = {}   # scenario id -> planned end (epoch)
        self._demo_tail_until = 0.0               # a detection starting before this is the demo's
        self._episode_sim = False                 # the current detection was started by the demo
        self._sim_cot_sent = False                # a demo UAV marker is (maybe) on the map
        if self.gemini is not None:
            self.gemini.on_result = self._on_gemini

    # ------------------------------------------------------------------------------------------
    # audio in
    # ------------------------------------------------------------------------------------------
    def feed(self, block: np.ndarray):
        block = np.asarray(block, dtype=np.float32)
        n = len(block)
        inj = self._inject
        if inj is not None and n:
            chunk = inj["sig"][inj["pos"]:inj["pos"] + n]
            mixed = block.copy()
            mixed[:len(chunk)] += chunk
            block = np.clip(mixed, -1.0, 1.0)
            inj["pos"] += n
            if inj["pos"] >= len(inj["sig"]) and self._inject is inj:
                self._inject = None                       # signature played out
                self._demo_tail_until = time.time() + demo.TAIL_S
        if n:
            self._peak = max(float(np.max(np.abs(block))), self._peak * 0.9)
        with self._lock:
            end = self._ring_pos + n
            if end <= len(self._ring):
                self._ring[self._ring_pos:end] = block
            else:
                k = len(self._ring) - self._ring_pos
                self._ring[self._ring_pos:] = block[:k]
                self._ring[: n - k] = block[k:]
            self._ring_pos = end % len(self._ring)
            self._total += n
            self._wf_pending += n
            while self._wf_pending >= dsp.WF_HOP:
                self._wf_pending -= dsp.WF_HOP
                frame = self._tail_locked(dsp.WF_NFFT, offset=self._wf_pending)
                self._wf_cols.append(dsp.waterfall_column(frame))
            finished = None
            if self._recording is not None:
                self._recording["chunks"].append(block.copy())
                self._recording["have"] += n
                if self._recording["have"] >= self._recording["need"]:
                    finished, self._recording = self._recording, None

            self._since_analysis += n
            if self._since_analysis >= HOP_SAMPLES:
                self._since_analysis = 0
                self._analysis_due.set()
        if finished is not None:
            # embedding the enrolled clips can take a second or two: keep it off the audio thread
            threading.Thread(target=self._finish_recording, args=(finished,), daemon=True).start()

    def _tail_locked(self, n: int, offset: int = 0) -> np.ndarray:
        """Last n samples ending `offset` samples before the newest one."""
        end = (self._ring_pos - offset) % len(self._ring)
        start = end - n
        if start >= 0:
            return self._ring[start:end].copy()
        return np.concatenate([self._ring[start:], self._ring[:end]])

    def tail(self, seconds: float) -> np.ndarray:
        n = int(min(seconds, RING_SECONDS) * SAMPLE_RATE)
        with self._lock:
            n = min(n, self._total)
            return self._tail_locked(n) if n else np.zeros(0, dtype=np.float32)

    def take_waterfall(self) -> list[bytes]:
        with self._lock:
            cols = list(self._wf_cols)
            self._wf_cols.clear()
        return cols

    # ------------------------------------------------------------------------------------------
    # analysis
    # ------------------------------------------------------------------------------------------
    def analyze_window(self, window: np.ndarray, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        feats = dsp.spectral_features(window, min_level_dbfs=self.s.min_level_dbfs)
        yres = self.yamnet.classify(window) if (self.yamnet is not None and self.yamnet.available) else None
        cal = self.calibration.score(window, yres.scores if yres is not None else None)
        quiet = feats.level_dbfs < self.s.min_level_dbfs
        components = {
            "spectral": feats.score,
            "yamnet": None if yres is None else (0.0 if quiet else yres.score),
            "calibration": None if cal is None else (0.0 if quiet else cal["score"]),
            "confuser": None if yres is None else yres.confuser,
        }
        raw = fuse(components)
        details = {"f0_hz": round(feats.f0_hz, 1), "components": {k: (None if v is None else round(v, 3))
                                                                  for k, v in components.items()}}
        if yres is not None:
            details["yamnet_top"] = yres.top[0][0]
        if self.source is not None and getattr(self.source, "kind", "") == "simulate":
            details["sim_truth"] = self.source.truth
        if self._inject is not None:
            details["sim_truth"] = "drone (demo)"
        prev_alert = self.hyst.last_alert_ts
        event = self.hyst.update(raw, now=now, details=details)
        if event is not None and event.kind in ("detect", "suppressed"):
            self._episode_sim = self._demo_running()
            if self._episode_sim:
                self.hyst.last_alert_ts = prev_alert   # a demo alert never starts the real cooldown

        state = {
            "ts": now,
            "raw": round(raw, 3),
            "confidence": round(self.hyst.smoothed, 3),
            "active": self.hyst.active,
            "components": details["components"],
            "spectral": feats.as_dict(),
            "yamnet": None if yres is None else yres.as_dict(),
            "calibration": cal,
        }
        if "sim_truth" in details:
            state["sim_truth"] = details["sim_truth"]
        self.state = state
        self.history.append((now, self.hyst.smoothed))
        if event is not None:
            self._handle_event(event)
        self._periodic_outputs(now)
        return state

    @property
    def source_kind(self) -> str:
        """The label on outputs: "simulate" (SIMULATED: / [SIM]) for anything the demo caused."""
        if self.demo_labelled:
            return "simulate"
        return getattr(self.source, "kind", "") or ""

    @property
    def demo_labelled(self) -> bool:
        """A demo scenario is running, or the current detection came from one."""
        return self._episode_sim or self._demo_running()

    @property
    def _parent_uid(self) -> str | None:
        """Our TAK registration (EUD) uid, linked from markers so OTS files them under it."""
        uid = getattr(self.cot, "uid", None)
        return uid if isinstance(uid, str) else None

    def _handle_event(self, event):
        entry = event.as_dict()
        if self._episode_sim:
            entry["sim"] = True
        self._append_log(entry)
        if event.kind == "detect":
            logger.warning("DRONE DETECTED conf %.2f f0 %.0f Hz", event.confidence, event.details.get("f0_hz", 0))
            self._send_detection_cot(event.confidence, event.details.get("f0_hz", 0.0))
            if self.mesh is not None and self.s.mesh_enabled:
                self.mesh.send(meshbus.alert_text(event.confidence, self.s.sensor_name, self.source_kind))
            if self.gemini is not None and self.gemini.enabled and self.s.gemini_auto:
                self.gemini.request(self.tail(GEMINI_CLIP_S), trigger="auto")
        elif event.kind == "suppressed":
            logger.info("Re-detection within cooldown; alert suppressed (conf %.2f)", event.confidence)
            self._send_detection_cot(event.confidence, event.details.get("f0_hz", 0.0))
        elif event.kind == "clear":
            logger.info("Clear (peak %.2f)", event.details.get("peak", 0))
            if self.mesh is not None and self.s.mesh_enabled and self.s.mesh_clear_msg:
                self.mesh.send(meshbus.clear_text(self.s.sensor_name, event.details.get("peak", 0),
                                                  self.source_kind))
            self._episode_sim = False

    def _append_log(self, entry: dict):
        self.log.appendleft(entry)
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.debug("Could not write detection log: %s", e)

    def _send_detection_cot(self, confidence: float, f0: float):
        if self.cot is None or not self.s.has_position:
            if self.cot is not None:
                logger.warning("SENSOR_LAT/SENSOR_LON not set; CoT detection marker not sent")
            return
        self._last_active_send = time.time()
        src = self.source_kind
        if self._episode_sim:
            self._sim_cot_sent = True
        self.cot.send(cot.detection_event(self.s.sensor_name, self.s.sensor_lat, self.s.sensor_lon,
                                          self.s.sensor_hae, confidence, self.s.detect_radius_m,
                                          self.s.cot_affiliation, f0, source=src,
                                          parent_uid=self._parent_uid, demo=self._episode_sim))

    def _periodic_outputs(self, now: float):
        if self.cot is None or not self.s.has_position:
            return
        if self.s.cot_sensor_marker and now - self._last_sensor_marker >= SENSOR_MARKER_EVERY_S:
            self._last_sensor_marker = now
            status = "DRONE DETECTED" if self.hyst.active else "listening"
            self.cot.send(cot.sensor_event(self.s.sensor_name, self.s.sensor_lat, self.s.sensor_lon,
                                           self.s.sensor_hae, status, source=self.source_kind,
                                           parent_uid=self._parent_uid))
        if self.hyst.active and now - self._last_active_send >= ACTIVE_RESEND_S:
            self._send_detection_cot(self.hyst.smoothed, self.state.get("spectral", {}).get("f0_hz", 0.0))

    def _analysis_loop(self):
        while not self._stop.is_set():
            if not self._analysis_due.wait(timeout=1.0):
                continue
            self._analysis_due.clear()
            with self._lock:
                if self._total < WINDOW_SAMPLES // 2:
                    continue
                window = self._tail_locked(WINDOW_SAMPLES)
            try:
                self.analyze_window(window)
            except Exception:  # noqa: BLE001
                logger.exception("Analysis failed")

    # ------------------------------------------------------------------------------------------
    # recording / calibration
    # ------------------------------------------------------------------------------------------
    def start_recording(self, label: str, seconds: float = 5.0) -> bool:
        with self._lock:
            if self._recording is not None:
                return False
            self._recording = {"label": label, "need": int(seconds * SAMPLE_RATE), "have": 0,
                               "chunks": [], "started": time.time(), "seconds": seconds}
        return True

    def _finish_recording(self, rec: dict):
        audio = np.concatenate(rec["chunks"])[: rec["need"]]
        try:
            path = self.calibration.add_clip(audio, rec["label"])
            self.last_record = {"ok": True, "label": rec["label"], "file": path.name, "ts": time.time(),
                                "warning": self.calibration.last_warning}
            self._append_log({"kind": "enrolled", "ts": time.time(), "confidence": 0.0,
                              "label": rec["label"], "file": path.name})
        except Exception as e:  # noqa: BLE001
            logger.exception("Saving sample failed")
            self.last_record = {"ok": False, "error": str(e), "ts": time.time()}

    def recording_status(self) -> dict | None:
        rec = self._recording
        if rec is None:
            return None
        return {"label": rec["label"], "progress": round(rec["have"] / rec["need"], 2), "seconds": rec["seconds"]}

    # ------------------------------------------------------------------------------------------
    # gemini
    # ------------------------------------------------------------------------------------------
    def ask_gemini(self) -> str:
        if self.gemini is None:
            return "disabled"
        return self.gemini.request(self.tail(GEMINI_CLIP_S), trigger="manual", force=True)

    def _on_gemini(self, result: dict):
        entry = {"kind": "gemini", "ts": result.get("ts", time.time()),
                 "confidence": result.get("confidence") or 0.0,
                 "drone_audible": result.get("drone_audible"), "best_guess": result.get("best_guess", ""),
                 "trigger": result.get("trigger"), "ok": result.get("ok")}
        self._append_log(entry)

    # ------------------------------------------------------------------------------------------
    # step-by-step demo (ui-kit/DEMO-TRIGGER-CONTRACT.md)
    # ------------------------------------------------------------------------------------------
    def demo_available(self) -> list[str]:
        kind = getattr(self.source, "kind", "none") if self.source is not None else "none"
        ids = ["drone-sound"]
        if kind == "mic" and self.speaker.available:   # the real mic has to hear the speaker
            ids.append("play-clip")
        return ids

    def demo_active(self) -> list[str]:
        now = time.time()
        ids = []
        if self._inject is not None:
            if now > self._demo_until.get("drone-sound", now) + 10.0:
                self._inject = None                        # no audio is flowing: give up on it
            else:
                ids.append("drone-sound")
        if self._demo_until.get("play-clip", 0.0) > now:
            ids.append("play-clip")
        return ids

    def _demo_running(self) -> bool:
        return bool(self.demo_active()) or time.time() < self._demo_tail_until

    def demo_scenarios(self) -> dict:
        return {"mode": self.s.demo_mode,
                "scenarios": [{"id": i, **demo.SCENARIOS[i]} for i in self.demo_available()],
                "active": self.demo_active()}

    def demo_trigger(self, sid: str) -> dict:
        if sid not in demo.SCENARIOS:
            raise KeyError(sid)
        if sid not in self.demo_available():
            raise ValueError("play-clip needs a speaker on this unit and the microphone as input")
        now, dur = time.time(), demo.SCENARIOS[sid]["duration_s"]
        sig = self._demo_sound(dur)
        if sid == "drone-sound":
            # a noisy room would drown it: keep the close-in part well above what the mic hears now
            room = self.tail(1.0)
            scale = demo.room_scale(float(np.sqrt(np.mean(room ** 2))) if len(room) else 0.0, sig)
            self._inject = {"sig": sig * scale, "pos": 0}     # restarts if already running
        elif not self.speaker.play(sig):
            raise RuntimeError("the speaker did not play")
        self._demo_until[sid] = now + dur
        self._demo_tail_until = max(self._demo_tail_until, now + dur + demo.TAIL_S)
        logger.info("Demo scenario %s started (%.0f s), outputs marked SIMULATED", sid, dur)
        self._append_log({"kind": "demo", "ts": now, "confidence": 0.0, "id": sid,
                          "label": demo.SCENARIOS[sid]["label"]})
        return {"ok": True, "id": sid, "until": round(now + dur, 3)}

    def _demo_sound(self, seconds: float) -> np.ndarray:
        """The taught drone if there is one (so the "My drone" score matches too), else synthetic."""
        if self.calibration.enrolled:
            try:
                sig = demo.taught_signature(self.calibration.dir / "drone", seconds)
                if sig is not None:
                    return sig
            except Exception:  # noqa: BLE001
                logger.exception("Could not build the demo sound from the taught clips")
        if self._signature is None:
            self._signature = demo.drone_signature(seconds)
        return self._signature

    def demo_clear(self) -> dict:
        """Stop every scenario and take the demo's marker off the map. Real detections are untouched:
        they never use the demo uid, and the detector state is only reset for a demo detection."""
        active = self.demo_active()
        was_demo = bool(active) or self._episode_sim or time.time() < self._demo_tail_until
        self._inject = None
        if "play-clip" in active:
            self.speaker.stop()
        self._demo_until.clear()
        if was_demo:
            # the last second of demo sound is still in the analysis window: keep labelling briefly
            self._demo_tail_until = time.time() + demo.TAIL_S
        if self._episode_sim:
            h = self.hyst
            peak = h.peak
            h.active, h._above, h._below, h.smoothed = False, 0, 0, 0.0
            self._append_log({"kind": "clear", "ts": time.time(), "confidence": 0.0, "peak": round(peak, 3),
                              "sim": True, "demo_cleared": True})
            self._episode_sim = False
        elif was_demo and not self.hyst.active:
            self.hyst.smoothed, self.hyst._above = 0.0, 0     # don't let the leftover sound trigger
        removed = self._remove_demo_marker()
        if active:
            logger.info("Demo cleared: %s", ", ".join(active))
        if active or removed:
            self._append_log({"kind": "demo_clear", "ts": time.time(), "confidence": 0.0, "cleared": len(active),
                              "marker_removed": removed})
        return {"ok": True, "cleared": len(active), "marker_removed": removed}

    def _remove_demo_marker(self) -> bool:
        if not self._sim_cot_sent or self.cot is None or not self.s.has_position:
            return False
        self._sim_cot_sent = False
        s, now = self.s, time.time()
        uid = cot.detection_uid(s.sensor_name, demo=True)
        sends = [
            cot.detection_event(s.sensor_name, s.sensor_lat, s.sensor_lon, s.sensor_hae, 0.0, s.detect_radius_m,
                                s.cot_affiliation, source="simulate", parent_uid=self._parent_uid, demo=True,
                                stale_s=0.0, note="Demo cleared.", now=now),
            cot.delete_event(uid, cot.detection_type(s.cot_affiliation), s.sensor_lat, s.sensor_lon,
                             source="simulate", now=now),
        ]
        registration = getattr(self.cot, "registration", None)
        if callable(registration):
            sends.append(registration(now=now))              # OTS marks us disconnected on a t-x-d-d
        for xml in sends:
            self.cot.send(xml, force=True)
        return True

    # ------------------------------------------------------------------------------------------
    # test alert + lifecycle
    # ------------------------------------------------------------------------------------------
    def test_alert(self) -> dict:
        mesh_ok = False
        if self.mesh is not None:
            mesh_ok = self.mesh.send(meshbus.test_text(self.s.sensor_name, self.source_kind))
        cot_sent = False
        if self.cot is not None and self.s.has_position:
            self.cot.send(cot.sensor_event(self.s.sensor_name, self.s.sensor_lat, self.s.sensor_lon,
                                           self.s.sensor_hae, "TEST alert", source=self.source_kind,
                                           parent_uid=self._parent_uid))
            cot_sent = True
        self._append_log({"kind": "test", "ts": time.time(), "confidence": 0.0})
        return {"mesh_sent": mesh_ok, "cot_queued": cot_sent}

    def list_devices(self) -> dict:
        from acoustic.sources import list_input_devices

        current = getattr(self.source, "device", None) if getattr(self.source, "kind", "") == "mic" else None
        try:
            devices = list_input_devices()
        except Exception as e:  # noqa: BLE001
            return {"devices": [], "current": None, "error": str(e), "switchable": False}
        return {"devices": devices, "current": current,
                "switchable": getattr(self.source, "kind", "") == "mic"}

    def switch_device(self, device) -> str:
        if getattr(self.source, "kind", "") != "mic":
            raise ValueError("input is not the microphone (running in simulate/file mode)")
        desc = self.source.switch(device)
        try:
            (Path(self.s.data_dir) / "audio_device.txt").write_text(str(self.source.requested))
        except OSError:
            pass
        return desc

    def heartbeat(self) -> dict:
        """Fields for the isrkit heartbeat (tool/ts are added by meshbus)."""
        kind = getattr(self.source, "kind", "none")
        if kind == "simulate":
            mode = "simulate"
        elif kind == "file":
            mode = "replay"
        elif self.gemini is not None and self.gemini.enabled:
            mode = "gemini"
        else:
            mode = "cpu"
        status, notes = "ok", []
        if getattr(self.source, "error", None) or (kind == "none"):
            status = "error"                      # no working audio input
            notes.append("no audio input")
        elif getattr(self.source, "finished", False):
            status = "degraded"                   # file replay ended
            notes.append("replay finished")
        if (self.cot is not None and self.s.has_position and not self.cot.connected
                and time.time() - self._started > 30):
            status = "degraded" if status == "ok" else status
            notes.append("TAK unreachable")
        if self.yamnet_unavailable:
            status = "degraded" if status == "ok" else status
            notes.append(UNAVAILABLE)             # prop-comb detection only
        active = self.demo_active()
        if active:
            notes.append("demo: " + ", ".join(active))
        hb = {"host": meshbus.station_host(), "url": meshbus.advertised_url(self.s.web_port, self.s.public_url),
              "mode": mode, "status": status, "demo_mode": self.s.demo_mode, "demo_active": active}
        if notes:
            hb["detail"] = "; ".join(notes)
        return hb

    @property
    def yamnet_unavailable(self) -> bool:
        """YAMNet was wanted (not --no-yamnet) but its runtime, model or class map is missing."""
        return self.s.use_yamnet and (self.yamnet is None or not self.yamnet.available)

    def start(self):
        threading.Thread(target=self._analysis_loop, name="analysis", daemon=True).start()
        if self.source is not None:
            threading.Thread(target=self.source.run, args=(self.feed, self._stop), name="audio",
                             daemon=True).start()
        return self

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------------------------------
    # status for the UI
    # ------------------------------------------------------------------------------------------
    def status(self) -> dict:
        src = self.source
        return {
            "sensor": {"name": self.s.sensor_name, "lat": self.s.sensor_lat, "lon": self.s.sensor_lon,
                       "has_position": self.s.has_position},
            "source": {"kind": getattr(src, "kind", "none"), "description": getattr(src, "description", ""),
                       "finished": getattr(src, "finished", False), "error": getattr(src, "error", None),
                       "peak": round(self._peak, 3), "clipping": self._peak >= 0.99},
            "detector": {
                "yamnet": bool(self.yamnet is not None and self.yamnet.available),
                "yamnet_reason": "" if self.yamnet is None else getattr(self.yamnet, "reason", ""),
                "yamnet_unavailable": self.yamnet_unavailable,
                "on": self.s.detect_on, "off": self.s.detect_off, "cooldown_s": self.s.alert_cooldown_s,
                "affiliation": self.s.cot_affiliation,
            },
            "calibration": self.calibration.status(),
            "gemini": self.gemini.status() if self.gemini is not None else {"enabled": False, "reason": "not configured"},
            "outputs": {
                "tak": None if self.cot is None else {"host": f"{self.s.tak_host}:{self.s.tak_port}",
                                                      "connected": self.cot.connected, "sent": self.cot.sent},
                "mesh": None if self.mesh is None else {"host": f"{self.s.mqtt_host}:{self.s.mqtt_port}",
                                                        "connected": self.mesh.connected, "sent": self.mesh.sent},
            },
            "recording": self.recording_status(),
            "last_record": self.last_record,
            "demo": {**self.demo_scenarios(), "until": {k: round(v, 3) for k, v in self._demo_until.items()
                                                        if k in self.demo_active()},
                     "labelled": self.demo_labelled},
        }
