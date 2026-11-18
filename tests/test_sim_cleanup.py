"""Simulated alert markers are taken off the map on shutdown and on start in on_demand mode."""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from acoustic.engine import Engine
from acoustic.simulator import SCENARIO, Synth
from acoustic.sources import SimSource
from acoustic.tak_client import TakClient
from conftest import FakeCot, FakeMesh, noise, tone_comb

REPO = Path(__file__).resolve().parent.parent


class MicLike:
    kind = "mic"
    description = "test mic"


def feed(e, audio):
    for i in range(len(audio) // 8000):
        e.feed(audio[i * 8000:(i + 1) * 8000])
        if i >= 2:
            e.analyze_window(e.tail(0.975), now=1000.0 + i * 0.5)


def events(msgs):
    return [ET.fromstring(m.encode()) for m in msgs]


def removals(msgs):
    """{uid: (stale-now marker sent, forcedelete sent)} from a run of CoT messages."""
    out = {}
    for ev in events(msgs):
        if ev.get("type") == "t-x-d-d" and ev.find("detail/__forcedelete") is not None:
            uid = ev.find("detail/link").get("uid")
            out.setdefault(uid, [False, False])[1] = True
        elif "A-M-F-Q" in ev.get("type") and ev.get("stale") == ev.get("time"):
            out.setdefault(ev.get("uid"), [False, False])[0] = True
    return {k: tuple(v) for k, v in out.items()}


def continuous_sim_run(settings):
    cot = FakeCot()
    e = Engine(settings, cot_client=cot, mesh=FakeMesh(), source=SimSource())
    synth = Synth()
    feed(e, np.concatenate([synth.block(1600) for _ in range(int(sum(d for _, d in SCENARIO) * 10))]))
    return e, cot


def test_continuous_sim_alert_is_recorded_and_removed_on_shutdown(settings):
    e, cot = continuous_sim_run(settings)
    assert "ACOUSTIC-MIC1-UAV" in [ev.get("uid") for ev in events(cot.messages)]
    assert json.loads((settings.data_dir / "sim_markers.json").read_text()) == ["ACOUSTIC-MIC1-UAV"]
    sent = len(cot.messages)
    assert e.remove_sim_markers() == ["ACOUSTIC-MIC1-UAV"]
    new = cot.messages[sent:]
    assert removals(new) == {"ACOUSTIC-MIC1-UAV": (True, True)}
    assert all(ev.get("uid") != "ACOUSTIC-MIC1" for ev in events(new))   # the sensor marker stays
    assert json.loads((settings.data_dir / "sim_markers.json").read_text()) == []
    assert e.remove_sim_markers() == []                                  # nothing left to remove


def test_on_demand_start_removes_what_continuous_left(settings):
    continuous_sim_run(settings)                    # that run ended without a clean shutdown
    cot = FakeCot()
    e = Engine(settings, cot_client=cot, mesh=FakeMesh(), source=SimSource(quiet=True))
    assert e.remove_sim_markers(startup=True) == ["ACOUSTIC-MIC1-UAV"]
    assert removals(cot.messages) == {"ACOUSTIC-MIC1-UAV": (True, True)}


def test_start_with_no_record_removes_both_alert_uids_once(settings):
    cot = FakeCot()
    e = Engine(settings, cot_client=cot, mesh=FakeMesh(), source=MicLike())
    assert sorted(e.remove_sim_markers(startup=True)) == ["ACOUSTIC-MIC1-UAV", "ACOUSTIC-MIC1-UAV-SIM"]
    e2 = Engine(settings, cot_client=FakeCot(), mesh=FakeMesh(), source=MicLike())
    assert e2.remove_sim_markers(startup=True) == []                     # now the history is known


def test_real_mic_alerts_are_never_removed(settings):
    (settings.data_dir / "sim_markers.json").write_text("[]")
    cot = FakeCot()
    e = Engine(settings, cot_client=cot, mesh=FakeMesh(), source=MicLike())
    feed(e, np.concatenate([noise(1.0), tone_comb(380.0, seconds=7.0, jitter=0.02)]))
    assert e.hyst.active and "ACOUSTIC-MIC1-UAV" in [ev.get("uid") for ev in events(cot.messages)]
    sent = len(cot.messages)
    assert e.remove_sim_markers() == [] and e.remove_sim_markers(startup=True) == []
    assert len(cot.messages) == sent


def test_demo_marker_is_recorded_too(settings):
    from test_demo import FakeSpeaker, run

    (settings.data_dir / "sim_markers.json").write_text("[]")
    e = Engine(settings, cot_client=FakeCot(), mesh=FakeMesh(), source=MicLike(), speaker=FakeSpeaker(False))
    run(e, 3)
    run(e, 9, trigger_at=0)
    assert json.loads((settings.data_dir / "sim_markers.json").read_text()) == ["ACOUSTIC-MIC1-UAV-SIM"]
    assert e.remove_sim_markers() == ["ACOUSTIC-MIC1-UAV-SIM"]


# -- with a fake TAK server ------------------------------------------------------------------
class FakeTak:
    """Accepts one CoT client at a time and keeps every event it is sent."""

    def __init__(self):
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(4)
        self.port = self.srv.getsockname()[1]
        self.data = b""
        self.lock = threading.Lock()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._read, args=(conn,), daemon=True).start()

    def _read(self, conn):
        while chunk := conn.recv(65536):
            with self.lock:
                self.data += chunk

    def messages(self):
        with self.lock:
            text = self.data.decode()
        return ["<?xml" + p for p in text.split("<?xml")[1:]]

    def wait_for(self, pred, timeout=15.0):
        end = time.time() + timeout
        while time.time() < end:
            if pred(self.messages()):
                return True
            time.sleep(0.1)
        return False

    def close(self):
        self.srv.close()


def test_shutdown_removal_reaches_tak_before_the_client_closes(settings):
    tak = FakeTak()
    client = TakClient("127.0.0.1", tak.port, "acoustic-detector", "test", "ACOUSTIC",
                       lat=52.0, lon=-1.0, register_delay_s=0.0).start()
    try:
        (settings.data_dir / "sim_markers.json").write_text('["ACOUSTIC-MIC1-UAV"]')
        e = Engine(settings, cot_client=client, mesh=FakeMesh(), source=SimSource())
        assert tak.wait_for(lambda m: len(m) >= 1)                       # registered
        assert e.remove_sim_markers() == ["ACOUSTIC-MIC1-UAV"]
        assert client.flush(5.0)
        client.stop()                                                    # straight after, as on SIGTERM
        msgs = tak.messages()
        assert removals(msgs) == {"ACOUSTIC-MIC1-UAV": (True, True)}
        assert [ev.get("type") for ev in events(msgs)][-1] == "a-f-G-E-S"
    finally:
        client.stop()
        tak.close()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_process_cleans_up_on_on_demand_start_and_on_sigterm(tmp_path):
    """The real entry point: start in on_demand with a leftover simulated alert, then SIGTERM."""
    tak = FakeTak()
    (tmp_path / "sim_markers.json").write_text('["ACOUSTIC-OP1-UAV"]')
    env = {**os.environ, "DEMO_MODE": "on_demand", "DATA_DIR": str(tmp_path), "SENSOR_NAME": "OP1",
           "SENSOR_LAT": "52.1", "SENSOR_LON": "-1.1", "TAK_HOST": "127.0.0.1", "TAK_PORT": str(tak.port),
           "MQTT_HOST": "127.0.0.1", "MQTT_PORT": str(_free_port()), "GEMINI_API_KEY": "", "PYTHONPATH": str(REPO)}
    proc = subprocess.Popen([sys.executable, "-m", "acoustic", "--simulate", "--no-yamnet",
                             "--web-port", str(_free_port())], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        assert tak.wait_for(lambda m: "ACOUSTIC-OP1-UAV" in removals(m)), "no removal on start"
        assert removals(tak.messages()) == {"ACOUSTIC-OP1-UAV": (True, True)}
        start = len(tak.messages())
        # simulate a leftover from this run too, then stop it the way systemd does
        (tmp_path / "sim_markers.json").write_text("[]")
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=15) == 0
        assert removals(tak.messages()[start:]) == {}                    # nothing simulated was sent
        assert "ACOUSTIC-OP1" in [ev.get("uid") for ev in events(tak.messages())]   # sensor marker kept
    finally:
        if proc.poll() is None:
            proc.kill()
        tak.close()


def test_process_sigterm_removes_its_own_simulated_alert(tmp_path):
    """Continuous --simulate with an alert on the map: SIGTERM takes it off before exiting."""
    tak = FakeTak()
    (tmp_path / "sim_markers.json").write_text("[]")
    env = {**os.environ, "DEMO_MODE": "continuous", "DATA_DIR": str(tmp_path), "SENSOR_NAME": "OP1",
           "SENSOR_LAT": "52.1", "SENSOR_LON": "-1.1", "TAK_HOST": "127.0.0.1", "TAK_PORT": str(tak.port),
           "MQTT_HOST": "127.0.0.1", "MQTT_PORT": str(_free_port()), "GEMINI_API_KEY": "", "PYTHONPATH": str(REPO)}
    proc = subprocess.Popen([sys.executable, "-m", "acoustic", "--simulate", "--no-yamnet",
                             "--web-port", str(_free_port())], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        # the looping soundscape starts with 15 s of background, then the drone pass alerts
        assert tak.wait_for(lambda m: "ACOUSTIC-OP1-UAV" in [ev.get("uid") for ev in events(m)], timeout=40)
        assert json.loads((tmp_path / "sim_markers.json").read_text()) == ["ACOUSTIC-OP1-UAV"]
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=15) == 0
        assert removals(tak.messages()) == {"ACOUSTIC-OP1-UAV": (True, True)}
        assert json.loads((tmp_path / "sim_markers.json").read_text()) == []
    finally:
        if proc.poll() is None:
            proc.kill()
        tak.close()
