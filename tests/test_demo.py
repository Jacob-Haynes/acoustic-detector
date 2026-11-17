"""Step-by-step demo: ui-kit/DEMO-TRIGGER-CONTRACT.md."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from fastapi.testclient import TestClient

from acoustic import cot, demo, meshbus
from acoustic.config import SAMPLE_RATE, Settings
from acoustic.engine import Engine
from acoustic.simulator import SCENARIO
from acoustic.sources import SimSource
from conftest import FakeCot, FakeMesh, noise, tone_comb


class MicLike:
    kind = "mic"
    description = "test mic"


class FakeSpeaker:
    def __init__(self, available=True):
        self.available, self.played, self.stopped = available, [], 0

    def play(self, sig, sr=SAMPLE_RATE):
        self.played.append(len(sig) / sr)
        return True

    def stop(self):
        self.stopped += 1


def make(settings, source=None, speaker=None):
    cot_, mesh = FakeCot(), FakeMesh()
    e = Engine(settings, cot_client=cot_, mesh=mesh, source=source or MicLike(),
               speaker=speaker or FakeSpeaker(available=False))
    return e, cot_, mesh


def run(e, seconds, level=0.005, t0=[1000.0], trigger_at=None, sid="drone-sound", signal=None):
    """Feed `seconds` of room noise (or `signal`) in 0.5 s blocks and analyse each, like the live loop."""
    rng = np.random.default_rng(0)
    for i in range(int(seconds * 2)):
        if trigger_at is not None and i == int(trigger_at * 2):
            e.demo_trigger(sid)
        if signal is not None:
            block = signal[i * 8000:(i + 1) * 8000]
        else:
            block = (rng.standard_normal(8000) * level).astype(np.float32)
        e.feed(block)
        t0[0] += 0.5
        if i >= 2:
            e.analyze_window(e.tail(0.975), now=t0[0])


def remarks(xml):
    return ET.fromstring(xml.encode()).find("detail/remarks").text


def uids(msgs):
    return [ET.fromstring(m.encode()).get("uid") for m in msgs]


# -- mode ------------------------------------------------------------------------------------
def test_demo_mode_from_env(monkeypatch):
    assert Settings.from_env().demo_mode == "continuous"           # default: today's behaviour
    for raw, want in (("on_demand", "on_demand"), ("on-demand", "on_demand"), ("CONTINUOUS", "continuous"),
                      ("nonsense", "continuous")):
        monkeypatch.setenv("DEMO_MODE", raw)
        assert Settings.from_env().demo_mode == want


def test_on_demand_simulator_detects_nothing_on_its_own(settings):
    src = SimSource(quiet=True)
    e, cot_, mesh = make(settings, source=src)
    total = sum(d for _, d in SCENARIO)                            # a whole continuous loop's worth
    run(e, total, signal=np.concatenate([src.synth.block(1600) for _ in range(int(total * 10))]))
    assert [x for x in e.log if x["kind"] in ("detect", "suppressed")] == []
    assert mesh.texts == []
    assert not any("A-M-F-Q" in m for m in cot_.messages)
    assert SimSource().synth.scenario is SCENARIO                  # continuous keeps its drone pass


# -- drone-sound -----------------------------------------------------------------------------
def test_drone_sound_fires_the_real_pipeline_labelled_simulated(settings):
    e, cot_, mesh = make(settings)
    run(e, 3)
    assert e.demo_scenarios()["scenarios"][0]["id"] == "drone-sound"
    r = e.demo_trigger("drone-sound")
    assert r["ok"] and r["id"] == "drone-sound" and r["until"] > 0
    assert e.demo_active() == ["drone-sound"]
    run(e, 26)
    kinds = [x["kind"] for x in e.log]
    assert "detect" in kinds and "clear" in kinds
    det = next(x for x in e.log if x["kind"] == "detect")
    assert det["sim"] is True
    assert mesh.texts and mesh.texts[0].startswith("[SIM] ACOUSTIC: drone detected")
    uav = [m for m in cot_.messages if "A-M-F-Q" in m]
    assert uav and set(uids(uav)) == {"ACOUSTIC-MIC1-UAV-SIM"}     # never the real marker's uid
    assert all(remarks(m).startswith("SIMULATED: ") for m in uav)
    assert e.demo_active() == []                                   # ended on its own after ~20 s
    assert e.hyst.last_alert_ts is None                            # the real cooldown is untouched


def test_mic_stays_live_and_real_detections_stay_real(settings):
    e, cot_, mesh = make(settings)
    run(e, 8, signal=np.concatenate([noise(1.0), tone_comb(380.0, seconds=7.0, jitter=0.02)]))
    det = next(x for x in e.log if x["kind"] == "detect")
    assert "sim" not in det
    assert mesh.texts[0].startswith("ACOUSTIC: drone detected")
    assert "ACOUSTIC-MIC1-UAV" in uids(cot_.messages)
    sent = len(cot_.messages)
    r = e.demo_clear()                                             # never touches a real detection
    assert r == {"ok": True, "cleared": 0, "marker_removed": False}
    assert len(cot_.messages) == sent and e.hyst.active


def test_clear_stops_the_scenario_and_removes_its_marker(settings):
    e, cot_, mesh = make(settings)
    run(e, 3)
    run(e, 9, trigger_at=0)
    assert e.hyst.active and e._episode_sim
    sent = len(cot_.messages)
    r = e.demo_clear()
    assert r == {"ok": True, "cleared": 1, "marker_removed": True}
    assert e.demo_active() == [] and not e.hyst.active
    stale, delete = cot_.messages[sent:sent + 2]
    ev = ET.fromstring(stale.encode())
    assert ev.get("uid") == "ACOUSTIC-MIC1-UAV-SIM" and ev.get("stale") == ev.get("time")
    d = ET.fromstring(delete.encode())
    assert d.get("type") == "t-x-d-d" and d.find("detail/__forcedelete") is not None
    assert d.find("detail/link").get("uid") == "ACOUSTIC-MIC1-UAV-SIM"
    assert float(d.find("point").get("lat")) == settings.sensor_lat          # never 0,0
    assert next(x for x in e.log if x["kind"] == "clear")["sim"] is True
    # a second demo straight after alerts again (the demo never starts the cooldown)
    run(e, 3)
    n_detect = sum(x["kind"] == "detect" for x in e.log)
    run(e, 9, trigger_at=0)
    assert sum(x["kind"] == "detect" for x in e.log) == n_detect + 1


def test_drone_sound_uses_the_taught_drone(settings):
    e, _, _ = make(settings)
    e.calibration.add_clip(tone_comb(600.0, seconds=4.0, jitter=0.03), "drone")   # not the synth's ~375 Hz
    e.calibration.add_clip(noise(4.0), "background")
    run(e, 3)
    run(e, 12, trigger_at=0)
    assert any(x["kind"] == "detect" and x.get("sim") for x in e.log)
    assert e.state["components"]["calibration"] > 0.8


def test_signatures():
    sig = demo.drone_signature()
    assert len(sig) == int(demo.SIGNATURE_S * SAMPLE_RATE) and sig.dtype == np.float32
    assert np.array_equal(sig, demo.drone_signature())             # canned: same every time
    assert float(np.max(np.abs(sig))) < 0.9
    assert demo.room_scale(0.0, sig) == 1.0
    assert 1.0 < demo.room_scale(0.03, sig) <= 0.9 / float(np.max(np.abs(sig))) + 1e-6


# -- play-clip -------------------------------------------------------------------------------
def test_play_clip_needs_a_speaker_and_the_mic(settings):
    e, _, _ = make(settings, speaker=FakeSpeaker(available=False))
    assert [s["id"] for s in e.demo_scenarios()["scenarios"]] == ["drone-sound"]
    with pytest.raises(ValueError):
        e.demo_trigger("play-clip")
    spk = FakeSpeaker()
    e, _, _ = make(settings, speaker=spk)
    assert [s["id"] for s in e.demo_scenarios()["scenarios"]] == ["drone-sound", "play-clip"]
    assert e.demo_trigger("play-clip")["ok"] and spk.played == [demo.SIGNATURE_S]
    assert e.demo_active() == ["play-clip"] and e.source_kind == "simulate"   # what the mic hears is SIM
    assert e.demo_clear()["cleared"] == 1 and spk.stopped == 1
    e, _, _ = make(settings, source=SimSource(), speaker=FakeSpeaker())     # simulated input can't hear it
    assert "play-clip" not in [s["id"] for s in e.demo_scenarios()["scenarios"]]


class FakeSd:
    def __init__(self, devices, default_out):
        self._devices = devices
        self.default = type("D", (), {"device": [0, default_out]})()

    def query_devices(self, i=None):
        return self._devices if i is None else self._devices[i]


def test_speaker_selection_skips_hdmi():
    devs = [{"name": "USB mic", "max_output_channels": 0}, {"name": "vc4-hdmi-0", "max_output_channels": 2},
            {"name": "USB Audio speaker", "max_output_channels": 2}]
    assert demo.Speaker("auto", sd=FakeSd(devs, 1)).device() == 2
    assert demo.Speaker("auto", sd=FakeSd(devs[:2], 1)).device() is None        # HDMI only: no speaker
    assert demo.Speaker("off", sd=FakeSd(devs, 2)).device() is None
    assert demo.Speaker("1", sd=FakeSd(devs, 2)).device() == 1                   # explicit wins
    assert demo.Speaker("speaker", sd=FakeSd(devs, 1)).device() == 2


# -- outputs ---------------------------------------------------------------------------------
def test_heartbeat_carries_demo_fields(settings):
    e, _, _ = make(settings)
    hb = e.heartbeat()
    assert hb["demo_mode"] == "continuous" and hb["demo_active"] == []
    e.demo_trigger("drone-sound")
    hb = meshbus.heartbeat_payload(**e.heartbeat())
    assert hb["demo_active"] == ["drone-sound"] and "demo: drone-sound" in hb["detail"]
    assert "demo_mode" not in meshbus.heartbeat_payload("h", "u", "cpu", "ok")


def test_tak_force_skips_the_rate_limit_once():
    from acoustic.tak_client import TakClient

    t = [0.0]
    c = TakClient("127.0.0.1", 1, "acoustic-detector", "test", "ACOUSTIC", clock=lambda: t[0])
    xml = cot.detection_event("MIC1", 52.0, -1.0, 0.0, 0.9, 300, demo=True)
    assert c.send(xml) and not c.send(xml)
    assert c.send(cot.detection_event("MIC1", 52.0, -1.0, 0.0, 0.0, 300, demo=True, stale_s=0), force=True)


# -- HTTP + page -----------------------------------------------------------------------------
@pytest.fixture
def client(settings):
    from acoustic.gemini import GeminiChecker
    from acoustic.web import app as web

    web.engine = Engine(settings, cot_client=FakeCot(), mesh=FakeMesh(), source=MicLike(),
                        gemini=GeminiChecker(None, settings.gemini_model), speaker=FakeSpeaker(available=False))
    yield TestClient(web.app)
    web.engine = None


def test_demo_api(client):
    j = client.get("/api/demo/scenarios").json()
    assert j["mode"] == "continuous" and j["active"] == []
    assert j["scenarios"] == [{"id": "drone-sound", **demo.SCENARIOS["drone-sound"]}]
    assert set(j["scenarios"][0]) == {"id", "label", "hint", "duration_s"}
    assert client.post("/api/demo/trigger", json={"id": "nope"}).status_code == 404
    assert client.post("/api/demo/trigger", json={"id": "play-clip"}).status_code == 409
    r = client.post("/api/demo/trigger", json={"id": "drone-sound", "extra": 1}).json()
    assert r["ok"] and r["id"] == "drone-sound" and r["until"] > 0
    assert client.get("/api/demo/scenarios").json()["active"] == ["drone-sound"]
    s = client.get("/api/status").json()["demo"]
    assert s["labelled"] and "drone-sound" in s["until"]
    assert client.post("/api/demo/clear").json() == {"ok": True, "cleared": 1, "marker_removed": False}


def test_simulated_input_alone_is_not_a_demo(settings):
    e, _, _ = make(settings, source=SimSource(quiet=True))
    assert e.source_kind == "simulate" and e.status()["demo"]["labelled"] is False


def test_page_has_demo_card(client):
    html = client.get("/").text
    assert 'id="demo"' in html and ">Demo</h2>" in html and ">Clear demo</button>" in html
    assert "/api/demo/trigger" in html and "/api/demo/clear" in html
    assert html.count("isr-btn primary") == 1
