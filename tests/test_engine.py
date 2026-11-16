import time

import numpy as np

from acoustic.config import SAMPLE_RATE
from acoustic.engine import Engine
from acoustic.simulator import SCENARIO, Synth
from conftest import FakeCot, FakeMesh, tone_comb


def run_sim(engine, seconds):
    synth = Synth()
    for i in range(int(seconds * 2)):
        engine.feed(synth.block(SAMPLE_RATE // 2))
        if i >= 2:
            engine.analyze_window(engine.tail(0.975), now=1000.0 + i * 0.5)
    return synth


def test_simulated_drone_pass_alerts_once(settings):
    cot, mesh = FakeCot(), FakeMesh()
    e = Engine(settings, cot_client=cot, mesh=mesh)
    run_sim(e, sum(d for _, d in SCENARIO))     # one full loop
    kinds = [x["kind"] for x in e.log]
    assert kinds.count("detect") == 1, kinds
    assert mesh.texts and mesh.texts[0].startswith("ACOUSTIC: drone detected conf ")
    assert mesh.texts[0].endswith("@ OP1")
    assert len(mesh.texts) == 1                  # no alerts for lawnmower / generator
    assert any("a-u-A-M-F-Q" in m for m in cot.messages)
    assert (settings.data_dir / "detections.jsonl").exists()


def test_no_position_means_no_cot(settings):
    for lat, lon in ((None, None), (0.0, 0.0)):        # C4.2: unset or 0,0 never drawn
        settings.sensor_lat, settings.sensor_lon = lat, lon
        cot = FakeCot()
        e = Engine(settings, cot_client=cot, mesh=FakeMesh())
        run_sim(e, 40)
        assert cot.messages == []


def test_simulated_outputs_are_labelled_c43(settings):
    from acoustic.sources import SimSource
    import xml.etree.ElementTree as ET

    cot, mesh = FakeCot(), FakeMesh()
    e = Engine(settings, cot_client=cot, mesh=mesh, source=SimSource())   # not started: we feed it
    run_sim(e, sum(d for _, d in SCENARIO))
    assert mesh.texts and all(t.startswith("[SIM] ACOUSTIC: ") for t in mesh.texts)
    remarks = [ET.fromstring(m.encode()).find("detail/remarks").text for m in cot.messages]
    assert remarks and all(r.startswith("SIMULATED: ") for r in remarks)
    assert any("Acoustic drone sensor" in r for r in remarks)               # sensor marker too
    assert e.test_alert()["mesh_sent"] and mesh.texts[-1].startswith("[SIM] ACOUSTIC: TEST")


def test_ring_buffer_and_waterfall(settings):
    e = Engine(settings)
    e.feed(np.ones(SAMPLE_RATE, dtype=np.float32))
    assert len(e.tail(0.5)) == SAMPLE_RATE // 2
    cols = e.take_waterfall()
    assert len(cols) == SAMPLE_RATE // 512
    assert e.take_waterfall() == []
    # wrap around the 30 s ring
    for _ in range(35):
        e.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert len(e.tail(40)) == 30 * SAMPLE_RATE


def test_record_sample_enrols(settings):
    e = Engine(settings)
    assert e.start_recording("drone", seconds=1.0)
    assert not e.start_recording("drone", seconds=1.0)  # one at a time
    clip = tone_comb(seconds=1.2)
    for i in range(0, len(clip), 1600):
        e.feed(clip[i:i + 1600])
    for _ in range(50):
        if e.last_record:
            break
        time.sleep(0.05)
    assert e.last_record and e.last_record["ok"]
    assert e.calibration.counts["drone"] == 1
    assert e.recording_status() is None


def test_test_alert(settings):
    cot, mesh = FakeCot(), FakeMesh()
    e = Engine(settings, cot_client=cot, mesh=mesh)
    r = e.test_alert()
    assert r == {"mesh_sent": True, "cot_queued": True}
    assert "TEST" in mesh.texts[0]


def test_heartbeat_fields(settings):
    from acoustic.sources import SimSource

    e = Engine(settings, source=SimSource())
    hb = e.heartbeat()
    assert hb["mode"] == "simulate" and hb["status"] == "ok"
    assert hb["url"].endswith(":8100")
    assert Engine(settings).heartbeat()["status"] == "error"      # no audio input at all
