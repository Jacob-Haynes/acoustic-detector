import pytest

from acoustic.detector import Hysteresis, fuse


def test_fuse_spectral_only():
    assert fuse({"spectral": 0.9, "yamnet": None, "calibration": None}) == pytest.approx(0.9)


def test_yamnet_alone_cannot_trigger():
    assert fuse({"spectral": 0.0, "yamnet": 1.0, "calibration": None}) < 0.65


def test_yamnet_boosts_and_confuser_penalises():
    base = fuse({"spectral": 0.6})
    assert fuse({"spectral": 0.6, "yamnet": 0.8}) > base
    assert fuse({"spectral": 0.6, "confuser": 0.8}) < base


def test_calibration_averaged_with_spectral():
    assert fuse({"spectral": 0.4, "calibration": 1.0}) == pytest.approx(0.7)


def drive(h, values, t0=0.0, dt=0.5):
    events = []
    for i, v in enumerate(values):
        e = h.update(v, now=t0 + i * dt)
        if e:
            events.append(e)
    return events


def test_hysteresis_on_off():
    h = Hysteresis(0.65, 0.4, on_frames=2, off_frames=3, cooldown_s=60)
    ev = drive(h, [0.1] * 4 + [0.95] * 6)
    assert [e.kind for e in ev] == ["detect"]
    assert h.active
    # dipping between thresholds keeps it active
    assert drive(h, [0.55] * 10, t0=10) == []
    assert h.active
    ev = drive(h, [0.0] * 10, t0=20)
    assert [e.kind for e in ev] == ["clear"]
    assert not h.active


def test_single_spike_does_not_trigger():
    h = Hysteresis(0.65, 0.4, on_frames=2)
    assert drive(h, [0.1, 1.0, 0.1, 0.1, 0.1]) == []


def test_cooldown_suppresses_realert_then_allows():
    h = Hysteresis(0.65, 0.4, on_frames=2, off_frames=2, cooldown_s=60)
    kinds = [e.kind for e in drive(h, [1.0] * 4 + [0.0] * 6 + [1.0] * 4, t0=0)]
    assert kinds == ["detect", "clear", "suppressed"]
    kinds = [e.kind for e in drive(h, [0.0] * 6, t0=20) + drive(h, [1.0] * 4, t0=100)]
    assert kinds == ["clear", "detect"]


def test_bad_thresholds_rejected():
    with pytest.raises(ValueError):
        Hysteresis(0.3, 0.5)
