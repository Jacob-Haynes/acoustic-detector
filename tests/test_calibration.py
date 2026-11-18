import numpy as np

from acoustic.calibration import Calibration
from conftest import noise, tone_comb


def test_enrol_and_score(tmp_path):
    cal = Calibration(tmp_path)
    assert cal.score(tone_comb()) is None
    for f0 in (370.0, 385.0, 400.0):
        cal.add_clip(tone_comb(f0, seconds=3.0), "drone")
    assert cal.enrolled and cal.counts["drone"] == 3
    assert cal.last_warning is None
    cal.add_clip(noise(seconds=3.0), "background")
    drone = cal.score(tone_comb(380.0))["score"]
    bg = cal.score(noise(seed=7))["score"]
    assert drone > 0.7
    assert bg < 0.3


def test_profile_persists_and_clears(tmp_path):
    cal = Calibration(tmp_path)
    cal.add_clip(tone_comb(seconds=2.0), "drone")
    again = Calibration(tmp_path)
    assert again.enrolled and again.counts["drone"] == 1
    again.clear()
    assert not Calibration(tmp_path).enrolled
    assert not list((tmp_path / "samples" / "drone").glob("*.wav"))


def test_warns_on_drone_clip_without_prop_tone(tmp_path):
    cal = Calibration(tmp_path)
    cal.add_clip(noise(seconds=2.0), "drone")
    assert cal.last_warning and "No prop tone" in cal.last_warning
    assert cal.enrolled  # still usable as a fallback


def test_resamples_input(tmp_path):
    cal = Calibration(tmp_path)
    x48 = np.repeat(tone_comb(seconds=2.0), 3)  # crude 48k version
    path = cal.add_clip(x48, "drone", sr=48000)
    import soundfile as sf

    assert sf.info(str(path)).samplerate == 16000
