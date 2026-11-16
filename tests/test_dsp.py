import numpy as np

from acoustic import dsp
from acoustic.simulator import Synth
from conftest import SR, noise, tone_comb


def test_prop_comb_scores_high_and_finds_f0():
    f = dsp.spectral_features(tone_comb(380.0))
    assert f.score > 0.7
    assert abs(f.f0_hz - 380.0) < 10
    assert f.n_harmonics >= 5


def test_drifting_prop_tone_still_detected():
    # +-5% throttle wobble inside the window
    f = dsp.spectral_features(tone_comb(400.0, jitter=0.05))
    assert f.score > 0.6


def test_white_noise_scores_low():
    assert dsp.spectral_features(noise(level=0.05)).score < 0.1


def test_silence_is_gated():
    assert dsp.spectral_features(np.zeros(15600, dtype=np.float32)).score == 0.0
    quiet = tone_comb(380.0, level=1e-5)
    assert dsp.spectral_features(quiet).score < 0.05


def test_engine_firing_rate_comb_scores_low():
    # Lawnmower-like 28 Hz comb with dense harmonics: the comb sits below the prop band.
    s = Synth()
    s.t = int(55 * SR)  # inside the lawnmower phase
    assert s.current_label == "lawnmower"
    x = np.concatenate([s.block(1600) for _ in range(10)])
    assert dsp.spectral_features(x).score < 0.3


def test_resample_lengths():
    x = np.zeros(48000, dtype=np.float32)
    assert len(dsp.resample(x, 48000)) == 16000
    stereo = np.zeros((44100, 2), dtype=np.float32)
    assert len(dsp.resample(stereo, 44100)) == 16000


def test_waterfall_column_shape():
    col = dsp.waterfall_column(tone_comb(500.0, seconds=0.1)[: dsp.WF_NFFT])
    assert len(col) == dsp.WF_BINS
    assert max(col) > 100


def test_embedding_is_unit_and_loudness_invariant():
    a = dsp.spectral_embedding(tone_comb(380.0, level=0.05))
    b = dsp.spectral_embedding(tone_comb(380.0, level=0.2))
    assert abs(np.linalg.norm(a) - 1.0) < 1e-4
    assert dsp.cosine(a, b) > 0.95
