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


def speech_like(seconds=1.0, seed=0):
    """Voiced syllables (~200 Hz pitch, wandering intonation) separated by short gaps."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = np.zeros(n)
    pos = 0
    while pos < n:
        syl = int(rng.uniform(0.12, 0.25) * SR)
        gap = int(rng.uniform(0.04, 0.12) * SR)
        t = np.arange(syl) / SR
        f0 = rng.uniform(170, 240) * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(2, 5) * t))
        ph = 2 * np.pi * np.cumsum(f0) / SR
        seg = sum(np.sin(k * ph) / k for k in range(1, 12)) * np.hanning(syl)
        end = min(n, pos + syl)
        x[pos:end] += seg[: end - pos]
        pos += syl + gap
    return (0.05 * x / np.max(np.abs(x))).astype(np.float32)


def test_speech_like_scores_mostly_low():
    scores = [dsp.spectral_features(speech_like(seed=s)).score for s in range(20)]
    assert np.median(scores) < 0.35
    assert dsp.spectral_features(tone_comb(200.0)).score > 0.7


def test_speech_like_stream_does_not_trigger_without_yamnet(settings):
    from acoustic.engine import Engine

    e = Engine(settings)   # spectral-only: the hardest case
    x = speech_like(seconds=30.0, seed=11)
    for i in range(0, len(x) - 8000, 8000):
        e.feed(x[i:i + 8000])
        if i >= 8000:
            e.analyze_window(e.tail(0.975), now=i / SR)
    assert [ev["kind"] for ev in e.log] == []


def test_keyboard_and_aircon_score_low():
    rng = np.random.default_rng(5)
    clicks = rng.standard_normal(15600) * 0.002
    for start in range(500, 15000, 2600):
        clicks[start:start + 400] += rng.standard_normal(400) * np.exp(-np.arange(400) / 60) * 0.2
    assert dsp.spectral_features(clicks.astype(np.float32)).score < 0.1
    from scipy.signal import butter, lfilter

    t = np.arange(15600) / SR
    b, a = butter(2, [60 / (SR / 2), 3000 / (SR / 2)], "band")
    hvac = lfilter(b, a, rng.standard_normal(15600)) * 0.03 + 0.01 * np.sin(2 * np.pi * 100 * t)
    assert dsp.spectral_features(hvac.astype(np.float32)).score < 0.1
