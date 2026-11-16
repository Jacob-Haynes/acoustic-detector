"""Signal processing: resampling, waterfall columns and the hand-built "drone-ness" spectral score.

A multirotor's sound is dominated by a harmonic comb: each motor/prop spins at a few hundred
revolutions per second, giving a fundamental (rotation rate x blade count) somewhere around
150-800 Hz for a 5" FPV quad, plus a ladder of harmonics up to several kHz. Petrol engines
(lawnmowers, generators) also produce combs, but their fundamental (firing rate) is far lower
(~20-60 Hz). The score below looks for a strong comb whose fundamental sits in the prop band.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from acoustic.config import SAMPLE_RATE

# ---------------------------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------------------------


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x


def resample(x: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    """Band-limited polyphase resample to sr_out."""
    x = to_mono(x)
    if sr_in == sr_out:
        return x.astype(np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Waterfall (spectrogram columns for the web UI)
# ---------------------------------------------------------------------------------------------

WF_NFFT = 1024
WF_HOP = 512
WF_BINS = 128
WF_FMIN = 50.0
WF_FMAX = 8000.0
WF_DB_FLOOR = -110.0
WF_DB_CEIL = -30.0


def _log_band_edges(n_bands: int, fmin: float, fmax: float, nfft: int, sr: int) -> np.ndarray:
    edges_hz = np.geomspace(fmin, fmax, n_bands + 1)
    edges = np.round(edges_hz / (sr / nfft)).astype(int)
    edges = np.clip(edges, 1, nfft // 2)
    # guarantee each band covers at least one bin
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    return np.minimum(edges, nfft // 2 + 1)


_WF_EDGES = _log_band_edges(WF_BINS, WF_FMIN, WF_FMAX, WF_NFFT, SAMPLE_RATE)
_WF_WINDOW = np.hanning(WF_NFFT).astype(np.float32)
_WF_NORM = float(np.sum(_WF_WINDOW) ** 2)


def waterfall_column(frame: np.ndarray) -> bytes:
    """One log-frequency spectrogram column (WF_BINS bytes, 0-255) from WF_NFFT samples."""
    spec = np.abs(np.fft.rfft(frame * _WF_WINDOW)) ** 2 / _WF_NORM * 4.0
    bands = np.maximum.reduceat(spec, _WF_EDGES[:-1])[:WF_BINS]
    db = 10.0 * np.log10(bands + 1e-14)
    scaled = (db - WF_DB_FLOOR) / (WF_DB_CEIL - WF_DB_FLOOR)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------------------------

FEAT_NFFT = 2048
FEAT_HOP = 1024
F0_MIN = 80.0          # Hz. Below this is engine firing-rate territory.
F0_MAX = 800.0
F0_PRIOR_LO = 150.0    # typical 5" FPV fundamental range (comb spacing)
F0_PRIOR_HI = 700.0
COMB_FMAX = 5000.0
N_HARMONICS = 8
BAND_LO = 150.0
BAND_HI = 4000.0


@dataclass
class SpectralFeatures:
    level_dbfs: float
    band_ratio: float      # share of energy in the prop band (150-4000 Hz)
    f0_hz: float           # best comb spacing found, 0 if none
    comb_db: float         # mean harmonic prominence above the spectral floor (dB)
    n_harmonics: int       # harmonics with >6 dB prominence
    flatness: float        # spectral flatness (1 = white noise, 0 = pure tones)
    score: float           # 0..1 hand-built "drone-ness"

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


def power_spectrum(x: np.ndarray, nfft: int = FEAT_NFFT, hop: int = FEAT_HOP) -> np.ndarray:
    """Welch-style averaged power spectrum (nfft//2+1 bins)."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    win = np.hanning(nfft).astype(np.float32)
    norm = float(np.sum(win) ** 2)
    n_frames = 1 + (len(x) - nfft) // hop
    acc = np.zeros(nfft // 2 + 1, dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * hop: i * hop + nfft] * win
        acc += np.abs(np.fft.rfft(seg)) ** 2
    return (acc / max(n_frames, 1) / norm * 4.0).astype(np.float64)


def _running_median(x: np.ndarray, width: int) -> np.ndarray:
    from scipy.ndimage import median_filter

    return median_filter(x, size=width, mode="nearest")


def spectral_features(x: np.ndarray, sr: int = SAMPLE_RATE, min_level_dbfs: float = -65.0) -> SpectralFeatures:
    x = np.asarray(x, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
    level = 20.0 * np.log10(rms)

    ps = power_spectrum(x)
    freqs = np.fft.rfftfreq(FEAT_NFFT, 1.0 / sr)
    df = freqs[1]

    usable = (freqs >= 20) & (freqs <= sr / 2 - 200)
    total = float(ps[usable].sum()) + 1e-20
    band = (freqs >= BAND_LO) & (freqs <= BAND_HI)
    band_ratio = float(ps[band].sum()) / total

    sub = ps[usable]
    flatness = float(np.exp(np.mean(np.log(sub + 1e-20))) / (np.mean(sub) + 1e-20))

    # Harmonic comb, matched per short frame (~128 ms) so a drifting prop tone (throttle changes)
    # doesn't smear the harmonics, then summarised with the median over frames.
    best_f0, best_comb, best_n = _comb_search(x, df)

    harm = np.clip((best_comb - 5.0) / 7.0, 0.0, 1.0)   # random noise scores ~4 dB here
    if best_n < 3:
        harm *= best_n / 3.0
    prior = 1.0 if F0_PRIOR_LO <= best_f0 <= F0_PRIOR_HI else 0.6
    level_gate = float(np.clip((level - min_level_dbfs) / 10.0, 0.0, 1.0))
    score = float(harm * prior * (0.4 + 0.6 * band_ratio) * level_gate)

    return SpectralFeatures(
        level_dbfs=float(level),
        band_ratio=band_ratio,
        f0_hz=best_f0 if harm > 0 else 0.0,
        comb_db=float(best_comb),
        n_harmonics=best_n,
        flatness=flatness,
        score=score,
    )


_F0_GRID = np.arange(F0_MIN, F0_MAX, 2.0)
_KS = np.arange(1, N_HARMONICS + 1)


def _comb_tables(df: float, n_bins: int):
    on = np.rint(np.outer(_F0_GRID, _KS) / df).astype(int)
    off = np.rint(np.outer(_F0_GRID, _KS - 0.5) / df).astype(int)
    valid = (np.outer(_F0_GRID, _KS) <= COMB_FMAX)
    return np.clip(on, 0, n_bins - 1), np.clip(off, 0, n_bins - 1), valid


_COMB_CACHE: dict = {}


def _comb_search(x: np.ndarray, df: float) -> tuple[float, float, int]:
    """Best comb spacing f0, its mean harmonic prominence (dB) and harmonic count."""
    from scipy.ndimage import maximum_filter1d

    n_bins = FEAT_NFFT // 2 + 1
    key = (round(df, 6), n_bins)
    if key not in _COMB_CACHE:
        _COMB_CACHE[key] = _comb_tables(df, n_bins)
    on_idx, off_idx, valid = _COMB_CACHE[key]
    n_valid = valid.sum(axis=1)
    usable = n_valid >= 3

    win = np.hanning(FEAT_NFFT).astype(np.float32)
    if len(x) < FEAT_NFFT:
        x = np.pad(x, (0, FEAT_NFFT - len(x)))
    frames = [x[i:i + FEAT_NFFT] for i in range(0, len(x) - FEAT_NFFT + 1, FEAT_HOP)]
    f0s, combs, counts = [], [], []
    for fr in frames:
        db = 10.0 * np.log10(np.abs(np.fft.rfft(fr * win)) ** 2 + 1e-20)
        floor = _running_median(db, width=max(5, int(round(120.0 / df)) | 1))
        prom = maximum_filter1d(np.maximum(db - floor, 0.0), size=3)   # +-1 bin tolerance
        on = np.where(valid, prom[on_idx], 0.0)
        off = np.where(valid, prom[off_idx], 0.0)
        mean_on = on.sum(axis=1) / np.maximum(n_valid, 1)
        mean_off = off.sum(axis=1) / np.maximum(n_valid, 1)
        # Peaks between the teeth mean the true comb is at f0/2 (e.g. an engine). Penalise.
        comb = np.where(usable, mean_on - 0.8 * mean_off, -np.inf)
        i = int(np.argmax(comb))
        f0s.append(_F0_GRID[i])
        combs.append(comb[i])
        counts.append(int(np.sum(on[i] > 6.0)))
    if not combs:
        return 0.0, 0.0, 0
    order = np.argsort(combs)
    mid = order[len(order) // 2]
    return float(np.median(f0s)), float(np.median(combs)), int(counts[mid])


# ---------------------------------------------------------------------------------------------
# Spectral embedding for calibration
# ---------------------------------------------------------------------------------------------

EMB_BANDS = 64
_EMB_EDGES = _log_band_edges(EMB_BANDS, 100.0, 7000.0, FEAT_NFFT, SAMPLE_RATE)


def spectral_embedding(x: np.ndarray) -> np.ndarray:
    """64-d log-band spectral shape, mean-removed and L2-normalised (loudness independent)."""
    ps = power_spectrum(x)
    bands = np.add.reduceat(ps, _EMB_EDGES[:-1])[:EMB_BANDS]
    v = 10.0 * np.log10(bands + 1e-14)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
