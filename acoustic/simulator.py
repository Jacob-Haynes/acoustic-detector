"""Synthetic soundscape for --simulate: background, a drone pass, and two classic false-positive
sources (lawnmower, generator). Not realistic enough to tune thresholds on; good enough to show
the pipeline end to end on a laptop."""

from __future__ import annotations

import numpy as np

from acoustic.config import SAMPLE_RATE

# (label, seconds). The script loops.
SCENARIO = [
    ("background", 15.0),
    ("drone", 24.0),
    ("background", 10.0),
    ("lawnmower", 14.0),
    ("background", 8.0),
    ("generator", 12.0),
]


# DEMO_MODE=on_demand: --simulate plays only the quiet field, so nothing is detected on its own.
QUIET = [("background", 60.0)]


class Synth:
    noise_level = 0.004                  # quiet field background

    def __init__(self, seed: int = 1, sr: int = SAMPLE_RATE, scenario: list | None = None):
        self.sr = sr
        self.scenario = SCENARIO if scenario is None else scenario
        self.rng = np.random.default_rng(seed)
        self.t = 0                       # samples generated so far
        self._phases = np.zeros(256)     # oscillator phases, persistent across blocks
        self._pink = np.zeros(3)
        self.total = sum(d for _, d in self.scenario)

    # -- scenario bookkeeping ------------------------------------------------------------------
    def phase_at(self, seconds: float) -> tuple[str, float, float]:
        """(label, seconds into phase, phase length) for a time in the loop."""
        s = seconds % self.total
        for label, dur in self.scenario:
            if s < dur:
                return label, s, dur
            s -= dur
        return self.scenario[-1][0], 0.0, self.scenario[-1][1]

    @property
    def current_label(self) -> str:
        return self.phase_at(self.t / self.sr)[0]

    def drone_distance(self, frac: float) -> float:
        """Metres from the mic, frac = 0..1 through the drone phase: 150 m -> 8 m -> 150 m."""
        return 8.0 + 142.0 * abs(2 * frac - 1) ** 1.3

    # -- generators ----------------------------------------------------------------------------
    def _noise(self, n: int, level: float) -> np.ndarray:
        white = self.rng.standard_normal(n)
        # cheap pinkish noise: leaky integrator mix
        out = np.empty(n)
        b0, b1, b2 = self._pink
        for i in range(0, n, 256):
            w = white[i:i + 256]
            # vectorised enough for the block sizes we use
            b0 = 0.99 * b0 + 0.1 * w.mean()
            out[i:i + 256] = w * 0.6 + b0
        self._pink = np.array([b0, b1, b2])
        return out * level

    def _comb(self, n: int, f0s: list[float], amps: list[list[float]], slot: int) -> np.ndarray:
        t = np.arange(n) / self.sr
        out = np.zeros(n)
        idx = slot
        for f0, harm_amps in zip(f0s, amps):
            for k, a in enumerate(harm_amps, start=1):
                f = f0 * k
                if f >= self.sr / 2 - 100:
                    idx += 1
                    continue
                ph = self._phases[idx % 256]
                out += a * np.sin(2 * np.pi * f * t + ph)
                self._phases[idx % 256] = (ph + 2 * np.pi * f * n / self.sr) % (2 * np.pi)
                idx += 1
        return out

    def block(self, n: int) -> np.ndarray:
        label, into, dur = self.phase_at(self.t / self.sr)
        x = self._noise(n, self.noise_level)
        if label == "drone":
            dist = self.drone_distance(into / dur)                 # approach then recede
            gain = min(1.0, 6.0 / dist) * 0.25
            throttle = 1.0 + 0.06 * np.sin(2 * np.pi * self.t / self.sr / 3.1)
            base = [372.0, 381.0, 366.0, 377.0]                    # four motors, slightly unequal
            f0s = [b * throttle for b in base]
            amps = [[1.0 / k ** 0.8 for k in range(1, 11)] for _ in base]
            x += gain * 0.25 * self._comb(n, f0s, amps, slot=0)
            x += self._noise(n, gain * 0.15)                       # prop wash broadband
        elif label == "lawnmower":
            f0 = 28.0                                              # ~3400 rpm single cylinder
            amps = [[0.6 / (1 + 0.15 * k) for k in range(1, 40)]]
            x += 0.05 * self._comb(n, [f0], amps, slot=48)
            x += self._noise(n, 0.03)
        elif label == "generator":
            # 3000 rpm alternator: 50 Hz hum with odd-heavy harmonics + firing rate at 25 Hz
            amps = [[(0.8 if k % 2 else 0.3) / k for k in range(1, 30)]]
            x += 0.06 * self._comb(n, [25.0], amps, slot=48)
            x += self._noise(n, 0.012)
        self.t += n
        return np.clip(x, -1.0, 1.0).astype(np.float32)
