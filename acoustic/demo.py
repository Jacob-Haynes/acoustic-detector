"""Step-by-step demo (ui-kit/DEMO-TRIGGER-CONTRACT.md): presenter-fired scenarios.

drone-sound  mixes a drone sound into the live input for ~20 s, so the real pipeline (waterfall,
             detector, CoT, mesh) reacts exactly as it would to a real drone. Everything it raises
             is labelled SIMULATED / [SIM]. The sound is the drone taught on this unit (its clips'
             rotor-tone stretches) when there is one, so "My drone" matches; else a synthetic one.
play-clip    plays the same sound from a speaker on this machine so the real mic hears it.
             Offered only when there is a speaker output and the input is the microphone.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from acoustic import dsp
from acoustic.config import SAMPLE_RATE
from acoustic.simulator import Synth

logger = logging.getLogger(__name__)

MODES = ("continuous", "on_demand")
SIGNATURE_S = 20.0
PLATEAU_RMS = 0.063          # the close-in stretch, about -24 dBFS: a drone 10 to 15 m from the mic
WINDOW_S = 0.975
# After a scenario ends, a new detection is still put down to the demo for this long: the last
# second of injected sound is still in the analysis window, and the room is still hearing the clip.
TAIL_S = 2.0

SCENARIOS = {
    "drone-sound": {"label": "Drone sound", "duration_s": SIGNATURE_S,
                    "hint": "Adds a drone sound to the live sound for 20 s: the drone taught here, or a "
                            "synthetic one. Its alert, TAK marker and mesh text are marked SIMULATED."},
    "play-clip": {"label": "Play drone clip", "duration_s": SIGNATURE_S,
                  "hint": "Plays a drone sound from this unit's speaker for 20 s, so the real mic hears it. "
                          "Its alerts are marked SIMULATED."},
}


class DemoSynth(Synth):
    """One drone phase: closes from 120 m in 4 s, holds at 10 to 14 m, leaves in the last 4 s."""
    noise_level = 0.0005

    def __init__(self, seconds: float = SIGNATURE_S, sr: int = SAMPLE_RATE, seed: int = 7):
        super().__init__(seed=seed, sr=sr, scenario=[("drone", seconds)])
        self.seconds = seconds

    def drone_distance(self, frac: float) -> float:
        t, edge = frac * self.seconds, 4.0
        if t < edge:
            return 120.0 - 108.0 * t / edge
        if t > self.seconds - edge:
            return 12.0 + 108.0 * (t - (self.seconds - edge)) / edge
        return 12.0 + 2.0 * np.sin(2 * np.pi * t / 5.0)


def drone_signature(seconds: float = SIGNATURE_S, sr: int = SAMPLE_RATE) -> np.ndarray:
    """The canned drone sound, deterministic (same every time), float32 mono at `sr`."""
    synth, block = DemoSynth(seconds, sr), sr // 10
    n = int(seconds * sr)
    out = np.concatenate([synth.block(block) for _ in range(-(-n // block))])[:n]
    return out.astype(np.float32)


def taught_signature(sample_dir: Path, seconds: float = SIGNATURE_S, min_comb: float = 0.5) -> np.ndarray | None:
    """The taught drone (samples/drone/*.wav): its rotor-tone windows end to end, tiled to `seconds`,
    faded in and out like a pass, at the synthetic sound's level. None with under 2 s of tone."""
    import soundfile as sf

    from acoustic.calibration import windows

    parts = []
    for wav in sorted(Path(sample_dir).glob("*.wav")):
        try:
            audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        except Exception as e:  # noqa: BLE001
            logger.debug("Skipping %s: %s", wav, e)
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        parts += [w for w in windows(dsp.resample(audio, sr)) if dsp.spectral_features(w).score >= min_comb]
    if len(parts) * WINDOW_S < 2.0:
        return None
    x = np.concatenate(parts)
    n = int(seconds * SAMPLE_RATE)
    x = np.tile(x, -(-n // len(x)))[:n].astype(np.float32)
    t = np.arange(n) / SAMPLE_RATE
    env = np.clip(np.minimum(t, seconds - t) / 4.0, 0.08, 1.0)       # 4 s in, hold, 4 s out
    x = x * env
    loud = float(np.sqrt(np.mean(x[n // 3: 2 * n // 3] ** 2))) or 1.0
    x = x * (PLATEAU_RMS / loud)
    peak = float(np.max(np.abs(x)))
    return (x * (0.9 / peak) if peak > 0.9 else x).astype(np.float32)


def room_scale(room_rms: float, sig: np.ndarray) -> float:
    """Gain for the injected signature: its loudest stretch at least 4x the room's RMS, never
    quieter than as made, and never clipping."""
    loud = float(np.sqrt(np.mean(sig[len(sig) // 3: 2 * len(sig) // 3] ** 2)))
    peak = float(np.max(np.abs(sig))) or 1.0
    return float(np.clip(4.0 * room_rms / max(loud, 1e-6), 1.0, 0.9 / peak))


def _is_hdmi(name: str) -> bool:
    n = name.lower()
    return "hdmi" in n or "vc4" in n                   # Pi HDMI outputs exist with no speaker on them


class Speaker:
    """A speaker output for play-clip. DEMO_SPEAKER: "auto" (default: the first output that isn't
    HDMI), "off", or a device index or name substring."""

    def __init__(self, setting: str | None = "auto", sd=None):
        self.setting = (setting or "auto").strip()
        self._sd = sd
        self._cache: tuple[float, object] | None = None
        self._lock = threading.Lock()

    def _module(self):
        if self._sd is None:
            import sounddevice as sd

            self._sd = sd
        return self._sd

    def device(self):
        """Index of the speaker to use, or None. Cached for 30 s (querying ALSA is not free)."""
        if self.setting.lower() == "off":
            return None
        now = time.monotonic()
        if self._cache is not None and now - self._cache[0] < 30.0:
            return self._cache[1]
        found = None
        try:
            sd = self._module()
            outs = [(i, d) for i, d in enumerate(sd.query_devices()) if d.get("max_output_channels", 0) > 0]
            if self.setting.lower() == "auto":
                default = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None
                ordered = sorted(outs, key=lambda x: x[0] != default)
                found = next((i for i, d in ordered if not _is_hdmi(d.get("name", ""))), None)
            elif self.setting.isdigit():
                found = next((i for i, _ in outs if i == int(self.setting)), None)
            else:
                found = next((i for i, d in outs if self.setting.lower() in d.get("name", "").lower()), None)
        except Exception as e:  # noqa: BLE001 - no audio stack, no speaker
            logger.debug("No speaker output: %s", e)
        self._cache = (now, found)
        return found

    @property
    def available(self) -> bool:
        return self.device() is not None

    def play(self, sig: np.ndarray, sr: int = SAMPLE_RATE) -> bool:
        """Play `sig` (mono float32 at `sr`) on the speaker, without blocking."""
        dev = self.device()
        if dev is None:
            return False
        sd = self._module()
        try:
            rate = int(sd.query_devices(dev).get("default_samplerate") or 48000)
            n = int(len(sig) * rate / sr)
            clip = np.interp(np.arange(n) * sr / rate, np.arange(len(sig)), sig).astype(np.float32)
            clip = 0.5 * clip / max(1e-6, float(np.max(np.abs(clip))))
            with self._lock:
                sd.play(clip, rate, device=dev)       # non-blocking, its own output stream
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not play the drone clip: %s", e)
            return False

    def stop(self):
        if self._sd is None:
            return
        try:
            with self._lock:
                self._sd.stop()
        except Exception:  # noqa: BLE001
            pass
