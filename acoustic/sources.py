"""Audio inputs. Each source calls feed(block) with float32 mono 16 kHz blocks until stopped."""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from acoustic import dsp
from acoustic.config import SAMPLE_RATE

logger = logging.getLogger(__name__)

BLOCK = 1600  # 100 ms at 16 kHz

Feed = Callable[[np.ndarray], None]


class SimSource:
    kind = "simulate"

    def __init__(self, seed: int = 1):
        from acoustic.simulator import Synth

        self.synth = Synth(seed=seed)
        self.description = "synthetic soundscape"

    @property
    def truth(self) -> str:
        return self.synth.current_label

    def run(self, feed: Feed, stop: threading.Event):
        next_t = time.monotonic()
        while not stop.is_set():
            feed(self.synth.block(BLOCK))
            next_t += BLOCK / SAMPLE_RATE
            delay = next_t - time.monotonic()
            if delay > 0:
                stop.wait(delay)


def load_audio_file(path: str | Path) -> np.ndarray:
    """Read WAV/FLAC/OGG/MP3 (libsndfile >= 1.1 handles MP3) and return 16 kHz mono float32."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return dsp.resample(audio, sr)


class FileSource:
    kind = "file"

    def __init__(self, path: str | Path, loop: bool = False):
        self.path = Path(path)
        self.loop = loop
        self.audio = load_audio_file(self.path)
        self.description = f"{self.path.name} ({len(self.audio) / SAMPLE_RATE:.1f} s{', looping' if loop else ''})"
        self.finished = False

    def run(self, feed: Feed, stop: threading.Event):
        next_t = time.monotonic()
        while not stop.is_set():
            for start in range(0, len(self.audio), BLOCK):
                if stop.is_set():
                    return
                block = self.audio[start:start + BLOCK]
                if len(block) < BLOCK:
                    block = np.pad(block, (0, BLOCK - len(block)))
                feed(block)
                next_t += BLOCK / SAMPLE_RATE
                delay = next_t - time.monotonic()
                if delay > 0:
                    stop.wait(delay)
            if not self.loop:
                break
        self.finished = True
        logger.info("File replay finished: %s", self.path)


class StreamResampler:
    """Continuous (stateful) downsampler, so 100 ms blocks don't click at their edges.
    Integer ratios (48k, 32k) use a persistent FIR + decimate; others fall back to per-block
    polyphase resampling."""

    def __init__(self, rate_in: int, rate_out: int = SAMPLE_RATE):
        from scipy.signal import firwin

        self.rate_in, self.rate_out = rate_in, rate_out
        self.factor = rate_in // rate_out if rate_in % rate_out == 0 else None
        if self.factor:
            self.taps = firwin(97, 0.9 * (rate_out / 2), fs=rate_in)
            self.zi = np.zeros(len(self.taps) - 1)
            self.offset = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        if not self.factor:
            return dsp.resample(block, self.rate_in, self.rate_out)
        from scipy.signal import lfilter

        y, self.zi = lfilter(self.taps, 1.0, block, zi=self.zi)
        out = y[self.offset::self.factor]
        self.offset = (self.offset - len(block)) % self.factor
        return out.astype(np.float32)


def pick_usb_input(devices) -> int | None:
    """Prefer the first USB capture device (the Pi 5 has no built-in mic); else system default."""
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0 and "usb" in str(d.get("name", "")).lower():
            return idx
    return None


def _parse_device(device: str | None):
    if device is None:
        return None
    try:
        return int(device)
    except ValueError:
        return device


class MicSource:
    """USB microphone via ALSA/PortAudio. Opens at 16 kHz if the device allows it, otherwise at the
    device's native rate and resamples."""

    kind = "mic"

    def __init__(self, device: str | None = None):
        import sounddevice as sd

        self.sd = sd
        self.device = _parse_device(device)
        if self.device is None:
            self.device = pick_usb_input(sd.query_devices())
        info = sd.query_devices(self.device, "input")
        self.device_name = info["name"]
        self.native_rate = int(info["default_samplerate"])
        self.rate = SAMPLE_RATE
        try:
            sd.check_input_settings(device=self.device, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        except Exception:  # noqa: BLE001
            self.rate = self.native_rate
        self.description = f"{self.device_name} @ {self.rate} Hz"
        self.overflows = 0

    def run(self, feed: Feed, stop: threading.Event):
        q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)

        def callback(indata, frames, t, status):
            if status:
                self.overflows += 1
            try:
                q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                self.overflows += 1

        blocksize = int(self.rate * BLOCK / SAMPLE_RATE)
        resampler = StreamResampler(self.rate) if self.rate != SAMPLE_RATE else None
        while not stop.is_set():
            try:
                with self.sd.InputStream(device=self.device, channels=1, samplerate=self.rate,
                                         dtype="float32", blocksize=blocksize, callback=callback):
                    logger.info("Microphone open: %s", self.description)
                    while not stop.is_set():
                        try:
                            block = q.get(timeout=1.0)
                        except queue.Empty:
                            continue
                        if resampler is not None:
                            block = resampler.process(block)
                        feed(block)
            except Exception as e:  # noqa: BLE001 - USB mic unplugged etc.: retry
                logger.error("Microphone error (%s); retrying in 3 s", e)
                stop.wait(3.0)
