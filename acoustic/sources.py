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


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".m4a", ".aac"}
MAX_FILE_SECONDS = 600


class NoAudioTrack(ValueError):
    pass


def _load_with_av(path: Path, max_seconds: float) -> np.ndarray:
    """Decode the first audio stream of a video/audio container with PyAV (bundled ffmpeg)."""
    import av

    with av.open(str(path)) as container:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            raise NoAudioTrack(f"{path.name} has no audio track")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        chunks, total, limit = [], 0, int(max_seconds * SAMPLE_RATE)
        for frame in container.decode(streams[0]):
            for out in resampler.resample(frame):
                a = out.to_ndarray().reshape(-1)
                chunks.append(a)
                total += len(a)
            if total >= limit:
                break
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise NoAudioTrack(f"{path.name}: audio track decoded to nothing")
    return np.concatenate(chunks)[:limit].astype(np.float32)


def _load_with_ffmpeg(path: Path, max_seconds: float) -> np.ndarray:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("no decoder: pip install av, or install ffmpeg")
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
           "-t", str(max_seconds), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode != 0 or not r.stdout:
        msg = r.stderr.decode(errors="replace").strip()
        if "does not contain any stream" in msg or "matches no streams" in msg or not r.stdout:
            raise NoAudioTrack(f"{path.name} has no audio track")
        raise RuntimeError(f"ffmpeg failed: {msg[:200]}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def load_audio_file(path: str | Path, max_seconds: float = MAX_FILE_SECONDS) -> np.ndarray:
    """16 kHz mono float32 from WAV/FLAC/OGG/MP3 (libsndfile) or the audio track of a video
    (MP4/MOV, e.g. GoPro or DJI footage) via PyAV, falling back to the ffmpeg CLI."""
    path = Path(path)
    if path.suffix.lower() not in VIDEO_EXTS:
        try:
            import soundfile as sf

            audio, sr = sf.read(str(path), dtype="float32", always_2d=False,
                                frames=int(max_seconds * sf.info(str(path)).samplerate))
            return dsp.resample(audio, sr)
        except Exception:  # noqa: BLE001 - fall through to the container decoders
            pass
    try:
        return _load_with_av(path, max_seconds)
    except NoAudioTrack:
        raise
    except ImportError:
        return _load_with_ffmpeg(path, max_seconds)
    except Exception as e:  # noqa: BLE001
        logger.debug("PyAV failed on %s (%s); trying ffmpeg", path, e)
        return _load_with_ffmpeg(path, max_seconds)


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
    """Prefer the first USB capture device (the Pi 5 has no built-in mic); then the first real
    ALSA card ("... (hw:2,0)"), since some USB mics don't say USB in their name; else default."""
    inputs = [(i, str(d.get("name", ""))) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
    for idx, name in inputs:
        if "usb" in name.lower():
            return idx
    for idx, name in inputs:
        if "(hw:" in name and "hdmi" not in name.lower():
            return idx
    return None


def _parse_device(device: str | None):
    if device is None:
        return None
    try:
        return int(device)
    except ValueError:
        return device


def stable_device_name(name: str) -> str:
    import re

    return re.sub(r"\s*\(hw:\d+,\d+\)\s*$", "", name).strip() or name


def list_input_devices() -> list[dict]:
    import sounddevice as sd

    out = []
    for idx, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append({"index": idx, "name": d["name"], "channels": d["max_input_channels"],
                        "default_rate": int(d.get("default_samplerate", 0))})
    return out


class MicSource:
    """USB microphone via ALSA/PortAudio. Opens at 16 kHz if the device allows it, otherwise at the
    device's native rate and resamples. The device can be switched live from the UI."""

    kind = "mic"

    def __init__(self, device: str | int | None = None):
        import sounddevice as sd

        self.sd = sd
        self.error: str | None = None
        self.overflows = 0
        self._reopen = threading.Event()
        self.requested = _parse_device(device) if not isinstance(device, int) else device
        self._configure(self.requested)

    def _configure(self, device):
        sd = self.sd
        if device is None:
            device = pick_usb_input(sd.query_devices())
        info = sd.query_devices(device, "input")
        self.device = device
        self.device_name = info["name"]
        self.native_rate = int(info["default_samplerate"])
        self.rate = SAMPLE_RATE
        try:
            sd.check_input_settings(device=device, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        except Exception:  # noqa: BLE001
            self.rate = self.native_rate
        self.description = f"{self.device_name} @ {self.rate} Hz"

    def switch(self, device) -> str:
        """Change input device (index or name). Raises if the device can't be used."""
        dev = _parse_device(device) if not isinstance(device, int) else device
        self._configure(dev)
        # Remember it by name, minus ALSA's "(hw:2,0)" suffix, so a replug onto another card
        # number still finds it (sounddevice matches names by substring).
        self.requested = stable_device_name(self.device_name)
        self._reopen.set()
        return self.description

    def run(self, feed: Feed, stop: threading.Event):
        while not stop.is_set():
            q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)

            def callback(indata, frames, t, status):
                if status:
                    self.overflows += 1
                try:
                    q.put_nowait(indata[:, 0].copy())
                except queue.Full:
                    self.overflows += 1

            rate, device = self.rate, self.device
            blocksize = int(rate * BLOCK / SAMPLE_RATE)
            resampler = StreamResampler(rate) if rate != SAMPLE_RATE else None
            self._reopen.clear()
            try:
                with self.sd.InputStream(device=device, channels=1, samplerate=rate,
                                         dtype="float32", blocksize=blocksize, callback=callback):
                    logger.info("Microphone open: %s", self.description)
                    self.error = None
                    last_audio = time.monotonic()
                    while not stop.is_set() and not self._reopen.is_set():
                        try:
                            block = q.get(timeout=1.0)
                        except queue.Empty:
                            if time.monotonic() - last_audio > 5:
                                raise RuntimeError("no audio from device for 5 s")
                            continue
                        last_audio = time.monotonic()
                        if resampler is not None:
                            block = resampler.process(block)
                        feed(block)
            except Exception as e:  # noqa: BLE001 - USB mic unplugged etc.: retry
                self.error = str(e)[:200]
                logger.error("Microphone error (%s); retrying in 3 s", e)
                stop.wait(3.0)
                try:   # the card number may have changed after a replug
                    self._configure(self.requested)
                except Exception:  # noqa: BLE001
                    pass
