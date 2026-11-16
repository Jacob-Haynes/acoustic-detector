import threading

import numpy as np
import soundfile as sf

from acoustic.sources import FileSource, StreamResampler, load_audio_file


def test_file_source_resamples_and_replays(tmp_path):
    path = tmp_path / "clip.wav"
    stereo = np.zeros((int(44100 * 0.3), 2), dtype=np.float32)
    sf.write(str(path), stereo, 44100)
    assert len(load_audio_file(path)) == int(16000 * 0.3)
    src = FileSource(path)
    blocks = []
    src.run(blocks.append, threading.Event())
    assert src.finished
    assert sum(len(b) for b in blocks) >= int(16000 * 0.3)
    assert all(len(b) == 1600 for b in blocks)


def test_stream_resampler_continuous():
    t = np.arange(48000) / 48000
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    r = StreamResampler(48000)
    out = np.concatenate([r.process(x[i:i + 4800]) for i in range(0, len(x), 4800)])
    assert len(out) == 16000
    # no clicks at block joins: sample-to-sample steps stay small after the filter settles
    assert np.max(np.abs(np.diff(out[200:]))) < 0.25


def test_mp3_replay(tmp_path):
    path = tmp_path / "clip.mp3"
    t = np.arange(44100) / 44100
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * 400 * t)).astype(np.float32), 44100, format="MP3")
    assert abs(len(load_audio_file(path)) - 16000) < 1600


def test_pick_usb_input():
    from acoustic.sources import pick_usb_input

    devs = [{"name": "vc4-hdmi-0", "max_input_channels": 0},
            {"name": "USB PnP Sound Device: Audio (hw:2,0)", "max_input_channels": 1}]
    assert pick_usb_input(devs) == 1
    assert pick_usb_input(devs[:1]) is None
    alsa = [{"name": "default", "max_input_channels": 32},
            {"name": "Yeti Nano: Audio (hw:3,0)", "max_input_channels": 2}]
    assert pick_usb_input(alsa) == 1


def test_prefers_yeti_by_name_over_other_inputs():
    from acoustic.sources import pick_usb_input

    mac = [{"name": "MacBook Pro Microphone", "max_input_channels": 1},
           {"name": "HD Pro Webcam C920", "max_input_channels": 2},
           {"name": "Yeti Nano", "max_input_channels": 2}]
    assert pick_usb_input(mac) == 2
    alsa = [{"name": "USB PnP Sound Device: Audio (hw:1,0)", "max_input_channels": 1},
            {"name": "Yeti Nano: USB Audio (hw:2,0)", "max_input_channels": 2},
            {"name": "default", "max_input_channels": 32}]
    assert pick_usb_input(alsa) == 1
    assert pick_usb_input(alsa, prefer=()) == 0
    assert pick_usb_input(alsa, prefer=("pnp",)) == 0


class _FakeSD:
    """sounddevice stand-in: an ALSA hw: device that only accepts 48 kHz stereo."""

    def __init__(self, ok=((48000, 2),)):
        self.ok = ok

    def query_devices(self, device=None, kind=None):
        devs = [{"name": "Yeti Nano: USB Audio (hw:2,0)", "max_input_channels": 2,
                 "default_samplerate": 48000.0}]
        return devs if device is None else devs[0]

    def check_input_settings(self, device=None, samplerate=None, channels=None, dtype=None):
        if (samplerate, channels) not in self.ok:
            raise ValueError("Invalid sample rate / channels")


def test_negotiates_48k_stereo_on_alsa_hw(monkeypatch):
    import sys

    from acoustic.sources import MicSource

    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSD())
    m = MicSource(None)
    assert m.device_name.startswith("Yeti Nano")
    assert (m.rate, m.channels) == (48000, 2)
    assert "resampled to 16000" in m.description and "downmixed" in m.description
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSD(ok=((16000, 1), (48000, 2))))
    assert (MicSource(None).rate, MicSource(None, rate=48000).rate) == (16000, 48000)
