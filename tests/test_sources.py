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
