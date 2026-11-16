import numpy as np
import pytest

from acoustic.config import SAMPLE_RATE, Settings

SR = SAMPLE_RATE


def tone_comb(f0=380.0, seconds=1.0, harmonics=10, level=0.05, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        if k * f0 < SR / 2 - 100:
            x += np.sin(2 * np.pi * k * f0 * t * (1 + jitter * np.sin(2 * np.pi * 0.7 * t))) / k ** 0.8
    x = level * x / np.max(np.abs(x)) + rng.standard_normal(len(t)) * level * 0.05
    return x.astype(np.float32)


def noise(seconds=1.0, level=0.02, seed=1):
    return (np.random.default_rng(seed).standard_normal(int(seconds * SR)) * level).astype(np.float32)


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.data_dir = tmp_path
    s.sensor_lat, s.sensor_lon = 51.15, -1.75
    s.use_yamnet = False
    return s


class FakeCot:
    connected = True
    sent = 0

    def __init__(self):
        self.messages = []

    def send(self, xml):
        self.messages.append(xml)
        self.sent += 1


class FakeMesh:
    connected = True
    sent = 0

    def __init__(self):
        self.texts = []

    def send(self, text):
        self.texts.append(text)
        self.sent += 1
        return True
