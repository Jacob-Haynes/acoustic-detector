"""YAMNet (AudioSet, 521 classes) on the Pi CPU via LiteRT.

AudioSet has no "drone" class. Small multirotors are usually tagged as some mix of the aircraft
classes and the insect/buzz classes, so both groups count toward the drone score (insects with a
lower weight). Engines, power tools and vehicles are the classic confusers and pull it down.
Class names below are verified against yamnet_class_map.csv.
"""

from __future__ import annotations

import csv
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from acoustic.config import WINDOW_SAMPLES

logger = logging.getLogger(__name__)

CLASS_MAP = Path(__file__).parent / "data" / "yamnet_class_map.csv"

DRONE_CLASSES: dict[str, float] = {
    "Aircraft": 1.0,
    "Aircraft engine": 1.0,
    "Propeller, airscrew": 1.0,
    "Helicopter": 0.9,
    "Fixed-wing aircraft, airplane": 0.6,
    "Light engine (high frequency)": 0.6,
    "Buzz": 0.7,
    "Insect": 0.6,
    "Bee, wasp, etc.": 0.6,
    "Mosquito": 0.6,
    "Fly, housefly": 0.5,
    "Whir": 0.5,
}

CONFUSER_CLASSES = [
    "Lawn mower",
    "Chainsaw",
    "Power tool",
    "Motorcycle",
    "Motor vehicle (road)",
    "Car",
    "Vehicle",
    "Medium engine (mid frequency)",
    "Heavy engine (low frequency)",
    "Engine knocking",
    "Hair dryer",
    "Vacuum cleaner",
    "Speech",
]


def load_class_names(path: Path = CLASS_MAP) -> list[str]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [r["display_name"] for r in rows]


@dataclass
class YamnetResult:
    score: float                       # 0..1 drone-likeness from drone-type class scores
    confuser: float                    # 0..1 strength of the strongest confuser class
    drone_mass: float
    confuser_mass: float
    top: list[tuple[str, float]]
    scores: np.ndarray                 # raw 521 class scores (used as a calibration embedding)

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "confuser": round(self.confuser, 3),
            "drone_mass": round(self.drone_mass, 3),
            "confuser_mass": round(self.confuser_mass, 3),
            "top": [[n, round(s, 3)] for n, s in self.top],
        }


def _load_interpreter(model_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter  # LiteRT (successor to tflite-runtime)
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except ImportError:
            return None, "no LiteRT runtime installed (pip install ai-edge-litert)"
    try:
        interp = Interpreter(model_path=str(model_path), num_threads=2)
        interp.allocate_tensors()
        return interp, None
    except Exception as e:  # noqa: BLE001 - surface any load failure as "unavailable"
        return None, f"failed to load {model_path}: {e}"


class Yamnet:
    def __init__(self, model_path: Path):
        self.available = False
        self.reason = ""
        self._lock = threading.Lock()   # the interpreter is not thread-safe
        self.class_names = load_class_names()
        self._drone_idx = {self.class_names.index(n): w for n, w in DRONE_CLASSES.items()}
        self._conf_idx = [self.class_names.index(n) for n in CONFUSER_CLASSES]
        if not Path(model_path).exists():
            self.reason = f"model not found at {model_path} (run scripts/fetch_model.sh)"
            logger.warning("YAMNet disabled: %s", self.reason)
            return
        self._interp, err = _load_interpreter(Path(model_path))
        if self._interp is None:
            self.reason = err or "unknown"
            logger.warning("YAMNet disabled: %s", self.reason)
            return
        self._in = self._interp.get_input_details()[0]["index"]
        self._out = self._interp.get_output_details()[0]["index"]
        self.available = True
        logger.info("YAMNet loaded from %s", model_path)

    def classify(self, window: np.ndarray) -> YamnetResult | None:
        if not self.available:
            return None
        x = np.asarray(window, dtype=np.float32)
        if len(x) < WINDOW_SAMPLES:
            x = np.pad(x, (0, WINDOW_SAMPLES - len(x)))
        x = np.clip(x[-WINDOW_SAMPLES:], -1.0, 1.0)
        with self._lock:
            self._interp.set_tensor(self._in, x)
            self._interp.invoke()
            scores = np.array(self._interp.get_tensor(self._out)).reshape(-1)
        return self.score(scores)

    def score(self, scores: np.ndarray) -> YamnetResult:
        drone = max(scores[i] * w for i, w in self._drone_idx.items())
        confuser = max(scores[i] for i in self._conf_idx)
        # Sigmoid outputs: ~0.3+ is already a confident call for AudioSet's rarer classes.
        value = float(np.clip(float(drone) / 0.35, 0.0, 1.0))
        conf_value = float(np.clip((float(confuser) - 0.15) / 0.35, 0.0, 1.0)) if confuser > drone else 0.0
        top_idx = np.argsort(scores)[::-1][:3]
        top = [(self.class_names[i], float(scores[i])) for i in top_idx]
        return YamnetResult(value, conf_value, float(drone), float(confuser), top, scores.astype(np.float32))
