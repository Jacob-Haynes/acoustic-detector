"""Teach the sensor your own drone: enrol a few clips, keep a mean reference embedding, score live
audio by cosine similarity. Optionally enrol background clips too, so the score becomes "closer
to my drone than to this field's normal noise".

Two embeddings are kept when available:
  * spectral: 64 log-band spectral shape (always available, no ML)
  * yamnet:   the 521 YAMNet class-score vector (when the model is loaded)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from acoustic import dsp
from acoustic.config import SAMPLE_RATE, WINDOW_SAMPLES

logger = logging.getLogger(__name__)

LABELS = ("drone", "background")
HOP = WINDOW_SAMPLES // 2
MIN_DRONE_COMB = 0.25    # drone clips only contribute windows where a prop comb is visible


def windows(x: np.ndarray, min_level_dbfs: float = -65.0):
    """Analysis windows of a clip, skipping near-silent ones."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < WINDOW_SAMPLES:
        x = np.pad(x, (0, WINDOW_SAMPLES - len(x)))
    for start in range(0, len(x) - WINDOW_SAMPLES + 1, HOP):
        w = x[start:start + WINDOW_SAMPLES]
        level = 20 * np.log10(np.sqrt(np.mean(w.astype(np.float64) ** 2)) + 1e-12)
        if level >= min_level_dbfs:
            yield w


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class Calibration:
    def __init__(self, data_dir: Path, yamnet=None):
        self.dir = Path(data_dir) / "samples"
        self.profile_path = Path(data_dir) / "profile.json"
        self.yamnet = yamnet
        self._lock = threading.Lock()
        self.ref: dict[str, dict[str, np.ndarray]] = {}
        self.counts: dict[str, int] = {label: 0 for label in LABELS}
        self.last_warning: str | None = None
        self.last_stats: dict | None = None
        for label in LABELS:
            (self.dir / label).mkdir(parents=True, exist_ok=True)
        self._load()

    # -- persistence ---------------------------------------------------------------------------
    def _load(self):
        if not self.profile_path.exists():
            return
        try:
            data = json.loads(self.profile_path.read_text())
            self.counts = {label: int(data.get("counts", {}).get(label, 0)) for label in LABELS}
            self.ref = {
                label: {k: np.array(v, dtype=np.float32) for k, v in vecs.items()}
                for label, vecs in data.get("ref", {}).items()
            }
        except (OSError, ValueError) as e:
            logger.warning("Ignoring unreadable calibration profile: %s", e)

    def _save(self):
        data = {
            "updated": time.time(),
            "counts": self.counts,
            "ref": {label: {k: v.tolist() for k, v in vecs.items()} for label, vecs in self.ref.items()},
        }
        self.profile_path.write_text(json.dumps(data))

    # -- enrolment -----------------------------------------------------------------------------
    def add_clip(self, audio: np.ndarray, label: str, sr: int = SAMPLE_RATE) -> Path:
        import soundfile as sf

        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        if sr != SAMPLE_RATE:
            audio = dsp.resample(audio, sr)
        path = self.dir / label / f"{label}-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.wav"
        audio = np.asarray(audio, dtype=np.float32)
        self.last_warning = None
        ws = list(windows(audio))
        good = sum(1 for w in ws if dsp.spectral_features(w).score >= MIN_DRONE_COMB)
        self.last_stats = {"seconds": round(len(audio) / SAMPLE_RATE, 1), "windows": len(ws),
                           "prop_tone_windows": good,
                           "prop_tone_fraction": round(good / len(ws), 2) if ws else 0.0}
        if label == "drone":
            if not good:
                self.last_warning = ("No prop tone found in that clip. Saved, but it only counts if the "
                                     "drone is audible; check distance/throttle or delete and retry.")
        sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
        self.rebuild()
        return path

    def rebuild(self):
        """Recompute reference embeddings from every enrolled clip on disk."""
        import soundfile as sf

        ref: dict[str, dict[str, np.ndarray]] = {}
        counts = {label: 0 for label in LABELS}
        for label in LABELS:
            spec_vecs, yam_vecs = [], []
            fallback: list[np.ndarray] = []
            for wav in sorted((self.dir / label).glob("*.wav")):
                try:
                    audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Skipping %s: %s", wav, e)
                    continue
                audio = dsp.resample(audio, sr)
                counts[label] += 1
                for w in windows(audio):
                    if label == "drone" and dsp.spectral_features(w).score < MIN_DRONE_COMB:
                        fallback.append(w)
                        continue
                    spec_vecs.append(dsp.spectral_embedding(w))
                    if self.yamnet is not None and self.yamnet.available:
                        r = self.yamnet.classify(w)
                        if r is not None:
                            yam_vecs.append(_unit(np.sqrt(r.scores)))
            if label == "drone" and not spec_vecs and fallback:
                # nothing comb-like enrolled: fall back to the raw clips rather than nothing
                for w in fallback:
                    spec_vecs.append(dsp.spectral_embedding(w))
                    if self.yamnet is not None and self.yamnet.available:
                        r = self.yamnet.classify(w)
                        if r is not None:
                            yam_vecs.append(_unit(np.sqrt(r.scores)))
            vecs = {}
            if spec_vecs:
                vecs["spectral"] = _unit(np.mean(spec_vecs, axis=0)).astype(np.float32)
            if yam_vecs:
                vecs["yamnet"] = _unit(np.mean(yam_vecs, axis=0)).astype(np.float32)
            if vecs:
                ref[label] = vecs
        with self._lock:
            self.ref, self.counts = ref, counts
            self._save()
        logger.info("Calibration rebuilt: %s", counts)

    def clear(self):
        for label in LABELS:
            for wav in (self.dir / label).glob("*.wav"):
                wav.unlink()
        with self._lock:
            self.ref = {}
            self.counts = {label: 0 for label in LABELS}
            self._save()

    # -- scoring -------------------------------------------------------------------------------
    @property
    def enrolled(self) -> bool:
        return "drone" in self.ref

    def score(self, window: np.ndarray, yamnet_scores: np.ndarray | None = None) -> dict | None:
        """Similarity of a live window to the enrolled drone. None when nothing is enrolled."""
        with self._lock:
            ref = self.ref
        if "drone" not in ref:
            return None
        live = {"spectral": dsp.spectral_embedding(window)}
        if yamnet_scores is not None:
            live["yamnet"] = _unit(np.sqrt(np.asarray(yamnet_scores, dtype=np.float32)))
        sims_d, sims_b = [], []
        for key, vec in live.items():
            if key in ref["drone"]:
                sims_d.append(dsp.cosine(vec, ref["drone"][key]))
                if "background" in ref and key in ref["background"]:
                    sims_b.append(dsp.cosine(vec, ref["background"][key]))
        if not sims_d:
            return None
        sim_d = float(np.mean(sims_d))
        if sims_b:
            sim_b = float(np.mean(sims_b))
            value = float(np.clip(0.5 + 2.5 * (sim_d - sim_b), 0.0, 1.0))
        else:
            sim_b = None
            value = float(np.clip((sim_d - 0.55) / 0.35, 0.0, 1.0))
        return {"score": round(value, 3), "sim_drone": round(sim_d, 3),
                "sim_background": None if sim_b is None else round(sim_b, 3)}

    def status(self) -> dict:
        return {"enrolled": self.enrolled, "counts": dict(self.counts),
                "embeddings": sorted(self.ref.get("drone", {}).keys())}
