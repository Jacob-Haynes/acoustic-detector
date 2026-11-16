import numpy as np
import pytest

from acoustic.config import REPO_ROOT
from acoustic.yamnet import CONFUSER_CLASSES, DRONE_CLASSES, Yamnet, load_class_names

MODEL = REPO_ROOT / "models" / "yamnet.tflite"


def test_class_names_exist_in_map():
    names = load_class_names()
    assert len(names) == 521
    for n in list(DRONE_CLASSES) + CONFUSER_CLASSES:
        assert n in names


def test_missing_model_degrades_gracefully(tmp_path):
    y = Yamnet(tmp_path / "nope.tflite")
    assert not y.available and "not found" in y.reason
    assert y.classify(np.zeros(15600)) is None


def test_score_mapping():
    y = Yamnet.__new__(Yamnet)
    y.class_names = load_class_names()
    y._drone_idx = {y.class_names.index(n): w for n, w in DRONE_CLASSES.items()}
    y._conf_idx = [y.class_names.index(n) for n in CONFUSER_CLASSES]
    s = np.zeros(521, dtype=np.float32)
    s[y.class_names.index("Propeller, airscrew")] = 0.5
    r = y.score(s)
    assert r.score == 1.0 and r.confuser == 0.0
    s[:] = 0
    s[y.class_names.index("Lawn mower")] = 0.6
    r = y.score(s)
    assert r.score == 0.0 and r.confuser > 0.9


@pytest.mark.skipif(not MODEL.exists(), reason="run scripts/fetch_model.sh to test the real model")
def test_real_model_runs():
    pytest.importorskip("ai_edge_litert")
    y = Yamnet(MODEL)
    assert y.available
    r = y.classify(np.zeros(15600, dtype=np.float32))
    assert r is not None and r.scores.shape == (521,)
    assert r.top[0][0] == "Silence"
