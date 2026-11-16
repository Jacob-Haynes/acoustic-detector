"""Offline Pi: YAMNet runtime/model/class map may be missing. The sensor must degrade, never crash."""

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from acoustic.engine import Engine
from acoustic.yamnet import CLASS_MAP, UNAVAILABLE, Yamnet

REPO = Path(__file__).resolve().parents[1]


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_yamnet_missing_class_map_degrades(tmp_path, caplog):
    y = Yamnet(tmp_path / "none.tflite", class_map=tmp_path / "missing.csv")
    assert not y.available and "class map" in y.reason
    assert y.classify(__import__("numpy").zeros(15600)) is None
    assert sum(UNAVAILABLE in r.getMessage() for r in caplog.records) == 1     # one clear warning


def test_yamnet_missing_model_degrades(tmp_path):
    y = Yamnet(tmp_path / "none.tflite")
    assert not y.available and "model not found" in y.reason and "prefetch.sh" in y.reason


def test_corrupt_class_map_degrades(tmp_path):
    bad = tmp_path / "map.csv"
    bad.write_text("index,mid,display_name\n0,/m/x,Speech\n")       # truncated: classes missing
    assert not Yamnet(tmp_path / "none.tflite", class_map=bad).available


def test_heartbeat_and_status_report_unavailable(settings, tmp_path):
    from acoustic.sources import SimSource

    settings.use_yamnet = True
    e = Engine(settings, yamnet=Yamnet(tmp_path / "none.tflite", class_map=tmp_path / "x.csv"),
               source=SimSource())
    hb = e.heartbeat()
    assert hb["status"] == "degraded" and UNAVAILABLE in hb["detail"]
    det = e.status()["detector"]
    assert det["yamnet"] is False and det["yamnet_unavailable"] is True
    # Yamnet() raised in __main__ (yamnet=None) while wanted: still reported
    assert UNAVAILABLE in Engine(settings, source=SimSource()).heartbeat()["detail"]
    settings.use_yamnet = False                       # --no-yamnet is a choice, not a fault
    assert "detail" not in Engine(settings, source=SimSource()).heartbeat()


def test_install_rsync_keeps_package_data(tmp_path):
    """install.sh's rsync excludes must not drop acoustic/data/ (the real-Pi crash)."""
    if shutil.which("rsync") is None:
        pytest.skip("rsync not installed")
    script = (REPO / "deploy" / "install.sh").read_text()
    excludes = re.findall(r"--exclude '([^']+)'", script)
    assert "/data" in excludes and "data" not in excludes
    dest = tmp_path / "opt"
    subprocess.run(["rsync", "-a", *[f"--exclude={x}" for x in excludes], f"{REPO}/", f"{dest}/"],
                   check=True)
    assert (dest / "acoustic" / "data" / "yamnet_class_map.csv").stat().st_size > 10_000
    assert not (dest / ".git").exists() and not (dest / ".venv").exists()


def test_app_starts_with_assets_missing(tmp_path):
    """End to end: python -m acoustic with no model and no class map serves the UI and exits 0."""
    port = _free_port()
    env = {**os.environ, "YAMNET_MODEL": str(tmp_path / "none.tflite"),
           "YAMNET_CLASS_MAP": str(tmp_path / "none.csv"), "DATA_DIR": str(tmp_path / "data"),
           "WEB_PORT": str(port), "MQTT_HOST": "127.0.0.1", "MQTT_PORT": str(_free_port()),
           "PYTHONUNBUFFERED": "1"}
    env.pop("GEMINI_API_KEY", None)
    proc = subprocess.Popen([sys.executable, "-m", "acoustic", "--simulate", "--no-cot"], cwd=REPO,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        status = None
        deadline = time.time() + 30
        while time.time() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=1) as r:
                    status = json.load(r)
                    break
            except OSError:
                time.sleep(0.3)
        assert proc.poll() is None, "app exited at startup:\n" + (proc.stdout.read() if proc.stdout else "")
        assert status is not None, "UI never came up"
        assert status["detector"]["yamnet"] is False and status["detector"]["yamnet_unavailable"]
        time.sleep(1.5)                       # let it analyse a few windows
        assert proc.poll() is None
    finally:
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=15)
    assert proc.returncode == 0, out
    assert out.count(UNAVAILABLE) == 1, out
    assert "Traceback" not in out, out


def test_shipped_class_map_present():
    assert CLASS_MAP.exists() and sum(1 for _ in CLASS_MAP.open()) == 522
