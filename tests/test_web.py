import pytest
from fastapi.testclient import TestClient

from acoustic.engine import Engine
from acoustic.gemini import GeminiChecker
from acoustic.web import app as web
from conftest import FakeCot, FakeMesh


@pytest.fixture
def client(settings):
    web.engine = Engine(settings, cot_client=FakeCot(), mesh=FakeMesh(),
                        gemini=GeminiChecker(None, settings.gemini_model))
    yield TestClient(web.app)
    web.engine = None


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200 and "<h1>Acoustic</h1>" in r.text


# sha256 of ui-kit v3 (isr-ui.css, isr-ui.js, fonts/). The served copies must be byte-identical to the
# kit: change the kit at its source and re-copy, never edit the vendored files.
KIT_SHA256 = {
    "isr-ui.css": "287c3df0938d25c25b04b751007095c3cf08e1df0e2689d3dd8df8b3e9eb832c",
    "isr-ui.js": "79a61cb2a427302e45b62cbebe2d022eff2dcd0549df5edea77b13a6b478fe7d",
    "fonts/OFL-BarlowCondensed.txt": "186d750eb496a4c17a76385f82be6aea2ac1cf2de074a811d63786cf374ea73f",
    "fonts/OFL-IBMPlexMono.txt": "7e6b2818edbd8f6a01ae80641cc8f16a51080d08fb4e532be3a0b6f74adb07da",
    "fonts/barlow-condensed-500.woff2": "31ff5e6e96a096fde4d78bc838719e69b093db2d0b062b96e423c065a6d1b2b1",
    "fonts/barlow-condensed-600.woff2": "5bc49d41f32810a114765314392332ce2acae2276635eee265a50319a12c1b49",
    "fonts/ibm-plex-mono-400.woff2": "52de70091213c9eab340e2a7ac59e7b6b2b94e089bf69ef38479b22cfde4274c",
    "fonts/ibm-plex-mono-500.woff2": "f17f51948d10b67d907393ccbac551912f2c2bd2c5af19be3aabe2f456f8fe54",
}


def test_page_uses_isr_kit(client):
    import hashlib

    html = client.get("/").text
    assert '<link rel="stylesheet" href="isr-ui.css?v=3">' in html       # kit version, busts old caches
    assert '<script src="isr-ui.js?v=3"></script>' in html
    assert client.get("/").headers["cache-control"] == "no-cache"
    for name in ("isr-ui.css", "isr-ui.js"):                             # ?v= matches the file's own vN
        assert client.get("/" + name).text.startswith("/* isr-ui." + name.split(".")[1] + " v3")
    assert html.index("isr-ui.css") < html.index("<style>")      # kit loads before page CSS
    assert 'id="isr-status"' in html and 'id="isr-kit-link"' in html
    assert html.count("isr-btn primary") == 1
    for name, digest in KIT_SHA256.items():
        r = client.get("/" + name)
        assert r.status_code == 200
        assert hashlib.sha256(r.content).hexdigest() == digest, f"{name} differs from the kit"
    assert client.get("/fonts/barlow-condensed-500.woff2").headers["content-type"] == "font/woff2"
    for bad in ("/fonts/..%2Fapp.py", "/fonts/missing.woff2", "/fonts/index.html"):
        assert client.get(bad).status_code == 404


def test_status_shows_modes(client):
    s = client.get("/api/status").json()
    assert s["gemini"]["enabled"] is False
    assert s["detector"]["yamnet"] is False
    assert s["sensor"]["name"] == "MIC1"
    assert s["waterfall"]["bins"] == 128


def test_record_validation(client):
    assert client.post("/api/record", json={"label": "cat"}).status_code == 400
    assert client.post("/api/record", json={"label": "drone", "seconds": 3}).status_code == 200
    assert client.post("/api/record", json={"label": "drone"}).status_code == 409
    assert client.get("/api/status").json()["recording"]["label"] == "drone"


def test_gemini_offline_is_explicit(client):
    r = client.post("/api/gemini")
    assert r.status_code == 400 and "GEMINI_API_KEY" in r.json()["detail"]


def test_test_alert_and_log(client):
    assert client.post("/api/test-alert").json()["mesh_sent"] is True
    assert client.get("/api/log").json()[0]["kind"] == "test"


def test_websocket_snapshot(client):
    web.engine.feed(__import__("numpy").zeros(16000, dtype="float32"))
    snap = web.snapshot(web.engine, web.engine.take_waterfall())
    assert len(snap["wf"]) == 31 and "status" in snap
    with client.websocket_connect("/ws") as ws:
        ws.send_text("hi")


def _mp4(path, with_audio, signal=None):
    import av
    import numpy as np

    with av.open(str(path), "w") as c:
        v = c.add_stream("mpeg4", rate=10)
        v.width = v.height = 64
        v.pix_fmt = "yuv420p"
        a = None
        if with_audio:
            a = c.add_stream("aac", rate=48000)
            a.layout = "mono"
        for _ in range(10):
            for pk in v.encode(av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")):
                c.mux(pk)
        if a is not None:
            for i in range(0, len(signal), 1024):
                fr = av.AudioFrame.from_ndarray(signal[i:i + 1024].reshape(1, -1), format="flt", layout="mono")
                fr.sample_rate = 48000
                for pk in a.encode(fr):
                    c.mux(pk)
            for pk in a.encode(None):
                c.mux(pk)
        for pk in v.encode(None):
            c.mux(pk)


def test_enroll_from_wav_upload(client, tmp_path):
    import soundfile as sf

    from conftest import tone_comb

    wav = tmp_path / "fpv.wav"
    sf.write(str(wav), tone_comb(380.0, seconds=3.0), 16000)
    with open(wav, "rb") as fh:
        r = client.post("/api/enroll", files={"file": ("fpv.wav", fh, "audio/wav")}, data={"label": "drone"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["calibration"]["enrolled"] and j["stats"]["prop_tone_fraction"] > 0.8 and j["warning"] is None


def test_enroll_from_video_audio_track(client, tmp_path):
    import numpy as np

    pytest.importorskip("av")
    t = np.arange(48000 * 3) / 48000
    sig = sum(np.sin(2 * np.pi * 380 * k * t) / k for k in range(1, 9)).astype(np.float32) * 0.1
    mp4 = tmp_path / "DJI_0001.MP4"
    _mp4(mp4, True, sig)
    with open(mp4, "rb") as fh:
        r = client.post("/api/enroll", files={"file": ("DJI_0001.MP4", fh, "video/mp4")}, data={"label": "drone"})
    assert r.status_code == 200, r.text
    assert r.json()["stats"]["seconds"] >= 2.5


def test_enroll_video_without_audio_is_explained(client, tmp_path):
    pytest.importorskip("av")
    mp4 = tmp_path / "silent.mp4"
    _mp4(mp4, False)
    with open(mp4, "rb") as fh:
        r = client.post("/api/enroll", files={"file": ("silent.mp4", fh, "video/mp4")}, data={"label": "drone"})
    assert r.status_code == 422 and "no audio track" in r.json()["detail"]


def test_device_endpoints_in_non_mic_mode(client):
    d = client.get("/api/devices").json()
    assert d["switchable"] is False
    assert client.post("/api/device", json={"device": 0}).status_code == 400
