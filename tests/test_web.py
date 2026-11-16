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
    assert r.status_code == 200 and "ACOUSTIC DETECTOR" in r.text


def test_status_shows_modes(client):
    s = client.get("/api/status").json()
    assert s["gemini"]["enabled"] is False
    assert s["detector"]["yamnet"] is False
    assert s["sensor"]["name"] == "OP1"
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
