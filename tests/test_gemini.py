import time

import numpy as np

from acoustic.gemini import GeminiChecker, parse_reply, to_wav_bytes


def test_parse_reply_json_and_fenced():
    r = parse_reply('{"drone_audible": true, "confidence": 0.9, "best_guess": "quadcopter", "reason": "buzz"}')
    assert r["drone_audible"] is True and r["confidence"] == 0.9
    r = parse_reply('```json\n{"drone_audible": false, "confidence": 2}\n```')
    assert r["drone_audible"] is False and r["confidence"] == 1.0
    r = parse_reply("not json at all")
    assert r["drone_audible"] is None and "not json" in r["reason"]


def test_disabled_without_key():
    g = GeminiChecker(None, "gemini-3.8-flash")
    assert not g.enabled
    assert g.request(np.zeros(16000)) == "disabled"
    assert "offline" in g.status()["reason"]


def test_request_uses_client_and_rate_limits():
    calls = []

    class Resp:
        text = '{"drone_audible": true, "confidence": 0.8, "best_guess": "drone", "reason": "prop whine"}'

    class Models:
        def generate_content(self, model, contents, config):
            calls.append((model, contents))
            return Resp()

    class Client:
        models = Models()

    got = []
    g = GeminiChecker("key", "test-model", min_interval_s=60, client=Client())
    g.on_result = got.append
    assert g.request(np.zeros(16000, dtype=np.float32)) == "started"
    for _ in range(100):
        if got:
            break
        time.sleep(0.02)
    assert got and got[0]["ok"] and got[0]["drone_audible"] is True
    assert calls[0][0] == "test-model"
    assert g.request(np.zeros(16000)) == "rate-limited"
    assert g.request(np.zeros(16000), force=True) == "started"


def test_api_error_is_reported_not_raised():
    class Models:
        def generate_content(self, **kw):
            raise RuntimeError("no network")

    class Client:
        models = Models()

    got = []
    g = GeminiChecker("key", "m", client=Client())
    g.on_result = got.append
    g.request(np.zeros(16000))
    for _ in range(100):
        if got:
            break
        time.sleep(0.02)
    assert got and got[0]["ok"] is False and "no network" in got[0]["error"]


def test_wav_bytes():
    assert to_wav_bytes(np.zeros(1600))[:4] == b"RIFF"
