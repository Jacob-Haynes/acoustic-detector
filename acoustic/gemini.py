"""Optional cloud second opinion from Google Gemini. Off unless GEMINI_API_KEY is set.

Cloud mode uploads the audio clip to Google. Use it only with demo audio."""

from __future__ import annotations

import io
import json
import logging
import threading
import time

import numpy as np

from acoustic.config import SAMPLE_RATE

logger = logging.getLogger(__name__)

PROMPT = (
    "You are assisting an acoustic sensor used in a military training demo for detecting small "
    "drones (for example a 5-inch FPV quadcopter). Listen to this short clip recorded by a "
    "ground microphone. Decide whether a multirotor drone is audible. Common confusers are "
    "lawnmowers, generators, power tools, vehicles, insects and wind. Reply with JSON only: "
    '{"drone_audible": true|false, "confidence": 0.0-1.0, "best_guess": "<short sound label>", '
    '"reason": "<one sentence>"}'
)


def to_wav_bytes(audio: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def parse_reply(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        data = json.loads(text[start:end])
    except ValueError:
        return {"drone_audible": None, "confidence": None, "best_guess": "", "reason": text[:200]}
    conf = data.get("confidence")
    try:
        conf = None if conf is None else max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = None
    audible = data.get("drone_audible")
    return {
        "drone_audible": audible if isinstance(audible, bool) else None,
        "confidence": conf,
        "best_guess": str(data.get("best_guess", ""))[:80],
        "reason": str(data.get("reason", ""))[:300],
    }


class GeminiChecker:
    def __init__(self, api_key: str | None, model: str, min_interval_s: float = 30.0, client=None):
        self.model = model
        self.min_interval_s = min_interval_s
        self.enabled = bool(api_key) or client is not None
        self.reason = "" if self.enabled else "GEMINI_API_KEY not set (offline mode)"
        self._client = client
        self._api_key = api_key
        self._lock = threading.Lock()
        self.busy = False
        self.last_call = 0.0
        self.last_result: dict | None = None
        self.on_result = None            # callback(dict)

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def request(self, audio: np.ndarray, trigger: str = "manual", force: bool = False) -> str:
        """Queue a background check. Returns 'started', 'busy', 'rate-limited' or 'disabled'."""
        if not self.enabled:
            return "disabled"
        with self._lock:
            if self.busy:
                return "busy"
            if not force and time.time() - self.last_call < self.min_interval_s:
                return "rate-limited"
            self.busy = True
            self.last_call = time.time()
        threading.Thread(target=self._run, args=(np.array(audio), trigger), daemon=True).start()
        return "started"

    def check(self, audio: np.ndarray) -> dict:
        from google.genai import types

        client = self._get_client()
        resp = client.models.generate_content(
            model=self.model,
            contents=[PROMPT, types.Part.from_bytes(data=to_wav_bytes(audio), mime_type="audio/wav")],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
        )
        return parse_reply(resp.text)

    def _run(self, audio: np.ndarray, trigger: str):
        t0 = time.time()
        try:
            result = self.check(audio)
            result["ok"] = True
        except Exception as e:  # noqa: BLE001 - network/API errors must not kill the sensor
            logger.warning("Gemini check failed: %s", e)
            result = {"ok": False, "error": str(e)[:200]}
        result.update({"ts": time.time(), "trigger": trigger, "model": self.model,
                       "latency_s": round(time.time() - t0, 2), "clip_s": round(len(audio) / SAMPLE_RATE, 1)})
        with self._lock:
            self.last_result = result
            self.busy = False
        if self.on_result:
            try:
                self.on_result(result)
            except Exception:  # noqa: BLE001
                logger.exception("Gemini result callback failed")

    def status(self) -> dict:
        return {"enabled": self.enabled, "model": self.model if self.enabled else None,
                "reason": self.reason, "busy": self.busy, "last": self.last_result}
