"""FastAPI UI on port 8100: live waterfall, confidence meter, detection log, calibration controls."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from acoustic import dsp

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Acoustic Detector")

engine = None   # set by __main__ (or tests)
_clients: set[WebSocket] = set()


class RecordRequest(BaseModel):
    label: str = "drone"
    seconds: float = 5.0


def _engine():
    if engine is None:
        raise HTTPException(503, "engine not running")
    return engine


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/status")
async def status():
    e = _engine()
    return {**e.status(), "state": e.state,
            "waterfall": {"bins": dsp.WF_BINS, "fmin": dsp.WF_FMIN, "fmax": dsp.WF_FMAX,
                          "cols_per_s": 16000 / dsp.WF_HOP}}


@app.get("/api/log")
async def log():
    return list(_engine().log)


@app.post("/api/record")
async def record(req: RecordRequest):
    e = _engine()
    if req.label not in ("drone", "background"):
        raise HTTPException(400, "label must be 'drone' or 'background'")
    seconds = max(1.0, min(req.seconds, 15.0))
    if not e.start_recording(req.label, seconds):
        raise HTTPException(409, "already recording")
    return {"recording": req.label, "seconds": seconds}


@app.post("/api/calibration/clear")
async def clear_calibration():
    e = _engine()
    await asyncio.to_thread(e.calibration.clear)
    return e.calibration.status()


@app.post("/api/gemini")
async def gemini():
    result = _engine().ask_gemini()
    if result == "disabled":
        raise HTTPException(400, "Gemini is off (no GEMINI_API_KEY)")
    return {"result": result}


@app.post("/api/test-alert")
async def test_alert():
    return await asyncio.to_thread(_engine().test_alert)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


def snapshot(e, cols: list[bytes]) -> dict:
    return {
        "wf": [base64.b64encode(c).decode("ascii") for c in cols],
        "state": e.state,
        "history": [round(c, 3) for _, c in list(e.history)[-120:]],
        "status": e.status(),
        "log": list(e.log)[:50],
    }


async def periodic_broadcast(interval: float = 0.1):
    global _clients
    tick = 0
    while True:
        await asyncio.sleep(interval)
        if engine is None:
            continue
        cols = engine.take_waterfall()
        if not _clients:
            continue
        tick += 1
        if tick % 5 == 0:
            msg = snapshot(engine, cols)          # full state twice a second
        else:
            msg = {"wf": [base64.b64encode(c).decode("ascii") for c in cols]}
        data = json.dumps(msg, default=str)
        dead = set()
        for ws in list(_clients):
            try:
                await ws.send_text(data)
            except Exception:  # noqa: BLE001
                dead.add(ws)
        _clients -= dead
