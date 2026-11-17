"""FastAPI UI on port 8100: live waterfall, confidence meter, detection log, calibration controls."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from acoustic import dsp

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Acoustic Detector")

engine = None   # set by __main__ (or tests)
_clients: set[WebSocket] = set()


class DeviceRequest(BaseModel):
    device: int | str


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


# The shared ISR kit look, vendored unchanged from ui-kit/ (see UI-SPEC.md).
@app.get("/isr-ui.css")
async def isr_ui_css():
    return FileResponse(STATIC_DIR / "isr-ui.css", media_type="text/css")


@app.get("/isr-ui.js")
async def isr_ui_js():
    return FileResponse(STATIC_DIR / "isr-ui.js", media_type="text/javascript")


_FONT_TYPES = {".woff2": "font/woff2", ".txt": "text/plain; charset=utf-8"}


@app.get("/fonts/{name}")
async def isr_ui_font(name: str):
    """The kit's fonts (and their OFL licences), referenced by isr-ui.css as fonts/<file>."""
    path = STATIC_DIR / "fonts" / name
    if "/" in name or name.startswith(".") or path.suffix not in _FONT_TYPES or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type=_FONT_TYPES[path.suffix])


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


ENROL_MAX_SECONDS = 120.0
UPLOAD_MAX_BYTES = 2 * 1024 ** 3


def _enrol_file(path: str, label: str) -> dict:
    from acoustic.sources import NoAudioTrack, load_audio_file

    e = _engine()
    try:
        audio = load_audio_file(path, max_seconds=ENROL_MAX_SECONDS)
    except NoAudioTrack as err:
        raise HTTPException(422, f"{err}. Some FPV air units record video only; use a GoPro or "
                                 "phone recording of the drone instead.") from err
    except Exception as err:  # noqa: BLE001
        raise HTTPException(422, f"could not decode file: {err}") from err
    if len(audio) < 16000:
        raise HTTPException(422, "audio shorter than 1 s")
    saved = e.calibration.add_clip(audio, label)
    e._append_log({"kind": "enrolled", "ts": time.time(), "confidence": 0.0,
                   "label": label, "file": saved.name})
    return {"file": saved.name, "label": label, "warning": e.calibration.last_warning,
            "stats": e.calibration.last_stats, "calibration": e.calibration.status()}


@app.post("/api/enroll")
async def enroll(file: UploadFile = File(...), label: str = Form("drone")):
    """Enrol from an uploaded WAV/MP3/FLAC or video (MP4/MOV). The first 120 s are used."""
    if label not in ("drone", "background"):
        raise HTTPException(400, "label must be 'drone' or 'background'")
    suffix = Path(file.filename or "upload").suffix[:8] or ".bin"
    tmp_dir = Path(_engine().s.data_dir)      # not /tmp: often RAM-backed on the Pi
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=suffix, dir=tmp_dir, delete=True) as tmp:
        size = 0
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > UPLOAD_MAX_BYTES:
                raise HTTPException(413, "file too large (2 GB max)")
            tmp.write(chunk)
        tmp.flush()
        result = await asyncio.to_thread(_enrol_file, tmp.name, label)
    result["source_name"] = file.filename
    return result


@app.get("/api/devices")
async def devices():
    return await asyncio.to_thread(_engine().list_devices)


@app.post("/api/device")
async def set_device(req: DeviceRequest):
    try:
        return {"source": await asyncio.to_thread(_engine().switch_device, req.device)}
    except Exception as err:  # noqa: BLE001
        raise HTTPException(400, f"cannot use device {req.device!r}: {err}") from err


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
