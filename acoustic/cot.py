"""Plain CoT XML over TCP to the TAK server (OpenTAKServer on 127.0.0.1:8087 by default).

A background thread owns the socket and reconnects with exponential backoff, so a missing TAK
server never blocks or crashes detection."""

from __future__ import annotations

import logging
import queue
import re
import socket
import threading
import time
from datetime import datetime, timezone
from xml.sax.saxutils import escape, quoteattr

logger = logging.getLogger(__name__)

DETECTION_STALE_S = 90
SENSOR_STALE_S = 300
COT_MIN_INTERVAL_S = 5.0       # contract C4.4: at most one CoT per 5 s per uid

# Contract C4.3: remarks of synthetic / replayed output start with these. Live mic: no prefix.
_REMARK_PREFIX = {"simulate": "SIMULATED: ", "file": "RECORDED: "}


def remarks_prefix(source_kind: str | None) -> str:
    return _REMARK_PREFIX.get(source_kind or "", "")


def valid_position(lat, lon) -> bool:
    """Contract C4.2: never draw a point at 0,0 (an unset position) or out of range."""
    if lat is None or lon is None:
        return False
    if lat == 0 and lon == 0:
        return False
    return -90 <= lat <= 90 and -180 <= lon <= 180


def isotime(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def build_event(uid: str, cot_type: str, lat: float, lon: float, hae: float, callsign: str,
                remarks: str, stale_s: float, ce: float = 10.0, how: str = "m-g",
                now: float | None = None) -> str:
    now = time.time() if now is None else now
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<event version="2.0" uid={quoteattr(uid)} type={quoteattr(cot_type)} how={quoteattr(how)}'
        f' time="{isotime(now)}" start="{isotime(now)}" stale="{isotime(now + stale_s)}">'
        f'<point lat="{lat:.6f}" lon="{lon:.6f}" hae="{hae:.1f}" ce="{ce:.1f}" le="9999999.0"/>'
        f'<detail><contact callsign={quoteattr(callsign)}/>'
        f"<remarks>{escape(remarks)}</remarks></detail>"
        "</event>"
    )


def detection_event(sensor_name: str, lat: float, lon: float, hae: float, confidence: float,
                    radius_m: float, affiliation: str = "u", f0_hz: float = 0.0,
                    source: str = "", now: float | None = None) -> str:
    """UAV marker placed at the sensor (acoustics gives presence, not position). ce = rough range."""
    uid = f"ACOUSTIC-{sensor_name}-UAV"
    cot_type = f"a-{affiliation}-A-M-F-Q"        # standard CoT air type; -A-M-F-Q = UAV/drone
    remarks = remarks_prefix(source) + (
        f"ACOUSTIC drone detection by sensor {sensor_name}. conf {confidence:.2f}. "
        f"Marker is the SENSOR position: target is within ~{radius_m:.0f} m, bearing unknown."
    )
    if f0_hz:
        remarks += f" Prop tone ~{f0_hz:.0f} Hz."
    if source:
        remarks += f" Input: {'live microphone' if source == 'mic' else source}."
    # how="m-f": machine-derived/fused rather than a measured target position.
    return build_event(uid, cot_type, lat, lon, hae, f"UAV? (acoustic {sensor_name})", remarks,
                       DETECTION_STALE_S, ce=radius_m, how="m-f", now=now)


def sensor_event(sensor_name: str, lat: float, lon: float, hae: float, status: str,
                 source: str = "", now: float | None = None) -> str:
    uid = f"ACOUSTIC-{sensor_name}"
    remarks = remarks_prefix(source) + f"Acoustic drone sensor {sensor_name}: {status}"
    return build_event(uid, "a-f-G-E-S", lat, lon, hae, f"ACOUSTIC {sensor_name}", remarks,
                       SENSOR_STALE_S, ce=5.0, how="h-e", now=now)


_UID_RE = re.compile(r"""<event\b[^>]*\buid=(["'])(.*?)\1""")


class CotClient:
    def __init__(self, host: str, port: int, max_backoff: float = 30.0,
                 min_interval_s: float = COT_MIN_INTERVAL_S, clock=time.monotonic):
        self.host, self.port = host, port
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._last_by_uid: dict[str, float] = {}
        self.throttled = 0
        self.max_backoff = max_backoff
        self.connected = False
        self.sent = 0
        self._q: queue.Queue[str] = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cot", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def send(self, xml: str) -> bool:
        """Queue a CoT event. Returns False if the per-uid rate limit dropped it."""
        m = _UID_RE.search(xml)
        if m and self.min_interval_s > 0:
            uid, now = m.group(2), self._clock()
            last = self._last_by_uid.get(uid)
            if last is not None and now - last < self.min_interval_s:
                self.throttled += 1
                logger.debug("CoT rate limit for %s; dropped", uid)
                return False
            self._last_by_uid[uid] = now
        self._enqueue(xml)
        return True

    def _enqueue(self, xml: str):
        try:
            self._q.put_nowait(xml)
        except queue.Full:
            try:
                self._q.get_nowait()       # drop oldest, keep newest
            except queue.Empty:
                pass
            self._q.put_nowait(xml)

    def _run(self):
        backoff = 1.0
        sock: socket.socket | None = None
        while not self._stop.is_set():
            try:
                xml = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            while not self._stop.is_set():
                if sock is None:
                    try:
                        sock = socket.create_connection((self.host, self.port), timeout=5)
                        self.connected = True
                        backoff = 1.0
                        logger.info("TAK connected %s:%d", self.host, self.port)
                    except OSError as e:
                        self.connected = False
                        logger.warning("TAK %s:%d unavailable (%s); retry in %.0fs",
                                       self.host, self.port, e, backoff)
                        if self._stop.wait(backoff):
                            return
                        backoff = min(backoff * 2, self.max_backoff)
                        # keep only the freshest message while the server is down
                        try:
                            while True:
                                xml = self._q.get_nowait()
                        except queue.Empty:
                            pass
                        continue
                try:
                    sock.sendall(xml.encode("utf-8"))
                    self.sent += 1
                    break
                except OSError as e:
                    logger.warning("TAK send failed (%s); reconnecting", e)
                    try:
                        sock.close()
                    except OSError:
                        pass
                    sock = None
                    self.connected = False
        if sock is not None:
            sock.close()
