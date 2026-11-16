"""One long-lived CoT connection to the TAK server, registered as an EUD (isr-kit contract v5).

OpenTAKServer drops CoT from a sender that isn't a registered EUD, only binds a client's receive
queue if its first <contact> event arrives after its async channel is up (~3 s after connect), and
refuses a new uid whose callsign is already taken. So this client:

- C5.1 keeps ONE TCP connection open, reconnecting with exponential backoff;
- C5.2 on every (re)connect waits `register_delay_s` (3 s), then sends a registration SA with a
  stable uid ``isrkit-<tool>-<host>`` and callsign ``<LABEL> <host>``, re-sent every 60 s;
- C5.3 sends every marker on that same connection, only after registration. Markers keep their
  own uids. Per-uid rate limiting (contract C4.4) happens in ``send()``.

Self-contained: standard library only, so it can be copied into other isr-kit tools.
"""

from __future__ import annotations

import logging
import platform
import queue
import re
import socket
import threading
import time
from datetime import datetime, timezone
from xml.sax.saxutils import quoteattr

logger = logging.getLogger(__name__)

REGISTER_DELAY_S = 3.0          # OTS binds the receive queue only for a contact sent after ~3 s
REGISTER_EVERY_S = 60.0
REGISTER_STALE_S = 120.0
MIN_INTERVAL_S = 5.0            # contract C4.4: at most one CoT per 5 s per uid
QUEUE_MAX = 100

_UID_RE = re.compile(r"""<event\b[^>]*\buid=(["'])(.*?)\1""")


def isotime(ts: float) -> str:
    """ISO 8601 UTC with milliseconds and a trailing Z, e.g. 2026-11-16T15:01:02.345Z."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _valid_position(lat, lon) -> bool:
    if lat is None or lon is None or (lat == 0 and lon == 0):
        return False
    return -90 <= lat <= 90 and -180 <= lon <= 180


def registration_event(uid: str, callsign: str, lat: float | None, lon: float | None,
                       hae: float = 0.0, device: str = "", version: str = "",
                       now: float | None = None) -> str:
    """The EUD registration SA. With no position it uses the "unknown point" (0,0 with 9999999
    errors) and stale == start, so ATAK never draws it. This is the only 0,0 the kit allows."""
    now = time.time() if now is None else now
    if _valid_position(lat, lon):
        point = f'<point lat="{lat:.6f}" lon="{lon:.6f}" hae="{hae:.1f}" ce="10.0" le="9999999.0"/>'
        stale = now + REGISTER_STALE_S
    else:
        point = '<point lat="0" lon="0" hae="9999999" ce="9999999" le="9999999"/>'
        stale = now
    os_name = f"{platform.system()} {platform.release()}".strip()
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<event version="2.0" uid={quoteattr(uid)} type="a-f-G-E-S" how="h-e"'
        f' time="{isotime(now)}" start="{isotime(now)}" stale="{isotime(stale)}">'
        f"{point}"
        f'<detail><contact callsign={quoteattr(callsign)} endpoint="*:-1:stcp"/>'
        '<__group name="Cyan" role="Team Member"/>'
        f'<takv platform="isr-kit" device={quoteattr(device)} os={quoteattr(os_name)}'
        f" version={quoteattr(version)}/>"
        "</detail></event>"
    )


class TakClient:
    """Drop-in for the old one-shot client: ``send(xml) -> bool``, ``connected``, ``sent``."""

    def __init__(self, host: str, port: int, tool: str, station: str, label: str,
                 lat: float | None = None, lon: float | None = None, hae: float = 0.0,
                 version: str = "", register_delay_s: float = REGISTER_DELAY_S,
                 register_every_s: float = REGISTER_EVERY_S, max_backoff: float = 30.0,
                 min_interval_s: float = MIN_INTERVAL_S, clock=time.monotonic):
        self.host, self.port = host, port
        self.uid = f"isrkit-{tool}-{station}"            # stable: never random
        self.callsign = f"{label} {station}"
        self.tool, self.version = tool, version
        self.lat, self.lon, self.hae = lat, lon, hae
        self.register_delay_s = register_delay_s
        self.register_every_s = register_every_s
        self.max_backoff = max_backoff
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._last_by_uid: dict[str, float] = {}
        self.connected = False          # TCP connection up
        self.registered = False         # registration SA sent on this connection
        self.sent = 0                   # markers sent (not counting registrations)
        self.registrations = 0
        self.connects = 0
        self.throttled = 0
        self._q: queue.Queue[str] = queue.Queue(maxsize=QUEUE_MAX)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tak", daemon=True)

    # -- public ------------------------------------------------------------------------------
    def start(self):
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def send(self, xml: str) -> bool:
        """Queue a marker. Returns False if the per-uid rate limit dropped it."""
        m = _UID_RE.search(xml)
        if m and self.min_interval_s > 0:
            uid, now = m.group(2), self._clock()
            last = self._last_by_uid.get(uid)
            if last is not None and now - last < self.min_interval_s:
                self.throttled += 1
                logger.debug("CoT rate limit for %s; dropped", uid)
                return False
            self._last_by_uid[uid] = now
        try:
            self._q.put_nowait(xml)
        except queue.Full:
            try:
                self._q.get_nowait()       # drop oldest, keep newest
            except queue.Empty:
                pass
            self._q.put_nowait(xml)
        return True

    def registration(self, now: float | None = None) -> str:
        return registration_event(self.uid, self.callsign, self.lat, self.lon, self.hae,
                                  device=self.tool, version=self.version, now=now)

    # -- connection thread -------------------------------------------------------------------
    def _connect(self) -> socket.socket | None:
        try:
            sock = socket.create_connection((self.host, self.port), timeout=5)
        except OSError as e:
            logger.warning("TAK %s:%d unavailable (%s)", self.host, self.port, e)
            return None
        sock.settimeout(5)
        self.connected, self.registered = True, False
        self.connects += 1
        logger.info("TAK connected %s:%d; registering as %s in %.0f s",
                    self.host, self.port, self.callsign, self.register_delay_s)
        return sock

    def _drain(self, sock: socket.socket, closed: threading.Event):
        """Read and discard whatever the server pushes, so its send buffer never fills; notice EOF."""
        try:
            while not closed.is_set():
                try:
                    if not sock.recv(65536):
                        break
                except socket.timeout:
                    continue
        except OSError:
            pass
        closed.set()

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            sock = self._connect()
            if sock is None:
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, self.max_backoff)
                continue
            backoff = 1.0
            closed = threading.Event()
            threading.Thread(target=self._drain, args=(sock, closed), name="tak-rx", daemon=True).start()
            try:
                self._serve(sock, closed)
            except OSError as e:
                logger.warning("TAK connection lost (%s); reconnecting", e)
            finally:
                closed.set()
                self.connected = self.registered = False
                try:
                    sock.close()
                except OSError:
                    pass
            if not self._stop.is_set() and self._stop.wait(backoff):
                return

    def _serve(self, sock: socket.socket, closed: threading.Event):
        # C5.2: wait for the server's async channel, then register. Markers wait in the queue.
        if self._stop.wait(self.register_delay_s) or closed.is_set():
            return
        sock.sendall(self.registration(now=time.time()).encode("utf-8"))
        self.registered = True
        self.registrations += 1
        last_reg = time.monotonic()
        pending: str | None = None
        try:
            while not self._stop.is_set() and not closed.is_set():
                if time.monotonic() - last_reg >= self.register_every_s:
                    sock.sendall(self.registration(now=time.time()).encode("utf-8"))
                    self.registrations += 1
                    last_reg = time.monotonic()
                if pending is None:
                    try:
                        pending = self._q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                sock.sendall(pending.encode("utf-8"))   # raises on a dead socket
                pending = None
                self.sent += 1
        finally:
            if pending is not None:
                self._requeue(pending)                  # resend after the next registration

    def _requeue(self, xml: str):
        try:
            self._q.put_nowait(xml)
        except queue.Full:
            pass
