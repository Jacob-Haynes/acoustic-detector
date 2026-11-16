"""TakClient against a fake OpenTAKServer that enforces the three behaviours seen on OTS 1.8.3:

1. CoT from a connection with no registered EUD is dropped.
2. A client whose first <contact endpoint=...> arrives less than BIND_AFTER_S after connect is
   registered but never gets a receive queue (nothing is delivered to it).
3. A new uid claiming a callsign another uid already holds is refused (and so is not registered).
"""

import re
import socket
import threading
import time
import xml.etree.ElementTree as ET

import pytest

from acoustic import cot
from acoustic.tak_client import TakClient, registration_event

BIND_AFTER_S = 0.25      # fake server's async-channel delay (the real one is ~3 s)
DELAY_OK = 0.35          # client delay longer than that
EVENT_RE = re.compile(r"<event\b.*?</event>", re.S)


class FakeOTS:
    def __init__(self):
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.port = self.srv.getsockname()[1]
        self.callsigns: dict[str, str] = {}      # callsign -> uid (EUD table)
        self.accepted: list[tuple[str, ET.Element]] = []   # (sender_uid, event) stored
        self.dropped: list[str] = []             # reasons
        self.bound: list[str] = []               # uids whose receive queue got bound
        self.unbound: list[str] = []
        self.connections = 0
        self.conns: list[socket.socket] = []
        self.lock = threading.Lock()
        self._stop = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._stop:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            with self.lock:
                self.connections += 1
                self.conns.append(conn)
            threading.Thread(target=self._handle, args=(conn, time.monotonic()), daemon=True).start()

    def _handle(self, conn, t0):
        sender = None
        buf = ""
        conn.settimeout(0.2)
        while not self._stop:
            try:
                data = conn.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            if not data:
                return
            buf += data.decode()
            while (m := EVENT_RE.search(buf)) is not None:
                buf = buf[m.end():]
                ev = ET.fromstring(m.group(0))
                contact = ev.find("detail/contact")
                with self.lock:
                    if contact is not None and contact.get("endpoint"):     # EUD SA
                        uid, cs = ev.get("uid"), contact.get("callsign")
                        holder = self.callsigns.get(cs)
                        if holder is not None and holder != uid:
                            self.dropped.append(f"callsign {cs!r} taken by {holder}")    # rule 3
                            continue
                        if sender is None:
                            (self.bound if time.monotonic() - t0 >= BIND_AFTER_S
                             else self.unbound).append(uid)                         # rule 2
                        self.callsigns[cs] = uid
                        sender = uid
                        self.accepted.append((uid, ev))
                    elif sender is None:
                        self.dropped.append(f"unregistered sender: {ev.get('uid')}")  # rule 1
                    else:
                        self.accepted.append((sender, ev))

    def markers(self):
        with self.lock:
            return [(s, e) for s, e in self.accepted if e.find("detail/contact") is None
                    or not e.find("detail/contact").get("endpoint")]

    def drop_connections(self):
        with self.lock:
            for c in self.conns:
                try:
                    c.shutdown(socket.SHUT_RDWR)
                    c.close()
                except OSError:
                    pass
            self.conns.clear()

    def close(self):
        self._stop = True
        self.srv.close()
        self.drop_connections()


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def ots():
    s = FakeOTS()
    yield s
    s.close()


def client(port, delay=DELAY_OK, **kw):
    kw.setdefault("lat", 52.0)
    kw.setdefault("lon", -1.0)
    return TakClient("127.0.0.1", port, tool="acoustic-detector", station="isr-pi", label="ACOUSTIC",
                     register_delay_s=delay, max_backoff=0.2, min_interval_s=0, **kw)


def det(name="MIC1", source="simulate"):
    return cot.detection_event(name, 52.0, -1.0, 0, 0.8, 300, source=source)


def test_registers_then_delivers_on_one_connection(ots):
    c = client(ots.port)
    for i in range(3):
        c.send(det(f"MIC{i}"))                  # queued before registration: held, not lost
    c.start()
    try:
        assert wait_for(lambda: len(ots.markers()) == 3)
        assert ots.bound == ["isrkit-acoustic-detector-isr-pi"] and not ots.unbound
        assert not ots.dropped
        assert ots.connections == 1             # C5.1: not a connection per alert
        assert all(s == "isrkit-acoustic-detector-isr-pi" for s, _ in ots.markers())
        uids = {e.get("uid") for _, e in ots.markers()}
        assert uids == {"ACOUSTIC-MIC0-UAV", "ACOUSTIC-MIC1-UAV", "ACOUSTIC-MIC2-UAV"}  # own uids
        remarks = ots.markers()[0][1].find("detail/remarks").text
        assert remarks.startswith("SIMULATED: ")                                     # C4.3 kept
    finally:
        c.stop()


def test_fake_enforces_rules_against_naive_senders(ots):
    # rule 1: a one-shot sender with no registration is dropped
    s = socket.create_connection(("127.0.0.1", ots.port))
    s.sendall(det().encode())
    assert wait_for(lambda: ots.dropped)
    s.close()
    assert "unregistered sender" in ots.dropped[0] and not ots.markers()
    # rule 2: registering immediately on connect gets no receive queue
    early = client(ots.port, delay=0.0).start()
    try:
        assert wait_for(lambda: ots.unbound == ["isrkit-acoustic-detector-isr-pi"])
        assert not ots.bound
    finally:
        early.stop()


def test_duplicate_callsign_under_new_uid_is_refused(ots):
    squatter = socket.create_connection(("127.0.0.1", ots.port))
    time.sleep(BIND_AFTER_S + 0.05)
    squatter.sendall(registration_event("random-uuid-1234", "ACOUSTIC isr-pi", 52.0, -1.0).encode())
    assert wait_for(lambda: "ACOUSTIC isr-pi" in ots.callsigns)
    c = client(ots.port).start()
    try:
        c.send(det())
        assert wait_for(lambda: len(ots.dropped) >= 2)
        assert any("callsign 'ACOUSTIC isr-pi' taken" in d for d in ots.dropped)
        assert not ots.markers()
    finally:
        c.stop()
        squatter.close()


def test_reconnect_reregisters_with_same_uid(ots):
    c = client(ots.port).start()
    try:
        c.send(det("A"))
        assert wait_for(lambda: len(ots.markers()) == 1)
        ots.drop_connections()
        assert wait_for(lambda: ots.connections == 2 and c.registered, timeout=5)
        c.send(det("B"))
        assert wait_for(lambda: len(ots.markers()) == 2)
        assert ots.bound == ["isrkit-acoustic-detector-isr-pi"] * 2       # same stable uid
        assert not ots.dropped                                            # callsign not "taken"
    finally:
        c.stop()


def test_registration_resent_periodically(ots):
    c = client(ots.port, register_every_s=0.2).start()
    try:
        assert wait_for(lambda: c.registrations >= 3)
        assert ots.connections == 1
    finally:
        c.stop()


def test_registration_event_fields():
    ev = ET.fromstring(registration_event("isrkit-acoustic-detector-isr-pi", "ACOUSTIC isr-pi",
                                          52.0, -1.0, device="acoustic-detector", version="0.1.0",
                                          now=1_700_000_000.0))
    assert ev.get("type") == "a-f-G-E-S"
    for k in ("time", "start", "stale"):
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", ev.get(k))
    assert ev.get("stale") > ev.get("start")
    assert ev.find("point").get("lat") == "52.000000"
    contact = ev.find("detail/contact")
    assert contact.get("callsign") == "ACOUSTIC isr-pi" and contact.get("endpoint") == "*:-1:stcp"
    assert ev.find("detail/__group").attrib == {"name": "Cyan", "role": "Team Member"}
    takv = ev.find("detail/takv")
    assert takv.get("platform") == "isr-kit" and takv.get("device") == "acoustic-detector"
    assert takv.get("version") == "0.1.0" and takv.get("os")


def test_registration_without_position_is_never_drawn():
    for lat, lon in ((None, None), (0.0, 0.0)):
        ev = ET.fromstring(registration_event("u", "c", lat, lon, now=1_700_000_000.0))
        assert ev.find("point").attrib == {"lat": "0", "lon": "0", "hae": "9999999",
                                           "ce": "9999999", "le": "9999999"}
        assert ev.get("stale") == ev.get("start")


def test_server_down_never_blocks():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    c = client(port).start()
    for _ in range(300):              # more than the queue size: must not block or raise
        c.send(det())
    time.sleep(0.3)
    assert not c.connected and not c.registered
    c.stop()


def test_rate_limit_per_uid_c44():
    t = [0.0]
    c = TakClient("127.0.0.1", 1, tool="acoustic-detector", station="x", label="ACOUSTIC",
                  clock=lambda: t[0])                 # not started: just queues
    marker = cot.sensor_event("MIC1", 52.0, -1.0, 0, "listening")
    assert c.send(det()) and c.send(marker)           # different uids
    t[0] = 4.9
    assert not c.send(det()) and not c.send(marker)
    t[0] = 5.1
    assert c.send(det())
    assert c.throttled == 2 and c._q.qsize() == 3
