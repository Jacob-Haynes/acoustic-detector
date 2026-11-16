import socket
import threading
import time
import xml.etree.ElementTree as ET

from acoustic import cot


def test_detection_event_is_valid_cot():
    xml = cot.detection_event("MIC1", 52.0, -1.0, 100.0, 0.87, 300.0, "u", 380.0, source="mic", now=0)
    root = ET.fromstring(xml.encode())
    assert root.tag == "event" and root.get("version") == "2.0"
    assert root.get("type") == "a-u-A-M-F-Q"
    assert root.get("uid") == "ACOUSTIC-MIC1-UAV"
    pt = root.find("point")
    assert float(pt.get("lat")) == 52.0 and float(pt.get("lon")) == -1.0
    assert float(pt.get("ce")) == 300.0
    remarks = root.find("detail/remarks").text
    assert "ACOUSTIC" in remarks and "0.87" in remarks and "SENSOR position" in remarks
    assert root.get("stale") > root.get("time")


def test_hostile_affiliation_and_escaping():
    xml = cot.detection_event('OP<1>&"', 0, 0, 0, 0.5, 100, "h")
    root = ET.fromstring(xml.encode())
    assert root.get("type") == "a-h-A-M-F-Q"
    assert 'OP<1>&"' in root.find("detail/contact").get("callsign")


def test_sensor_event():
    root = ET.fromstring(cot.sensor_event("MIC1", 1.0, 2.0, 3.0, "listening").encode())
    assert root.get("type") == "a-f-G-E-S"
    assert root.find("detail/contact").get("callsign") == "ACOUSTIC MIC1"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_client_survives_server_down():
    c = cot.CotClient("127.0.0.1", _free_port(), max_backoff=0.2).start()
    for _ in range(300):          # more than the queue size: must not block or raise
        c.send("<event/>")
    time.sleep(0.5)
    assert not c.connected
    c.stop()


def test_client_delivers_and_reconnects():
    port = _free_port()
    received = []
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)

    def accept():
        conn, _ = srv.accept()
        conn.settimeout(3)
        data = b""
        while b"</event>" not in data:
            data += conn.recv(4096)
        received.append(data.decode())
        conn.close()

    t = threading.Thread(target=accept, daemon=True)
    t.start()
    c = cot.CotClient("127.0.0.1", port, max_backoff=0.2).start()
    c.send(cot.sensor_event("MIC1", 1, 2, 3, "test"))
    t.join(5)
    srv.close()
    c.stop()
    assert received and "ACOUSTIC-MIC1" in received[0]


def test_remarks_prefix_c43():
    sim = ET.fromstring(cot.detection_event("MIC1", 52.0, -1.0, 0, 0.8, 300, source="simulate").encode())
    assert sim.find("detail/remarks").text.startswith("SIMULATED: ")
    rec = ET.fromstring(cot.detection_event("MIC1", 52.0, -1.0, 0, 0.8, 300, source="file").encode())
    assert rec.find("detail/remarks").text.startswith("RECORDED: ")
    live = ET.fromstring(cot.detection_event("MIC1", 52.0, -1.0, 0, 0.8, 300, source="mic").encode())
    assert live.find("detail/remarks").text.startswith("ACOUSTIC drone detection")
    marker = ET.fromstring(cot.sensor_event("MIC1", 52.0, -1.0, 0, "listening", source="simulate").encode())
    assert marker.find("detail/remarks").text.startswith("SIMULATED: ")
    assert ET.fromstring(cot.sensor_event("MIC1", 1, 2, 3, "x").encode()).find("detail/remarks").text.startswith("Acoustic")


def test_valid_position_c42():
    assert not cot.valid_position(None, None)
    assert not cot.valid_position(0.0, 0.0)
    assert not cot.valid_position(95.0, 1.0)
    assert cot.valid_position(52.0, -1.0)
    assert cot.valid_position(0.0, -1.0)          # equator is a real place


def test_cot_rate_limit_per_uid_c44():
    t = [0.0]
    c = cot.CotClient("127.0.0.1", 1, clock=lambda: t[0])       # not started: just queues
    det = cot.detection_event("MIC1", 52.0, -1.0, 0, 0.8, 300)
    marker = cot.sensor_event("MIC1", 52.0, -1.0, 0, "listening")
    assert c.send(det)
    assert c.send(marker)                  # different uid: allowed
    t[0] = 4.9
    assert not c.send(det)
    assert not c.send(marker)
    t[0] = 5.1
    assert c.send(det)
    assert c.throttled == 2 and c._q.qsize() == 3
