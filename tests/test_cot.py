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
