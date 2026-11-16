import json
import socket

from acoustic import meshbus


def test_payload_contract():
    p = meshbus.build_payload(meshbus.alert_text(0.873, "OP1"))
    assert p == {"text": "ACOUSTIC: drone detected conf 0.87 @ OP1", "to": None, "channel": 0,
                 "source": "acoustic-detector"}


def test_payload_truncated_under_200():
    p = meshbus.build_payload("x" * 500, to="!abcd1234", channel=2)
    assert len(p["text"]) < 200
    assert p["to"] == "!abcd1234" and p["channel"] == 2


def test_no_broker_does_not_raise():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    bus = meshbus.MeshBus("127.0.0.1", port, "guest", "guest").start()
    assert bus.send("ACOUSTIC: test") is False
    bus.stop()


def test_publishes_to_meshbus_tx():
    published = []

    class Info:
        rc = 0

    class FakeClient:
        def publish(self, topic, payload, qos=0):
            published.append((topic, json.loads(payload)))
            return Info()

    bus = meshbus.MeshBus("127.0.0.1", 1883, "guest", "guest", to="!1234", channel=1)
    bus._client = FakeClient()
    bus.connected = True
    assert bus.send("ACOUSTIC: drone detected conf 0.90 @ OP1")
    assert published == [("meshbus/tx", {"text": "ACOUSTIC: drone detected conf 0.90 @ OP1",
                                          "to": "!1234", "channel": 1, "source": "acoustic-detector"})]


def test_heartbeat_contract(monkeypatch):
    monkeypatch.setattr(meshbus.socket, "gethostname", lambda: "isr-pi.lan")
    hb = meshbus.heartbeat_payload(meshbus.short_hostname(), meshbus.advertised_url(8100), "cpu", "ok", ts=12.7)
    assert hb == {"tool": "acoustic-detector", "host": "isr-pi", "url": "http://isr-pi.local:8100",
                  "mode": "cpu", "status": "ok", "ts": 12}
    assert meshbus.advertised_url(8100, "http://192.168.4.1:8100") == "http://192.168.4.1:8100"


def test_heartbeat_published_retained_and_stopped_on_exit():
    import threading

    published = []
    done = threading.Event()

    class Info:
        rc = 0

        def wait_for_publish(self, timeout=None):
            return True

    class FakeClient:
        def publish(self, topic, payload, qos=0, retain=False):
            published.append((topic, json.loads(payload), retain))
            done.set()
            return Info()

        def loop_stop(self):
            pass

        def disconnect(self):
            pass

    bus = meshbus.MeshBus("127.0.0.1", 1883, "guest", "guest")
    bus._client = FakeClient()
    bus.connected = True
    bus.start_heartbeat(lambda: {"host": "isr-pi", "url": "http://isr-pi.local:8100", "mode": "simulate",
                                 "status": "ok"}, every_s=60)
    assert done.wait(2)
    bus.stop()
    topics = {t for t, _, _ in published}
    assert topics == {"isrkit/heartbeat/acoustic-detector"}
    assert all(r for _, _, r in published)
    assert published[0][1]["status"] == "ok" and published[-1][1]["status"] == "stopped"
    assert set(published[0][1]) == {"tool", "host", "url", "mode", "status", "ts"}
