"""MQTT on the Pi hub: mesh alerts via the mesh bridge, plus the isrkit heartbeat.
Never touches the Meshtastic serial port.

Publishes {"text", "to", "channel", "source"} to meshbus/tx. If the broker is down the message is
logged and dropped; paho reconnects in the background.

Heartbeat: retained JSON on isrkit/heartbeat/acoustic-detector every 10 s, with a retained
"error" last-will (and a retained "stopped" on clean exit) so the Pi launcher sees it go."""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

TX_TOPIC = "meshbus/tx"
HEARTBEAT_TOPIC = "isrkit/heartbeat/acoustic-detector"
HEARTBEAT_EVERY_S = 10.0
MAX_TEXT = 200
MESH_MIN_INTERVAL_S = 30.0    # contract C4.4: at most one mesh text per 30 s per tool
SOURCE = "acoustic-detector"


def build_payload(text: str, to: str | None = None, channel: int = 0) -> dict:
    text = text.strip()
    if len(text) > MAX_TEXT - 1:
        text = text[: MAX_TEXT - 2] + "~"
    return {"text": text, "to": to, "channel": channel, "source": SOURCE}


HEARTBEAT_MODES = ("cpu", "cuda", "mps", "gemini", "vlm", "simulate", "replay")
HEARTBEAT_STATUSES = ("ok", "degraded", "error", "stopped")


def short_hostname() -> str:
    return socket.gethostname().split(".")[0]


def station_host() -> str:
    """Contract C4.1: env ISR_HOST if set, else the lowercased short hostname."""
    return os.environ.get("ISR_HOST", "").strip() or short_hostname().lower()


# Contract C4.3: synthetic or replayed input is labelled everywhere it leaves the box.
# Live microphone input (including drone audio played through a speaker) carries no tag.
_MESH_TAGS = {"simulate": "[SIM] ", "file": "[REC] "}


def mesh_tag(source_kind: str | None) -> str:
    return _MESH_TAGS.get(source_kind or "", "")


def heartbeat_payload(host: str, url: str, mode: str, status: str, ts: float | None = None,
                      detail: str | None = None, demo_mode: str | None = None,
                      demo_active: list | None = None) -> dict:
    """isrkit heartbeat contract: {"tool","host","url","mode","status","ts"}, plus an optional
    short "detail" string the launcher shows under the row (e.g. why we're degraded)."""
    hb = {"tool": SOURCE, "host": host, "url": url, "mode": mode, "status": status,
          "ts": int(time.time() if ts is None else ts)}
    if detail:
        hb["detail"] = detail[:120]
    if demo_mode is not None:                 # DEMO-TRIGGER-CONTRACT: the launcher shows these
        hb["demo_mode"] = demo_mode
        hb["demo_active"] = list(demo_active or [])
    return hb


def advertised_url(port: int, public_url: str | None = None) -> str:
    """PUBLIC_URL if set, else http://<short-hostname>.local:<port> (e.g. isr-pi.local)."""
    return public_url or f"http://{station_host()}.local:{port}"


def alert_text(confidence: float, sensor_name: str, source_kind: str | None = None) -> str:
    return f"{mesh_tag(source_kind)}ACOUSTIC: drone detected conf {confidence:.2f} @ {sensor_name}"


def clear_text(sensor_name: str, peak: float, source_kind: str | None = None) -> str:
    return f"{mesh_tag(source_kind)}ACOUSTIC: clear @ {sensor_name} (peak {peak:.2f})"


def test_text(sensor_name: str, source_kind: str | None = None) -> str:
    return f"{mesh_tag(source_kind)}ACOUSTIC: TEST alert @ {sensor_name} (not a detection)"


class MeshBus:
    def __init__(self, host: str, port: int, user: str, password: str,
                 to: str | None = None, channel: int = 0,
                 min_interval_s: float = MESH_MIN_INTERVAL_S, clock: Callable[[], float] = time.monotonic):
        self.host, self.port = host, port
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._last_sent: float | None = None
        self.throttled = 0
        self.to, self.channel = to, channel
        self.connected = False
        self.sent = 0
        self._client = None
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="acoustic-detector")
            self._client.username_pw_set(user, password)
            # Unclean exit (crash, power loss): broker publishes a retained "error" heartbeat.
            self._client.will_set(HEARTBEAT_TOPIC, json.dumps(
                heartbeat_payload(station_host(), "", "cpu", "error")), qos=1, retain=True)
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        except Exception as e:  # noqa: BLE001
            logger.warning("MQTT client unavailable: %s", e)
            self._client = None

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = not getattr(reason_code, "is_failure", False)
        if getattr(self, "_hb_stop", None) is not None and self.connected:
            self._hb_started = False      # re-publish the heartbeat promptly after (re)connect
        logger.info("MQTT %s:%d connect: %s", self.host, self.port, reason_code)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = False

    def start(self):
        if self._client is None:
            return self
        try:
            self._client.connect_async(self.host, self.port, keepalive=30)
            self._client.loop_start()
        except Exception as e:  # noqa: BLE001
            logger.warning("MQTT start failed (%s); mesh alerts disabled until restart", e)
        return self

    def stop(self):
        if getattr(self, "_hb_stop", None) is not None:
            self._hb_stop.set()
            # clean exit: leave a retained "stopped" heartbeat for the launcher
            try:
                hb = {**self._hb_fn(), "status": "stopped"}
                if self._client is not None and self.connected:
                    info = self._client.publish(HEARTBEAT_TOPIC, json.dumps(heartbeat_payload(**hb)),
                                                qos=1, retain=True)
                    info.wait_for_publish(timeout=2.0)
            except Exception as e:  # noqa: BLE001
                logger.debug("stopped heartbeat not sent: %s", e)
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass

    def publish_json(self, topic: str, payload: dict, retain: bool = False) -> bool:
        if self._client is None or not self.connected:
            return False
        try:
            return self._client.publish(topic, json.dumps(payload), qos=1, retain=retain).rc == 0
        except Exception as e:  # noqa: BLE001
            logger.debug("MQTT publish to %s failed: %s", topic, e)
            return False

    def start_heartbeat(self, status_fn: Callable[[], dict], every_s: float = HEARTBEAT_EVERY_S):
        """status_fn() returns {"host", "url", "mode", "status"}; published retained every_s."""
        def loop():
            while not self._hb_stop.wait(0 if not self._hb_started else every_s):
                self._hb_started = True
                try:
                    hb = status_fn()
                    self.publish_json(HEARTBEAT_TOPIC, heartbeat_payload(**hb), retain=True)
                except Exception:  # noqa: BLE001
                    logger.exception("heartbeat failed")
        self._hb_fn = status_fn
        self._hb_stop = threading.Event()
        self._hb_started = False
        threading.Thread(target=loop, name="heartbeat", daemon=True).start()
        return self

    def send(self, text: str) -> bool:
        payload = build_payload(text, self.to, self.channel)
        now = self._clock()
        if self._last_sent is not None and now - self._last_sent < self.min_interval_s:
            self.throttled += 1
            logger.warning("Mesh rate limit (1 per %.0f s); not sent: %s", self.min_interval_s, payload["text"])
            return False
        if self._client is None or not self.connected:
            logger.warning("MQTT broker not connected; mesh alert not sent: %s", payload["text"])
            return False
        try:
            info = self._client.publish(TX_TOPIC, json.dumps(payload), qos=1)
            ok = info.rc == 0
        except Exception as e:  # noqa: BLE001
            logger.warning("MQTT publish failed: %s", e)
            return False
        if ok:
            self.sent += 1
            self._last_sent = now
            logger.info("Mesh alert sent: %s", payload["text"])
        return ok
