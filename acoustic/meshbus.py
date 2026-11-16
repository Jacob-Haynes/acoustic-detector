"""Mesh alerts via the blufor-tracker bridge's MQTT bus. Never touches the Meshtastic serial port.

Publishes {"text", "to", "channel", "source"} to meshbus/tx. If the broker is down the message is
logged and dropped; paho reconnects in the background."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

TX_TOPIC = "meshbus/tx"
MAX_TEXT = 200
SOURCE = "acoustic-detector"


def build_payload(text: str, to: str | None = None, channel: int = 0) -> dict:
    text = text.strip()
    if len(text) > MAX_TEXT - 1:
        text = text[: MAX_TEXT - 2] + "~"
    return {"text": text, "to": to, "channel": channel, "source": SOURCE}


def alert_text(confidence: float, sensor_name: str) -> str:
    return f"ACOUSTIC: drone detected conf {confidence:.2f} @ {sensor_name}"


def clear_text(sensor_name: str, peak: float) -> str:
    return f"ACOUSTIC: clear @ {sensor_name} (peak {peak:.2f})"


class MeshBus:
    def __init__(self, host: str, port: int, user: str, password: str,
                 to: str | None = None, channel: int = 0):
        self.host, self.port = host, port
        self.to, self.channel = to, channel
        self.connected = False
        self.sent = 0
        self._client = None
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="acoustic-detector")
            self._client.username_pw_set(user, password)
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        except Exception as e:  # noqa: BLE001
            logger.warning("MQTT client unavailable: %s", e)
            self._client = None

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = not getattr(reason_code, "is_failure", False)
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
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass

    def send(self, text: str) -> bool:
        payload = build_payload(text, self.to, self.channel)
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
            logger.info("Mesh alert sent: %s", payload["text"])
        return ok
