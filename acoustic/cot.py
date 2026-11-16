"""CoT XML builders for the acoustic markers. Transport (one registered, long-lived connection to
the TAK server) lives in acoustic.tak_client."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from xml.sax.saxutils import escape, quoteattr

logger = logging.getLogger(__name__)

DETECTION_STALE_S = 90
SENSOR_STALE_S = 300

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
