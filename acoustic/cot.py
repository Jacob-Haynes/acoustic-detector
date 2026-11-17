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
                now: float | None = None, parent_uid: str | None = None) -> str:
    """A marker OTS 1.8.3 both relays and stores (contract C5.5): a-* type, no <takv>, no contact
    endpoint, a real position (callers never send 0,0), and a <link> only to our own EUD uid."""
    now = time.time() if now is None else now
    link = ""
    if parent_uid:
        link = (f'<link uid={quoteattr(parent_uid)} relation="p-p" type="a-f-G-E-S"'
                f' production_time="{isotime(now)}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<event version="2.0" uid={quoteattr(uid)} type={quoteattr(cot_type)} how={quoteattr(how)}'
        f' time="{isotime(now)}" start="{isotime(now)}" stale="{isotime(now + stale_s)}">'
        f'<point lat="{lat:.6f}" lon="{lon:.6f}" hae="{hae:.1f}" ce="{ce:.1f}" le="9999999.0"/>'
        f'<detail><contact callsign={quoteattr(callsign)}/>'
        f"<remarks>{escape(remarks)}</remarks>{link}</detail>"
        "</event>"
    )


def detection_uid(sensor_name: str, demo: bool = False) -> str:
    """Demo detections get their own uid, so clearing the demo can never remove a real marker."""
    return f"ACOUSTIC-{sensor_name}-UAV" + ("-SIM" if demo else "")


def detection_type(affiliation: str = "u") -> str:
    return f"a-{affiliation}-A-M-F-Q"            # standard CoT air type; -A-M-F-Q = UAV/drone


def detection_event(sensor_name: str, lat: float, lon: float, hae: float, confidence: float,
                    radius_m: float, affiliation: str = "u", f0_hz: float = 0.0,
                    source: str = "", now: float | None = None, parent_uid: str | None = None,
                    demo: bool = False, stale_s: float = DETECTION_STALE_S, note: str = "") -> str:
    """UAV marker placed at the sensor (acoustics gives presence, not position). ce = rough range."""
    uid = detection_uid(sensor_name, demo)
    cot_type = detection_type(affiliation)
    remarks = remarks_prefix(source) + (
        f"ACOUSTIC drone detection by sensor {sensor_name}. conf {confidence:.2f}. "
        f"Marker is the SENSOR position: target is within ~{radius_m:.0f} m, bearing unknown."
    )
    if f0_hz:
        remarks += f" Prop tone ~{f0_hz:.0f} Hz."
    if source:
        remarks += f" Input: {'live microphone' if source == 'mic' else source}."
    if note:
        remarks += " " + note
    # how="m-f": machine-derived/fused rather than a measured target position.
    return build_event(uid, cot_type, lat, lon, hae, f"UAV? (acoustic {sensor_name})", remarks,
                       stale_s, ce=radius_m, how="m-f", now=now, parent_uid=parent_uid)


def delete_event(target_uid: str, target_type: str, lat: float, lon: float, source: str = "",
                 now: float | None = None) -> str:
    """t-x-d-d with __forcedelete: ATAK removes the linked marker at once (as op-sentry does).
    OTS 1.8.3 relays it but keeps its stored copy, so send the marker with stale = now first; and it
    marks the sender disconnected until its next registration SA, so resend that straight after.
    The point is the sensor, never 0,0 (OTS stores it against the sender)."""
    now = time.time() if now is None else now
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<event version="2.0" uid={quoteattr(f"{target_uid}-del-{int(now * 1000)}")} type="t-x-d-d"'
        f' how="h-g-i-g-o" time="{isotime(now)}" start="{isotime(now)}" stale="{isotime(now + 10)}">'
        f'<point lat="{lat:.6f}" lon="{lon:.6f}" hae="0.0" ce="9999999.0" le="9999999.0"/>'
        f'<detail><link uid={quoteattr(target_uid)} relation="none" type={quoteattr(target_type)}/>'
        f"<__forcedelete/><remarks>{escape(remarks_prefix(source) + 'Demo cleared.')}</remarks></detail>"
        "</event>"
    )


def sensor_event(sensor_name: str, lat: float, lon: float, hae: float, status: str,
                 source: str = "", now: float | None = None, parent_uid: str | None = None) -> str:
    uid = f"ACOUSTIC-{sensor_name}"
    remarks = remarks_prefix(source) + f"Acoustic drone sensor {sensor_name}: {status}"
    return build_event(uid, "a-f-G-E-S", lat, lon, hae, f"ACOUSTIC {sensor_name}", remarks,
                       SENSOR_STALE_S, ce=5.0, how="h-e", now=now, parent_uid=parent_uid)
