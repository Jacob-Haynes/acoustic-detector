"""Fuse the component scores into one confidence and turn it into detect/clear events with
hysteresis and an alert cooldown."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

EMA_ALPHA = 0.5
YAMNET_BOOST = 0.45       # max a confident YAMNet "aircraft/buzz" call can add
CONFUSER_PENALTY = 0.5    # how hard a confident "lawn mower / vehicle / speech" call pulls down


def fuse(components: dict[str, float | None]) -> float:
    """Combine component scores (each 0..1, None = unavailable) into one confidence.

    base     = spectral comb score, averaged 50/50 with calibration similarity when a drone has
               been enrolled
    + boost  = YAMNet drone-class score x YAMNet_BOOST (evidence, never a veto)
    - penalty= YAMNet confuser-class score x CONFUSER_PENALTY

    So YAMNet alone (no prop comb, no calibration match) can't cross the default 0.65 threshold,
    and a strong comb that YAMNet confidently calls "lawn mower" gets pulled down.
    """
    spectral = components.get("spectral") or 0.0
    cal = components.get("calibration")
    base = spectral if cal is None else 0.5 * spectral + 0.5 * cal
    boost = YAMNET_BOOST * (components.get("yamnet") or 0.0)
    penalty = CONFUSER_PENALTY * (components.get("confuser") or 0.0)
    fused = base + boost * (1.0 - base) - penalty
    return max(0.0, min(1.0, fused))


@dataclass
class Event:
    kind: str                    # detect | clear | suppressed
    ts: float
    confidence: float
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "ts": self.ts, "confidence": round(self.confidence, 3), **self.details}


class Hysteresis:
    """ON after `on_frames` windows >= on_thr, OFF after `off_frames` windows < off_thr.
    A fresh ON within `cooldown_s` of the previous alert is reported as "suppressed" (no alert)."""

    def __init__(self, on_thr: float, off_thr: float, on_frames: int = 2, off_frames: int = 6,
                 cooldown_s: float = 60.0):
        if off_thr > on_thr:
            raise ValueError("off threshold must be <= on threshold")
        self.on_thr, self.off_thr = on_thr, off_thr
        self.on_frames, self.off_frames = on_frames, off_frames
        self.cooldown_s = cooldown_s
        self.active = False
        self.smoothed = 0.0
        self._above = 0
        self._below = 0
        self.last_alert_ts: float | None = None
        self.peak = 0.0

    def update(self, confidence: float, now: float | None = None, details: dict | None = None) -> Event | None:
        now = time.time() if now is None else now
        self.smoothed = EMA_ALPHA * confidence + (1 - EMA_ALPHA) * self.smoothed
        c = self.smoothed
        details = details or {}
        if not self.active:
            self._above = self._above + 1 if c >= self.on_thr else 0
            if self._above >= self.on_frames:
                self.active = True
                self._below = 0
                self.peak = c
                if self.last_alert_ts is not None and now - self.last_alert_ts < self.cooldown_s:
                    return Event("suppressed", now, c, details)
                self.last_alert_ts = now
                return Event("detect", now, c, details)
        else:
            self.peak = max(self.peak, c)
            self._below = self._below + 1 if c < self.off_thr else 0
            if self._below >= self.off_frames:
                self.active = False
                self._above = 0
                return Event("clear", now, c, {**details, "peak": round(self.peak, 3)})
        return None
