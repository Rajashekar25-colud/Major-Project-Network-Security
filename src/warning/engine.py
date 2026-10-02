"""Centralised early-warning engine.

Turns the raw per-window forecast (risk = max attack probability over the horizon,
plus a predicted ATT&CK stage) into analyst-grade alerts with:

  * severity states  : OK -> WATCH -> WARNING -> CRITICAL
  * persistence      : an alert opens only after N consecutive windows above threshold
  * hysteresis       : it stays open until risk < clear_ratio*thr for M consecutive windows
  * de-duplication   : while open, new above-threshold windows *update* the alert
  * cooldown         : after closing, an alert for the same stage cannot re-open for K windows
  * stage progression: the ordered list of stages predicted during the alert
The same engine is used by the dashboard and by the offline lead-time evaluation,
so reported lead times describe exactly the alerts an analyst would have seen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SEVERITY_RANK = {"OK": 0, "WATCH": 1, "WARNING": 2, "CRITICAL": 3}


@dataclass
class Alert:
    id: int
    stage: str
    severity: str
    opened_idx: int
    opened_time: Any
    last_idx: int
    last_time: Any
    peak_risk: float
    eta_seconds: float
    status: str = "OPEN"
    n_windows: int = 1
    stages: list[str] = field(default_factory=list)
    peak_severity: str = "WARNING"
    closed_idx: int | None = None
    closed_time: Any = None
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stages"] = " > ".join(self.stages)
        d["duration_windows"] = (self.last_idx - self.opened_idx) + 1
        return d


class WarningEngine:
    def __init__(self, threshold: float, wcfg: dict[str, Any], window_seconds: float, horizon: int):
        self.thr = float(threshold)
        self.cfg = wcfg
        self.wsec = float(window_seconds)
        self.horizon = int(horizon)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.alerts: list[Alert] = []
        self.open: Alert | None = None
        self._above = 0
        self._below = 0
        self._cooldown: dict[str, int] = {}      # stage -> idx until which re-alerts are suppressed
        self._next_id = 1
        self.level = "OK"

    def new_segment(self, idx: int, time: Any) -> None:
        """Capture discontinuity: close any open alert, clear streaks, keep the alert history."""
        if self.open is not None:
            self.open.status = "CLOSED"; self.open.closed_idx, self.open.closed_time = idx, time
            self.open = None
        self._above = self._below = 0
        self._cooldown.clear()
        self.level = "OK"

    def level_of(self, risk: float) -> str:
        t = self.thr
        if risk >= t + float(self.cfg["critical_margin"]) * (1.0 - t):
            return "CRITICAL"
        if risk >= t:
            return "WARNING"
        if risk >= float(self.cfg["watch_ratio"]) * t:
            return "WATCH"
        return "OK"

    # ------------------------------------------------------------------
    def update(self, idx: int, time: Any, risk: float, stage: str, probs: list[float] | None = None,
               context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Feed one window's forecast. Returns {level, events, alert}."""
        events: list[dict[str, Any]] = []
        level = self.level_of(risk)
        self.level = level
        above = SEVERITY_RANK[level] >= SEVERITY_RANK["WARNING"]
        self._above = self._above + 1 if above else 0
        clear = risk < float(self.cfg["clear_ratio"]) * self.thr
        self._below = self._below + 1 if clear else 0

        a = self.open
        if a is None:
            cooling = self._cooldown.get(stage, -1) >= idx
            if above and self._above >= int(self.cfg["persistence_windows"]) and not cooling:
                eta = self.wsec
                if probs:
                    for k, p in enumerate(probs, start=1):
                        if p >= self.thr:
                            eta = k * self.wsec
                            break
                a = Alert(id=self._next_id, stage=stage, severity=level, peak_severity=level, opened_idx=idx,
                          opened_time=time, last_idx=idx, last_time=time, peak_risk=risk, eta_seconds=eta,
                          n_windows=self._above, stages=[stage], context=context or {})
                self._next_id += 1
                self.alerts.append(a); self.open = a
                events.append({"type": "opened", "alert_id": a.id, "idx": idx, "severity": level, "stage": stage})
        else:
            if above:
                a.last_idx, a.last_time = idx, time
                a.n_windows += 1
                if risk > a.peak_risk:
                    a.peak_risk = risk
                if stage not in a.stages and stage != "Benign":
                    a.stages.append(stage)
                if SEVERITY_RANK[level] > SEVERITY_RANK[a.peak_severity]:
                    a.peak_severity = level
                    events.append({"type": "escalated", "alert_id": a.id, "idx": idx, "severity": level})
                a.severity = level
                if context:
                    a.context = context
            if self._below >= int(self.cfg["clear_windows"]):
                a.status = "CLOSED"; a.closed_idx, a.closed_time = idx, time
                self._cooldown[a.stage] = idx + int(self.cfg["cooldown_windows"])
                for s in a.stages:
                    self._cooldown[s] = idx + int(self.cfg["cooldown_windows"])
                events.append({"type": "closed", "alert_id": a.id, "idx": idx})
                self.open = None
        return {"level": level, "events": events, "alert": self.open}
