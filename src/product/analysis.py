"""Full-range analysis of a traffic source: forecasts -> alerts -> incidents -> summary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..explain.explainer import explain_window
from ..models.inference import Forecaster
from ..replay import ReplaySession
from .config import SEVERITY_LABEL
from .knowledge import driver_sentences
from .store import incident_id


@dataclass
class Analysis:
    source: str
    states: pd.DataFrame
    session: ReplaySession
    incidents: pd.DataFrame
    summary: dict[str, Any]
    threshold: float


def build_incidents(rs: ReplaySession, source: str, fc: Forecaster) -> pd.DataFrame:
    rows = []
    st = rs.states
    labelled = bool(st["is_attack"].any())
    atk = st["is_attack"].to_numpy()
    for a in rs.engine.alerts:
        row_open = int(rs.ends[a.opened_idx])
        row_last = int(rs.ends[min(a.last_idx, len(rs.ends) - 1)])
        lo = max(row_open - fc.H, 0); hi = min(row_last + fc.H, len(atk) - 1)
        confirmed = int(atk[lo:hi + 1].any()) if labelled else -1
        rows.append(dict(id=incident_id(source, a.opened_time, a.stage), source=source, opened_time=a.opened_time,
                         closed_time=a.closed_time if a.closed_time is not None else pd.NaT, stage=a.stage,
                         progression=" > ".join(a.stages), severity=SEVERITY_LABEL[a.peak_severity], peak_risk=a.peak_risk,
                         eta_seconds=a.eta_seconds, windows=int(a.n_windows), confirmed=confirmed, row_open=row_open, k_open=int(a.opened_idx)))
    return pd.DataFrame(rows)


def summarize(rs: ReplaySession, inc: pd.DataFrame, fc: Forecaster) -> dict[str, Any]:
    n = len(rs.ends)
    risk = rs.dec["risk"][:n]
    hours = len(rs.states) * fc.wsec / 3600
    sev = inc["severity"].value_counts().to_dict() if len(inc) else {}
    top = inc["stage"].value_counts().idxmax() if len(inc) else None
    labelled = bool(rs.states["is_attack"].any())
    s = {"windows": int(len(rs.states)), "hours": float(hours), "incidents": int(len(inc)), "by_severity": sev,
         "top_tactic": top, "peak_risk": float(risk.max()) if n else 0.0, "mean_risk": float(risk.mean()) if n else 0.0,
         "pct_elevated": float((risk >= rs.thr * 0.7).mean()) if n else 0.0, "labelled": labelled,
         "start": rs.states["timestamp"].min(), "end": rs.states["timestamp"].max()}
    if labelled and len(inc):
        s["confirmed_share"] = float((inc["confirmed"] == 1).mean())
    return s


def run_analysis(states: pd.DataFrame, source: str, fc: Forecaster, cfg: dict[str, Any], thr: float) -> Analysis:
    rs = ReplaySession(states, fc, cfg, thr)
    rs.step(10 ** 9)
    inc = build_incidents(rs, source, fc)
    return Analysis(source, states, rs, inc, summarize(rs, inc, fc), thr)


def incident_drivers(a: Analysis, fc: Forecaster, k_open: int, top_k: int = 8) -> dict[str, Any]:
    seq = a.session.sequence(k_open)
    ex = explain_window(fc, seq, top_k)
    ex["sentences"] = driver_sentences(ex["top_features"], 4)
    return ex
