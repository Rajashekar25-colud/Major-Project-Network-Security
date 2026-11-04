"""One source of truth for a New Analysis. Calls the existing components (Forecaster -> ReplaySession -> WarningEngine ->
explainers) and exposes read-only views for the UI. No model logic is duplicated here."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..replay import ReplaySession
from .integrity import fingerprint

STATE_LABEL = {"OK": "NORMAL", "WATCH": "WATCH", "WARNING": "WARNING", "CRITICAL": "HIGH RISK"}
STATE_RANK = {"NORMAL": 0, "WATCH": 1, "WARNING": 2, "HIGH RISK": 3}


@dataclass
class AnalysisResult:
    id: str
    name: str
    created: datetime
    states: pd.DataFrame
    rs: ReplaySession
    thr: float
    fc: Any
    cfg: dict[str, Any]
    info: dict[str, Any]
    stats: dict[str, Any]
    labelled: bool
    integrity_before: dict[str, Any] | None = None
    integrity_after: dict[str, Any] | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.rs.ends))


def run_analysis(states: pd.DataFrame, name: str, fc, cfg: dict[str, Any], thr: float | None = None, info: dict[str, Any] | None = None,
                 stats: dict[str, Any] | None = None, models_dir=None, progress: Callable[[str, int, int], None] | None = None,
                 analysis_id: str | None = None, created: datetime | None = None) -> AnalysisResult:
    """Forecast every window with the trained model, then replay the alert engine. progress(stage, done, total) is real:
    stage in {"sequence","forecast","risk"}. The model is never trained or modified."""
    emit = progress or (lambda *_: None)
    info = info or {}
    thr = float(fc.threshold if thr is None else thr)
    before = fingerprint(models_dir) if models_dir else None
    t0 = time.time(); emit("sequence", 0, 1); seen = {"f": False}

    def _fp(done: int, total: int) -> None:
        if not seen["f"]:
            seen["f"] = True; emit("sequence", 1, 1)
        emit("forecast", done, total)

    rs = ReplaySession(states, fc, cfg, thr, progress=_fp)
    t1 = time.time()
    if not seen["f"]:
        emit("sequence", 1, 1)
    n = len(rs.ends); chunk = max(25, n // 40); done = 0
    emit("risk", 0, n)
    while not rs.finished:
        done += rs.step(chunk); emit("risk", min(done, n), n)
    after = fingerprint(models_dir) if models_dir else None
    return AnalysisResult(id=analysis_id or uuid.uuid4().hex[:10], name=name, created=created or datetime.now(), states=rs.states, rs=rs, thr=thr,
                          fc=fc, cfg=cfg, info=info, stats=stats or {}, labelled=bool(info.get("labelled", False)),
                          integrity_before=before, integrity_after=after, timings={"forecast_s": t1 - t0, "alerts_s": time.time() - t1})


def apply_threshold(res: AnalysisResult, thr: float) -> None:
    """Re-decode and re-run the alert engine at another threshold (model outputs unchanged)."""
    res.rs.set_threshold(float(thr)); res.thr = float(thr); res.rs.step(10 ** 9)


def level_label(res: AnalysisResult, risk: float) -> str:
    return STATE_LABEL[res.rs.engine.level_of(float(risk))]


def window_time(res: AnalysisResult, k: int) -> pd.Timestamp:
    return pd.Timestamp(res.states["timestamp"].iat[int(res.rs.ends[k])])


def forecast_frame(res: AnalysisResult, k: int) -> pd.DataFrame:
    """The K-step forecast issued at window k: one row per future step, straight from the model's outputs."""
    rs, fc, st = res.rs, res.fc, res.states
    probs, sp = rs.attack_prob[k], rs.stage_prob[k]
    row = int(rs.ends[k]); multi = len(fc.stages) > 1; rows = []
    for s in range(fc.H):
        p = float(probs[s]); top = int(sp[s][1:].argmax()) + 1 if multi else 0
        r: dict[str, Any] = {"step": s + 1, "seconds": (s + 1) * fc.wsec, "label": f"+{(s + 1) * fc.wsec}s", "risk": p,
                             "state": STATE_LABEL[rs.engine.level_of(p)], "above": p >= res.thr, "stage": fc.stages[top] if p >= res.thr else "None",
                             "likely_stage": fc.stages[top], "stage_conf": float(sp[s][top])}
        t = row + s + 1
        if res.labelled and t < len(st) and st["segment"].iat[t] == st["segment"].iat[row]:
            a = bool(st["is_attack"].iat[t])
            r.update(actual_attack=a, actual_stage=str(st["stage"].iat[t]) if a else "None", target_time=pd.Timestamp(st["timestamp"].iat[t]))
        else:
            r.update(actual_attack=None, actual_stage=None, target_time=None)
        rows.append(r)
    return pd.DataFrame(rows)


def forecast_span(frame: pd.DataFrame) -> tuple[str, str] | None:
    a = frame[frame["above"]]
    return None if a.empty else (a["label"].iloc[0], a["label"].iloc[-1])


def expected_now(res: AnalysisResult, k: int) -> dict[str, Any]:
    """What the model forecast one window EARLIER for the current window (None when no earlier forecast exists)."""
    rs = res.rs; pos = rs.pos_of.get(int(rs.ends[k]) - 1)
    if pos is None:
        return {"risk": None, "state": None, "stage": None}
    p = float(rs.attack_prob[pos, 0]); sp = rs.stage_prob[pos, 0]
    top = int(sp[1:].argmax()) + 1 if len(sp) > 1 else 0
    return {"risk": p, "state": level_label(res, p), "stage": res.fc.stages[top] if p >= res.thr else "None"}


def current_summary(res: AnalysisResult, k: int) -> dict[str, Any]:
    rs, st = res.rs, res.states
    row = int(rs.ends[k]); risk = float(rs.dec["risk"][k]); head = str(rs.dec["headline_stage"][k])
    out = {"k": k, "row": row, "window_no": k + 1, "n": res.n, "time": window_time(res, k), "risk": risk, "state": level_label(res, risk),
           "stage": "None" if head == "Benign" else head, "flows": float(st["flow_count"].iat[row]), "packets": float(st["packets_total"].iat[row]),
           "bytes": float(st["bytes_total"].iat[row]), "filled": bool(st["filled"].iat[row])}
    if res.labelled:
        out["actual_attack"] = bool(st["is_attack"].iat[row]); out["actual_stage"] = str(st["stage"].iat[row]) if out["actual_attack"] else "None"
    return out


def observed_frame(res: AnalysisResult, k: int) -> pd.DataFrame:
    """The L windows the model sees at window k (oldest first) + the +1 forecast issued for each one window earlier."""
    rs, st, L = res.rs, res.states, res.fc.L
    row = int(rs.ends[k]); rows = []
    for r in range(row - L + 1, row + 1):
        pos = rs.pos_of.get(r - 1)
        rows.append({"offset": r - row, "time": pd.Timestamp(st["timestamp"].iat[r]), "flows": float(st["flow_count"].iat[r]),
                     "packets": float(st["packets_total"].iat[r]), "bytes": float(st["bytes_total"].iat[r]),
                     "prior_forecast": float(rs.attack_prob[pos, 0]) if pos is not None else np.nan,
                     "actual_attack": bool(st["is_attack"].iat[r]) if res.labelled else None})
    return pd.DataFrame(rows)


def alert_records(res: AnalysisResult) -> list[dict[str, Any]]:
    """Alerts produced by the project's WarningEngine (ordinal numbering only - no invented IDs)."""
    rs, fc = res.rs, res.fc; out = []
    for n, a in enumerate(rs.engine.alerts, start=1):
        probs = rs.attack_prob[a.opened_idx]; cross = np.flatnonzero(probs >= res.thr)
        s0 = int(cross[0]) if len(cross) else int(np.argmax(probs))
        out.append({"n": n, "k_open": int(a.opened_idx), "k_last": int(a.last_idx), "k_close": None if a.closed_idx is None else int(a.closed_idx),
                    "opened_time": a.opened_time, "closed_time": a.closed_time, "stage": a.stage, "stages": list(a.stages),
                    "severity": STATE_LABEL.get(a.peak_severity, a.peak_severity), "peak_risk": float(a.peak_risk), "step": s0 + 1,
                    "seconds_ahead": (s0 + 1) * fc.wsec, "risk_at_step": float(probs[s0]), "windows": int(a.n_windows), "status": a.status})
    return out


def alert_open_at(res: AnalysisResult, k: int) -> dict[str, Any] | None:
    for a in alert_records(res):
        if a["k_open"] <= k and (a["k_close"] is None or k < a["k_close"]):
            return a
    return None


def alert_reason(res: AnalysisResult, a: dict[str, Any]) -> str:
    p = res.cfg["warning"]["persistence_windows"]
    return (f"The forecast for +{a['seconds_ahead']}s reached {a['risk_at_step']:.0%}, at or above the configured alert threshold ({res.thr:.0%}), "
            f"and stayed above it for {p} consecutive windows (persistence rule).")


def risk_text(res: AnalysisResult, k: int, frame: pd.DataFrame | None = None) -> str:
    frame = forecast_frame(res, k) if frame is None else frame
    span = forecast_span(frame); peak = frame.loc[frame["risk"].idxmax()]; horizon = res.fc.H * res.fc.wsec
    if span:
        first = frame[frame["above"]].iloc[0]
        return (f"The model forecasts attack activity ({first['stage']}) from {span[0]}: probability {first['risk']:.0%} against an alert threshold of "
                f"{res.thr:.0%}; the highest value in the next {horizon}s is {peak['risk']:.0%} at {peak['label']}.")
    return (f"No forecast step reaches the alert threshold ({res.thr:.0%}). The highest probability in the next {horizon}s is {peak['risk']:.0%} at {peak['label']}.")


def traffic_evidence(res: AnalysisResult, k: int, features: list[str], limit: int = 8) -> pd.DataFrame:
    """Observed raw values of the given features in the model's input window: current vs mean of the earlier windows."""
    st, L = res.states, res.fc.L; row = int(res.rs.ends[k]); out = []
    for f in features:
        if f not in st.columns:
            continue
        v = st[f].to_numpy(dtype=float)[row - L + 1:row + 1]
        cur = float(v[-1]); base = float(v[:-1].mean()) if len(v) > 1 else float("nan")
        chg = (cur - base) / abs(base) if np.isfinite(base) and base != 0.0 else None
        out.append({"feature": f, "current": cur, "earlier_mean": base, "change": chg})
        if len(out) >= limit:
            break
    return pd.DataFrame(out, columns=["feature", "current", "earlier_mean", "change"])


def explain_step(res: AnalysisResult, k: int, step: int, top_k: int = 10) -> dict[str, Any]:
    """Real gradient x input + attention explanation of forecast step ``step`` (1-based) at window k."""
    from ..explain.explainer import explain_window       # lazy: needs torch
    ex = explain_window(res.fc, res.rs.sequence(k), top_k, horizon_idx=step - 1)
    try:
        res.fc.model.zero_grad(set_to_none=True)          # gradients flow to the INPUT only; weights are never updated
    except Exception:
        pass
    ex["evidence"] = traffic_evidence(res, k, [f["feature"] for f in ex["top_features"]])
    ex["attention_available"] = str(res.fc.meta.get("model", {}).get("type", "lstm")).lower() != "transformer"
    return ex


def shap_step(res: AnalysisResult, k: int, step: int, top_k: int = 10) -> dict[str, Any]:
    from pathlib import Path

    from ..config import MODELS_DIR
    from ..explain.explainer import explain_shap
    bgp = Path(MODELS_DIR) / "background_sequences.npy"
    out = explain_shap(res.fc, res.rs.sequence(k), np.load(bgp) if bgp.exists() else None, top_k, horizon_idx=step - 1)
    try:
        res.fc.model.zero_grad(set_to_none=True)
    except Exception:
        pass
    return out


def attack_mapping(meta: dict[str, Any], cfg: dict[str, Any], stage: str) -> dict[str, Any]:
    """Stage -> ATT&CK tactic exactly as configured. Techniques are NOT modelled."""
    tid = meta.get("attack_tactics", {}).get(stage) or cfg.get("attack_tactics", {}).get(stage)
    sm = cfg.get("stage_mapping", {})
    return {"stage": stage, "tactic_id": tid, "url": f"https://attack.mitre.org/tactics/{tid}/" if tid else None,
            "dataset_labels": [k for k, v in sm.get("exact", {}).items() if v == stage],
            "label_patterns": [p for p, v in sm.get("regex", []) if v == stage], "techniques_modelled": False}
