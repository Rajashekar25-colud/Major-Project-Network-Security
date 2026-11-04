"""Forecast validation: forecasts vs what actually happened - ONLY when the uploaded file carries ground-truth labels.
All numbers are computed from THIS run; nothing is read from the saved evaluation artifact."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.metrics import binary_metrics, lead_time_report, onset_masks
from ..models.sequences import future_targets, sample_ends, stage_ids
from .workflow import AnalysisResult

NO_TRUTH = ("Ground truth unavailable. Forecasting can be performed, but prediction-vs-actual metrics cannot be calculated for this file.")


def _clean(m: dict[str, Any]) -> dict[str, Any]:
    """Undefined ratios become None (n/a) rather than a misleading 0."""
    m = dict(m)
    if m["tp"] + m["fp"] == 0:
        m["precision"] = None
    if m["support_positive"] == 0:
        m["recall"] = None; m["f1"] = None
    if m["fp"] + m["tn"] == 0:
        m["false_positive_rate"] = None
    return m


def window_outcomes(res: AnalysisResult) -> pd.DataFrame | None:
    st, fc, rs = res.states, res.fc, res.rs
    ends = sample_ends(st, fc.L, fc.H)
    if len(ends) == 0:
        return None
    ks = np.array([rs.pos_of[int(e)] for e in ends])
    is_atk = st["is_attack"].to_numpy(); stages = st["stage"].to_numpy(dtype=object)
    _, y = future_targets(stage_ids(st, fc.stages), is_atk, ends, fc.H)
    act_stage = []
    for e in ends:
        fut = [stages[e + j] for j in range(1, fc.H + 1) if is_atk[e + j]]
        act_stage.append(max(set(fut), key=fut.count) if fut else "None")
    risk = rs.dec["risk"][ks]; pred = risk >= res.thr; act = y.max(axis=1) > 0
    return pd.DataFrame({"k": ks, "time": st["timestamp"].to_numpy()[ends], "forecast_risk": risk, "predicted_attack": pred,
                         "predicted_stage": np.where(pred, rs.dec["headline_stage"][ks], "None"), "actual_attack": act,
                         "actual_stage": act_stage, "correct": pred == act})


def validate_forecast(res: AnalysisResult) -> dict[str, Any]:
    if not res.labelled:
        return {"available": False, "reason": NO_TRUTH}
    st, fc, rs = res.states, res.fc, res.rs
    ends = sample_ends(st, fc.L, fc.H)
    if len(ends) == 0:
        return {"available": False, "reason": f"Not enough windows after the {fc.L}-window history to compare a full {fc.H}-step forecast with what actually happened."}
    ks = np.array([rs.pos_of[int(e)] for e in ends]); is_atk = st["is_attack"].to_numpy()
    _, y = future_targets(stage_ids(st, fc.stages), is_atk, ends, fc.H)
    P = rs.attack_prob[ks]; risk = P.max(axis=1)
    out: dict[str, Any] = {"available": True, "n_origins": int(len(ends)), "threshold": res.thr, "attack_windows_in_file": int(is_atk.sum()),
                           "has_attacks": bool(is_atk.any()), "forecast": _clean(binary_metrics(y, P, res.thr))}
    quiet, pos = onset_masks(is_atk, ends, y)
    out["onset"] = _clean(binary_metrics(pos[quiet], risk[quiet], res.thr)) if quiet.any() else None
    tl = pd.DataFrame({"idx": rs.ends, "timestamp": st["timestamp"].to_numpy()[rs.ends], "risk": rs.dec["risk"], "headline_stage": rs.dec["headline_stage"]})
    for j in range(fc.H):
        tl[f"p_t{j + 1}"] = rs.attack_prob[:, j]
    try:
        out["lead_time"] = lead_time_report(tl, st, res.thr, res.cfg["warning"], fc.wsec, fc.H, fc.L)
    except Exception as e:           # pragma: no cover
        out["lead_time"] = {"available": False, "error": str(e)}
    out["outcomes"] = window_outcomes(res)
    return out
