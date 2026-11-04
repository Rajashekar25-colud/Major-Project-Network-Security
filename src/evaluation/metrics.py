"""Forecast evaluation: binary / onset / stage metrics, baselines and warning lead time."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from ..warning.engine import WarningEngine


def binary_metrics(y_true, y_prob, thr: float) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int).ravel(); p = np.asarray(y_prob).ravel()
    pred = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"accuracy": float((y == pred).mean()), "precision": float(pr), "recall": float(rc), "f1": float(f1),
           "false_positive_rate": float(fp / max(fp + tn, 1)), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
           "support_positive": int(y.sum()), "support_total": int(len(y)), "threshold": float(thr)}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p)); out["pr_auc"] = float(average_precision_score(y, p))
        out["brier"] = float(np.mean((p - y) ** 2))
    return out


def onset_masks(is_attack: np.ndarray, ends: np.ndarray, y_future_attack: np.ndarray):
    """Samples whose *current* window is benign: positives = an attack begins within the horizon."""
    cur = is_attack[ends].astype(bool)
    quiet = ~cur
    pos = y_future_attack.max(axis=1) > 0
    return quiet, pos


def choose_threshold(risk: np.ndarray, quiet: np.ndarray, pos: np.ndarray, target_fpr: float,
                     fallback: float) -> tuple[float, dict[str, Any]]:
    """Validation-set threshold: max onset-recall s.t. onset-FPR <= target (else best onset-F1)."""
    q = quiet
    y = (pos & q).astype(int)[q]; r = risk[q]
    if y.sum() < 3 or (y == 0).sum() < 3:
        return float(fallback), {"method": "fallback (validation split has too few onset examples)",
                                 "onset_positives": int(y.sum()), "onset_negatives": int((y == 0).sum())}
    cands = np.unique(np.round(np.concatenate([np.linspace(0.05, 0.95, 91), np.quantile(r, np.linspace(0.5, 0.999, 60))]), 4))
    best, best_key = None, None
    for t in cands:
        pred = r >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum()); tn = int(((~pred) & (y == 0)).sum())
        fpr = fp / max(fp + tn, 1); rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        key = (fpr <= target_fpr, rec if fpr <= target_fpr else f1, -fpr)
        if best_key is None or key > best_key:
            best_key, best = key, (float(t), fpr, rec, prec, f1)
    t, fpr, rec, prec, f1 = best
    return t, {"method": "max recall s.t. onset-FPR<=target" if best_key[0] else "best onset-F1 (target FPR unreachable)",
               "val_onset_fpr": fpr, "val_onset_recall": rec, "val_onset_precision": prec, "val_onset_f1": f1,
               "target_fpr": target_fpr, "onset_positives": int(y.sum()), "onset_negatives": int((y == 0).sum())}


def stage_metrics(y_true: np.ndarray, y_pred: np.ndarray, vocab: list[str], unseen_id: int = -1) -> dict[str, Any]:
    """Stage forecast metrics computed ONLY over classes that have support in the hold-out."""
    yt = y_true.ravel(); yp = y_pred.ravel()
    labels_all = list(range(len(vocab)))
    counts = {vocab[i]: int((yt == i).sum()) for i in labels_all}
    unseen = int((yt == unseen_id).sum())
    m = yt >= 0
    yt, yp = yt[m], yp[m]
    present = [i for i in labels_all if counts[vocab[i]] > 0]
    if not len(yt):
        return {"note": "no labelled samples"}
    pr, rc, f1, sup = precision_recall_fscore_support(yt, yp, labels=present, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=labels_all)
    return {
        "accuracy": float((yt == yp).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)) if len(set(yt)) > 1 else None,
        "macro_precision": float(pr.mean()), "macro_recall": float(rc.mean()), "macro_f1": float(f1.mean()),
        "per_class": {vocab[i]: {"precision": float(a), "recall": float(b), "f1": float(c), "support": int(d)}
                      for i, a, b, c, d in zip(present, pr, rc, f1, sup)},
        "classes_with_support": [vocab[i] for i in present],
        "classes_without_support": [vocab[i] for i in labels_all if counts[vocab[i]] == 0],
        "support": counts, "unseen_stage_samples": unseen,
        "confusion_matrix": {"labels": vocab, "matrix": cm.tolist()},
        "note": "Accuracy/macro metrics use only classes present in the hold-out; classes without support are not scored.",
    }


# ----------------------------------------------------------------------
def episodes(is_attack: np.ndarray, segment: np.ndarray, min_gap: int = 2) -> list[tuple[int, int]]:
    """Contiguous attack episodes (row index ranges), merged across gaps <= min_gap windows."""
    out: list[list[int]] = []
    for i in np.flatnonzero(is_attack):
        if out and segment[i] == segment[out[-1][1]] and i - out[-1][1] - 1 <= min_gap:
            out[-1][1] = int(i)
        else:
            out.append([int(i), int(i)])
    return [(a, b) for a, b in out]


def lead_time_report(timeline: pd.DataFrame, states: pd.DataFrame, threshold: float, wcfg: dict[str, Any],
                     window_seconds: int, horizon: int, seq_len: int, stage_col: str = "headline_stage",
                     eval_mask: np.ndarray | None = None) -> dict[str, Any]:
    """Replay the forecast stream through the WarningEngine and score every attack episode.

    Qualifying warning : an alert OPENED before the episode start and no earlier than
                         ``lookback_windows`` (default = horizon) before it.
    Late detection     : the first alert opens at/after the onset, inside the episode.
    Missed             : no alert overlapping the episode.
    Not forecastable   : the episode starts before the first window that has a full input sequence.
    False alerts       : alerts that overlap no episode and no lookback interval.
    """
    lookback = int(wcfg.get("lookback_windows") or horizon)
    is_atk = states["is_attack"].to_numpy().copy(); seg = states["segment"].to_numpy()
    if eval_mask is not None:
        is_atk &= eval_mask
    eng_alerts: list[dict[str, Any]] = []
    for sid, g in timeline.groupby(states["segment"].to_numpy()[timeline["idx"].to_numpy()], sort=True):
        eng = WarningEngine(threshold, wcfg, window_seconds, horizon)
        for row in g.itertuples(index=False):
            eng.update(int(row.idx), row.timestamp, float(row.risk), getattr(row, stage_col),
                       [getattr(row, f"p_t{k + 1}") for k in range(horizon)])
        for a in eng.alerts:
            eng_alerts.append({"opened_idx": a.opened_idx, "last_idx": a.last_idx, "closed_idx": a.closed_idx, "segment": int(sid)})
    eps = episodes(is_atk, seg)
    tl_seg = seg[timeline["idx"].to_numpy()]
    first_idx = {int(k): int(v) for k, v in timeline["idx"].groupby(tl_seg).min().items()}
    rows = []
    used = set()
    for (s, e) in eps:
        sid = int(seg[s])
        if sid not in first_idx or s <= first_idx[sid]:
            rows.append({"start": s, "end": e, "status": "not_forecastable"}); continue
        early = [a for a in eng_alerts if a["segment"] == sid and s - lookback <= a["opened_idx"] < s]
        late = [a for a in eng_alerts if a["segment"] == sid and s <= a["opened_idx"] <= e]
        if early:
            a = min(early, key=lambda x: x["opened_idx"]); used.add(id(a))
            rows.append({"start": s, "end": e, "status": "warned_early",
                         "lead_seconds": float((s - a["opened_idx"]) * window_seconds)})
        elif late:
            a = min(late, key=lambda x: x["opened_idx"]); used.add(id(a))
            rows.append({"start": s, "end": e, "status": "detected_late",
                         "delay_seconds": float((a["opened_idx"] - s) * window_seconds)})
        else:
            rows.append({"start": s, "end": e, "status": "missed"})
    forecastable = [r for r in rows if r["status"] != "not_forecastable"]
    lead = np.array([r["lead_seconds"] for r in rows if r["status"] == "warned_early"], dtype=float)
    # false alerts: no overlap with any episode / its look-back interval
    covered = np.zeros(len(is_atk), dtype=bool)
    for (s, e) in eps:
        covered[max(s - lookback, 0):e + 1 + lookback] = True
    false_alerts = [a for a in eng_alerts if not covered[a["opened_idx"]]]
    hours = max(len(timeline) * window_seconds / 3600.0, 1e-9)
    return {
        "available": True, "attack_episodes": len(eps), "forecastable_episodes": len(forecastable),
        "not_forecastable_episodes": len(rows) - len(forecastable),
        "warned_early": int(len(lead)), "detected_late": sum(r["status"] == "detected_late" for r in rows),
        "missed": sum(r["status"] == "missed" for r in rows),
        "warning_coverage": float(len(lead) / len(forecastable)) if forecastable else None,
        "detection_coverage_any": float((len(lead) + sum(r["status"] == "detected_late" for r in rows)) / len(forecastable)) if forecastable else None,
        "mean_lead_seconds": float(lead.mean()) if len(lead) else None,
        "median_lead_seconds": float(np.median(lead)) if len(lead) else None,
        "min_lead_seconds": float(lead.min()) if len(lead) else None,
        "max_lead_seconds": float(lead.max()) if len(lead) else None,
        "total_alerts": len(eng_alerts), "false_alerts": len(false_alerts),
        "false_alerts_per_hour": float(len(false_alerts) / hours),
        "lookback_windows": lookback, "episodes": rows[:200],
        "definition": "Lead time = episode start - alert opening time for alerts opened <= lookback windows before onset.",
    }
