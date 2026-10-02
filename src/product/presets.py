"""Sensitivity presets: alert thresholds derived from the model's own validation data."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..config import ROOT, load_config
from ..evaluation.metrics import choose_threshold, onset_masks
from ..models.inference import Forecaster
from ..models.sequences import future_targets, sample_ends, split_labels, stage_ids
from ..service import load_states_csv

TARGETS = {"low_noise": 0.02, "early_warning": 0.15}


def _fallback(base: float) -> dict[str, float]:
    return {"low_noise": float(min(base + 0.04, 0.98)), "balanced": float(base), "early_warning": float(max(base * 0.7, 0.2))}


def get_presets(fc: Forecaster) -> dict[str, Any]:
    """Cached per trained model. Falls back to simple offsets if the base states file is unavailable."""
    cache = ROOT / "data" / "cache" / f"presets_{str(fc.meta.get('trained_at', 'x')).replace(':', '')}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"thresholds": _fallback(fc.threshold), "method": "offset (validation data not found)"}
    src = ROOT / "data" / "states" / "cicids2018_states.csv"
    if not src.exists():
        cands = [p for p in (ROOT / "data" / "states").glob("*.csv") if "_info" not in p.name and "test" not in p.name and "guide" not in p.name]
        src = cands[0] if cands else None
    try:
        cfg = load_config(); cfg["window"] = fc.meta["window"]; cfg["split"] = fc.meta["split"]
        st = load_states_csv(src, cfg)
        part = split_labels(st, cfg)
        va = sample_ends(st, fc.L, fc.H, part, "val")
        if len(va) < 50:
            raise ValueError("too few validation samples")
        X = fc.scale(st)
        pr = fc.predict_batch(X, va)["attack_prob"].max(axis=1)
        sid = stage_ids(st, fc.stages)
        _, y = future_targets(sid, st["is_attack"].to_numpy(), va, fc.H)
        q, p = onset_masks(st["is_attack"].to_numpy(), va, y)
        th = {k: choose_threshold(pr, q, p, t, fc.threshold)[0] for k, t in TARGETS.items()}
        th["balanced"] = float(fc.threshold)
        th["low_noise"] = float(max(th["low_noise"], th["balanced"]))
        th["early_warning"] = float(th["early_warning"] if th["early_warning"] < th["balanced"] else th["balanced"] * 0.75)
        out = {"thresholds": th, "method": "validation-calibrated (onset false-positive targets 2% / model default / 15%)"}
    except Exception as e:
        out["note"] = str(e)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def effective_threshold(fc: Forecaster, settings: dict[str, Any]) -> float:
    if settings.get("custom_threshold"):
        return float(settings["custom_threshold"])
    return float(get_presets(fc)["thresholds"].get(settings.get("sensitivity", "balanced"), fc.threshold))
