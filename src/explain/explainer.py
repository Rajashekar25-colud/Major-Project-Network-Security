"""Explanations for a forecast: temporal attention, gradient x input, optional SHAP."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..models.inference import Forecaster


def explain_window(fc: Forecaster, seq: np.ndarray, top_k: int = 10, horizon_idx: int | None = None) -> dict[str, Any]:
    """seq: (L, F) scaled states. Returns temporal attention + per-feature attribution.

    attribution = sum over time of (d risk / d x) * x  (gradient x input): signed
    contribution of every traffic feature to the forecast risk. Deterministic, fast,
    and needs no background data.
    """
    m = fc.model.eval()
    x = torch.from_numpy(np.asarray(seq, dtype=np.float32)).unsqueeze(0).requires_grad_(True)
    with torch.enable_grad():
        o = m(x, need_attention=True)
        p = torch.sigmoid(o["attack_logit"])
        risk = p.max(1).values if horizon_idx is None else p[:, horizon_idx]
        risk.sum().backward()
    grad = x.grad[0].numpy()
    attr = grad * seq
    attr[:, ~fc.pre.active] = 0.0
    feat = attr.sum(axis=0)
    mag = np.abs(attr).sum(axis=0)
    total = mag.sum() + 1e-12
    att = o["attention"][0].detach().mean(0)[-1].numpy()      # last query over history, heads averaged
    order = np.argsort(-np.abs(feat))[:top_k]
    last = seq[-1]
    top = [{"feature": fc.names[i], "attribution": float(feat[i]), "share": float(mag[i] / total),
            "direction": "raises risk" if feat[i] > 0 else "lowers risk", "current_z": float(last[i])} for i in order]
    # which window in the past mattered most (by attribution magnitude)
    per_window = np.abs(attr).sum(axis=1)
    return {"method": "gradient x input + temporal attention", "risk": float(risk.detach()),
            "temporal_attention": att.tolist(), "temporal_attribution": (per_window / (per_window.sum() + 1e-12)).tolist(),
            "top_features": top, "attribution_matrix": attr}


def explain_shap(fc: Forecaster, seq: np.ndarray, background: np.ndarray | None, top_k: int = 10, n_background: int = 30) -> dict[str, Any]:
    """True SHAP values (GradientExplainer) for the max-horizon attack risk.

    Optional: requires ``pip install shap``. Background = scaled training sequences
    of shape (n, L, F) saved by the trainer.
    """
    try:
        import shap
    except Exception as e:  # pragma: no cover
        return {"available": False, "reason": f"shap not installed ({e}). Run: python -m pip install shap"}
    L = fc.L
    if background is None or len(background) < 5:
        return {"available": False, "reason": "no background sequences available"}
    rng = np.random.default_rng(0)
    bg = np.asarray(background, dtype=np.float32)
    bg = bg[rng.choice(len(bg), min(n_background, len(bg)), replace=False)]

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):
            return torch.sigmoid(self.m(x)["attack_logit"]).max(1, keepdim=True).values

    w = Wrap(fc.model.eval())
    # cuDNN LSTM backward requires train mode; CPU is fine either way.
    with torch.backends.cudnn.flags(enabled=False):
        e = shap.GradientExplainer(w, torch.from_numpy(bg))
        sv = e.shap_values(torch.from_numpy(np.asarray(seq, dtype=np.float32)).unsqueeze(0))
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv).reshape(L, -1)
    sv[:, ~fc.pre.active] = 0.0
    feat = sv.sum(axis=0); mag = np.abs(sv).sum(axis=0)
    order = np.argsort(-np.abs(feat))[:top_k]
    return {"available": True, "method": "SHAP GradientExplainer", "top_features": [
        {"feature": fc.names[i], "shap_value": float(feat[i]), "share": float(mag[i] / (mag.sum() + 1e-12))} for i in order]}
