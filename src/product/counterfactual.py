"""Counterfactual network-state generation, future roll-out and trajectory comparison.

The world model is rolled forward autoregressively from the current traffic history. For each
defensive scenario the *predicted* future network states are modified (in raw feature units) before being
fed back, so the model "imagines" the traffic that would follow. The attack-probability trajectories of
all scenarios are then compared with the no-action roll-out.

IMPORTANT: this is a model-based what-if simulation. The scenario effects are assumptions (configs/product.yaml);
the output is not evidence that a real control would behave this way.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from ..models.inference import Forecaster

RATIO_FEATURES = {"scan_score", "failed_flow_ratio", "small_flow_ratio", "syn_ratio", "rst_ratio", "well_known_dst_port_ratio"}


@torch.no_grad()
def rollout(fc: Forecaster, seq: np.ndarray, steps: int, scale: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    """seq: (L, F) standardised states. Returns per-step attack probability, tactic probabilities and raw states."""
    m = fc.model.eval()
    x = torch.from_numpy(np.asarray(seq, dtype=np.float32)).unsqueeze(0)
    idx = {n: i for i, n in enumerate(fc.names)}
    probs, stages, raws = [], [], []
    for _ in range(steps):
        o = m(x)
        nxt = o["state"][:, 0].numpy().astype(np.float64)
        raw = fc.pre.inverse_array(nxt)
        if scale:
            for name, f in scale.items():
                if name in idx:
                    v = raw[:, idx[name]] * float(f)
                    raw[:, idx[name]] = np.clip(v, 0, 1) if name in RATIO_FEATURES else np.maximum(v, 0)
            nxt = fc.pre.transform_array(raw).astype(np.float64)
        raws.append(raw[0]); probs.append(float(torch.sigmoid(o["attack_logit"][0, 0])))
        stages.append(torch.softmax(o["stage_logits"][0, 0], -1).numpy())
        x = torch.cat([x[:, 1:], torch.from_numpy(nxt.astype(np.float32)).unsqueeze(1)], dim=1)
    return {"attack_prob": np.array(probs), "stage_prob": np.array(stages), "raw_states": np.array(raws)}


def compare(fc: Forecaster, seq: np.ndarray, steps: int, scenarios: dict[str, dict[str, Any]], thr: float) -> dict[str, Any]:
    base = rollout(fc, seq, steps)
    runs = {"no_action": {"label": "No action", **base}}
    for k, sc in scenarios.items():
        runs[k] = {"label": sc["label"], **rollout(fc, seq, steps, sc["scale"])}
    rows = []
    for k, r in runs.items():
        p = r["attack_prob"]
        above = np.flatnonzero(p >= thr)
        rows.append({"scenario": r["label"], "key": k, "mean_risk": float(p.mean()), "peak_risk": float(p.max()),
                     "windows_above_threshold": int((p >= thr).sum()),
                     "first_alert_step": int(above[0] + 1) if len(above) else None,
                     "risk_reduction_vs_no_action": float(base["attack_prob"].mean() - p.mean())})
    return {"runs": runs, "summary": pd.DataFrame(rows), "steps": steps, "wsec": fc.wsec}
