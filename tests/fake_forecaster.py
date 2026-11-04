"""Torch-free stand-in for ``Forecaster`` used ONLY by the workflow/UI-layer tests (ingest, forecast table, alerts, labelled
validation). Same interface and same decode rule as the real class; the real trained model is exercised in
test_analysis_real_model.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import feature_names
from src.features.preprocess import Preprocessor, validate_features


class FakeForecaster:
    def __init__(self, cfg, states: pd.DataFrame, L=12, H=6, threshold=0.6):
        self.names = feature_names(cfg); self.pre = Preprocessor(self.names).fit(states)
        self.stages = ["Benign", "Initial Access", "Credential Access", "Lateral Movement"]
        self.L, self.H, self.wsec, self.threshold = L, H, int(cfg["window"]["seconds"]), threshold
        self.meta = {"attack_tactics": {"Initial Access": "TA0001", "Credential Access": "TA0006", "Lateral Movement": "TA0008"}, "model": {"type": "lstm"}, "stage_names": self.stages}
        self._i = self.names.index("flow_count"); self._j = self.names.index("failed_flow_ratio")

    def validate(self, states):
        return validate_features(states.columns, self.pre, states)

    def scale(self, states):
        return self.pre.transform(states)

    def predict_batch(self, X, ends, batch=1024, want_state=False):
        sig = lambda z: 1 / (1 + np.exp(-z))
        last = X[ends]; base = 1.5 * last[:, self._i] + 1.0 * last[:, self._j]
        P = np.stack([sig(base + 0.15 * s - 1.0) for s in range(self.H)], 1)
        lg = np.stack([np.stack([np.zeros_like(base), base, 0.5 * base, -base], 1) for _ in range(self.H)], 1)
        e = np.exp(lg - lg.max(-1, keepdims=True))
        return {"attack_prob": P, "stage_prob": e / e.sum(-1, keepdims=True)}

    def decode(self, attack_prob, stage_prob, thr=None):          # same rule as Forecaster.decode
        thr = self.threshold if thr is None else thr
        risk = attack_prob.max(axis=-1); top = stage_prob[..., 1:].argmax(-1) + 1
        top_p = np.take_along_axis(stage_prob, top[..., None], -1)[..., 0]
        idx = np.where(attack_prob >= thr, top, 0); names = np.array(self.stages, dtype=object)[idx]
        head = [self.stages[int(top[k, np.flatnonzero(attack_prob[k] >= thr)[0]])] if (attack_prob[k] >= thr).any() else "Benign" for k in range(attack_prob.shape[0])]
        return {"risk": risk, "stage_idx": idx, "stage_names": names, "headline_stage": np.array(head, dtype=object), "top_stage_prob": top_p}
