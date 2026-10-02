"""Model loading and batched forecasting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..features.preprocess import Preprocessor, validate_features
from .sequences import inference_ends
from .world_model import WorldModel

ARTIFACTS = ("world_model.pt", "metadata.json")


def artifacts_present(models_dir: str | Path) -> bool:
    d = Path(models_dir)
    return all((d / a).exists() for a in ARTIFACTS)


class Forecaster:
    def __init__(self, model: WorldModel, meta: dict[str, Any], device: str = "cpu"):
        self.model = model.to(device).eval()
        self.meta = meta
        self.device = device
        self.pre = Preprocessor.from_dict(meta["preprocessing"])
        self.names: list[str] = meta["feature_names"]
        self.stages: list[str] = meta["stage_names"]
        self.L = int(meta["window"]["sequence_length"])
        self.H = int(meta["window"]["forecast_horizon"])
        self.wsec = int(meta["window"]["seconds"])
        self.threshold = float(meta["warning_threshold"])
        self.active = torch.tensor(self.pre.active, dtype=torch.bool)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, models_dir: str | Path, device: str = "cpu") -> "Forecaster":
        d = Path(models_dir)
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        hp = meta["model"]
        model = WorldModel(len(meta["feature_names"]), len(meta["stage_names"]), int(meta["window"]["forecast_horizon"]),
                           hidden_size=hp["hidden_size"], layers=hp["layers"], dropout=hp["dropout"], heads=hp["attention_heads"])
        model.load_state_dict(torch.load(d / "world_model.pt", map_location=device, weights_only=True))
        return cls(model, meta, device)

    # ------------------------------------------------------------------
    def validate(self, states: pd.DataFrame) -> dict[str, Any]:
        return validate_features(states.columns, self.pre, states)

    def scale(self, states: pd.DataFrame) -> np.ndarray:
        return self.pre.transform(states)

    @torch.no_grad()
    def predict_batch(self, X: np.ndarray, ends: np.ndarray, batch: int = 1024, want_state: bool = False) -> dict[str, np.ndarray]:
        """Forecast from scaled states X at each end index (needs L past windows)."""
        ap, sp, st = [], [], []
        for i in range(0, len(ends), batch):
            e = ends[i:i + batch]
            xin = X[e[:, None] + np.arange(-self.L + 1, 1)[None, :]]
            o = self.model(torch.from_numpy(xin).to(self.device))
            ap.append(torch.sigmoid(o["attack_logit"]).cpu().numpy())
            logits = o["stage_logits"]
            sp.append(torch.softmax(logits, -1).cpu().numpy())
            if want_state:
                st.append(o["state"].cpu().numpy())
        out = {"attack_prob": np.concatenate(ap) if ap else np.zeros((0, self.H)),
               "stage_prob": np.concatenate(sp) if sp else np.zeros((0, self.H, len(self.stages)))}
        if want_state:
            out["state"] = np.concatenate(st)
        return out

    def decode(self, attack_prob: np.ndarray, stage_prob: np.ndarray, thr: float | None = None) -> dict[str, Any]:
        """Per-step predicted stage: dominant attack stage if p>=thr else Benign."""
        thr = self.threshold if thr is None else thr
        risk = attack_prob.max(axis=-1)
        if len(self.stages) > 1:
            att = stage_prob[..., 1:]
            top = att.argmax(-1) + 1
            top_p = np.take_along_axis(stage_prob, top[..., None], -1)[..., 0]
        else:
            top = np.zeros(attack_prob.shape, dtype=int); top_p = attack_prob
        stage_idx = np.where(attack_prob >= thr, top, 0)
        names = np.array(self.stages, dtype=object)[stage_idx]
        # headline stage = stage of the earliest above-threshold step, else strongest attack stage
        head = []
        for k in range(attack_prob.shape[0]):
            above = np.flatnonzero(attack_prob[k] >= thr)
            if len(above):
                head.append(self.stages[int(top[k, above[0]])] if len(self.stages) > 1 else "Attack")
            else:
                head.append("Benign")
        return {"risk": risk, "stage_idx": stage_idx, "stage_names": names, "headline_stage": np.array(head, dtype=object),
                "top_stage_prob": top_p}

    def timeline(self, states: pd.DataFrame, X: np.ndarray | None = None) -> pd.DataFrame:
        """Forecast for every forecastable window of a state table (vectorised)."""
        X = self.scale(states) if X is None else X
        ends = inference_ends(states, self.L)
        p = self.predict_batch(X, ends)
        d = self.decode(p["attack_prob"], p["stage_prob"])
        df = pd.DataFrame({"idx": ends, "timestamp": states["timestamp"].to_numpy()[ends], "risk": d["risk"],
                           "headline_stage": d["headline_stage"]})
        for k in range(self.H):
            df[f"p_t{k + 1}"] = p["attack_prob"][:, k]
        return df
