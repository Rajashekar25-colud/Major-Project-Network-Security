"""Feature scaling (train-only fit), active-feature mask and feature-mismatch validation."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..data.schema import IP_LEVEL_FEATURES, LOG_FEATURES, PACKET_LEVEL_FEATURES


def _slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class Preprocessor:
    """signed-log1p (heavy-tailed features) -> standardise -> clip.

    Features that are constant in the training data (e.g. TTL when training on a
    flow-only CSV) are marked INACTIVE: they are neutralised to 0 at inference,
    so a model trained without a telemetry type is never fed out-of-distribution
    values for it (e.g. PCAP TTLs) and the dashboard can report the mismatch.
    """

    def __init__(self, names: list[str], clip: float = 10.0):
        self.names = list(names)
        self.clip = float(clip)
        self.log_mask = np.array([n in set(LOG_FEATURES) for n in names])
        self.mean = np.zeros(len(names)); self.std = np.ones(len(names))
        self.active = np.ones(len(names), dtype=bool)
        self.fitted = False

    def _log(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64, copy=True)
        X[:, self.log_mask] = _slog(X[:, self.log_mask])
        return X

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        X = self._log(df[self.names].to_numpy(dtype=np.float64))
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.active = std > 1e-8
        self.std = np.where(self.active, std, 1.0)
        self.fitted = True
        return self

    def transform_array(self, X: np.ndarray) -> np.ndarray:
        Z = (self._log(np.asarray(X)) - self.mean) / self.std
        Z[:, ~self.active] = 0.0
        return np.clip(np.nan_to_num(Z, nan=0.0, posinf=self.clip, neginf=-self.clip), -self.clip, self.clip).astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        missing = [n for n in self.names if n not in df.columns]
        if missing:
            raise ValueError(f"Input is missing {len(missing)} model feature(s): {missing[:8]}")
        return self.transform_array(df[self.names].to_numpy(dtype=np.float64))

    def inverse_array(self, X: np.ndarray) -> np.ndarray:
        """Convert model-space features back to raw feature units."""
        Z = np.asarray(X, dtype=np.float64) * self.std + self.mean
        if self.log_mask.any():
            Z[:, self.log_mask] = np.sign(Z[:, self.log_mask]) * np.expm1(np.abs(Z[:, self.log_mask]))
        return Z

    def to_dict(self) -> dict[str, Any]:
        return {"names": self.names, "clip": self.clip, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "active": self.active.tolist(), "log_features": [n for n, m in zip(self.names, self.log_mask) if m]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Preprocessor":
        p = cls(d["names"], d.get("clip", 10.0))
        p.mean = np.asarray(d["mean"], dtype=np.float64); p.std = np.asarray(d["std"], dtype=np.float64)
        p.active = np.asarray(d["active"], dtype=bool); p.fitted = True
        return p


def validate_features(columns: Iterable[str], pre: Preprocessor, df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Feature-mismatch report between an input state table and a trained model."""
    cols = set(columns)
    names = pre.names
    missing = [n for n in names if n not in cols]
    extra = [c for c in cols if c not in set(names) and c not in {
        "timestamp", "stage", "is_attack", "attack_flows", "attack_fraction", "source", "filled", "segment", "label"}]
    report: dict[str, Any] = {"missing": missing, "extra": sorted(extra), "ok": not missing,
                              "inactive_in_model": [n for n, a in zip(names, pre.active) if not a],
                              "unavailable_in_input": [], "novel_in_input": []}
    if df is not None and not missing:
        act = {n for n, a in zip(names, pre.active) if a}
        nz = (df[names].abs().sum(axis=0) > 0)
        report["unavailable_in_input"] = [n for n in names if n in act and not nz[n]]
        report["novel_in_input"] = [n for n in names if n not in act and nz[n]
                                    and n in set(PACKET_LEVEL_FEATURES + IP_LEVEL_FEATURES)]
    return report
