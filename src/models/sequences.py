"""Segment-aware sequence indexing, leakage-free splits and forecast targets."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def split_labels(states: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    """Per-row split label: 'train' | 'val' | 'test' | 'gap'.

    blocked (default): each source file (capture day) is cut into time blocks assigned to
    train/val/test by a repeating pattern. A sample's whole span [i-L+1, i+H] must lie inside
    ONE block (see ``sample_ends``), so no window is ever shared between splits.
    day_chronological / chronological: classic time-ordered fractions with a (L+H) gap.
    """
    L = int(config["window"]["sequence_length"]); H = int(config["window"]["forecast_horizon"])
    G = L + H
    sp = config["split"]
    lab = np.array(["gap"] * len(states), dtype=object)
    if sp["strategy"] == "chronological":
        groups = [np.arange(len(states))]
    else:
        groups = [np.flatnonzero((states["source"] == s).to_numpy()) for s in states["source"].unique()]
    if sp["strategy"] == "blocked":
        pat = {"T": "train", "V": "val", "E": "test"}
        pattern = [pat[c] for c in str(sp.get("block_pattern", "TTTVTTTETE")).upper()]
        for idx in groups:
            n = len(idx)
            B = int(max(G + 12, min(int(sp.get("block_windows", 90)), n // len(pattern))))
            blk = np.arange(n) // B
            lab[idx] = np.array(pattern, dtype=object)[blk % len(pattern)]
        return lab

    tf = float(sp["train_fraction"]); vf = float(sp["val_fraction"])
    for idx in groups:
        n = len(idx)
        a = int(n * tf); b = int(n * (tf + vf))
        lab[idx[:max(a, 0)]] = "train"
        lab[idx[min(a + G, n):max(b, 0)]] = "val"
        lab[idx[min(b + G, n):]] = "test"
    return lab


def holdout_blocks(states: pd.DataFrame, part: np.ndarray, label: str = "test",
                   min_windows: int = 1) -> list[np.ndarray]:
    """Return contiguous, segment-safe blocks belonging to one split label."""
    if len(states) != len(part):
        raise ValueError("states and split labels must have the same length")
    mask = np.asarray(part) == label
    if not mask.any():
        return []
    indices = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    groups = np.split(indices, breaks)
    return [g for g in groups if len(g) >= min_windows]


def sample_ends(states: pd.DataFrame, L: int, H: int, part: np.ndarray | None = None,
                label: str | None = None) -> np.ndarray:
    """Row indices ``i`` whose span [i-L+1, i+H] is inside one segment (and one split part)."""
    seg = states["segment"].to_numpy()
    n = len(seg)
    ends = np.arange(L - 1, n - H)
    ok = (seg[ends - L + 1] == seg[ends]) & (seg[ends] == seg[ends + H])
    if part is not None and label is not None:
        # all rows in the span must carry the same split label
        p = (part == label).astype(np.int8)
        cs = np.concatenate([[0], np.cumsum(p)])
        span = cs[ends + H + 1] - cs[ends - L + 1]
        ok &= span == (L + H)
    return ends[ok]


def inference_ends(states: pd.DataFrame, L: int) -> np.ndarray:
    """Rows with L past windows in the same segment (forecast possible; no future needed)."""
    seg = states["segment"].to_numpy()
    n = len(seg)
    ends = np.arange(L - 1, n)
    return ends[seg[ends - L + 1] == seg[ends]]


def stage_vocab(states: pd.DataFrame, part: np.ndarray, config: dict[str, Any]) -> list[str]:
    """Stages the model can predict = attack stages present in the TRAIN split only.
    (A class the model never saw cannot be learned; it is reported as 'unseen' at evaluation.)"""
    seen = set(states.loc[(part == "train") & states["is_attack"].to_numpy(), "stage"].unique())
    return ["Benign"] + [s for s in config["stage_order"][1:] if s in seen]


def stage_ids(states: pd.DataFrame, vocab: list[str]) -> np.ndarray:
    """Stage id per row; attack stages missing from the vocabulary -> -1 (unseen in training)."""
    m = {s: i for i, s in enumerate(vocab)}
    st = states["stage"].where(states["is_attack"], "Benign")
    return st.map(lambda s: m.get(s, -1)).to_numpy(dtype=np.int64)


def gather(X: np.ndarray, ends: np.ndarray, L: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    off_in = np.arange(-L + 1, 1); off_out = np.arange(1, H + 1)
    return X[ends[:, None] + off_in[None, :]], X[ends[:, None] + off_out[None, :]]


def future_targets(sid: np.ndarray, is_attack: np.ndarray, ends: np.ndarray, H: int) -> tuple[np.ndarray, np.ndarray]:
    off = np.arange(1, H + 1)
    idx = ends[:, None] + off[None, :]
    return sid[idx], is_attack[idx].astype(np.float32)
