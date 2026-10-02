"""High-level services shared by the dashboard and the CLI scripts."""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .config import MODELS_DIR
from .data.pcap import pcap_to_flows
from .data.schema import feature_names
from .data.windows import build_states, regularize, states_from_flows
from .models.inference import Forecaster, artifacts_present


def load_forecaster(models_dir: str | Path = MODELS_DIR) -> Forecaster | None:
    return Forecaster.load(models_dir) if artifacts_present(models_dir) else None


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_states(states: pd.DataFrame, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    states.to_csv(p, index=False)


def load_states_csv(path_or_buf, config: dict[str, Any]) -> pd.DataFrame:
    """Load a *precomputed* state table (produced by scripts/prepare_states.py)."""
    df = pd.read_csv(path_or_buf, parse_dates=["timestamp"])
    for c, d in (("stage", "Benign"), ("is_attack", False), ("source", "states"), ("segment", 0), ("filled", False),
                 ("attack_flows", 0.0), ("attack_fraction", 0.0)):
        if c not in df.columns:
            df[c] = d
    df["is_attack"] = df["is_attack"].astype(bool); df["segment"] = df["segment"].astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


def is_state_table(df: pd.DataFrame, config: dict[str, Any]) -> bool:
    names = feature_names(config)
    return sum(n in df.columns for n in names) >= int(0.8 * len(names))


def states_from_upload(name: str, data: bytes, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Uploaded file (state table CSV, flow CSV or PCAP) -> regularised states + info."""
    suffix = Path(name).suffix.lower()
    if suffix in (".pcap", ".cap"):
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data); tmp = f.name
        try:
            flows = pcap_to_flows(tmp, config)
        finally:
            Path(tmp).unlink(missing_ok=True)
        st = regularize(states_from_flows(flows, config, Path(name).name), config)
        return st, {"kind": "pcap", "flows": int(len(flows)), "windows": int(len(st))}
    head = pd.read_csv(io.BytesIO(data), nrows=5)
    if is_state_table(head, config):
        st = load_states_csv(io.BytesIO(data), config)
        return st, {"kind": "state-table", "windows": int(len(st))}
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(data); tmp = f.name
    try:
        st, infos = build_states([tmp], config, progress=None)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return st, {"kind": "flow-csv", "windows": int(len(st)), "files": infos[:1]}
