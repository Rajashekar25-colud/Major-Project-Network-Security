"""Configuration loading and project paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
MODELS_DIR = ROOT / "models"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    w = cfg["window"]
    for k in ("seconds", "sequence_length", "forecast_horizon"):
        if int(w[k]) < 1:
            raise ValueError(f"window.{k} must be >= 1")
    if cfg["split"]["train_fraction"] + cfg["split"]["val_fraction"] >= 0.95:
        raise ValueError("split.train_fraction + split.val_fraction must leave a test hold-out")
    if cfg["stage_order"][0] != "Benign":
        raise ValueError("stage_order[0] must be 'Benign'")
