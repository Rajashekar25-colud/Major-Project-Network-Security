"""Product configuration + persisted user settings (data/settings.json)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from ..config import ROOT, load_config

PRODUCT_CFG = ROOT / "configs" / "product.yaml"
SETTINGS_PATH = ROOT / "data" / "settings.json"
SEVERITY_LABEL = {"OK": "Normal", "WATCH": "Elevated", "WARNING": "High", "CRITICAL": "Critical"}
SEVERITY_ORDER = ["Normal", "Elevated", "High", "Critical"]
SEV_COLOR = {"Normal": "#22c55e", "Elevated": "#eab308", "High": "#f97316", "Critical": "#ef4444"}
SENSITIVITY = {"low_noise": "Low noise", "balanced": "Balanced", "early_warning": "Early warning"}


def load_product() -> dict[str, Any]:
    with open(PRODUCT_CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings() -> dict[str, Any]:
    base = {"sensitivity": load_product()["defaults"]["sensitivity"], "custom_threshold": None,
            "persistence_windows": None, "cooldown_windows": None}
    if SETTINGS_PATH.exists():
        try:
            base.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return base


def save_settings(s: dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(s, indent=2), encoding="utf-8")


def runtime_config(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Engine config with the user's settings applied (a copy; the YAML is untouched)."""
    cfg = copy.deepcopy(load_config())
    s = settings or load_settings()
    if s.get("persistence_windows"):
        cfg["warning"]["persistence_windows"] = int(s["persistence_windows"])
    if s.get("cooldown_windows") is not None:
        cfg["warning"]["cooldown_windows"] = int(s["cooldown_windows"])
    return cfg
