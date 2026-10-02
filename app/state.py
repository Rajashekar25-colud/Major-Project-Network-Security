"""Shared, cached state for the console (model, sources, analyses)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd
import streamlit as st

from src.config import MODELS_DIR
from src.models.inference import Forecaster, artifacts_present
from src.product import store
from src.product.analysis import Analysis, run_analysis
from src.product.config import load_product, load_settings, runtime_config
from src.product.presets import effective_threshold, get_presets
from src.service import load_states_csv

STATES_DIR = ROOT / "data" / "states"
UPLOAD_DIR = ROOT / "data" / "uploads"
FRIENDLY = {"cicids2018_states": "CSE-CIC-IDS2018 - all capture days", "test_replay": "Hold-out test traffic",
            "guide_demo": "Demonstration scenarios", "demo_states": "Sample network (synthetic)", "pcap_states": "Packet capture sample"}


def model_key() -> tuple:
    f = MODELS_DIR / "world_model.pt"
    return (f.stat().st_mtime if f.exists() else 0, (MODELS_DIR / "metadata.json").stat().st_mtime if (MODELS_DIR / "metadata.json").exists() else 0)


def has_model() -> bool:
    return artifacts_present(MODELS_DIR)


@st.cache_resource(show_spinner=False)
def _fc(key: tuple) -> Forecaster:
    return Forecaster.load(MODELS_DIR)


def forecaster() -> Forecaster:
    return _fc(model_key())


def sources() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in (STATES_DIR, UPLOAD_DIR):
        if d.exists():
            for p in sorted(d.glob("*.csv")):
                if "_info" in p.name:
                    continue
                label = FRIENDLY.get(p.stem, p.stem.replace("_", " ").strip().title())
                if d == UPLOAD_DIR:
                    label = "Uploaded - " + label
                out[label] = p
    return out


def default_source(srcs: dict[str, Path]) -> str | None:
    for k, p in srcs.items():
        if p.stem == "cicids2018_states":
            return k
    return next(iter(srcs), None)


@st.cache_data(show_spinner=False)
def load_states(path: str, mtime: float) -> pd.DataFrame:
    return load_states_csv(path, runtime_config())


@st.cache_resource(show_spinner=False, max_entries=3)
def _analysis(path: str, mtime: float, thr: float, persist, cool, mkey: tuple) -> Analysis:
    cfg = runtime_config()
    fc = forecaster()
    st_ = load_states(path, mtime)
    a = run_analysis(st_, Path(path).name, fc, cfg, thr)
    store.upsert_incidents(a.incidents.drop(columns=["row_open", "k_open"]))
    return a


def threshold() -> float:
    return effective_threshold(forecaster(), load_settings())


def current_analysis() -> tuple[Analysis | None, str | None]:
    srcs = sources()
    label = st.session_state.get("source_label") or default_source(srcs)
    if not label or label not in srcs:
        return None, None
    p = srcs[label]
    s = load_settings()
    a = _analysis(str(p), p.stat().st_mtime, round(threshold(), 4), s.get("persistence_windows"), s.get("cooldown_windows"), model_key())
    return a, label


def clear_model_caches() -> None:
    _fc.clear(); _analysis.clear()
