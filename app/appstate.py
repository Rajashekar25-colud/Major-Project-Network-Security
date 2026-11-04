"""Centralised application state: session keys, cached model loading, navigation helpers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st

from src.analysis.integrity import fingerprint
from src.auth import UserStore, audit as _audit
from src.config import MODELS_DIR, load_config
from src.models.inference import Forecaster, artifacts_present
from src.service import read_json

DEFAULTS = {"page": "new", "analysis": None, "k": 0, "sel_step": 1, "playing": False, "speed": 1, "phase": "upload", "report": None,
            "thr_override": None, "cache_ex": {}, "uploader_n": 0, "validation_cache": {}, "auth": None}


def init() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v.copy() if isinstance(v, dict) else v


@st.cache_resource(show_spinner=False)
def config():
    return load_config()


@st.cache_resource(show_spinner="Loading trained model ...")
def _forecaster(wm: float, mm: float) -> Forecaster:
    return Forecaster.load(MODELS_DIR)


def model_available() -> bool:
    return artifacts_present(MODELS_DIR)


def _mt() -> tuple[float, float]:
    return (MODELS_DIR / "world_model.pt").stat().st_mtime, (MODELS_DIR / "metadata.json").stat().st_mtime


def forecaster() -> Forecaster:
    return _forecaster(*_mt())


@st.cache_data(show_spinner=False)
def _fp(wm: float, mm: float) -> dict:
    return fingerprint(MODELS_DIR)


def current_fingerprint() -> dict:
    return _fp(*_mt())


def metadata() -> dict:
    return read_json(MODELS_DIR / "metadata.json") or {}


def evaluation() -> dict | None:
    return read_json(MODELS_DIR / "evaluation.json")


def model_label(fc) -> str:
    return "Transformer" if str(fc.meta.get("model", {}).get("type", "lstm")).lower() == "transformer" else "LSTM + Attention"


def dataset_label(meta: dict) -> str:
    """Name of the training data as recorded in models/metadata.json (never guessed)."""
    if meta.get("synthetic_demo_data"):
        return "synthetic demo data"
    ds = meta.get("dataset") or {}
    srcs = ds.get("sources") or []
    if any("CICFlowMeter" in str(s) for s in srcs):
        return f"CSE-CIC-IDS2018 · {len(srcs)} capture days"
    return f"{ds.get('windows_total', 'N/A')} training windows" if ds else "not recorded"


def threshold(fc) -> float:
    o = st.session_state.get("thr_override")
    return float(o) if o else float(fc.threshold)


def has_analysis() -> bool:
    return st.session_state.get("analysis") is not None


def goto(page: str, **state) -> None:
    for k, v in state.items():
        st.session_state[k] = v
    st.session_state["page"] = page
    st.session_state["playing"] = False


def goto_rerun(page: str, **state) -> None:
    """Navigate from inside a fragment (forces a full-app rerun)."""
    goto(page, **state)
    st.rerun(scope="app")


def start_new() -> None:
    ss = st.session_state
    ss.update(page="new", phase="upload", report=None, playing=False, uploader_n=int(ss.get("uploader_n", 0)) + 1)


def default_step(res, k: int) -> int:
    probs = res.rs.attack_prob[k]
    above = [i for i, p in enumerate(probs) if p >= res.thr]
    return (above[0] if above else int(probs.argmax())) + 1


def set_analysis(res) -> None:
    from src.analysis.workflow import alert_records
    ss = st.session_state
    al = alert_records(res)
    k = int(al[0]["k_open"]) if al else (int(res.rs.dec["risk"].argmax()) if res.n else 0)
    ss.update(analysis=res, k=k, sel_step=default_step(res, k), playing=False, cache_ex={}, validation_cache={})


# ---------------------------------------------------------------------------------------------- accounts / roles
@st.cache_resource(show_spinner=False)
def users() -> UserStore:
    store = UserStore()
    store.ensure_admin()
    return store


def current_user() -> dict | None:
    return st.session_state.get("auth")


def username() -> str:
    u = current_user()
    return u["username"] if u else "anonymous"


def is_admin() -> bool:
    u = current_user()
    return bool(u and u.get("role") == "admin")


def sign_in(user: dict) -> None:
    ss = st.session_state
    ss.update(auth=user, page="admin" if user["role"] == "admin" else "new", analysis=None, phase="upload", report=None, playing=False, cache_ex={}, validation_cache={})


def sign_out() -> None:
    ss = st.session_state
    _audit(username(), "logout", "")
    for k, v in DEFAULTS.items():
        ss[k] = v.copy() if isinstance(v, dict) else v
    ss["uploader_n"] = int(ss.get("uploader_n", 0)) + 1


def log(action: str, detail: str = "") -> None:
    _audit(username(), action, detail)


def refresh_user() -> None:
    """Re-read the signed-in account (e.g. after a password change) and drop the session if it was deactivated."""
    u = current_user()
    if not u:
        return
    fresh = users().get(u["username"])
    if fresh is None or not fresh.get("active", True):
        sign_out()
    else:
        st.session_state["auth"] = fresh
