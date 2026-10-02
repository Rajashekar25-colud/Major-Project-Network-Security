from __future__ import annotations

import streamlit as st

import state
from src.product.config import SENSITIVITY, load_product, load_settings, save_settings
from src.product.presets import get_presets
from src.product import learning
from ui import kpi_row, page_header, section


def render():
    fc = state.forecaster()
    prod = load_product(); s = load_settings()
    page_header("Settings", "Tune how sensitive the system is and how alerts behave.")
    pres = get_presets(fc)["thresholds"]
    st.markdown("<div class='small'>Higher sensitivity warns earlier but raises more false alarms. Thresholds come from the model's own validation data.</div>", unsafe_allow_html=True)
    st.write("")
    keys = list(SENSITIVITY)
    cur = s.get("sensitivity", "balanced")
    cols = st.columns(3)
    desc = {"low_noise": "Fewest false alarms. Best for small teams.", "balanced": "Recommended default.", "early_warning": "Warns earliest. Best with an on-call SOC."}
    for c, k in zip(cols, keys):
        with c.container(border=True):
            st.markdown(f"**{SENSITIVITY[k]}**" + ("  ✓ active" if cur == k and not s.get("custom_threshold") else ""))
            st.markdown(f"<div class='small'>{desc[k]}<br>Alert threshold <b>{pres[k]:.0%}</b></div>", unsafe_allow_html=True)
            if st.button("Use this", key=f"sens_{k}", width="stretch", disabled=(cur == k and not s.get("custom_threshold"))):
                s.update(sensitivity=k, custom_threshold=None); save_settings(s); state.clear_model_caches(); st.rerun()
    with st.expander("Advanced"):
        use = st.toggle("Set a custom threshold", value=bool(s.get("custom_threshold")))
        thr = st.slider("Alert threshold", 0.05, 0.99, float(s.get("custom_threshold") or pres[cur]), 0.01, disabled=not use)
        p = st.number_input("Windows above threshold before an alert opens", 1, 10, int(s.get("persistence_windows") or 2))
        cd = st.number_input("Cool-down after an alert closes (windows)", 0, 60, int(s.get("cooldown_windows") if s.get("cooldown_windows") is not None else 6))
        if st.button("Save advanced settings", type="primary"):
            s.update(custom_threshold=float(thr) if use else None, persistence_windows=int(p), cooldown_windows=int(cd)); save_settings(s)
            state.clear_model_caches(); st.toast("Settings saved"); st.rerun()
        if st.button("Reset to defaults"):
            save_settings({"sensitivity": prod["defaults"]["sensitivity"], "custom_threshold": None, "persistence_windows": None, "cooldown_windows": None})
            state.clear_model_caches(); st.rerun()
    st.write("")
    section("About")
    kpi_row([("Product", prod["branding"]["product_name"], prod["branding"]["tagline"]), ("Model version", learning.active_version().split("_")[0], f"{fc.meta.get('model', {}).get('parameters', 0):,} parameters"),
             ("Forecast horizon", f"{fc.H * fc.wsec} s", f"{fc.wsec}-second windows"), ("Access control", "On" if prod["auth"]["enabled"] else "Off", "see configs/product.yaml")])
