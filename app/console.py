"""Aegis Forecast - predictive network threat detection console.

Run:  python -m streamlit run app/console.py
Fully offline: no external services, no telemetry.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st

from src.product.config import SENSITIVITY, load_product, load_settings

prod = load_product()
brand = prod["branding"]
st.set_page_config(page_title=f"{brand['product_name']} - {brand['tagline']}", page_icon=":material/shield:", layout="wide", initial_sidebar_state="expanded")

import state
import ui
from views import analyze, incidents, live, model, overview, settings

ui.inject_css()
st.logo(str(ROOT / "app" / "assets" / "logo.svg"), size="large")

if prod["auth"]["enabled"] and not st.session_state.get("authed"):
    ui.page_header(brand["product_name"], "Sign in to continue")
    pw = st.text_input("Password", type="password")
    if st.button("Sign in", type="primary"):
        if hashlib.sha256(pw.encode()).hexdigest() == prod["auth"]["password_sha256"]:
            st.session_state["authed"] = True; st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

if not state.has_model():
    ui.page_header(brand["product_name"], "Setup required")
    st.error("No trained model was found in the `models` folder. Train one with `python scripts/train.py --states data/states/cicids2018_states.csv`.")
    st.stop()

pages = [st.Page(overview.render, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
         st.Page(live.render, title="Live Monitor", icon=":material/monitor_heart:", url_path="live"),
         st.Page(incidents.render, title="Incidents", icon=":material/crisis_alert:", url_path="incidents"),
         st.Page(analyze.render, title="Analyze Traffic", icon=":material/upload_file:", url_path="analyze"),
         st.Page(model.render, title="Model & Learning", icon=":material/neurology:", url_path="model"),
         st.Page(settings.render, title="Settings", icon=":material/tune:", url_path="settings")]
nav = st.navigation(pages)

with st.sidebar:
    srcs = state.sources()
    if srcs:
        labels = list(srcs)
        if st.session_state.get("source_label") not in labels:
            st.session_state["source_label"] = state.default_source(srcs)
        st.selectbox("Data source", labels, key="source_label")
    s = load_settings()
    st.caption(f"Sensitivity: **{SENSITIVITY.get(s['sensitivity'], 'Balanced') if not s.get('custom_threshold') else 'Custom'}**")
    if brand.get("organisation"):
        st.caption(brand["organisation"])

nav.run()
