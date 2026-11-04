"""AI Network Security - Attack Forecasting & SOC Intelligence.   Run:  python -m streamlit run app/main.py

Sign in -> Upload -> Analyze -> Forecast -> Alert -> Explain -> ATT&CK -> Validate.  Fully offline; the trained model is never modified.
Two roles: Analyst (runs and reviews analyses) and Administrator (users, all analyses, model health, evaluation, audit log).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st

st.set_page_config(page_title="AI Network Security · Attack Forecasting", page_icon=":material/shield:", layout="wide", initial_sidebar_state="expanded")

import appstate as A
import theme as T
from screens import admin_pages as AD
from screens import analysis_pages as P
from screens import auth_pages as AU
from screens import history_settings as HS
from screens import new_analysis as NA
from screens import validate_pages as VP

T.inject()
A.init()

if not A.model_available():
    T.header("Setup required", "No trained model was found in the models folder.")
    st.error("Train one with `python scripts/train.py --data <CIC-IDS2018 folder>` (or create the demo model with `python scripts/make_demo.py`), then reload.")
    st.stop()

if A.current_user() is None:                    # ---- landing / sign-in
    AU.login()
    st.stop()

A.refresh_user()
if A.current_user() is None:
    st.rerun()

fc = A.forecaster()
ss = st.session_state
admin = A.is_admin()
PAGES = {"new": NA.render, "overview": P.overview, "traffic": P.traffic, "forecast": P.forecast, "alerts": P.alerts, "explain": P.explain, "attack": P.attack,
         "validation": VP.forecast_validation, "evaluation": VP.evaluation, "history": HS.history, "settings": HS.settings, "profile": AU.profile,
         "admin": AD.dashboard, "users": AD.users, "audit": AD.audit_log}
ANALYSIS_ONLY = {"overview", "traffic", "forecast", "alerts", "explain", "attack", "validation"}
ADMIN_ONLY = {"admin", "users", "audit", "settings", "evaluation"}

if ss["page"] not in PAGES or (ss["page"] in ADMIN_ONLY and not admin):
    ss["page"] = "admin" if admin else "new"
if not A.has_analysis() and ss["page"] in ANALYSIS_ONLY:
    ss["page"] = "new"                           # state-aware: analysis pages exist only once an analysis exists
if A.current_user().get("must_change"):
    ss["page"] = "profile"                       # forced password change (default / reset passwords)


def _nav(label: str, page: str) -> None:
    st.button(label, key=f"nav_{page}", width="stretch", type="primary" if ss["page"] == page else "secondary", on_click=A.goto, args=(page,), disabled=bool(A.current_user().get("must_change")))


def _sec(name: str) -> None:
    st.markdown(f'<div class="nav-sec">{name}</div>', unsafe_allow_html=True)


u = A.current_user()
meta = A.metadata()
with st.sidebar:
    st.markdown('<div class="brand"><div class="mark">🛡</div><div class="t">AI NETWORK SECURITY<span>Attack Forecasting · SOC</span></div></div>', unsafe_allow_html=True)
    initials = (u.get("full_name") or u["username"])[:1].upper()
    st.markdown(f'<div class="userchip"><div class="av">{T.esc(initials)}</div><div><div class="nm">{T.esc(u.get("full_name") or u["username"])}</div>'
                f'<div class="rl">{"Administrator" if admin else "Analyst"}</div></div></div>', unsafe_allow_html=True)
    st.button("＋  New Analysis", key="nav_new", width="stretch", on_click=A.start_new, disabled=bool(u.get("must_change")))
    if A.has_analysis():
        a = ss["analysis"]
        st.markdown(f'<div class="ctx">Current analysis<br><b>{T.esc(a.name)}</b><br>{a.n:,} windows · {len(a.rs.engine.alerts)} alert(s)</div>', unsafe_allow_html=True)
    if admin:
        _sec("Console"); _nav("Dashboard", "admin"); _nav("Users", "users"); _nav("All Analyses", "history"); _nav("Audit Log", "audit")
        _sec("Model"); _nav("Model Health", "settings"); _nav("Evaluation", "evaluation")
    if A.has_analysis():
        _sec("Current analysis")
        for lab, pg in (("Overview", "overview"), ("Traffic", "traffic"), ("Forecast", "forecast"), ("Alerts", "alerts")):
            _nav(lab, pg)
        _sec("Investigate"); _nav("Explain", "explain"); _nav("MITRE ATT&CK", "attack")
        _sec("Validate"); _nav("Forecast Validation", "validation")
    st.markdown('<div class="nav-sep"></div>', unsafe_allow_html=True)
    if not admin:
        _nav("History", "history")
    _nav("Profile", "profile")
    st.markdown('<div class="nav-sep"></div>', unsafe_allow_html=True)
    sysrows = [(True, "Model", A.model_label(fc)), (True, "Forecast engine", f"{fc.H} steps · {fc.wsec}s windows"), (bool(A.config().get("warning")), "Warning engine", f"threshold {A.threshold(fc):.0%}"),
               (None if meta.get("synthetic_demo_data") else True, "Dataset", A.dataset_label(meta))]
    st.markdown('<div class="sys">' + "".join(f'<div><span class="dot {"ok" if ok else "warn" if ok is None else "bad"}"></span><b>{T.esc(a)}</b> · {T.esc(b)}</div>' for ok, a, b in sysrows) + "</div>", unsafe_allow_html=True)
    st.markdown("&nbsp;")
    st.button("Sign out", key="logout", width="stretch", on_click=A.sign_out)

PAGES[ss["page"]]()
