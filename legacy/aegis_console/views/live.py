from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import streamlit as st

import state
from charts import gauge, horizon_bars, risk_timeline, tactic_probs
from src.explain.explainer import explain_window
from src.product.config import SEV_COLOR, SEVERITY_LABEL
from src.product.knowledge import driver_sentences, feature_label, playbook
from panels import flagged_traffic
from ui import banner, fmt_secs, kpi_row, page_header, section


def _live_session(a):
    key = ("live", a.source, round(a.threshold, 4), len(a.states))
    if st.session_state.get("live_key") != key:
        ls = copy.copy(a.session)
        ls.reset()
        st.session_state["live"] = ls; st.session_state["live_key"] = key; st.session_state["playing"] = False
    return st.session_state["live"]


def render():
    fc = state.forecaster()
    a, label = state.current_analysis()
    if a is None:
        page_header("Live monitor", "No traffic source yet"); st.info("Add traffic under **Analyze Traffic**."); return
    ls = _live_session(a)
    thr = a.threshold
    ss = st.session_state
    ss.setdefault("playing", False)
    page_header("Live monitor", "Stream the traffic window by window and watch forecasts and alerts update in real time.")

    with st.container(border=True):
        c = st.columns([1.1, 1, 1, 1, 2.4, 2.4])
        if c[0].button("⏸ Pause" if ss["playing"] else "▶ Start", type="primary", width="stretch", key="b_start"):
            if ls.finished:
                ls.reset()
            ss["playing"] = not ss["playing"]
        if c[1].button("Step", width="stretch", key="b_step"):
            ss["playing"] = False; ls.step(1)
        if c[2].button("+10", width="stretch", key="b_10"):
            ss["playing"] = False; ls.step(10)
        if c[3].button("Reset", width="stretch", key="b_reset"):
            ss["playing"] = False; ls.reset()
        speed = c[4].select_slider("Speed (windows per second)", [1, 2, 5, 10, 25, 50, 100], value=5, key="speed")
        n = len(ls.ends)
        span = c[5].slider("Playback range", 0, n - 1, (0, n - 1), key=f"span_{a.source}")
        if (ls.lo, ls.hi) != span:
            ls.set_range(*span); ss["playing"] = False
        ts_lo, ts_hi = a.states["timestamp"].iloc[ls.ends[span[0]]], a.states["timestamp"].iloc[ls.ends[span[1]]]
        st.caption(f"{ts_lo:%d %b %Y %H:%M:%S} → {ts_hi:%d %b %Y %H:%M:%S} · each window = {fc.wsec} s of network traffic")

    def view():
        if ss["playing"]:
            ls.step(max(1, int(round(speed * 0.6))))
            if ls.finished:
                ss["playing"] = False
                st.rerun()
        cur = ls.current()
        if cur is None:
            st.info("Press **Start** to stream the traffic. Every window is scored by the model and passed through the alert engine as it plays.")
            return
        al = cur["open_alert"]
        risk, level = cur["risk"], SEVERITY_LABEL[cur["level"]]
        att = np.asarray(cur["stage_prob"])[int(np.argmax(cur["probs"]))]
        top = int(np.argmax(att[1:])) + 1 if len(att) > 1 else 0
        likely = fc.stages[top]
        if al is not None:
            col = SEV_COLOR[SEVERITY_LABEL[al.peak_severity]]
            banner(f"{SEVERITY_LABEL[al.severity]} alert · likely {al.stage}", f"Attack activity expected within about {fmt_secs(al.eta_seconds)} · peak risk {al.peak_risk:.0%} · open for {al.n_windows} windows", col, pulse=True)
        elif cur["level"] == "WATCH":
            banner("Elevated risk", f"Forecast risk {risk:.0%} is rising but below the alert threshold ({thr:.0%}). Most likely tactic: {likely}.", SEV_COLOR["Elevated"])
        else:
            banner("Normal", f"No attack progression forecast (risk {risk:.0%}).", SEV_COLOR["Normal"])

        done = ls.cursor - ls.lo + 1
        kpi_row([("Forecast risk", f"{risk:.0%}", f"alert threshold {thr:.0%}", SEV_COLOR[level]), ("Most likely tactic", likely, "if an attack occurs"),
                 ("Alerts raised", len(ls.engine.alerts), f"{sum(x.status == 'OPEN' for x in ls.engine.alerts)} open"),
                 ("Progress", f"{done / ls.n_total:.0%}", f"{done:,} of {ls.n_total:,} windows"), ("Stream time", f"{pd.Timestamp(cur['time']):%H:%M:%S}", f"{pd.Timestamp(cur['time']):%d %b %Y}")])
        st.progress(min(done / max(ls.n_total, 1), 1.0))
        h = ls.history().tail(500)
        c1, c2 = st.columns([3, 1])
        with c1.container(border=True):
            st.plotly_chart(risk_timeline(h["timestamp"].reset_index(drop=True), h["risk"].to_numpy(), np.zeros(len(h), int), thr, None,
                                          h["truth"].to_numpy() if a.summary["labelled"] else None, height=300, title="Forecast risk"), width="stretch", key=f"lr_{ls.cursor}")
        with c2.container(border=True):
            st.plotly_chart(gauge(risk, thr), width="stretch", key=f"lg_{ls.cursor}")
        d1, d2 = st.columns(2)
        ft = ls.forecast_table()
        with d1.container(border=True):
            st.plotly_chart(horizon_bars(ft, thr), width="stretch", key=f"lh_{ls.cursor}")
        with d2.container(border=True):
            st.plotly_chart(tactic_probs(fc.stages, att), width="stretch", key=f"lt_{ls.cursor}")
        w1, w2 = st.columns(2)
        with w1.container(border=True):
            section("Why this forecast")
            ex = explain_window(fc, ls.sequence(), 6)
            for sline in driver_sentences(ex["top_features"], 4):
                st.markdown(f"<div class='why'>• {sline}</div>", unsafe_allow_html=True)
        with w2.container(border=True):
            tgt = al.stage if al is not None else (likely if risk >= thr * 0.7 else None)
            section("Recommended response" if tgt else "Status")
            if tgt:
                pb = playbook(tgt)
                st.markdown(f"<div class='small'>{pb['meaning']}</div>", unsafe_allow_html=True)
                for x in pb["actions"][:3]:
                    st.markdown(f"<div class='act'>{x}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='small'>No action needed. The system keeps scoring each new window.</div>", unsafe_allow_html=True)

        with st.container(border=True):
            section("Flagged traffic patterns")
            hh = ls.history().tail(300)
            sel = hh[hh["risk"] >= thr * 0.7]
            flagged_traffic(a.states, sel["row"].to_numpy(), sel["risk"].to_numpy(), 8)

    st.fragment(run_every=0.6 if ss["playing"] else None)(view)()
