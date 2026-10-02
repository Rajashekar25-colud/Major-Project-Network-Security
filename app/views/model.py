from __future__ import annotations

import json
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import state
from src.product import learning, store
from src.product.config import load_product, runtime_config
from src.product.knowledge import feature_label
from src.service import read_json, states_from_upload
from ui import GRID, PLOT, banner, chip, fmt_secs, kpi_row, page_header, section


def _perf_tab(fc):
    ev = read_json(state.MODELS_DIR / "evaluation.json")
    if not ev or "forecast" not in ev:
        st.info("No evaluation results available yet."); return
    f, o, lt = ev["forecast"], ev["onset"], ev["lead_time"]
    st.markdown("<div class='small'>Measured on traffic the model never saw during training.</div>", unsafe_allow_html=True)
    st.write("")
    kpi_row([("Alert precision", f"{f['precision']:.0%}", "alerts that were real attacks"), ("Attacks caught", f"{f['recall']:.0%}", "recall over the forecast window"),
             ("False-alarm rate", f"{f['false_positive_rate']:.1%}", "of normal windows"), ("F1 score", f"{f['f1']:.2f}", "balance of the two")])
    st.write("")
    fc_ = lt.get("forecastable_episodes") or 0
    early, late, miss = lt["warned_early"], lt["detected_late"], lt["missed"]
    kpi_row([("Attack episodes tested", lt["attack_episodes"], f"{fc_} could be forecast"), ("Warned in advance", early, fmt_secs(lt["mean_lead_seconds"]) + " average lead" if early else "none yet"),
             ("Detected at onset", late, "alert opened after the attack began"), ("Not alerted", miss, "no alert during the episode")])
    if fc_ and early / fc_ < 0.3:
        st.warning("Most attack episodes in this evaluation were detected as they began rather than minutes ahead. Raise the sensitivity in **Settings** to trade more false alarms for earlier warnings, and use **Learn from new data** to adapt the model to your traffic.")
    ph = pd.DataFrame(ev["per_horizon"])
    fig = go.Figure()
    for m, n in (("precision", "Precision"), ("recall", "Recall")):
        fig.add_scatter(x=ph["seconds_ahead"], y=ph[m], name=n, mode="lines+markers")
    fig.update_xaxes(title="How far ahead (seconds)", **GRID); fig.update_yaxes(range=[0, 1.02], tickformat=".0%", **GRID)
    fig.update_layout(**PLOT, height=280, title=dict(text="Forecast quality by lead time", font=dict(size=13)), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")
    with st.expander("Technical details"):
        meta = fc.meta
        st.json({"trained_at": meta.get("trained_at"), "model": {k: meta["model"].get(k) for k in ("architecture", "parameters", "hidden_size", "layers", "dropout", "learning_rate")},
                 "tactics": meta.get("stage_names"), "window_seconds": fc.wsec, "history_windows": fc.L, "forecast_windows": fc.H,
                 "alert_threshold": meta.get("warning_threshold"), "lineage": meta.get("lineage")}, expanded=False)


def _drift_tab(fc):
    a, label = state.current_analysis()
    if a is None:
        st.info("Select a data source in the sidebar."); return
    dr = learning.drift_report(fc, a.states)
    if not dr.get("available"):
        st.info(dr.get("reason")); return
    col = {"stable": "#22c55e", "moderate": "#eab308", "significant": "#ef4444"}[dr["overall"]]
    banner(f"Traffic drift: {dr['overall']}", dr["advice"], col)
    t = dr["table"].head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=t["psi"], y=[feature_label(x) for x in t["feature"]], orientation="h",
                           marker_color=["#ef4444" if v >= 0.25 else "#eab308" if v >= 0.1 else "#22c55e" for v in t["psi"]]))
    fig.add_vline(x=0.25, line=dict(color="#ef4444", dash="dot")); fig.add_vline(x=0.10, line=dict(color="#eab308", dash="dot"))
    fig.update_xaxes(title="Shift vs. training data (population stability index)", **GRID); fig.update_yaxes(**GRID)
    fig.update_layout(**PLOT, height=380, title=dict(text="Traffic characteristics that changed the most", font=dict(size=13)))
    st.plotly_chart(fig, width="stretch")
    st.caption("Below 0.10 stable · 0.10-0.25 moderate · above 0.25 significant. Compared with a sample of the data the model was trained on.")


def _learn_tab(fc):
    lc = load_product()["learning"]
    st.markdown("<div class='small'>Teach the model about traffic it has not seen before. Upload <b>labelled</b> flow data (a CSV with a Label column). "
                "The model is fine-tuned into a <b>candidate</b>, compared against the live model on held-out traffic, and only replaces it when you approve.</div>", unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        up = st.file_uploader("Labelled flow data (CSV)", type=["csv"], key="learn_up")
        c1, c2 = st.columns(2)
        epochs = c1.number_input("Training passes", 1, 20, int(lc["epochs"]))
        go_ = c2.button("Learn from this data", type="primary", disabled=up is None, width="stretch")
    if up is not None and go_:
        with st.status("Learning from new data ...", expanded=True) as status:
            try:
                logs = []
                states, info = states_from_upload(up.name, up.getvalue(), runtime_config())
                st.write(f"Prepared {len(states):,} windows")
                res = learning.fine_tune(states, up.name, log=lambda m: (logs.append(m), st.write(m)), epochs=int(epochs), lr=float(lc["learning_rate"]),
                                         anchor=float(lc["anchor_strength"]), min_windows=int(lc["min_windows"]))
                st.session_state["learn_result"] = res
                status.update(label="Candidate model ready", state="complete")
            except Exception as e:
                status.update(label="Could not learn from this data", state="error"); st.error(str(e))
    res = st.session_state.get("learn_result")
    if res:
        section(f"Candidate {res['version']}")
        b, a_ = res["before"], res["after"]
        tbl = pd.DataFrame({"Metric": ["Alert precision", "Attacks caught (recall)", "F1", "False-alarm rate", "Early-warning F1 (onset)"],
                            "Current model": [b["forecast"]["precision"], b["forecast"]["recall"], b["forecast"]["f1"], b["forecast"]["false_positive_rate"], b["onset"]["f1"]],
                            "Candidate": [a_["forecast"]["precision"], a_["forecast"]["recall"], a_["forecast"]["f1"], a_["forecast"]["false_positive_rate"], a_["onset"]["f1"]]})
        tbl["Change"] = tbl["Candidate"] - tbl["Current model"]
        st.markdown("**On the new data (held-out blocks)**")
        st.dataframe(tbl.round(3), hide_index=True, width="stretch")
        if res.get("forgetting"):
            fb, fa = res["forgetting"]["before"], res["forgetting"]["after"]
            drop = fa["f1"] - fb["f1"]
            st.markdown(f"**On the original data (forgetting check):** F1 {fb['f1']:.3f} → {fa['f1']:.3f} "
                        + chip(("no loss" if drop >= -0.02 else f"{drop:+.3f}"), "#22c55e" if drop >= -0.02 else "#ef4444"), unsafe_allow_html=True)
        if res["new_tactics"]:
            st.markdown(f"New tactics learned: **{', '.join(res['new_tactics'])}**")
        better = a_["forecast"]["f1"] >= b["forecast"]["f1"] - 0.005 and (not res.get("forgetting") or res["forgetting"]["after"]["f1"] >= res["forgetting"]["before"]["f1"] - 0.03)
        (st.success if better else st.warning)("The candidate looks safe to activate." if better else "The candidate is not clearly better. Review the numbers before activating.")
        c1, c2, _ = st.columns([1.3, 1.3, 4])
        if c1.button("Activate candidate", type="primary", width="stretch"):
            learning.promote(res["version"]); state.clear_model_caches(); st.session_state.pop("learn_result", None); st.toast("Model updated"); st.rerun()
        if c2.button("Discard", width="stretch"):
            st.session_state.pop("learn_result", None); st.rerun()


def _versions_tab():
    vs = learning.list_versions()
    df = pd.DataFrame(vs)
    df["Version"] = df["id"].str.split("_").str[0]
    df["Created"] = pd.to_datetime(df["created"], errors="coerce").dt.strftime("%d %b %Y %H:%M")
    st.dataframe(df[["Version", "status", "Created", "note", "threshold"]].rename(columns={"status": "Status", "note": "Notes", "threshold": "Threshold"}), hide_index=True, width="stretch")
    cand = [v for v in vs if v["status"] != "active"]
    if cand:
        pick = st.selectbox("Switch to version", [v["id"] for v in cand], format_func=lambda x: f"{x.split('_')[0]} - {next(v['note'] for v in cand if v['id'] == x)}")
        if st.button("Activate selected version"):
            learning.promote(pick); state.clear_model_caches(); st.toast("Model switched"); st.rerun()
    section("Activity log")
    log = store.audit_log(30)
    st.dataframe(log, hide_index=True, width="stretch") if len(log) else st.caption("No activity yet.")


def render():
    fc = state.forecaster()
    page_header("Model & learning", f"Active model {learning.active_version().split('_')[0]} · forecasts {fc.H * fc.wsec} seconds ahead")
    t1, t2, t3, t4 = st.tabs(["Detection quality", "Traffic drift", "Learn from new data", "Versions"])
    with t1:
        _perf_tab(fc)
    with t2:
        _drift_tab(fc)
    with t3:
        _learn_tab(fc)
    with t4:
        _versions_tab()
