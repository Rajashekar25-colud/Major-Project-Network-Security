from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import state
from src.product.config import load_product
from src.product.counterfactual import compare
from ui import GRID, PLOT, banner, kpi_row, page_header, section

COLORS = {"no_action": "#ef4444", "rate_limit": "#eab308", "isolate_host": "#22c55e"}


def render():
    fc = state.forecaster()
    a, label = state.current_analysis()
    page_header("Defence simulation", "Compare what the model expects to happen with no action versus defensive responses.")
    if a is None:
        st.info("Add traffic under **Analyze Traffic** first."); return
    prod = load_product()["counterfactual"]
    scen = prod["scenarios"]; rs = a.session
    inc = a.incidents.sort_values("opened_time")
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 2, 3])
        opts = ["Highest-risk moment"] + [f"{r.id} · {r.stage} · {pd.Timestamp(r.opened_time):%d %b %H:%M:%S}" for r in inc.itertuples()]
        pick = c1.selectbox("Start from", opts)
        steps = c2.slider("Windows to simulate", 4, 24, int(prod["steps"]))
        chosen = c3.multiselect("Responses to compare", list(scen), default=list(scen), format_func=lambda k: scen[k]["label"])
    k = int(np.argmax(rs.dec["risk"])) if pick == opts[0] else int(inc.iloc[opts.index(pick) - 1]["k_open"])
    seq = rs.sequence(k)
    res = compare(fc, seq, steps, {c: scen[c] for c in chosen}, a.threshold)
    t = pd.Timestamp(a.states["timestamp"].iloc[rs.ends[k]])
    sm = res["summary"]
    best = sm.iloc[1:].sort_values("mean_risk").iloc[0] if len(sm) > 1 else None
    if best is not None and best["risk_reduction_vs_no_action"] > 0.05:
        banner(f"Best simulated response: {best['scenario']}", f"Average attack probability drops from {sm.iloc[0]['mean_risk']:.0%} to {best['mean_risk']:.0%} over the next {steps * fc.wsec} seconds (simulation).", "#22c55e")
    else:
        banner("No simulated response clearly reduces the risk", "In this simulation none of the selected responses lowers the forecast materially.", "#eab308")
    kpi_row([("Simulation start", f"{t:%H:%M:%S}", f"{t:%d %b %Y}"), ("No-action peak risk", f"{sm.iloc[0]['peak_risk']:.0%}", f"mean {sm.iloc[0]['mean_risk']:.0%}"),
             ("Windows above alert threshold", int(sm.iloc[0]["windows_above_threshold"]), f"of {steps} (no action)"), ("Simulated horizon", f"{steps * fc.wsec} s", f"{steps} windows")])
    st.write("")
    x = [f"+{(i + 1) * fc.wsec}s" for i in range(steps)]
    fig = go.Figure()
    for key, r in res["runs"].items():
        fig.add_scatter(x=x, y=r["attack_prob"], mode="lines+markers", name=r["label"], line=dict(color=COLORS.get(key, "#38bdf8"), width=3 if key == "no_action" else 2))
    fig.add_hline(y=a.threshold, line=dict(color="#f97316", dash="dash"), annotation_text="alert threshold")
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%", **GRID); fig.update_xaxes(**GRID)
    fig.update_layout(**PLOT, height=360, title=dict(text="Forecast attack probability along each simulated future", font=dict(size=13)), legend=dict(orientation="h", y=1.12))
    with st.container(border=True):
        st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns([3, 2])
    with c1.container(border=True):
        section("Comparison")
        show = sm.rename(columns={"scenario": "Response", "mean_risk": "Mean risk", "peak_risk": "Peak risk", "windows_above_threshold": "Windows above threshold",
                                  "first_alert_step": "First alert (window)", "risk_reduction_vs_no_action": "Risk reduction"}).drop(columns=["key"])
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "Mean risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Peak risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Risk reduction": st.column_config.NumberColumn(format="%.0f%%" if False else "%.2f")})
    with c2.container(border=True):
        section("Generated traffic (connections per window)")
        i = fc.names.index("flow_count")
        f2 = go.Figure()
        for key, r in res["runs"].items():
            f2.add_scatter(x=x, y=r["raw_states"][:, i], name=r["label"], mode="lines", line=dict(color=COLORS.get(key, "#38bdf8")))
        f2.update_xaxes(**GRID); f2.update_yaxes(**GRID)
        f2.update_layout(**PLOT, height=250, showlegend=False)
        st.plotly_chart(f2, width="stretch")
    for key in chosen:
        st.caption(f"**{scen[key]['label']}** - {scen[key]['description']}")
    st.info("This is a model-based what-if simulation. The effect of each response is an assumption (editable in configs/product.yaml); "
            "it shows how the forecast would change, not proof that a real firewall rule or isolation would behave this way.")
