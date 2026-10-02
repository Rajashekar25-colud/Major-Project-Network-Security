from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import state
from charts import risk_timeline, tactic_donut
from src.product.config import SEV_COLOR, SEVERITY_ORDER, load_product
from src.product.report import build_report
from src.product import learning
from ui import banner, kpi_row, page_header, section, sev_chip


def render():
    fc = state.forecaster()
    a, label = state.current_analysis()
    if a is None:
        page_header("Security overview", "No traffic source yet")
        st.info("Add traffic under **Analyze Traffic** to see your threat overview.")
        return
    s = a.summary; inc = a.incidents; thr = a.threshold
    page_header("Security overview", f"{label} · {pd.Timestamp(s['start']):%d %b %Y %H:%M} → {pd.Timestamp(s['end']):%d %b %Y %H:%M}")

    by = s["by_severity"]
    crit, high = by.get("Critical", 0), by.get("High", 0)
    if crit:
        banner("Critical activity detected", f"{crit} critical and {high} high-severity incidents in this period. Predicted tactics: "
               f"{', '.join(inc['stage'].value_counts().head(3).index)}.", SEV_COLOR["Critical"], pulse=True)
    elif high:
        banner("Elevated threat level", f"{high} high-severity incidents in this period.", SEV_COLOR["High"])
    elif len(inc):
        banner("Low-level activity", f"{len(inc)} incidents of lower severity in this period.", SEV_COLOR["Elevated"])
    else:
        banner("No significant threats", "No attack progression was forecast for this traffic.", SEV_COLOR["Normal"])

    kpi_row([("Incidents", f"{s['incidents']:,}", f"{crit} critical · {high} high"),
             ("Peak forecast risk", f"{s['peak_risk']:.0%}", f"alert threshold {thr:.0%}"),
             ("Top predicted tactic", s["top_tactic"] or "-", "most frequent in incidents"),
             ("Time at elevated risk", f"{s['pct_elevated']:.0%}", "share of the period"),
             ("Traffic analysed", f"{s['hours']:.1f} h", f"{s['windows']:,} ten-second windows")])
    st.write("")
    c1, c2 = st.columns([5, 2])
    with c1:
        with st.container(border=True):
            section("Forecast risk over time")
            rs = a.session
            rows = rs.ends
            show_truth = False
            if s["labelled"]:
                show_truth = st.toggle("Show known attack activity from dataset labels", value=True)
            fig = risk_timeline(a.states["timestamp"].iloc[rows].reset_index(drop=True), rs.dec["risk"], a.states["segment"].to_numpy()[rows], thr, inc,
                                a.states["is_attack"].to_numpy()[rows] if show_truth else None, height=360)
            st.plotly_chart(fig, width="stretch")
    with c2:
        with st.container(border=True):
            if len(inc):
                st.plotly_chart(tactic_donut(inc["stage"].value_counts().to_dict()), width="stretch")
            else:
                st.markdown("<div class='small'>No incidents to chart.</div>", unsafe_allow_html=True)

    st.write("")
    section("Latest incidents")
    if len(inc):
        d = inc.sort_values("opened_time", ascending=False).head(10)
        view = pd.DataFrame({"Incident": d["id"], "Opened": pd.to_datetime(d["opened_time"]).dt.strftime("%d %b %H:%M:%S"),
                             "Severity": d["severity"], "Predicted tactic": d["stage"], "Peak risk": d["peak_risk"].clip(0, 1),
                             "Expected in": d["eta_seconds"].map(lambda x: f"{x:.0f} s"),
                             "Verified": d["confirmed"].map({1: "Matches known attack", 0: "No known attack", -1: "-"})})
        st.dataframe(view, hide_index=True, width="stretch", column_config={
            "Peak risk": st.column_config.ProgressColumn("Peak risk", format="percent", min_value=0, max_value=1)})
        st.caption("Open **Incidents** to triage, see why each was flagged and view the recommended response.")
    else:
        st.markdown("<div class='small'>Nothing to show.</div>", unsafe_allow_html=True)

    st.write("")
    x1, x2, _ = st.columns([1.2, 1.2, 4])
    prod = load_product()
    html_ = build_report(prod["branding"], label, s, inc, a.session.dec["risk"], thr, learning.active_version())
    x1.download_button("Download report (HTML)", html_.encode("utf-8"), "threat-report.html", "text/html", width="stretch")
    x2.download_button("Export incidents (CSV)", inc.drop(columns=["row_open", "k_open"]).to_csv(index=False).encode(), "incidents.csv", "text/csv", width="stretch")
