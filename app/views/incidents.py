from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import state
from charts import drivers_bar, risk_timeline
from src.explain.explainer import explain_shap
from src.product import store
from src.product.analysis import incident_drivers
from src.product.knowledge import feature_label, playbook
from panels import flagged_traffic
from ui import STATUS_COLOR, chip, fmt_secs, kpi_row, page_header, section, sev_chip


def render():
    fc = state.forecaster()
    a, label = state.current_analysis()      # also registers this source's incidents in the store
    page_header("Incidents", "Triage forecast threats, understand why they were flagged and follow the recommended response.")
    df = store.list_incidents()
    if df.empty:
        st.info("No incidents recorded yet."); return
    f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
    status = f1.multiselect("Status", store.STATUSES, default=["New", "Investigating"])
    sev = f2.multiselect("Severity", ["Critical", "High", "Elevated"], default=["Critical", "High", "Elevated"])
    tactic = f3.multiselect("Tactic", sorted(df["stage"].unique()))
    scope = f4.selectbox("Source", ["Current source", "All sources"])
    d = df.copy()
    if scope == "Current source" and a is not None:
        d = d[d["source"] == a.source]
    d = d[d["status"].isin(status) & d["severity"].isin(sev)]
    if tactic:
        d = d[d["stage"].isin(tactic)]
    d = d.sort_values(["opened_time"], ascending=False).reset_index(drop=True)
    sev_n = d["severity"].value_counts()
    kpi_row([("Open queue", len(d), "matching filters"), ("Critical", int(sev_n.get("Critical", 0)), ""), ("High", int(sev_n.get("High", 0)), ""),
             ("New", int((d["status"] == "New").sum()), "not yet reviewed"), ("False positives", int((df["status"] == "False positive").sum()), "all time")])
    st.write("")
    if d.empty:
        st.success("Nothing matches these filters."); return
    view = pd.DataFrame({"Incident": d["id"], "Opened": pd.to_datetime(d["opened_time"]).dt.strftime("%d %b %Y %H:%M:%S"), "Severity": d["severity"],
                         "Predicted tactic": d["stage"], "Peak risk": d["peak_risk"].clip(0, 1), "Status": d["status"]})
    ev = st.dataframe(view, hide_index=True, width="stretch", height=min(60 + 36 * len(view), 380), on_select="rerun", selection_mode="single-row",
                      column_config={"Peak risk": st.column_config.ProgressColumn("Peak risk", format="percent", min_value=0, max_value=1)}, key="inc_table")
    c1, c2, _ = st.columns([1.2, 1.2, 4])
    c1.download_button("Export (CSV)", d.to_csv(index=False).encode(), "incidents.csv", "text/csv", width="stretch")
    c2.download_button("Export (JSON)", d.to_json(orient="records", date_format="iso").encode(), "incidents.json", "application/json", width="stretch")
    sel = ev.selection.rows if ev and ev.selection else []
    if not sel:
        st.caption("Select an incident to see details."); return
    r = d.iloc[sel[0]]
    st.write("")
    with st.container(border=True):
        st.markdown(f"### {r['id']} &nbsp; {sev_chip(r['severity'])} &nbsp; {chip(r['status'], STATUS_COLOR.get(r['status'], '#64748b'))}", unsafe_allow_html=True)
        left, right = st.columns([3, 2])
        with left:
            pb = playbook(r["stage"])
            st.markdown(f"**Predicted tactic:** {r['stage']} ({pb['id']}) &nbsp;·&nbsp; **Progression:** {r['progression']}")
            st.markdown(f"**Opened:** {pd.to_datetime(r['opened_time']):%d %b %Y %H:%M:%S} &nbsp;·&nbsp; **Attack expected within:** {fmt_secs(r['eta_seconds'])} "
                        f"&nbsp;·&nbsp; **Peak risk:** {r['peak_risk']:.0%} &nbsp;·&nbsp; **Duration:** {int(r['windows']) * fc.wsec} s")
            if r["confirmed"] == 1:
                st.markdown(chip("Matches known attack activity in the data", "#22c55e"), unsafe_allow_html=True)
            elif r["confirmed"] == 0:
                st.markdown(chip("No known attack activity in the data at this time", "#eab308"), unsafe_allow_html=True)
            st.markdown("")
            section("What this means")
            st.markdown(f"<div class='why'>{pb['meaning']}</div>", unsafe_allow_html=True)
            section("Recommended response")
            for x in pb["actions"]:
                st.markdown(f"<div class='act'>{x}</div>", unsafe_allow_html=True)
        with right:
            section("Triage")
            ns = st.selectbox("Status", store.STATUSES, index=store.STATUSES.index(r["status"]), key=f"s_{r['id']}")
            note = st.text_area("Notes", value=r["note"] or "", height=110, key=f"n_{r['id']}", placeholder="Findings, actions taken, ticket reference ...")
            if st.button("Save", type="primary", key=f"save_{r['id']}", width="stretch"):
                store.update_incident(r["id"], ns, note); st.toast("Incident updated"); st.rerun()

        match = a.incidents[a.incidents["id"] == r["id"]] if a is not None else pd.DataFrame()
        if len(match):
            m = match.iloc[0]
            ex = incident_drivers(a, fc, int(m["k_open"]), 8)
            names = {x["feature"]: feature_label(x["feature"]) for x in ex["top_features"]}
            t1, t2 = st.columns([3, 2])
            with t1:
                st.plotly_chart(drivers_bar(ex["top_features"], names), width="stretch")
            with t2:
                k = int(m["k_open"]); rs = a.session
                lo, hi = max(k - 40, 0), min(k + 60, len(rs.ends) - 1)
                rows = rs.ends[lo:hi + 1]
                st.plotly_chart(risk_timeline(a.states["timestamp"].iloc[rows].reset_index(drop=True), rs.dec["risk"][lo:hi + 1], a.states["segment"].to_numpy()[rows],
                                              a.threshold, match, None, height=300, title="Risk around this incident"), width="stretch")
            section("Flagged traffic patterns during this incident")
            k0 = int(m["k_open"]); k1 = min(k0 + max(int(m["windows"]), 3) + 2, len(a.session.ends) - 1)
            ks = np.arange(k0, k1 + 1)
            flagged_traffic(a.states, a.session.ends[ks], a.session.dec["risk"][ks], 12)
            with st.expander("Advanced: exact SHAP explanation"):
                if st.button("Compute SHAP values", key=f"shap_{r['id']}"):
                    sh = explain_shap(fc, a.session.sequence(int(m["k_open"])), __import__("numpy").load(state.MODELS_DIR / "background_sequences.npy") if (state.MODELS_DIR / "background_sequences.npy").exists() else None, 10)
                    if sh.get("available"):
                        st.plotly_chart(drivers_bar(sh["top_features"], {x["feature"]: feature_label(x["feature"]) for x in sh["top_features"]}), width="stretch")
                    else:
                        st.warning(sh.get("reason"))
        else:
            st.caption("Load this incident's data source (sidebar) to see the full explanation.")
