from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

import state
from src.product.config import runtime_config
from src.service import save_states, states_from_upload
from ui import banner, kpi_row, page_header, section


def render():
    fc = state.forecaster()
    page_header("Analyze traffic", "Upload a network capture or flow export. The system converts it to traffic windows, forecasts threats and opens incidents.")
    with st.container(border=True):
        up = st.file_uploader("Network capture (PCAP / PCAPNG) or flow export (CSV)", type=["pcap", "pcapng", "cap", "csv"])
        name = st.text_input("Name for this data set", value=Path(up.name).stem if up else "", placeholder="e.g. Branch office - Monday")
        go = st.button("Analyze", type="primary", disabled=up is None)
    if up is not None and go:
        with st.status("Analysing traffic ...", expanded=True) as status:
            try:
                st.write("Reading and converting traffic ...")
                t0 = time.time()
                states, info = states_from_upload(up.name, up.getvalue(), runtime_config())
                st.write(f"{info['kind']}: {len(states):,} ten-second windows ({time.time() - t0:.0f}s)")
                if len(states) < fc.L + 2:
                    status.update(label="Capture too short", state="error")
                    st.error(f"Need at least {(fc.L + 2) * fc.wsec // 60 + 1} minutes of traffic for a forecast (got {len(states) * fc.wsec} s).")
                    return
                rep = fc.validate(states)
                if not rep["ok"]:
                    status.update(label="Incompatible data", state="error")
                    st.error("This data does not match the model's expected traffic features."); return
                stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in (name or Path(up.name).stem)).strip().replace(" ", "_") or "upload"
                path = state.UPLOAD_DIR / f"{stem}.csv"
                save_states(states, path)
                st.session_state["source_label"] = "Uploaded - " + stem.replace("_", " ").strip().title()
                status.update(label="Analysis complete", state="complete")
                st.session_state["analyzed"] = True
                state.load_states.clear()
            except Exception as e:
                status.update(label="Could not analyse this file", state="error"); st.error(str(e)); return
        a, label = state.current_analysis()
        s = a.summary
        st.write("")
        if s["incidents"]:
            banner(f"{s['incidents']} incidents detected", f"Peak risk {s['peak_risk']:.0%}. Predicted tactics: {', '.join(a.incidents['stage'].value_counts().head(3).index)}.", "#f97316")
        else:
            banner("No incidents detected", "No attack progression was forecast for this traffic.", "#22c55e")
        kpi_row([("Incidents", s["incidents"], ""), ("Peak risk", f"{s['peak_risk']:.0%}", ""), ("Traffic analysed", f"{s['hours'] * 60:.0f} min", ""),
                 ("Windows", f"{s['windows']:,}", "")])
        st.info("Open **Overview** for charts and the report, or **Incidents** to triage.")
        if rep["novel_in_input"]:
            st.caption("Note: this data includes packet-level details (TTL, retransmissions) that the current model does not use; results rely on flow-level behaviour.")
    section("Supported inputs")
    st.markdown("<div class='small'>• <b>PCAP / PCAPNG</b> - IPv4 and IPv6 packet captures (needs at least a few minutes of traffic).<br>"
                "• <b>CSV</b> - CICFlowMeter-style flow exports (Timestamp, Dst Port, Protocol and flow statistics). A Label column is optional and only used to verify results.</div>",
                unsafe_allow_html=True)
