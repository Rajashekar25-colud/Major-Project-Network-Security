"""Reusable dashboard panels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.product.knowledge import PORT_SERVICES


def flagged_traffic(states: pd.DataFrame, rows: np.ndarray, risk: np.ndarray | None = None, limit: int = 15) -> None:
    """'Flagged traffic patterns': the traffic indicators of the windows behind a forecast / incident."""
    rows = np.asarray(rows)[-limit:][::-1]
    if len(rows) == 0:
        st.caption("No flagged windows."); return
    S = states.iloc[rows]
    pc = [c for c in S.columns if c.startswith("dst_port_") and c != "dst_port_other"]
    dom = S[pc].idxmax(axis=1).str.replace("dst_port_", "")
    share = S[pc].max(axis=1)
    svc = [f"{p} ({PORT_SERVICES[int(p)]})" if p.isdigit() and int(p) in PORT_SERVICES and sh > 0 else ("other" if sh == 0 else p) for p, sh in zip(dom, share)]
    out = pd.DataFrame({"Time": S["timestamp"].dt.strftime("%H:%M:%S").to_numpy(), "Connections": S["flow_count"].astype(int).to_numpy(),
                        "Distinct ports": S["unique_dst_ports"].astype(int).to_numpy(), "New-connection share": S["syn_ratio"].to_numpy(),
                        "Unanswered share": S["failed_flow_ratio"].to_numpy(), "Scan pattern": S["scan_score"].to_numpy(), "Main service": svc})
    if risk is not None:
        out.insert(1, "Risk", np.asarray(risk)[-limit:][::-1])
    cfg = {"New-connection share": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
           "Unanswered share": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
           "Scan pattern": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)}
    if risk is not None:
        cfg["Risk"] = st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)
    st.dataframe(out, hide_index=True, width="stretch", column_config=cfg)
