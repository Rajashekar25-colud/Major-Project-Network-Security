"""History (persisted analyses) and Settings (minimal user settings; developer details collapsed)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import appstate as A
import theme as T
from src.analysis import history as H
from src.analysis import workflow as W
from src.analysis.ingest import traffic_stats
from src.config import MODELS_DIR


def _duration(m: dict) -> str:
    try:
        return T.secs((pd.Timestamp(m["end"]) - pd.Timestamp(m["start"])).total_seconds())
    except Exception:
        return "N/A"


def _open(m: dict) -> None:
    fc, cfg = A.forecaster(), A.config()
    with st.spinner("Re-running the trained model on the saved windows ..."):
        states = H.load_states(m["id"])
        info = dict(m.get("info", {})); info["labelled"] = bool(m.get("labelled"))
        info["feature_report"] = fc.validate(states)
        res = W.run_analysis(states, m["filename"], fc, cfg, A.threshold(fc), info=info, stats=traffic_stats(states, info, fc), models_dir=MODELS_DIR,
                             analysis_id=m["id"], created=pd.Timestamp(m["created"]).to_pydatetime())
    A.set_analysis(res); A.log("analysis_opened", m["id"]); A.goto("overview"); st.rerun()


def history() -> None:
    admin = A.is_admin(); me = A.username()
    T.header("All Analyses" if admin else "History", "Every stored analysis, from all accounts." if admin else "Your previous analyses. Reopening one re-runs the fixed trained model on the saved windows.")
    items = H.list_history(owner=None if admin else me)
    if not items:
        st.info("No analyses yet. Use New Analysis to upload traffic." if not admin else "No analyses have been stored yet."); return
    cols = (["Analysis", "Owner"] if admin else ["Analysis"]) + ["File", "Time", "Duration", "Risk", "Status", "Alerts"]
    rows = []
    for m in items:
        r = [f'<span class="mono">{T.esc(m["id"][:24])}</span>'] + ([T.esc(m.get("owner") or "N/A")] if admin else [])
        r += [T.esc(m["filename"]), T.esc(str(m["created"]).replace("T", " ")), _duration(m), T.pct(m.get("peak_risk")), T.state_chip(m["state"]) if m.get("state") else "N/A", f'{int(m.get("alerts", 0))}']
        rows.append(r)
    T.table(cols, rows)
    st.caption("Risk = peak forecast probability over the whole file; Status = highest risk state reached. N/A means the analysis was saved before these fields were recorded.")
    T.section("Open or delete an analysis")
    ids = [m["id"] for m in items]
    names = {m["id"]: f'{m["filename"]} · {str(m["created"]).replace("T", " ")}' + (f' · {m.get("owner") or "no owner"}' if admin else "") for m in items}
    pick = st.selectbox("Analysis", ids, format_func=lambda i: names[i], label_visibility="collapsed")
    m = next(x for x in items if x["id"] == pick)
    c = st.columns([1, 1, 4])
    if c[0].button("Open", type="primary", width="stretch", key="hist_open"):
        _open(m)
    if c[1].button("Delete", width="stretch", key="hist_del"):
        H.delete_analysis(m["id"]); A.log("analysis_deleted", m["id"])
        if A.has_analysis() and st.session_state["analysis"].id == m["id"]:
            st.session_state["analysis"] = None
        st.rerun()


def settings() -> None:
    fc, meta = A.forecaster(), A.metadata()
    T.header("Model Health", "Model integrity, data health and forecast configuration.", [T.chip("ADMIN", T.ACCENT)])
    fp = A.current_fingerprint()
    need = ("feature_names", "stage_names", "window", "preprocessing", "warning_threshold")
    meta_ok = all(k in meta for k in need)
    feat_ok = list(fc.names) == list(fc.pre.names) and len(fc.names) == int(meta.get("n_features", len(fc.names)))
    left, right = st.columns(2, gap="large")
    with left:
        T.section("Model health")
        st.markdown('<div class="card">' + T.status_line(True, "Model loaded", A.model_label(fc)) + T.status_line(bool(fc.pre.fitted), "Scaler loaded", "fitted on training data only" if fc.pre.fitted else "not fitted")
                    + T.status_line(meta_ok, "Metadata valid", "all required keys present" if meta_ok else "required keys missing") +
                    T.status_line(feat_ok, "Feature compatibility", f"{len(fc.names)} features · {int(fc.pre.active.sum())} active") + "</div>", unsafe_allow_html=True)
    with right:
        T.section("Data health")
        if A.has_analysis():
            res = st.session_state["analysis"]; s_ = res.stats; fr_ = res.info.get("feature_report", {}); un = fr_.get("unavailable_in_input", [])
            kind = {"flow-csv": "Flow CSV", "pcap": "Packet capture", "state-table": "Prepared network-state table"}.get(res.info.get("kind"), res.info.get("kind") or "N/A")
            st.markdown('<div class="card">' + T.status_line(True, "Input type", kind) + T.status_line(True, "Records", T.num(s_.get("records")))
                        + T.status_line(True, "Time range", f'{s_["start"]:%Y-%m-%d %H:%M:%S} → {s_["end"]:%H:%M:%S}') +
                        T.status_line(None if un else True, "Feature availability", f"{len(un)} model feature(s) empty in this file" if un else "all model features derived") + "</div>", unsafe_allow_html=True)
            st.caption(f"Current analysis: {res.name}")
        else:
            st.markdown('<div class="card">' + T.status_line(None, "No analysis loaded", "run or open an analysis to see data health") + "</div>", unsafe_allow_html=True)
    T.section("Model fingerprint (SHA-256)")
    T.kv([("Combined", f'<span class="mono">{fp["combined"]}</span>')] + [(k, f'<span class="mono">{v}</span>') for k, v in fp["files"].items()])
    if A.has_analysis():
        res = st.session_state["analysis"]
        from src.analysis.integrity import unchanged
        ok = unchanged(res.integrity_before, res.integrity_after)
        st.caption("Integrity check for the current analysis: " + ("model weights and scaler were unchanged by the analysis." if ok else "no before/after fingerprint is available for this analysis."))
    T.section("Training dataset")
    ds = meta.get("dataset") or {}
    T.kv([("Dataset", T.esc(A.dataset_label(meta))), ("Windows used", f'{ds.get("windows_total", "N/A"):,}' if isinstance(ds.get("windows_total"), int) else "N/A"),
          ("Period", f'{T.esc(str(ds.get("start", "N/A")))} → {T.esc(str(ds.get("end", "N/A")))}'), ("Trained", T.esc(str(meta.get("trained_at", "N/A")).replace("T", " "))),
          ("Project version", T.esc(str(meta.get("project_version", "N/A"))))])
    if meta.get("synthetic_demo_data"):
        st.warning("This model was trained on synthetic demo data.")
    T.section("Forecast configuration")
    c = st.columns(3)
    c[0].markdown(T.stat("Window size", f"{fc.wsec} s", "fixed by the trained model"), unsafe_allow_html=True)
    c[1].markdown(T.stat("History used", f"{fc.L} windows", f"{fc.L * fc.wsec} s"), unsafe_allow_html=True)
    c[2].markdown(T.stat("Forecast horizon", f"{fc.H} windows", f"{fc.H * fc.wsec} s ahead"), unsafe_allow_html=True)
    st.caption("Window size, history and horizon are properties of the trained model; changing them requires retraining.")
    T.section("Alert threshold")
    default = float(fc.threshold); cur = min(max(A.threshold(fc), 0.05), 0.99)
    val = st.slider("Alert threshold", 0.05, 0.99, cur, 0.01, help="Forecast probability that must be reached (for the configured number of consecutive windows) to open an alert.")
    t = st.columns([1, 1, 4])
    if t[0].button("Apply", type="primary"):
        st.session_state["thr_override"] = None if abs(val - default) < 1e-9 else float(val)
        A.log("threshold_changed", f"{val:.2f}")
        if A.has_analysis():
            W.apply_threshold(st.session_state["analysis"], val); st.session_state["validation_cache"] = {}; st.session_state["cache_ex"] = {}
        st.rerun()
    if t[1].button("Reset to default"):
        st.session_state["thr_override"] = None
        A.log("threshold_changed", f"reset to {default:.2f}")
        if A.has_analysis():
            W.apply_threshold(st.session_state["analysis"], default); st.session_state["validation_cache"] = {}
        st.rerun()
    st.caption(f"Default {default:.2f} was calibrated on the validation split during training. Changing it only changes which forecasts raise alerts; it does not alter the model's probabilities. "
               "The override applies to this browser session.")
    T.section("Model information")
    T.kv([("Architecture", A.model_label(fc)), ("Input features", f"{len(fc.names)} ({int(fc.pre.active.sum())} active)"), ("Sequence length", f"{fc.L} windows"),
          ("Forecast horizon", f"{fc.H} windows"), ("Parameters", f'{meta.get("model", {}).get("parameters", 0):,}')])
    inactive = [n for n, a in zip(fc.names, fc.pre.active) if not a]
    if inactive:
        st.caption("Features the model ignores (constant in its training data): " + ", ".join(inactive))
    with st.expander("Developer / Advanced"):
        mm = meta.get("model", {})
        st.json({"hyperparameters": {k: mm.get(k) for k in ("hidden_size", "layers", "dropout", "attention_heads", "learning_rate", "weight_decay", "batch_size", "epochs")},
                 "optimizer": meta.get("optimizer"), "seed": meta.get("seed"), "epochs_run": meta.get("epochs_run"), "device": meta.get("device"),
                 "split": meta.get("split"), "environment": meta.get("environment")}, expanded=False)
        st.caption("Read-only. Training is done offline with scripts/train.py and is never triggered by the application.")
