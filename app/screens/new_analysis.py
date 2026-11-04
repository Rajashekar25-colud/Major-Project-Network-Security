"""New Analysis: upload -> Traffic ready (validation) -> processing pipeline."""
from __future__ import annotations

import streamlit as st

import appstate as A
import components as C
import theme as T
from src.analysis import history as H
from src.analysis import workflow as W
from src.analysis.ingest import inspect_traffic
from src.config import MODELS_DIR, ROOT

SAMPLES = {"Sample flow CSV (labelled, CSE-CIC-IDS2018 format)": ROOT / "data" / "sample" / "test_flows_cic_format.csv",
           "Sample packet capture (PCAP, unlabelled)": ROOT / "data" / "sample" / "sample_traffic.pcap"}


STEPS = [("Upload traffic", "CSV · PCAP · PCAPNG"), ("Validate traffic", "format, fields, model fit"), ("Traffic ready", "records, range, windows"), ("Begin analysis", "forecast, risk, ATT&CK")]


def render() -> None:
    ss = st.session_state
    ok = ss.get("report") is not None and ss["report"].ok
    T.steps(4 if ss["phase"] == "processing" else 3 if ss["phase"] == "ready" and ok else 2 if ss["phase"] == "ready" else 1, STEPS)
    if ss["phase"] == "ready" and ss.get("report") is not None:
        _ready()
    elif ss["phase"] == "processing" and ss.get("report") is not None:
        _process()
    else:
        _upload()


def _upload() -> None:
    ss = st.session_state
    T.header("New Analysis", "Upload network traffic to forecast future attack progression.")
    up = st.file_uploader("Drop traffic file here", type=["csv", "pcap", "cap", "pcapng"], key=f"uploader_{ss['uploader_n']}", label_visibility="collapsed",
                          help="CSV (flow records), PCAP or PCAPNG")
    st.markdown('<div class="hint">Drop traffic file here &nbsp;·&nbsp; CSV · PCAP · PCAPNG</div>', unsafe_allow_html=True)
    st.markdown('<div class="card" style="margin-top:1rem"><b>The trained model will analyze this traffic.</b><br><span class="hint">The uploaded data will not be used for retraining.</span></div>', unsafe_allow_html=True)
    name = data = None
    if up is not None:
        name, data = up.name, up.getvalue()
    avail = {k: v for k, v in SAMPLES.items() if v.exists()}
    if avail:
        with st.expander("No file at hand? Use a bundled sample"):
            for label, path in avail.items():
                if st.button(label, key=f"sample_{path.name}"):
                    name, data = path.name, path.read_bytes()
    if name is not None:
        _validate(name, data)


def _validate(name: str, data: bytes) -> None:
    fc, cfg = A.forecaster(), A.config()
    with st.status(f"Reading and validating {name} ...", expanded=True) as status:
        ph = st.empty()
        rep = inspect_traffic(name, data, fc, cfg, progress=lambda m: ph.markdown(f"`{m.strip()}`"))
        status.update(label="Validation finished", state="complete" if rep.ok else "error", expanded=False)
    st.session_state.update(report=rep, phase="ready")
    st.rerun()


def _ready() -> None:
    ss = st.session_state; rep = ss["report"]
    if not rep.ok:
        T.header("This file cannot be analyzed", rep.filename)
        st.markdown(f'<div class="card">{C.checks_html(rep.checks)}</div>', unsafe_allow_html=True)
        st.error(rep.error)
        st.button("Choose another file", type="primary", on_click=A.start_new)
        return
    T.header("Traffic ready", rep.filename)
    st.markdown(f'<div class="card">{C.checks_html(rep.checks)}</div>', unsafe_allow_html=True)
    s = rep.stats
    st.markdown("&nbsp;")
    c = st.columns(5)
    c[0].markdown(T.stat("Records", T.num(s["records"]), "flow records" if s["kind"] != "state-table" else "flows in pre-aggregated windows"), unsafe_allow_html=True)
    c[1].markdown(T.stat("Time range", f'<span style="font-size:1.05rem">{s["start"]:%H:%M:%S} → {s["end"]:%H:%M:%S}</span>', f'{s["start"]:%Y-%m-%d}'), unsafe_allow_html=True)
    c[2].markdown(T.stat("Duration", T.secs(s["duration_s"])), unsafe_allow_html=True)
    c[3].markdown(T.stat("Time windows", f'{s["windows"]:,}', f'{s["forecastable_windows"]:,} with full history'), unsafe_allow_html=True)
    c[4].markdown(T.stat("Window size", f'{s["window_seconds"]} s', f'history {s["history_windows"]} · forecast {s["horizon_windows"]}'), unsafe_allow_html=True)
    if s["labelled"]:
        st.info("Ground-truth labels were found in this file, so Forecast Validation (forecast vs actual) will be available after the analysis.")
    else:
        st.caption("No ground-truth labels in this file: it can be forecast, but prediction-vs-actual metrics cannot be calculated.")
    if s["gap_filled_windows"]:
        st.caption(f'{s["gap_filled_windows"]:,} idle windows inside short gaps were represented as quiet (all-zero) windows.')
    b = st.columns([1, 1, 4])
    if b[0].button("Begin Analysis", type="primary", width="stretch"):
        ss["phase"] = "processing"; st.rerun()
    b[1].button("Cancel", width="stretch", on_click=A.start_new)


def _process() -> None:
    ss = st.session_state; rep, fc, cfg = ss["report"], A.forecaster(), A.config()
    thr = A.threshold(fc)
    T.header("Analyzing new traffic", rep.filename)
    ph, bar, note = st.empty(), st.progress(0.0), st.empty()
    prog = {k: ("todo", "") for k in ("sequence", "forecast", "risk", "map", "explain")}

    def draw() -> None:
        rows = [("done", "File validation", ""), ("done", "Preprocessing", ""), ("done", "Feature extraction", f"{len(fc.names)} features per window"),
                ("done", "Time-window construction", f'{rep.stats["windows"]:,} windows of {fc.wsec} s'),
                (*prog["sequence"][:1], "Sequence construction", prog["sequence"][1]), (*prog["forecast"][:1], "Forecasting", prog["forecast"][1]),
                (*prog["risk"][:1], "Risk assessment and alert logic", prog["risk"][1]), (*prog["map"][:1], "Attack-stage mapping", prog["map"][1]),
                (*prog["explain"][:1], "Explanation", prog["explain"][1])]
        ph.markdown(f'<div class="card">{C.pipeline_html(rows)}</div>', unsafe_allow_html=True)

    def cb(stage: str, done: int, total: int) -> None:
        if stage == "sequence":
            prog["sequence"] = ("done", f"{fc.L} windows of history per forecast") if done >= total else ("now", "")
            if done >= total:
                prog["forecast"] = ("now", "")
        elif stage == "forecast":
            prog["forecast"] = ("done" if done >= total else "now", f"{done:,} / {total:,} windows")
            if done >= total:
                prog["risk"] = ("now", "")
            bar.progress(min(0.5 * done / max(total, 1), 0.5))
        elif stage == "risk":
            fin = done >= total
            prog["risk"] = ("done" if fin else "now", f"Processing window {done:,} / {total:,}")
            if fin:
                prog["map"] = ("done", "stages come from the model's stage head")
                prog["explain"] = ("done", "computed for the selected forecast step when opened")
            bar.progress(0.5 + 0.5 * done / max(total, 1))
        draw()

    draw()
    try:
        res = W.run_analysis(rep.states, rep.filename, fc, cfg, thr, info=rep.info, stats=rep.stats, models_dir=MODELS_DIR, progress=cb, analysis_id=H.new_id(rep.filename))
    except Exception as e:
        note.error(f"The analysis failed: {e}")
        st.button("Back", on_click=A.start_new)
        return
    try:
        H.save_analysis(res, owner=A.username()); A.log("analysis_created", f"{res.id} · {res.n} windows · {len(res.rs.engine.alerts)} alert(s)")
    except Exception as e:
        st.warning(f"Analysis finished but could not be saved to History: {e}")
    A.set_analysis(res)
    ss.update(page="overview", phase="upload", report=None, uploader_n=int(ss["uploader_n"]) + 1)
    st.rerun()
