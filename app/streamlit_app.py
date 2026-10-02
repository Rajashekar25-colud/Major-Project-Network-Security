"""AI-Based Network Attack Forecasting - offline SOC dashboard (Streamlit).

Run:  streamlit run app/streamlit_app.py
Everything runs locally/offline: no network calls, no external services.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import MODELS_DIR, load_config
from src.explain.explainer import explain_shap, explain_window
from src.models.inference import Forecaster, artifacts_present
from src.replay import ReplaySession
from src.service import load_states_csv, read_json, states_from_upload

st.set_page_config(page_title="Network Attack Forecasting", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

SEV_COLOR = {"OK": "#22c55e", "WATCH": "#eab308", "WARNING": "#f97316", "CRITICAL": "#ef4444"}
STAGE_COLOR = {"Benign": "#334155", "Reconnaissance": "#38bdf8", "Initial Access": "#a78bfa", "Credential Access": "#f472b6",
               "Discovery": "#2dd4bf", "Lateral Movement": "#fb923c", "Command and Control": "#f43f5e",
               "Exfiltration": "#e11d48", "Impact": "#ef4444", "Attack": "#f97316"}
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,26,46,0.6)",
            margin=dict(l=10, r=10, t=36, b=10), font=dict(color="#cbd5e1"))

st.markdown("""
<style>
.block-container{padding-top:1.2rem;max-width:1500px}
.hero{display:flex;justify-content:space-between;align-items:center;padding:14px 20px;border-radius:14px;
      background:linear-gradient(135deg,#0f1c33,#13294b);border:1px solid #1e3a5f;margin-bottom:12px}
.hero h1{font-size:1.45rem;margin:0;color:#f1f5f9}.hero p{margin:2px 0 0;color:#94a3b8;font-size:.86rem}
.status{padding:14px 18px;border-radius:12px;font-weight:600;font-size:1.02rem;margin:6px 0 12px;border:1px solid}
.kpi{background:#111a2e;border:1px solid #1e2b45;border-radius:12px;padding:12px 14px}
.kpi .l{color:#94a3b8;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em}
.kpi .v{font-size:1.55rem;font-weight:700;color:#f8fafc;line-height:1.25}.kpi .s{color:#64748b;font-size:.75rem}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.75rem;font-weight:600;color:#0b1220}
.demo{background:#3b2a05;border:1px solid #a16207;color:#fde68a;padding:8px 14px;border-radius:10px;font-size:.85rem;margin-bottom:8px}
div[data-testid="stTabs"] button{font-weight:600}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner=False)
def _config():
    return load_config()


@st.cache_resource(show_spinner="Loading model ...")
def _forecaster(mtime: float):
    return Forecaster.load(MODELS_DIR)


@st.cache_data(show_spinner="Reading state table ...")
def _states_file(path: str, mtime: float):
    return load_states_csv(path, _config())


def _kpi(label, value, sub="", color=None):
    c = f"color:{color};" if color else ""
    return f'<div class="kpi"><div class="l">{label}</div><div class="v" style="{c}">{value}</div><div class="s">{sub}</div></div>'


def _fmt_secs(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "n/a"
    s = float(s)
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.1f} min"


cfg = _config()
have_model = artifacts_present(MODELS_DIR)
meta = read_json(MODELS_DIR / "metadata.json") or {}
evaluation = read_json(MODELS_DIR / "evaluation.json")

st.markdown(f"""<div class="hero"><div><h1>🛡️ AI-Based Network Attack Forecasting</h1>
<p>Temporal world model · multi-step forecast · MITRE ATT&amp;CK stage · explainable early warning · fully offline</p></div>
<div style="text-align:right;color:#94a3b8;font-size:.8rem">MLRITM · Project batch MP-C04<br>Guide: Dr. Y. Madhusekhar</div></div>""", unsafe_allow_html=True)

if not have_model:
    st.error("No trained model found in `models/`. Train one with `python scripts/train.py --data <CIC-IDS2018 folder>` "
             "or create the demo model with `python scripts/make_demo.py`.")
    st.stop()
fc = _forecaster((MODELS_DIR / "world_model.pt").stat().st_mtime)
if meta.get("synthetic_demo_data"):
    st.markdown('<div class="demo">⚠️ <b>DEMO MODEL</b> - trained on <b>synthetic</b> CIC-IDS2018-format data so the dashboard runs out of the box. '
                "Metrics shown are not research results. Train on the real dataset with <code>scripts/train.py</code>.</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- sidebar: data
ss = st.session_state
ss.setdefault("playing", False); ss.setdefault("session", None); ss.setdefault("key", None)
with st.sidebar:
    st.markdown("### 📂 Traffic source")
    state_files = sorted((ROOT / "data" / "states").glob("*.csv"))
    state_files = [p for p in state_files if "_info" not in p.name]
    choices = [p.name for p in state_files] + ["Upload file (state CSV / flow CSV / PCAP)"]
    default_i = next((i for i, p in enumerate(state_files) if "cicids2018" in p.name), 0)
    src_choice = st.selectbox("Replay source", choices, index=min(default_i, len(choices) - 1))
    states = None; src_label = ""
    if src_choice.startswith("Upload"):
        up = st.file_uploader("Flow CSV, prepared state CSV or PCAP / PCAPNG", type=["csv", "pcap", "pcapng", "cap"])
        sample = ROOT / "data" / "sample" / "sample_traffic.pcap"
        if sample.exists() and st.button("Use bundled sample PCAP", width="stretch"):
            ss["upload_name"], ss["upload_bytes"] = sample.name, sample.read_bytes()
        if up is not None:
            ss["upload_name"], ss["upload_bytes"] = up.name, up.getvalue()
        if ss.get("upload_bytes"):
            try:
                with st.spinner("Converting traffic to network states ..."):
                    states, uinfo = states_from_upload(ss["upload_name"], ss["upload_bytes"], cfg)
                src_label = f"{ss['upload_name']} ({uinfo['kind']})"
            except Exception as e:
                st.error(f"Could not read this file: {e}")
                st.caption("Tip: run `python scripts/pcap_info.py --pcap <file>` to see what the capture contains.")
    else:
        p = ROOT / "data" / "states" / src_choice
        states = _states_file(str(p), p.stat().st_mtime); src_label = src_choice

    st.markdown("### ⚙️ Warning threshold")
    thr_default = float(fc.threshold)
    thr = st.slider("Alert threshold (calibrated on validation)", 0.05, 0.95, thr_default, 0.01,
                    help="Default is chosen automatically on the validation split. Changing it restarts the replay.")

if states is None or len(states) == 0:
    st.info("Choose a traffic source in the sidebar to begin."); st.stop()

report = fc.validate(states)
if not report["ok"]:
    st.error(f"**Feature mismatch** - the input is missing {len(report['missing'])} model feature(s): "
             f"{', '.join(report['missing'][:10])}. Rebuild states with `scripts/prepare_states.py` using the same config.")
    st.stop()
if len(states) < fc.L + 2:
    st.warning(f"Need at least {fc.L + 2} windows ({(fc.L + 2) * fc.wsec} s) of traffic; got {len(states)}."); st.stop()

key = (src_label, len(states), round(thr, 3), str(states["timestamp"].iat[0]))
if ss["key"] != key:
    with st.spinner("Preparing forecasts ..."):
        ss["session"] = ReplaySession(states, fc, cfg, thr)
    ss["key"] = key; ss["playing"] = False
rs: ReplaySession = ss["session"]

# ----------------------------------------------------------------------------- sidebar: replay controls
with st.sidebar:
    st.markdown("### ▶️ Replay controls")
    src_days = states["source"].astype(str).to_numpy()[rs.ends]
    days = ["All"] + list(dict.fromkeys(src_days))
    day = st.selectbox("Capture day / file", days)
    lo_k, hi_k = (0, len(rs.ends) - 1) if day == "All" else (int(np.flatnonzero(src_days == day)[0]), int(np.flatnonzero(src_days == day)[-1]))
    ts_all = states["timestamp"].to_numpy()[rs.ends]
    span = st.slider("Time range (window index)", lo_k, hi_k, (lo_k, hi_k), key=f"rng_{day}_{key[1]}")
    st.caption(f"{pd.Timestamp(ts_all[span[0]]):%Y-%m-%d %H:%M:%S}  →  {pd.Timestamp(ts_all[span[1]]):%Y-%m-%d %H:%M:%S}  "
               f"({span[1] - span[0] + 1:,} windows)")
    if (rs.lo, rs.hi) != span:
        rs.set_range(*span); ss["playing"] = False
    speed = st.select_slider("Speed (windows / second)", [1, 2, 5, 10, 25, 50, 100, 250], value=5,
                             help=f"1 window = {fc.wsec} s of network traffic")
    c1, c2 = st.columns(2)
    if c1.button("▶ Start" if not ss["playing"] else "⏸ Pause", type="primary", width="stretch"):
        if rs.finished:
            rs.reset()
        ss["playing"] = not ss["playing"]
    if c2.button("⏭ Step", width="stretch"):
        ss["playing"] = False; rs.step(1)
    c3, c4 = st.columns(2)
    if c3.button("⏩ +10", width="stretch"):
        ss["playing"] = False; rs.step(10)
    if c4.button("⟲ Reset", width="stretch"):
        ss["playing"] = False; rs.reset()
    jump = st.number_input("Jump to window", lo_k, hi_k, span[0], help="Runs the replay silently up to this window (alerts included).")
    if st.button("Go", width="stretch"):
        ss["playing"] = False; rs.reset(); rs.step(max(int(jump) - rs.lo + 1, 0))

    st.markdown("---")
    st.caption(f"Model: LSTM + attention · {meta.get('model', {}).get('parameters', 0):,} params\n\n"
               f"Input {fc.L}×{fc.wsec}s · forecast {fc.H}×{fc.wsec}s ({fc.H * fc.wsec}s ahead)")


# ----------------------------------------------------------------------------- live view (auto-refreshing fragment)
def _risk_figure(h: pd.DataFrame, thr: float):
    fig = make_subplots(rows=1, cols=1)
    if len(h) == 0:
        fig.update_layout(**PLOT, height=330, title="Forecast risk timeline"); return fig
    hh = h.iloc[-600:]
    # ground-truth attack shading
    t = hh["truth"].to_numpy(); ts = hh["timestamp"]
    starts = np.flatnonzero(np.diff(np.concatenate([[0], t.astype(int), [0]])) == 1)
    ends = np.flatnonzero(np.diff(np.concatenate([[0], t.astype(int), [0]])) == -1) - 1
    for s_, e_ in zip(starts, ends):
        fig.add_vrect(x0=ts.iloc[s_], x1=ts.iloc[min(e_, len(ts) - 1)], fillcolor="rgba(239,68,68,0.20)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=ts, y=hh["risk"], mode="lines", name="Forecast risk (max over horizon)", line=dict(color="#22d3ee", width=2.4),
                             fill="tozeroy", fillcolor="rgba(34,211,238,0.10)"))
    fig.add_hline(y=thr, line=dict(color="#f97316", dash="dash"), annotation_text=f"threshold {thr:.2f}", annotation_position="top left")
    fig.update_yaxes(range=[0, 1.02], title="infiltration probability"); fig.update_xaxes(title=None)
    fig.update_layout(**PLOT, height=330, title="Infiltration-probability timeline (red bands = labelled attack windows in the data)",
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig


def _horizon_figure(rs: ReplaySession, thr: float):
    ft = rs.forecast_table()
    colors = [STAGE_COLOR.get(s, "#22d3ee") if p >= thr else "#475569" for s, p in zip(ft["most_likely_attack_stage"], ft["attack_probability"])]
    fig = go.Figure(go.Bar(x=[f"+{s}s" for s in ft["seconds_ahead"]], y=ft["attack_probability"], marker_color=colors,
                           text=[f"{p:.0%}" for p in ft["attack_probability"]], textposition="outside",
                           customdata=ft["most_likely_attack_stage"], hovertemplate="%{x}: %{y:.1%}<br>stage: %{customdata}<extra></extra>"))
    fig.add_hline(y=thr, line=dict(color="#f97316", dash="dash"))
    fig.update_yaxes(range=[0, 1.15], title="P(attack)"); fig.update_layout(**PLOT, height=300, title="K-step forecast (next windows)")
    return ft, fig


def _stage_figure(cur, fc):
    sp = np.asarray(cur["stage_prob"])[int(np.argmax(cur["probs"]))]
    names = [s for s in fc.stages]
    order = np.argsort(sp)
    fig = go.Figure(go.Bar(x=sp[order], y=[names[i] for i in order], orientation="h",
                           marker_color=[STAGE_COLOR.get(names[i], "#22d3ee") for i in order],
                           text=[f"{sp[i]:.0%}" for i in order], textposition="outside"))
    fig.update_xaxes(range=[0, 1.1]); fig.update_layout(**PLOT, height=300, title="ATT&CK stage distribution (peak-risk step)")
    return fig


def live_view():
    rs: ReplaySession = ss["session"]
    if ss["playing"]:
        rs.step(max(1, int(round(speed * 0.6))))
        if rs.finished:
            ss["playing"] = False
            st.rerun()
    cur = rs.current()
    if cur is None:
        st.info("Press **▶ Start** (or ⏭ Step) in the sidebar to replay the traffic. Nothing is pre-computed for display - "
                "each window is forecast, scored and passed through the warning engine as it is replayed.")
        return
    level = cur["level"]; col = SEV_COLOR[level]; a = cur["open_alert"]
    if a is not None:
        msg = (f"🚨 <b>{a.severity} ALERT #{a.id}</b> - forecast stage <b>{a.stage}</b> ({fc.meta['attack_tactics'].get(a.stage, '')}) · "
               f"peak risk {a.peak_risk:.0%} · attack expected within ~{_fmt_secs(a.eta_seconds)} · open for {a.n_windows} windows")
        col = SEV_COLOR[a.severity]
    elif level == "WATCH":
        msg = f"👁️ <b>WATCH</b> - risk {cur['risk']:.0%} is rising but below the alert threshold ({thr:.0%})"
    elif level in ("WARNING", "CRITICAL"):
        msg = f"⏳ Risk {cur['risk']:.0%} above threshold - waiting for persistence ({cfg['warning']['persistence_windows']} consecutive windows) before alerting"
    else:
        msg = f"✅ <b>NORMAL</b> - no attack progression forecast (risk {cur['risk']:.0%})"
    st.markdown(f'<div class="status" style="background:{col}22;border-color:{col};color:{col}">{msg}</div>', unsafe_allow_html=True)

    h = rs.history()
    done = rs.cursor - rs.lo + 1
    n_alerts = len(rs.engine.alerts)
    k = st.columns(6)
    k[0].markdown(_kpi("Forecast risk", f"{cur['risk']:.0%}", f"threshold {thr:.0%}", col), unsafe_allow_html=True)
    k[1].markdown(_kpi("Predicted stage", cur["stage"] if cur["risk"] >= thr else "None", fc.meta['attack_tactics'].get(cur["stage"], "MITRE ATT&CK") if cur["risk"] >= thr else "-"), unsafe_allow_html=True)
    k[2].markdown(_kpi("Severity", level, "current window", SEV_COLOR[level]), unsafe_allow_html=True)
    k[3].markdown(_kpi("Alerts raised", n_alerts, f"{sum(x.status == 'OPEN' for x in rs.engine.alerts)} open"), unsafe_allow_html=True)
    k[4].markdown(_kpi("Replay progress", f"{done / rs.n_total:.0%}", f"{done:,} / {rs.n_total:,} windows"), unsafe_allow_html=True)
    k[5].markdown(_kpi("Stream time", f"{pd.Timestamp(cur['time']):%H:%M:%S}", f"{pd.Timestamp(cur['time']):%Y-%m-%d}"), unsafe_allow_html=True)
    st.progress(min(done / max(rs.n_total, 1), 1.0))

    st.plotly_chart(_risk_figure(h, thr), width="stretch", key=f"risk_{rs.cursor}")
    c1, c2 = st.columns([3, 2])
    ft, fig = _horizon_figure(rs, thr)
    c1.plotly_chart(fig, width="stretch", key=f"hz_{rs.cursor}")
    c2.plotly_chart(_stage_figure(cur, fc), width="stretch", key=f"sg_{rs.cursor}")

    st.markdown("##### 🚩 Flagged traffic patterns (most recent windows)")
    rows = h.tail(400)
    rows = rows[(rows["risk"] >= thr * fc.meta["warning"]["watch_ratio"]) | rows["truth"]].tail(12).iloc[::-1]
    if len(rows):
        S = rs.states.iloc[rows["row"].to_numpy()]
        flag = pd.DataFrame({"time": rows["timestamp"].dt.strftime("%H:%M:%S").to_numpy(), "risk": rows["risk"].round(3).to_numpy(),
                             "forecast stage": rows["stage"].to_numpy(), "flows": S["flow_count"].to_numpy(),
                             "distinct dst ports": S["unique_dst_ports"].to_numpy(), "SYN ratio": S["syn_ratio"].round(2).to_numpy(),
                             "failed-flow ratio": S["failed_flow_ratio"].round(2).to_numpy(), "scan score": S["scan_score"].round(2).to_numpy(),
                             "dominant port": S[[c for c in S.columns if c.startswith("dst_port_")]].idxmax(axis=1).str.replace("dst_port_", "").to_numpy(),
                             "dataset label": rows["truth_stage"].to_numpy()})
        st.dataframe(flag, width="stretch", hide_index=True)
    else:
        st.caption("No flagged windows yet.")


frag = st.fragment(run_every=0.6 if ss["playing"] else None)(live_view)

tab_live, tab_explain, tab_alerts, tab_health, tab_eval, tab_about = st.tabs(
    ["📡 Live SOC view", "🔍 Explanation", "🚨 Alert history", "🩺 Model & data health", "📊 Evaluation", "ℹ️ Methodology"])

with tab_live:
    frag()

# ----------------------------------------------------------------------------- explanation
with tab_explain:
    cur = rs.current()
    if cur is None:
        st.info("Step or start the replay first - explanations are computed for the current window.")
    else:
        st.markdown(f"**Explaining the forecast at {pd.Timestamp(cur['time']):%Y-%m-%d %H:%M:%S}** - risk {cur['risk']:.0%}, forecast stage *{cur['stage']}*")
        seq = rs.sequence()
        ex = explain_window(fc, seq, int(cfg["inference"]["top_features"]) + 6)
        c1, c2 = st.columns([3, 2])
        tf = pd.DataFrame(ex["top_features"]).iloc[::-1]
        fig = go.Figure(go.Bar(x=tf["attribution"], y=tf["feature"], orientation="h",
                               marker_color=["#f97316" if v > 0 else "#38bdf8" for v in tf["attribution"]],
                               hovertemplate="%{y}<br>attribution %{x:.4f}<extra></extra>"))
        fig.update_layout(**PLOT, height=430, title="Traffic features driving the forecast (gradient × input; orange raises risk)")
        c1.plotly_chart(fig, width="stretch")
        L = len(ex["temporal_attention"])
        lab = [f"t-{L - 1 - i}" if i < L - 1 else "now" for i in range(L)]
        fig2 = go.Figure()
        fig2.add_bar(x=lab, y=ex["temporal_attention"], name="attention", marker_color="#22d3ee")
        fig2.add_scatter(x=lab, y=ex["temporal_attribution"], name="attribution share", mode="lines+markers", line=dict(color="#f97316"))
        fig2.update_layout(**PLOT, height=430, title="Which past windows mattered (attention / attribution)", legend=dict(orientation="h", y=1.1))
        c2.plotly_chart(fig2, width="stretch")
        st.dataframe(tf.iloc[::-1][["feature", "direction", "attribution", "share", "current_z"]].rename(
            columns={"current_z": "current value (z-score)", "share": "share of total attribution"}), width="stretch", hide_index=True)

        st.markdown("##### SHAP (optional, exact SHAP values via GradientExplainer)")
        bgp = MODELS_DIR / "background_sequences.npy"
        if st.button("Compute SHAP for this window"):
            with st.spinner("Computing SHAP values ..."):
                sh = explain_shap(fc, seq, np.load(bgp) if bgp.exists() else None, int(cfg["inference"]["top_features"]))
            if sh.get("available"):
                sd = pd.DataFrame(sh["top_features"]).iloc[::-1]
                f3 = go.Figure(go.Bar(x=sd["shap_value"], y=sd["feature"], orientation="h",
                                      marker_color=["#f97316" if v > 0 else "#38bdf8" for v in sd["shap_value"]]))
                f3.update_layout(**PLOT, height=380, title="SHAP values (sum over the input sequence)")
                st.plotly_chart(f3, width="stretch")
            else:
                st.warning(sh.get("reason"))

        st.markdown("##### World-model rollout (imagined future, autoregressive)")
        import torch
        steps = st.slider("Rollout steps", 2, 24, 12)
        ro = fc.model.rollout(torch.from_numpy(seq).unsqueeze(0), steps)
        pr = ro["attack_prob"][0].numpy()
        f4 = go.Figure(go.Scatter(x=[f"+{(i + 1) * fc.wsec}s" for i in range(steps)], y=pr, mode="lines+markers", line=dict(color="#a78bfa")))
        f4.add_hline(y=thr, line=dict(color="#f97316", dash="dash"))
        f4.update_layout(**PLOT, height=280, title="Attack probability along the model's own predicted state trajectory", yaxis=dict(range=[0, 1.02]))
        st.plotly_chart(f4, width="stretch")

# ----------------------------------------------------------------------------- alerts
with tab_alerts:
    ad = rs.alerts_df()
    if ad.empty:
        st.info("No alerts have been raised in this replay yet.")
    else:
        m = st.columns(4)
        m[0].metric("Alerts", len(ad)); m[1].metric("Open", int((ad["status"] == "OPEN").sum()))
        m[2].metric("Critical", int((ad["peak_severity"] == "CRITICAL").sum()))
        m[3].metric("Mean alert length", f"{ad['duration_windows'].mean() * fc.wsec:.0f} s")
        show = ad[["id", "status", "peak_severity", "stage", "stages", "opened_time", "closed_time", "duration_windows", "peak_risk", "eta_seconds"]].copy()
        show["peak_risk"] = show["peak_risk"].round(3)
        show = show.rename(columns={"peak_severity": "severity", "stages": "stage progression", "eta_seconds": "eta (s)", "duration_windows": "windows"})
        st.dataframe(show.iloc[::-1], width="stretch", hide_index=True)
        st.download_button("Download alert history (CSV)", show.to_csv(index=False).encode(), "alerts.csv", "text/csv")
    st.caption(f"Engine: opens after {cfg['warning']['persistence_windows']} consecutive above-threshold windows · closes after "
               f"{cfg['warning']['clear_windows']} windows below {cfg['warning']['clear_ratio']:.0%}×threshold · "
               f"{cfg['warning']['cooldown_windows']}-window cooldown per stage · updates instead of duplicating.")

# ----------------------------------------------------------------------------- health
with tab_health:
    d = meta.get("dataset", {})
    st.markdown("##### Model")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Trained at", str(meta.get("trained_at", "n/a")).replace("T", " "))
    h2.metric("Evaluated at", str(meta.get("evaluated_at") or (evaluation or {}).get("evaluated_at", "n/a")).replace("T", " "))
    h3.metric("Preprocessing", meta.get("preprocessing_version", "n/a")); h4.metric("Epochs run", meta.get("epochs_run", "n/a"))
    mm = meta.get("model", {})
    st.json({"architecture": mm.get("architecture"), "parameters": mm.get("parameters"),
             "hyperparameters": {k: mm.get(k) for k in ("hidden_size", "layers", "dropout", "attention_heads", "learning_rate", "weight_decay", "batch_size", "epochs")},
             "optimizer": meta.get("optimizer"), "seed": meta.get("seed"), "device": meta.get("device"),
             "warning_threshold": meta.get("warning_threshold"), "calibration": meta.get("calibration", {}).get("method"),
             "stage_vocabulary": meta.get("stage_names"), "window": meta.get("window")}, expanded=False)
    st.markdown("##### Training dataset")
    st.json({"sources": d.get("sources"), "period": [d.get("start"), d.get("end")], "windows": d.get("windows_total"), "segments": d.get("segments"),
             "attack_stage_windows": d.get("stage_window_counts"), "split": d.get("split_summary"), "synthetic": meta.get("synthetic_demo_data", False)}, expanded=False)
    st.markdown("##### Input compatibility (feature-mismatch validation)")
    r = report
    x1, x2, x3 = st.columns(3)
    x1.metric("Model features present", f"{len(fc.names) - len(r['missing'])}/{len(fc.names)}")
    x2.metric("Features unavailable in this input", len(r["unavailable_in_input"]))
    x3.metric("Telemetry the model never saw", len(r["novel_in_input"]))
    if r["unavailable_in_input"]:
        st.warning("The model was trained with these features but they are all-zero in this input (e.g. a flow CSV without packet telemetry): "
                   + ", ".join(r["unavailable_in_input"]))
    if r["novel_in_input"]:
        st.info("Present in this input but neutralised because the model was trained without them (packet-level fields on flow-only training): "
                + ", ".join(r["novel_in_input"]))
    if not (r["unavailable_in_input"] or r["novel_in_input"]):
        st.success("Feature schema matches the trained model.")
    st.markdown("##### Data health of the loaded traffic")
    fs = rs.states
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Windows", f"{len(fs):,}"); q2.metric("Gap-filled (idle) windows", f"{int(fs['filled'].sum()):,}")
    q3.metric("Segments", int(fs["segment"].nunique())); q4.metric("Labelled attack windows", int(fs["is_attack"].sum()))
    st.caption("Flow-level telemetry drives the model on CSE-CIC-IDS2018 CSVs. Packet-level fields (TTL, retransmissions) are populated only from PCAP input; "
               "TCP window comes from CICFlowMeter's initial window bytes on CSV input.")

# ----------------------------------------------------------------------------- evaluation
with tab_eval:
    if not evaluation or "forecast" not in evaluation:
        st.info("No evaluation.json found. Run `python scripts/train.py ...` or `python scripts/evaluate.py ...`.")
    else:
        ev = evaluation
        if meta.get("synthetic_demo_data"):
            st.markdown('<div class="demo">These numbers come from the synthetic demo data - not a research result.</div>', unsafe_allow_html=True)
        st.caption(f"Chronological per-day hold-out · {ev['n_samples']:,} samples · warning threshold {ev['threshold']:.3f} · evaluated {ev.get('evaluated_at', '')}")
        f, o = ev["forecast"], ev["onset"]
        st.markdown("##### Attack forecast (all horizons)  vs  true onset early-warning")
        cols = st.columns(6)
        for c, (lab_, key_) in zip(cols, [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("False-positive rate", "false_positive_rate"), ("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc")]):
            c.metric(lab_, f"{f.get(key_, float('nan')):.3f}", f"onset {o.get(key_, float('nan')):.3f}", delta_color="off")
        st.caption(o.get("definition", ""))
        b = ev.get("baselines", {}).get("logistic_sequence", {})
        if b and "forecast" in b:
            cmp_ = pd.DataFrame({"Model": ["World model (LSTM+attention)", "Logistic regression (sequence)"] ,
                                 "Forecast F1": [f["f1"], b["forecast"]["f1"]], "Forecast recall": [f["recall"], b["forecast"]["recall"]],
                                 "Forecast FPR": [f["false_positive_rate"], b["forecast"]["false_positive_rate"]],
                                 "Onset F1": [o["f1"], b["onset"]["f1"]], "Onset recall": [o["recall"], b["onset"]["recall"]],
                                 "Onset FPR": [o["false_positive_rate"], b["onset"]["false_positive_rate"]]})
            st.markdown("##### Baseline comparison (same inputs, targets, split and threshold rule)")
            st.dataframe(cmp_.round(3), width="stretch", hide_index=True)
        ph = pd.DataFrame(ev["per_horizon"])
        fig = go.Figure()
        for m_ in ("precision", "recall", "f1"):
            fig.add_scatter(x=ph["seconds_ahead"], y=ph[m_], name=m_, mode="lines+markers")
        fig.update_layout(**PLOT, height=300, title="Forecast quality vs how far ahead", xaxis_title="seconds ahead", yaxis=dict(range=[0, 1.02]))
        st.plotly_chart(fig, width="stretch")
        lt = ev["lead_time"]
        st.markdown("##### Warning lead time")
        l = st.columns(6)
        l[0].metric("Attack episodes", lt["attack_episodes"]); l[1].metric("Warned early", lt["warned_early"])
        l[2].metric("Detected late", lt["detected_late"]); l[3].metric("Missed", lt["missed"])
        l[4].metric("Mean lead", _fmt_secs(lt["mean_lead_seconds"])); l[5].metric("False alerts / hour", f"{lt['false_alerts_per_hour']:.2f}")
        st.caption(f"Early-warning coverage {lt['warning_coverage'] if lt['warning_coverage'] is None else round(lt['warning_coverage'], 3)} of forecastable episodes "
                   f"({lt['not_forecastable_episodes']} episode(s) begin before a full input sequence exists). {lt['definition']}")
        sm = ev["stage"]
        st.markdown("##### ATT&CK stage forecast")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Accuracy", f"{sm['accuracy']:.3f}"); s2.metric("Accuracy given attack", f"{sm.get('accuracy_given_attack', float('nan')):.3f}")
        s3.metric("Macro F1 (supported classes)", f"{sm['macro_f1']:.3f}"); s4.metric("Balanced accuracy", f"{sm['balanced_accuracy']:.3f}" if sm.get("balanced_accuracy") is not None else "n/a")
        st.warning(f"Stage metrics only score classes that actually occur in the hold-out: **{', '.join(sm['classes_with_support'])}**. "
                   f"Classes with zero support ({', '.join(sm['classes_without_support']) or 'none'}) are not scored - accuracy here is NOT nine-class accuracy.")
        cm = sm["confusion_matrix"]; labs = cm["labels"]
        fig = go.Figure(go.Heatmap(z=np.log1p(np.array(cm["matrix"])), x=labs, y=labs, text=cm["matrix"], texttemplate="%{text}", colorscale="Blues", showscale=False))
        fig.update_layout(**PLOT, height=420, title="Stage confusion matrix (rows = true, columns = predicted)", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        sf = ev["state_forecast"]
        st.caption(f"State forecasting (standardised active features): MAE {sf['mae']:.3f} vs persistence {sf['persistence_mae']:.3f} · RMSE {sf['rmse']:.3f} vs {sf['persistence_rmse']:.3f}")

# ----------------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**Pipeline** — *Network traffic → feature extraction → time-window network state → LSTM/attention world model → multi-step forecast → MITRE ATT&CK stage mapping → warning engine + explanation → dashboard.*

* **Network state**: each 10-second window becomes one 60-dimension vector (flow/packet counts, bytes, rates, durations, inter-arrival times, TCP flags,
  TCP window, TTL & retransmissions when packet data exist, port/protocol mix, scan indicators).
* **World model**: encoder → 2-layer LSTM → temporal self-attention → heads that predict the next *K* network states, the ATT&CK stage and the attack probability of each future window;
  the learned dynamics can also be unrolled autoregressively.
* **Warning engine**: severity levels (WATCH/WARNING/CRITICAL), persistence, hysteresis, cooldown and de-duplication; the offline lead-time evaluation replays the same engine.
* **Explainability**: temporal attention, gradient×input attribution and optional exact SHAP values.
* **Honest evaluation**: chronological per-day hold-out with gaps (no leakage), threshold calibrated on validation data, forecasting baselines, onset-only metrics, lead time with missed episodes reported.

**Labels**: public datasets label the attack itself, not its preparatory steps. Dataset labels are mapped to ATT&CK tactics through a transparent table in `configs/config.yaml`
(review it for your dataset version). The forecast therefore predicts *which labelled attack class is about to appear*, not a verified attacker kill chain.
""")
