"""Plotly figures built only from real analysis arrays passed in by the caller."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from theme import ACCENT, INK, MUTED, STAGE_COLOR, STATE_COLOR

GRID = "rgba(120,116,104,0.14)"
BASE = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="-apple-system, Segoe UI, Inter, sans-serif", color=INK, size=12), margin=dict(l=8, r=8, t=30, b=8),
            hoverlabel=dict(bgcolor="#fff", bordercolor="#e6e3da"))


def _layout(fig, height: int, title: str = "", **kw):
    fig.update_layout(**BASE, height=height, title=dict(text=title, font=dict(size=13, color=MUTED), x=0.0), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False); fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _ds(df: pd.DataFrame, n: int = 2500) -> pd.DataFrame:
    return df if len(df) <= n else df.iloc[:: int(np.ceil(len(df) / n))]


def _vmark(fig, x) -> None:
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="y domain", line=dict(color=ACCENT, width=1.5, dash="dot"))


def _yes(a) -> bool:
    return a is not None and a == a and bool(a)


def forecast_timeline(obs: pd.DataFrame, frame: pd.DataFrame, thr: float, sel_step: int, labelled: bool):
    """OBSERVED windows -> NOW -> FORECAST steps on one connected axis, with observed flow counts underneath."""
    L, H = len(obs), len(frame)
    ocat = ["NOW" if o == 0 else f"W{o}" for o in obs["offset"]]; fcat = list(frame["label"]); cats = ocat + fcat
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.05)
    fig.add_vrect(x0=L - 0.5, x1=L + H - 0.5, fillcolor="rgba(201,100,66,0.07)", line_width=0, row="all", col=1)
    fig.add_trace(go.Scatter(x=ocat, y=obs["prior_forecast"], mode="lines+markers", name="Earlier +1 forecast for observed windows",
                             line=dict(color="#a8a59b", width=1.6), marker=dict(size=5, color="#a8a59b"), hovertemplate="%{x}: %{y:.0%}<extra>earlier forecast</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fcat, y=frame["risk"], mode="lines+markers+text", name="Forecast issued now", line=dict(color=ACCENT, width=2.4),
                             marker=dict(size=[15 if s == sel_step else 9 for s in frame["step"]], color=[STATE_COLOR[s] for s in frame["state"]], line=dict(color="#fff", width=1.5)),
                             text=[f"{p:.0%}" for p in frame["risk"]], textposition="top center", customdata=np.stack([frame["state"], frame["stage"]], 1),
                             hovertemplate="%{x}: %{y:.1%}<br>%{customdata[0]} - stage: %{customdata[1]}<extra>forecast</extra>"), row=1, col=1)
    fig.add_hline(y=thr, line=dict(color="#d9692b", dash="dash", width=1), row=1, col=1, annotation_text=f"alert threshold {thr:.0%}",
                  annotation_position="bottom left", annotation_font=dict(size=10, color="#d9692b"))
    if labelled:
        ax = [c for c, a in zip(ocat, obs["actual_attack"]) if _yes(a)] + [c for c, a in zip(fcat, frame["actual_attack"]) if _yes(a)]
        if ax:
            fig.add_trace(go.Scatter(x=ax, y=[1.0] * len(ax), mode="markers", name="Actual attack (dataset label)",
                                     marker=dict(symbol="diamond", size=11, color="#c0392b", line=dict(color="#fff", width=1))), row=1, col=1)
    fig.add_trace(go.Bar(x=ocat, y=obs["flows"], name="Observed flows per window", marker_color="#bdb9ac", hovertemplate="%{x}: %{y:,.0f} flows<extra>observed</extra>"), row=2, col=1)
    fig.update_xaxes(categoryorder="array", categoryarray=cats)
    fig.update_yaxes(range=[0, 1.14], tickformat=".0%", title_text="attack probability", row=1, col=1); fig.update_yaxes(title_text="flows", row=2, col=1)
    fig.add_annotation(x=(L - 1) / 2, xref="x", y=1.0, yref="y domain", text="OBSERVED", showarrow=False, font=dict(size=11, color=MUTED), yshift=14)
    fig.add_annotation(x=L + (H - 1) / 2, xref="x", y=1.0, yref="y domain", text="FORECAST", showarrow=False, font=dict(size=11, color=ACCENT), yshift=14)
    _layout(fig, 470, legend=dict(orientation="h", y=-0.12, x=0, font=dict(size=11)))
    fig.update_layout(margin=dict(l=8, r=8, t=34, b=8))
    return fig


def stage_probs(names, probs, title: str = ""):
    pairs = sorted(((n, float(p)) for n, p in zip(names, probs) if n != "Benign"), key=lambda x: x[1])
    fig = go.Figure(go.Bar(x=[p for _, p in pairs], y=[n for n, _ in pairs], orientation="h", marker_color=[STAGE_COLOR.get(n, ACCENT) for n, _ in pairs],
                           text=[f"{p:.0%}" for _, p in pairs], textposition="outside", cliponaxis=False))
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%")
    return _layout(fig, max(200, 38 * len(pairs) + 70), title)


def traffic_charts(states: pd.DataFrame, now) -> dict:
    d = _ds(states); t = d["timestamp"]; out = {}
    f = go.Figure(go.Scatter(x=t, y=d["flow_count"], mode="lines", line=dict(color="#4a7fb5", width=1.4), fill="tozeroy", fillcolor="rgba(74,127,181,.10)",
                             hovertemplate="%{x|%H:%M:%S}: %{y:,.0f} flows<extra></extra>"))
    _vmark(f, now); out["flows"] = _layout(f, 230, "Flow activity (flows per window)")
    p = make_subplots(specs=[[{"secondary_y": True}]])
    p.add_trace(go.Scatter(x=t, y=d["packets_total"], name="packets", mode="lines", line=dict(color="#7a5fb0", width=1.3)), secondary_y=False)
    p.add_trace(go.Scatter(x=t, y=d["bytes_total"], name="bytes", mode="lines", line=dict(color="#3d8f8a", width=1.3)), secondary_y=True)
    p.update_yaxes(title_text="packets", secondary_y=False); p.update_yaxes(title_text="bytes", secondary_y=True, showgrid=False)
    _vmark(p, now); out["volume"] = _layout(p, 230, "Packet and byte activity per window", legend=dict(orientation="h", y=1.15, x=0))
    pc = [c for c in states.columns if c.startswith("dst_port_") and c != "dst_port_other"]
    share = states[pc].mean().sort_values(ascending=False); po = go.Figure()
    for c in [c for c in share.index[:5] if share[c] > 0]:
        po.add_trace(go.Scatter(x=t, y=d[c], name=f"port {c.replace('dst_port_', '')}", mode="lines", stackgroup="p", line=dict(width=0.6)))
    if "dst_port_other" in d:
        po.add_trace(go.Scatter(x=t, y=d["dst_port_other"], name="other ports", mode="lines", stackgroup="p", line=dict(width=0.6, color="#c9c6bb")))
    po.update_yaxes(tickformat=".0%", range=[0, 1.0])
    _vmark(po, now); out["ports"] = _layout(po, 240, "Destination-port activity (share of flows)", legend=dict(orientation="h", y=1.18, x=0))
    tc = go.Figure()
    for c, name, col in (("syn_ratio", "new-connection (SYN) share", "#c27a2c"), ("failed_flow_ratio", "unanswered-flow share", "#b5483f"), ("scan_score", "scan pattern", "#4a7fb5")):
        if c in d:
            tc.add_trace(go.Scatter(x=t, y=d[c], name=name, mode="lines", line=dict(color=col, width=1.3)))
    tc.update_yaxes(range=[0, 1.02])
    _vmark(tc, now); out["pattern"] = _layout(tc, 240, "Traffic-pattern indicators", legend=dict(orientation="h", y=1.18, x=0))
    return out


def contributions(features: list[dict], key: str, title: str):
    from src.product.knowledge import feature_label
    f = features[::-1]; v = [x[key] for x in f]
    fig = go.Figure(go.Bar(x=v, y=[feature_label(x["feature"]) for x in f], orientation="h", marker_color=["#c0592b" if a > 0 else "#4a7fb5" for a in v],
                           hovertemplate="%{y}<br>%{x:.4f}<extra></extra>"))
    return _layout(fig, max(260, 32 * len(f) + 70), title)


def temporal(attention, attribution, attention_available: bool):
    L = len(attribution); labels = ["NOW" if i == L - 1 else f"W-{L - 1 - i}" for i in range(L)]
    fig = go.Figure()
    if attention_available:
        fig.add_bar(x=labels, y=attention, name="attention weight", marker_color="#c9a99a")
    fig.add_scatter(x=labels, y=attribution, name="attribution share", mode="lines+markers", line=dict(color=ACCENT))
    fig.update_xaxes(categoryorder="array", categoryarray=labels)
    return _layout(fig, 300, "Which past windows influenced this forecast", legend=dict(orientation="h", y=1.15, x=0))


def per_horizon(ph: pd.DataFrame):
    fig = go.Figure()
    for m, c in (("precision", "#4a7fb5"), ("recall", "#c0592b"), ("f1", "#2f855a")):
        fig.add_scatter(x=[f"+{int(s)}s" for s in ph["seconds_ahead"]], y=ph[m], name=m, mode="lines+markers", line=dict(color=c))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    return _layout(fig, 280, "Held-out quality by how far ahead the forecast looks", legend=dict(orientation="h", y=1.15, x=0))


# ---------------------------------------------------------------------------------------------- SOC views (all inputs are real analysis data)
def horizon_bars(frame: pd.DataFrame, thr: float, sel_step: int):
    """Forecast probability per horizon step with the alert threshold as a horizontal line; bars are coloured by risk state."""
    fig = go.Figure(go.Bar(x=frame["label"], y=frame["risk"], marker_color=[STATE_COLOR[s] for s in frame["state"]],
                           marker_line=dict(color=[INK if s == sel_step else "rgba(0,0,0,0)" for s in frame["step"]], width=2),
                           text=[f"{p:.0%}" for p in frame["risk"]], textposition="outside", cliponaxis=False,
                           customdata=np.stack([frame["state"], frame["stage"]], 1),
                           hovertemplate="%{x}: %{y:.1%}<br>%{customdata[0]} · stage: %{customdata[1]}<extra></extra>"))
    fig.add_hline(y=thr, line=dict(color=INK, dash="dash", width=1.2), annotation_text=f"Alert threshold {thr:.0%}", annotation_position="top left",
                  annotation_font=dict(size=11, color=INK))
    fig.update_yaxes(range=[0, 1.15], tickformat=".0%", title_text="Forecast probability"); fig.update_xaxes(title_text="Forecast horizon")
    return _layout(fig, 300, "Forecast probability by horizon", showlegend=False)


def _hbar(names, values, colors, title: str, fmt: str = ".0%", height: int | None = None):
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors, text=[format(v, fmt) if fmt != "d" else f"{int(v):,}" for v in values],
                           textposition="outside", cliponaxis=False))
    fig.update_yaxes(autorange="reversed")
    if fmt == ".0%":
        fig.update_xaxes(range=[0, max(1.0, max(values, default=0) * 1.1)], tickformat=".0%")
    return _layout(fig, height or max(190, 34 * len(names) + 64), title)


def traffic_mix(states: pd.DataFrame, labelled: bool) -> dict:
    """Protocol mix and destination-port mix (flow-weighted over the whole file) and, if labelled, the attack / benign split."""
    out = {}
    w = states["flow_count"].astype(float); tot = float(w.sum())
    if tot > 0:
        pc = [c for c in ("protocol_tcp", "protocol_udp", "protocol_icmp", "protocol_other") if c in states.columns]
        if pc:
            sh = {c.replace("protocol_", "").upper(): float((states[c] * w).sum() / tot) for c in pc}
            sh = {k: v for k, v in sh.items() if v > 0}
            if sh:
                out["protocol"] = _hbar(list(sh), list(sh.values()), "#4a7fb5", "Protocol distribution (share of flows)", height=max(170, 40 * len(sh) + 60))
        dc = [c for c in states.columns if c.startswith("dst_port_")]
        if dc:
            sh = {c.replace("dst_port_", "port "): float((states[c] * w).sum() / tot) for c in dc}
            sh = dict(sorted(((k, v) for k, v in sh.items() if v > 0), key=lambda kv: -kv[1])[:8])
            if sh:
                out["ports"] = _hbar(list(sh), list(sh.values()), "#7a5fb0", "Destination-port distribution (share of flows)")
    if labelled and "is_attack" in states.columns:
        att = states[states["is_attack"].astype(bool)]
        names = ["Benign windows"] + [str(s) for s in att["stage"].value_counts().index]
        vals = [int((~states["is_attack"].astype(bool)).sum())] + [int(v) for v in att["stage"].value_counts().values]
        cols = [STAGE_COLOR["Benign"]] + [STAGE_COLOR.get(n, ACCENT) for n in names[1:]]
        out["labels"] = _hbar(names, vals, cols, "Ground-truth labels (windows)", fmt="d")
    return out
