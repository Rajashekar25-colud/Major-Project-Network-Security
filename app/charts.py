"""Plotly figures used by the console."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ui import GRID, PLOT, SEV_COLOR

STAGE_COLORS = ["#38bdf8", "#a78bfa", "#f472b6", "#2dd4bf", "#fb923c", "#f43f5e", "#e11d48", "#ef4444"]
TACTIC_COLOR = {"Reconnaissance": "#38bdf8", "Initial Access": "#a78bfa", "Credential Access": "#f472b6", "Discovery": "#2dd4bf",
                "Lateral Movement": "#fb923c", "Command and Control": "#f43f5e", "Exfiltration": "#e11d48", "Impact": "#ef4444", "Benign": "#334155"}


def _gapped(ts: pd.Series, y: np.ndarray, seg: np.ndarray, max_points: int = 4000):
    step = max(len(y) // max_points, 1)
    ts, y, seg = ts.iloc[::step], y[::step], seg[::step]
    x, v = [], []
    prev = None
    for t, a, s in zip(ts, y, seg):
        if prev is not None and s != prev:
            x.append(t); v.append(None)
        x.append(t); v.append(a); prev = s
    return x, v


def risk_timeline(ts: pd.Series, risk: np.ndarray, seg: np.ndarray, thr: float, incidents: pd.DataFrame | None = None,
                  truth: np.ndarray | None = None, height: int = 340, title: str = "") -> go.Figure:
    fig = go.Figure()
    if truth is not None and truth.any():
        t = truth.astype(int)
        s_ = np.flatnonzero(np.diff(np.concatenate([[0], t, [0]])) == 1); e_ = np.flatnonzero(np.diff(np.concatenate([[0], t, [0]])) == -1) - 1
        for a, b in list(zip(s_, e_))[:250]:
            fig.add_vrect(x0=ts.iloc[a], x1=ts.iloc[min(b, len(ts) - 1)], fillcolor="rgba(239,68,68,0.13)", line_width=0, layer="below")
    x, v = _gapped(ts, risk, seg)
    fig.add_trace(go.Scatter(x=x, y=v, mode="lines", name="Forecast risk", line=dict(color="#2dd4bf", width=2), connectgaps=False,
                             fill="tozeroy", fillcolor="rgba(45,212,191,0.10)", hovertemplate="%{x|%d %b %H:%M:%S}<br>risk %{y:.0%}<extra></extra>"))
    fig.add_hline(y=thr, line=dict(color="#f97316", dash="dash", width=1.2), annotation_text=f"alert threshold {thr:.0%}",
                  annotation_position="top left", annotation_font_color="#f97316")
    if incidents is not None and len(incidents):
        d = incidents.copy()
        fig.add_trace(go.Scatter(x=d["opened_time"], y=np.minimum(d["peak_risk"], 1.0), mode="markers", name="Incident",
                                 marker=dict(size=9, color=[SEV_COLOR.get(s, "#ef4444") for s in d["severity"]], line=dict(color="#0a0f1c", width=1.5)),
                                 text=d["stage"], hovertemplate="%{text}<br>%{x|%d %b %H:%M:%S}<extra>Incident</extra>"))
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%", **GRID)
    fig.update_xaxes(**GRID)
    fig.update_layout(**PLOT, height=height, title=dict(text=title, font=dict(size=13)), showlegend=False)
    return fig


def horizon_bars(ft: pd.DataFrame, thr: float) -> go.Figure:
    cols = [TACTIC_COLOR.get(s, "#2dd4bf") if p >= thr else "#2b3a55" for s, p in zip(ft["most_likely_attack_stage"], ft["attack_probability"])]
    fig = go.Figure(go.Bar(x=[f"+{int(s)}s" for s in ft["seconds_ahead"]], y=ft["attack_probability"], marker_color=cols,
                           text=[f"{p:.0%}" for p in ft["attack_probability"]], textposition="outside", cliponaxis=False,
                           customdata=ft["most_likely_attack_stage"], hovertemplate="%{x}: %{y:.0%}<br>likely tactic: %{customdata}<extra></extra>"))
    fig.add_hline(y=thr, line=dict(color="#f97316", dash="dash", width=1))
    fig.update_yaxes(range=[0, 1.18], tickformat=".0%", **GRID); fig.update_xaxes(**GRID)
    fig.update_layout(**PLOT, height=270, title=dict(text="Attack probability - next minute", font=dict(size=13)))
    return fig


def tactic_probs(names: list[str], probs: np.ndarray) -> go.Figure:
    pairs = [(n, p) for n, p in zip(names, probs) if n != "Benign"]
    pairs.sort(key=lambda x: x[1])
    fig = go.Figure(go.Bar(x=[p for _, p in pairs], y=[n for n, _ in pairs], orientation="h",
                           marker_color=[TACTIC_COLOR.get(n, "#2dd4bf") for n, _ in pairs],
                           text=[f"{p:.0%}" for _, p in pairs], textposition="outside", cliponaxis=False))
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%", **GRID); fig.update_yaxes(**GRID)
    fig.update_layout(**PLOT, height=270, title=dict(text="Most likely attack tactic", font=dict(size=13)))
    return fig


def gauge(risk: float, thr: float) -> go.Figure:
    c = "#22c55e" if risk < 0.7 * thr else "#eab308" if risk < thr else "#f97316" if risk < thr + 0.5 * (1 - thr) else "#ef4444"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=risk * 100, number=dict(suffix="%", font=dict(size=40, color="#f1f5f9")),
                                 gauge=dict(axis=dict(range=[0, 100], tickcolor="#475569"), bar=dict(color=c, thickness=0.28), bgcolor="#111a2d",
                                            borderwidth=0, threshold=dict(line=dict(color="#f97316", width=3), thickness=0.85, value=thr * 100))))
    fig.update_layout(**{**PLOT, "margin": dict(l=20, r=20, t=30, b=0)}, height=210)
    return fig


def tactic_donut(counts: dict[str, int]) -> go.Figure:
    fig = go.Figure(go.Pie(labels=list(counts), values=list(counts.values()), hole=0.62, sort=True,
                           marker=dict(colors=[TACTIC_COLOR.get(k, "#64748b") for k in counts], line=dict(color="#0a0f1c", width=2)),
                           textinfo="percent", hovertemplate="%{label}: %{value} incidents<extra></extra>"))
    fig.update_layout(**PLOT, height=340, title=dict(text="Incidents by predicted tactic", font=dict(size=13)),
                      legend=dict(orientation="h", y=-0.05, font=dict(size=11)))
    return fig


def drivers_bar(features: list[dict], labels: dict[str, str]) -> go.Figure:
    f = features[::-1]
    fig = go.Figure(go.Bar(x=[x.get("attribution", x.get("shap_value", 0)) for x in f], y=[labels[x["feature"]] for x in f], orientation="h",
                           marker_color=["#f97316" if x.get("attribution", x.get("shap_value", 0)) > 0 else "#38bdf8" for x in f]))
    fig.update_xaxes(**GRID); fig.update_yaxes(**GRID)
    fig.update_layout(**PLOT, height=max(260, 30 * len(f) + 60), title=dict(text="What drove this forecast (orange = raises risk)", font=dict(size=13)))
    return fig
