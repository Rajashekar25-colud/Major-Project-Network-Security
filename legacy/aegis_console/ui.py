"""Design system: CSS theme + small HTML components."""
from __future__ import annotations

import html

import streamlit as st

SEV_COLOR = {"Normal": "#22c55e", "Elevated": "#eab308", "High": "#f97316", "Critical": "#ef4444"}
STATUS_COLOR = {"New": "#38bdf8", "Investigating": "#a78bfa", "Resolved": "#22c55e", "False positive": "#64748b"}
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=8, r=8, t=34, b=8), font=dict(family="Segoe UI, system-ui, sans-serif", color="#a8b5cc", size=12),
            hoverlabel=dict(bgcolor="#101827", bordercolor="#1f2c44"))
GRID = dict(gridcolor="rgba(148,163,184,0.10)", zerolinecolor="rgba(148,163,184,0.10)")

CSS = """
<style>
html, body, [class*="css"] { font-family: "Segoe UI", Inter, system-ui, -apple-system, sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"], .stAppDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.stApp { background: radial-gradient(1200px 500px at 85% -10%, rgba(45,212,191,0.07), transparent 60%), #0a0f1c; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1480px; }
section[data-testid="stSidebar"] { background:#0d1424; border-right:1px solid #1a2438; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
[data-testid="stSidebarNav"] a { border-radius:10px; padding:.45rem .7rem; margin:2px 0; color:#a8b5cc; }
[data-testid="stSidebarNav"] a:hover { background:#141d31; color:#e6edf7; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background:linear-gradient(90deg, rgba(45,212,191,.16), rgba(59,130,246,.10)); color:#e6edf7; font-weight:600; border:1px solid rgba(45,212,191,.25); }
h1,h2,h3,h4 { color:#e6edf7; letter-spacing:-0.01em; }
.ph { display:flex; justify-content:space-between; align-items:flex-end; margin:0 0 1.1rem 0; padding-bottom:.9rem; border-bottom:1px solid #1a2438; }
.ph h1 { font-size:1.65rem; margin:0; font-weight:700; }
.ph p { margin:.25rem 0 0; color:#7f8ea8; font-size:.9rem; }
.chip { display:inline-block; padding:3px 11px; border-radius:999px; font-size:.74rem; font-weight:700; letter-spacing:.02em; border:1px solid; }
.kpi { background:linear-gradient(180deg,#111a2d,#0f1727); border:1px solid #1c2940; border-radius:14px; padding:14px 16px; height:100%; }
.kpi .l { color:#7f8ea8; font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; font-weight:600; }
.kpi .v { font-size:1.75rem; font-weight:700; color:#f1f5f9; line-height:1.2; margin-top:2px; }
.kpi .s { color:#64748b; font-size:.78rem; margin-top:2px; }
.banner { border-radius:14px; padding:14px 18px; border:1px solid; display:flex; align-items:center; gap:14px; margin-bottom:14px; }
.banner .t { font-weight:700; font-size:1.05rem; } .banner .d { color:#b6c2d6; font-size:.9rem; margin-top:2px; }
.dot { width:11px; height:11px; border-radius:50%; flex:none; box-shadow:0 0 0 4px rgba(255,255,255,.06); }
.pulse { animation:pulse 1.6s infinite; } @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(239,68,68,.55);} 70%{box-shadow:0 0 0 12px rgba(239,68,68,0);} 100%{box-shadow:0 0 0 0 rgba(239,68,68,0);} }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:#1c2940 !important; background:#0f1727; border-radius:14px !important; }
.sec { color:#a8b5cc; font-size:.78rem; text-transform:uppercase; letter-spacing:.09em; font-weight:700; margin:.2rem 0 .5rem; }
.act { background:#0d1a2b; border-left:3px solid #2dd4bf; border-radius:8px; padding:8px 12px; margin:6px 0; color:#cbd5e1; font-size:.9rem; }
.why { color:#cbd5e1; font-size:.92rem; margin:4px 0; }
.stButton>button { border-radius:10px; font-weight:600; border:1px solid #243452; }
.stButton>button[kind="primary"] { background:linear-gradient(135deg,#14b8a6,#2563eb); border:none; color:white; }
div[data-testid="stTabs"] button { font-weight:600; }
[data-testid="stMetric"] { background:#0f1727; border:1px solid #1c2940; border-radius:12px; padding:10px 14px; }
.small { color:#7f8ea8; font-size:.82rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", right: str = "") -> None:
    st.markdown(f'<div class="ph"><div><h1>{html.escape(title)}</h1><p>{subtitle}</p></div><div>{right}</div></div>', unsafe_allow_html=True)


def chip(text: str, color: str) -> str:
    return f'<span class="chip" style="color:{color};border-color:{color}55;background:{color}18">{html.escape(str(text))}</span>'


def sev_chip(sev: str) -> str:
    return chip(sev, SEV_COLOR.get(sev, "#64748b"))


def kpi_html(label: str, value, sub: str = "", color: str | None = None) -> str:
    c = f' style="color:{color}"' if color else ""
    return f'<div class="kpi"><div class="l">{html.escape(label)}</div><div class="v"{c}>{value}</div><div class="s">{sub}</div></div>'


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(kpi_html(*it), unsafe_allow_html=True)


def banner(title: str, detail: str, color: str, pulse: bool = False) -> None:
    st.markdown(f'<div class="banner" style="background:{color}14;border-color:{color}66"><div class="dot{" pulse" if pulse else ""}" '
                f'style="background:{color}"></div><div><div class="t" style="color:{color}">{title}</div><div class="d">{detail}</div></div></div>',
                unsafe_allow_html=True)


def section(text: str) -> None:
    st.markdown(f'<div class="sec">{html.escape(text)}</div>', unsafe_allow_html=True)


def fmt_secs(s) -> str:
    if s is None:
        return "n/a"
    s = float(s)
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.1f} min"
