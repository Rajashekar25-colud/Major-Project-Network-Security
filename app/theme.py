"""Look & feel: restrained SOC-console theme (warm off-white, charcoal text) with green / amber / red state semantics,
plus small HTML helpers shared by every page. Colour is used only to communicate state."""
from __future__ import annotations

import html

import streamlit as st

INK, MUTED, LINE, BG, SIDE, ACCENT = "#1f1e1d", "#6b6a68", "#e6e3da", "#faf9f5", "#f3f1ea", "#c96442"
STATE_COLOR = {"NORMAL": "#2f855a", "WATCH": "#b7791f", "WARNING": "#d9692b", "HIGH RISK": "#c0392b"}
STAGE_COLOR = {"Benign": "#9a9892", "None": "#9a9892", "Reconnaissance": "#4a7fb5", "Initial Access": "#7a5fb0", "Credential Access": "#b0557a",
               "Discovery": "#3d8f8a", "Lateral Movement": "#c27a2c", "Command and Control": "#b5483f", "Exfiltration": "#8f2f3a", "Impact": "#a33b2e"}

CSS = f"""
<style>
:root {{ --ink:{INK}; --muted:{MUTED}; --line:{LINE}; --bg:{BG}; --side:{SIDE}; --accent:{ACCENT}; --r:10px; --shadow:0 1px 2px rgba(31,30,29,.05); }}
html, body, [class*="css"], .stApp {{ font-family:-apple-system,"Segoe UI",Inter,system-ui,sans-serif; color:var(--ink); }}
.stApp {{ background:var(--bg); }}
header[data-testid="stHeader"] {{ background:transparent; height:2.2rem; }}
#MainMenu, footer, .stAppDeployButton {{ visibility:hidden; }}
.block-container {{ padding-top:1.4rem; padding-bottom:3rem; max-width:1280px; padding-left:2rem; padding-right:2rem; }}
h1,h2,h3,h4 {{ font-family:-apple-system,"Segoe UI",Inter,system-ui,sans-serif; font-weight:650; letter-spacing:-0.01em; color:var(--ink); }}
h1 {{ font-size:1.75rem; margin-bottom:.1rem; padding-top:.2rem; }} h3 {{ font-size:1.12rem; margin-top:.3rem; }} h4 {{ font-size:1rem; margin-top:.2rem; }}
/* sidebar */
section[data-testid="stSidebar"] {{ background:var(--side); border-right:1px solid var(--line); }}
section[data-testid="stSidebar"][aria-expanded="true"] {{ width:250px; min-width:250px; }}
section[data-testid="stSidebar"] .block-container {{ padding-top:1rem; }}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap:.15rem; }}
.brand {{ display:flex; align-items:center; gap:.6rem; margin:.1rem 0 .8rem .1rem; }}
.brand .mark {{ width:30px; height:30px; border-radius:8px; background:var(--ink); color:#fff; display:flex; align-items:center; justify-content:center; font-size:1rem; }}
.brand .t {{ font-weight:700; font-size:.82rem; letter-spacing:.05em; line-height:1.15; }} .brand .t span {{ display:block; font-weight:500; color:var(--muted); font-size:.68rem; letter-spacing:.04em; }}
.userchip {{ display:flex; align-items:center; gap:.55rem; border:1px solid var(--line); background:#fff; border-radius:var(--r); padding:.45rem .6rem; margin:.1rem 0 .55rem; }}
.userchip .av {{ width:26px; height:26px; border-radius:50%; background:#e6e2d6; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:.78rem; }}
.userchip .nm {{ font-size:.82rem; font-weight:600; line-height:1.15; }} .userchip .rl {{ font-size:.66rem; color:var(--muted); letter-spacing:.06em; text-transform:uppercase; }}
.ctx {{ border:1px dashed #d8d4c7; border-radius:var(--r); padding:.4rem .6rem; margin:.5rem 0 .1rem; font-size:.74rem; color:var(--muted); line-height:1.35; }}
.ctx b {{ color:var(--ink); font-size:.78rem; word-break:break-all; }}
.nav-sec {{ font-size:.64rem; letter-spacing:.12em; color:var(--muted); font-weight:700; margin:.9rem 0 .15rem .35rem; text-transform:uppercase; }}
.nav-sep {{ height:1px; background:var(--line); margin:.8rem 0 .5rem; }}
section[data-testid="stSidebar"] .stButton > button {{ justify-content:flex-start; background:transparent; border:none; color:var(--ink);
    font-weight:500; font-size:.9rem; padding:.3rem .6rem; min-height:2rem; border-radius:8px; box-shadow:none; }}
section[data-testid="stSidebar"] .stButton > button:hover {{ background:#e9e6dc; }}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{ background:#e6e2d6; font-weight:700; color:var(--ink); box-shadow:inset 3px 0 0 var(--accent); }}
section[data-testid="stSidebar"] .st-key-nav_new button {{ background:var(--accent); color:#fff; font-weight:600; border-radius:var(--r); padding:.45rem .7rem; justify-content:center; box-shadow:none; }}
section[data-testid="stSidebar"] .st-key-nav_new button:hover {{ background:#b5573a; color:#fff; }}
section[data-testid="stSidebar"] .st-key-logout button {{ border:1px solid var(--line); background:#fff; justify-content:center; font-size:.82rem; }}
.sys {{ font-size:.74rem; color:var(--muted); line-height:1.7; margin:.2rem 0 0 .25rem; }} .sys b {{ color:var(--ink); font-weight:600; }}
.dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:.45rem; vertical-align:middle; }}
.dot.ok {{ background:#2f855a; }} .dot.warn {{ background:#b7791f; }} .dot.bad {{ background:#c0392b; }}
/* page header */
.pagehead {{ display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin:0 0 .9rem; padding-bottom:.7rem; border-bottom:1px solid var(--line); }}
.pagehead .ttl {{ font-size:1.7rem; font-weight:700; letter-spacing:-.015em; line-height:1.15; }}
.pagehead .sub {{ color:var(--muted); font-size:.92rem; margin-top:.2rem; }}
.pagehead .rt {{ display:flex; gap:.4rem; flex-wrap:wrap; justify-content:flex-end; }}
.sub {{ color:var(--muted); font-size:.92rem; margin:-.1rem 0 1rem; }}
.sec {{ font-size:.7rem; letter-spacing:.11em; text-transform:uppercase; color:var(--muted); font-weight:700; margin:1.3rem 0 .5rem; }}
/* cards */
.card {{ background:#fff; border:1px solid var(--line); border-radius:var(--r); padding:14px 16px; height:100%; box-shadow:var(--shadow); }}
.card.accent {{ border-top:3px solid var(--c, var(--line)); }}
.lbl {{ font-size:.66rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); font-weight:700; }}
.val {{ font-size:1.65rem; font-weight:650; line-height:1.15; margin-top:.2rem; }}
.hint {{ color:var(--muted); font-size:.8rem; margin-top:.25rem; line-height:1.35; }}
.chip {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.7rem; font-weight:700; letter-spacing:.05em; border:1px solid; white-space:nowrap; }}
.chip.neutral {{ color:var(--muted); border-color:var(--line); background:#fff; }}
.banner {{ border-radius:var(--r); padding:12px 16px; border:1px solid; margin:.2rem 0 1rem; }}
.banner .t {{ font-weight:700; }} .banner .d {{ color:#4a4946; font-size:.9rem; margin-top:.15rem; }}
.chk {{ padding:.25rem 0; font-size:.92rem; }} .chk .ok {{ color:#2f855a; font-weight:700; }} .chk .warn {{ color:#b7791f; font-weight:700; }}
.chk .fail {{ color:#c0392b; font-weight:700; }} .chk small {{ color:var(--muted); }}
.pipe {{ padding:.2rem 0; font-size:.95rem; }} .pipe .done {{ color:#2f855a; }} .pipe .now {{ color:var(--accent); }} .pipe .todo {{ color:#b8b5aa; }}
.flow {{ display:flex; align-items:center; justify-content:center; gap:.7rem; flex-wrap:wrap; margin:.2rem 0 .8rem; font-size:.82rem; color:var(--muted); }}
.flow .box {{ border:1px solid var(--line); background:#fff; border-radius:8px; padding:.35rem .8rem; color:var(--ink); font-weight:600; }}
.flow .model {{ background:#f6e9e3; border-color:#e9c9bb; }}
.kpair {{ display:flex; justify-content:space-between; gap:1rem; padding:.28rem 0; border-bottom:1px solid #f0ede4; font-size:.9rem; }}
.kpair span:first-child {{ color:var(--muted); }} .kpair:last-child {{ border-bottom:none; }}
/* tables */
table.dt {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:var(--r); overflow:hidden; font-size:.86rem; box-shadow:var(--shadow); }}
table.dt th {{ text-align:left; font-size:.64rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); background:#f7f5ee; padding:.55rem .75rem; border-bottom:1px solid var(--line); }}
table.dt td {{ padding:.5rem .75rem; border-bottom:1px solid #f0ede4; vertical-align:middle; }} table.dt tr:last-child td {{ border-bottom:none; }}
table.dt tr:hover td {{ background:#fcfbf7; }} .tblwrap {{ overflow-x:auto; }}
.mono {{ font-family:ui-monospace,"Cascadia Mono",Consolas,monospace; font-size:.82rem; }}
/* stepper */
.steps {{ display:flex; gap:.5rem; margin:.2rem 0 1.1rem; flex-wrap:wrap; }}
.steps .s {{ flex:1; min-width:150px; border:1px solid var(--line); background:#fff; border-radius:var(--r); padding:.5rem .8rem; font-size:.8rem; color:var(--muted); }}
.steps .s b {{ display:block; color:var(--ink); font-size:.86rem; }} .steps .s .n {{ font-size:.62rem; letter-spacing:.12em; font-weight:700; }}
.steps .s.done {{ border-color:#bfdcc9; background:#f3faf5; }} .steps .s.done .n {{ color:#2f855a; }}
.steps .s.now {{ border-color:var(--accent); box-shadow:inset 0 -3px 0 var(--accent); }} .steps .s.now .n {{ color:var(--accent); }}
/* forecast horizon tiles */
.hz {{ display:flex; gap:.5rem; margin:.2rem 0 .4rem; }} .hz .t {{ flex:1; background:#fff; border:1px solid var(--line); border-top:3px solid var(--c); border-radius:var(--r); padding:.5rem .6rem; text-align:center; box-shadow:var(--shadow); }}
.hz .t .h {{ font-size:.7rem; color:var(--muted); font-weight:700; letter-spacing:.08em; }} .hz .t .p {{ font-size:1.25rem; font-weight:650; color:var(--c); }} .hz .t .g {{ font-size:.68rem; color:var(--muted); }}
/* ATT&CK ladder */
.ladder {{ display:flex; gap:.35rem; flex-wrap:wrap; margin:.3rem 0 .8rem; }}
.ladder .st {{ flex:1; min-width:120px; border:1px solid var(--line); background:#fff; border-radius:8px; padding:.5rem .6rem; font-size:.78rem; color:var(--muted); }}
.ladder .st b {{ display:block; color:var(--ink); font-size:.82rem; }} .ladder .st .id {{ font-size:.66rem; letter-spacing:.06em; }}
.ladder .st.on {{ border:2px solid var(--c); background:color-mix(in srgb, var(--c) 8%, #fff); color:var(--ink); }}
.bigstage {{ background:#fff; border:1px solid var(--line); border-left:5px solid var(--c); border-radius:var(--r); padding:1rem 1.2rem; box-shadow:var(--shadow); }}
.bigstage .nm {{ font-size:1.7rem; font-weight:700; color:var(--c); line-height:1.2; }}
/* login */
.loginbrand {{ background:#1f1e1d; color:#f3f1ea; border-radius:14px; padding:2.2rem 2rem; min-height:440px; }}
.loginbrand h2 {{ color:#fff; font-size:1.55rem; margin:.8rem 0 .2rem; line-height:1.2; }} .loginbrand .tag {{ color:#d9a58f; font-size:.82rem; letter-spacing:.08em; text-transform:uppercase; font-weight:600; }}
.loginbrand p {{ color:#c9c6bb; font-size:.9rem; line-height:1.5; margin:.9rem 0 1.4rem; }}
.loginbrand .sysrow {{ font-size:.82rem; color:#e6e3da; padding:.18rem 0; }}
.loginbrand .pipeline {{ font-size:.72rem; color:#9a9892; letter-spacing:.03em; margin-top:1.4rem; line-height:1.6; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{ background:var(--accent); border-color:var(--accent); }}
.stButton > button, .stFormSubmitButton > button {{ border-radius:8px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:var(--r); border-color:var(--line); background:#fff; }}
div[data-testid="stDataFrame"] {{ border:1px solid var(--line); border-radius:var(--r); }}
@media (max-width:1100px) {{ .block-container {{ padding-left:1rem; padding-right:1rem; }} .hz {{ flex-wrap:wrap; }} .hz .t {{ min-width:30%; }} .pagehead {{ flex-direction:column; align-items:flex-start; }} }}
</style>
"""

HIDE_SIDEBAR = """<style>section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display:none !important; }
.block-container { max-width:1020px; padding-top:3rem; }</style>"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def chip(text: str, color: str | None = None) -> str:
    if color is None:
        return f'<span class="chip neutral">{esc(text)}</span>'
    return f'<span class="chip" style="color:{color};border-color:{color}55;background:{color}14">{esc(text)}</span>'


def state_chip(state: str) -> str:
    return chip(state, STATE_COLOR.get(state, MUTED))


def stat(label: str, value: str, hint: str = "", color: str | None = None) -> str:
    c = f' style="color:{color}"' if color else ""
    top = f' accent" style="--c:{color}' if color else ""
    h = f'<div class="hint">{hint}</div>' if hint else ""
    return f'<div class="card{top}"><div class="lbl">{esc(label)}</div><div class="val"{c}>{value}</div>{h}</div>'


def banner(title: str, detail: str, color: str) -> None:
    st.markdown(f'<div class="banner" style="background:{color}12;border-color:{color}66"><div class="t" style="color:{color}">{title}</div><div class="d">{detail}</div></div>', unsafe_allow_html=True)


def header(title: str, subtitle: str = "", chips: list[str] | None = None) -> None:
    """Page title block. ``chips`` are pre-built HTML chips shown at the right (context only, never decoration)."""
    rt = f'<div class="rt">{"".join(chips)}</div>' if chips else ""
    sub = f'<div class="sub" style="margin:.2rem 0 0">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="pagehead"><div><div class="ttl">{esc(title)}</div>{sub}</div>{rt}</div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="sec">{esc(title)}</div>', unsafe_allow_html=True)


def kv(rows: list[tuple[str, str]]) -> None:
    st.markdown('<div class="card">' + "".join(f'<div class="kpair"><span>{esc(a)}</span><span>{b}</span></div>' for a, b in rows) + "</div>", unsafe_allow_html=True)


def table(columns: list[str], rows: list[list[str]], mono: tuple[int, ...] = ()) -> None:
    """HTML table. Cells are trusted HTML: callers must pass T.esc(...) for any free text."""
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f'<td{" class=mono" if i in mono else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    st.markdown(f'<div class="tblwrap"><table class="dt"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


def steps(active: int, labels: list[tuple[str, str]]) -> None:
    """Workflow stepper: steps before ``active`` are done, ``active`` is current (1-based)."""
    out = []
    for i, (t, d) in enumerate(labels, 1):
        cls = "done" if i < active else "now" if i == active else ""
        out.append(f'<div class="s {cls}"><span class="n">STEP {i}{" ✓" if i < active else ""}</span><b>{esc(t)}</b>{esc(d)}</div>')
    st.markdown(f'<div class="steps">{"".join(out)}</div>', unsafe_allow_html=True)


def status_line(ok: bool | None, text: str, detail: str = "") -> str:
    cls = "ok" if ok else "warn" if ok is None else "bad"
    d = f' <span style="color:{MUTED}">· {esc(detail)}</span>' if detail else ""
    return f'<div class="chk"><span class="dot {cls}"></span>{esc(text)}{d}</div>'


def pct(x) -> str:
    return "N/A" if x is None or (isinstance(x, float) and x != x) else f"{x:.0%}"


def f3(x) -> str:
    return "N/A" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"


def num(x, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "N/A"
    x = float(x)
    for lim, suf in ((1e9, "B"), (1e6, "M"), (1e4, "k")):
        if abs(x) >= lim:
            return f"{x / lim:.1f}{suf}"
    return f"{x:,.{digits}f}"


def secs(s) -> str:
    if s is None or (isinstance(s, float) and s != s):
        return "N/A"
    s = float(s)
    return f"{s:.0f} s" if s < 120 else (f"{s / 60:.1f} min" if s < 7200 else f"{s / 3600:.1f} h")
