"""Reusable UI components: replay bar, forecast-step selector, alert card, pipeline/check lists, fragment wrapper."""
from __future__ import annotations

import streamlit as st

import appstate as A
import theme as T
from src.analysis import workflow as W


def _set_k(v: int) -> None:
    st.session_state["k"] = int(v)


def _slider_changed() -> None:
    st.session_state["k"] = int(st.session_state["k_slider"]) - 1


def _speed_changed() -> None:
    st.session_state["speed"] = int(st.session_state["speed_sel"])


def replay_bar(res) -> None:
    """OFFLINE REPLAY control: the uploaded file is replayed window by window (it is not a live network connection)."""
    ss = st.session_state
    ss["k"] = max(0, min(int(ss["k"]), res.n - 1))
    cur = W.current_summary(res, ss["k"]); pg = ss["page"]
    with st.container(border=True):
        top = st.columns([3.2, 0.9, 0.9, 1.0, 1.1])
        top[0].markdown(f'{T.chip("OFFLINE REPLAY")} &nbsp; <b>Window {cur["window_no"]:,} / {res.n:,}</b> &nbsp;·&nbsp; {cur["time"]:%Y-%m-%d %H:%M:%S} &nbsp; {T.state_chip(cur["state"])}', unsafe_allow_html=True)
        top[1].button("◀ Back", key=f"rb_back_{pg}", width="stretch", on_click=_set_k, args=(max(ss["k"] - 1, 0),))
        top[2].button("Step ▶", key=f"rb_fwd_{pg}", width="stretch", on_click=_set_k, args=(min(ss["k"] + 1, res.n - 1),))
        if top[3].button("⏸ Pause" if ss["playing"] else "▶ Play", key=f"rb_play_{pg}", type="primary", width="stretch"):
            if not ss["playing"] and ss["k"] >= res.n - 1:
                ss["k"] = 0
            ss["playing"] = not ss["playing"]
            st.rerun(scope="app")
        ss["speed_sel"] = ss["speed"] if ss["speed"] in (1, 2, 5, 10) else 1
        top[4].selectbox("Speed", [1, 2, 5, 10], key="speed_sel", format_func=lambda x: f"{x} win/s", label_visibility="collapsed", on_change=_speed_changed)
        ss["k_slider"] = ss["k"] + 1
        st.slider("Replay position", 1, max(res.n, 2), key="k_slider", on_change=_slider_changed, label_visibility="collapsed")


def advance_if_playing(res) -> bool:
    ss = st.session_state
    if not ss.get("playing"):
        return False
    nxt = ss["k"] + int(ss.get("speed", 1))
    if nxt >= res.n - 1:
        ss["k"] = res.n - 1; ss["playing"] = False
        return True
    ss["k"] = nxt
    return False


def live_page(body, replay: bool = True) -> None:
    """Render an analysis page inside a fragment so OFFLINE REPLAY advances without re-running the whole app."""
    ss = st.session_state; res = ss["analysis"]

    @st.fragment(run_every=1.0 if ss.get("playing") else None)
    def _frag() -> None:
        ended = advance_if_playing(res)
        if replay:
            replay_bar(res)
        body(res)
        if ended:
            st.rerun(scope="app")

    _frag()


def step_selector(res, key: str) -> int:
    """Forecast step (+10s ... +Ks) shared by Forecast, Explain and ATT&CK."""
    ss = st.session_state
    labels = [f"+{(i + 1) * res.fc.wsec}s" for i in range(res.fc.H)]
    ss["sel_step"] = max(1, min(int(ss["sel_step"]), res.fc.H)); ss[key] = labels[ss["sel_step"] - 1]

    def _cb() -> None:
        v = st.session_state.get(key)
        if v in labels:
            st.session_state["sel_step"] = labels.index(v) + 1

    if hasattr(st, "segmented_control"):
        st.segmented_control("Forecast step", labels, key=key, on_change=_cb, selection_mode="single", label_visibility="collapsed")
    else:
        st.radio("Forecast step", labels, key=key, on_change=_cb, horizontal=True, label_visibility="collapsed")
    return int(ss["sel_step"])


def alert_card(res, a: dict, key: str, navigable: bool = True) -> None:
    col = T.STATE_COLOR.get(a["severity"], T.STATE_COLOR["WARNING"])
    st.markdown(
        f'<div class="banner" style="background:{col}10;border-color:{col}66"><div class="lbl" style="color:{col}">FORECAST ALERT</div>'
        f'<div class="t" style="font-size:1.2rem;margin:.15rem 0 .5rem">Potential attack progression detected</div>'
        f'<div class="kpair"><span>Predicted stage</span><span><b>{T.esc(a["stage"])}</b></span></div>'
        f'<div class="kpair"><span>Forecast step</span><span><b>+{a["seconds_ahead"]}s</b></span></div>'
        f'<div class="kpair"><span>Risk at that step</span><span><b>{a["risk_at_step"]:.0%}</b> &nbsp;(peak {a["peak_risk"]:.0%})</span></div>'
        f'<div class="kpair"><span>Opened</span><span>{a["opened_time"]:%Y-%m-%d %H:%M:%S}</span></div>'
        f'<div class="d" style="margin-top:.6rem"><b>Reason.</b> {T.esc(W.alert_reason(res, a))}</div></div>', unsafe_allow_html=True)
    if navigable:
        c = st.columns([1, 1, 1, 3]); k, step = a["k_open"], a["step"]
        if c[0].button("View Why", key=f"{key}_why"):
            A.goto_rerun("explain", k=k, sel_step=step)
        if c[1].button("View ATT&CK", key=f"{key}_att"):
            A.goto_rerun("attack", k=k, sel_step=step)
        if c[2].button("View Traffic", key=f"{key}_trf"):
            A.goto_rerun("traffic", k=k, sel_step=step)


def pipeline_html(rows) -> str:
    icon = {"done": "✓", "now": "●", "todo": "○"}
    return "".join(f'<div class="pipe"><span class="{s}">{icon[s]}</span> &nbsp;{T.esc(label)}' + (f' &nbsp;<small style="color:#6b6a68">{T.esc(d)}</small>' if d else "") + "</div> " for s, label, d in rows)


def checks_html(checks) -> str:
    sym = {"ok": "✓", "warn": "!", "fail": "✕"}
    return "".join(f'<div class="chk"><span class="{c.status}">{sym[c.status]}</span> &nbsp;{T.esc(c.name)}' + (f'<br><small>&nbsp;&nbsp;&nbsp;&nbsp;{T.esc(c.detail)}</small>' if c.detail else "") + "</div>" for c in checks)


def flow_strip(res) -> None:
    st.markdown(f'<div class="flow"><span class="box">Observed traffic ({res.fc.L} windows)</span>→<span class="box model">{A.model_label(res.fc)}</span>'
                f'→<span class="box">Forecast (next {res.fc.H} windows, {res.fc.H * res.fc.wsec}s)</span></div>', unsafe_allow_html=True)
