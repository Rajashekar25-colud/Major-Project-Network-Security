"""Overview, Traffic, Forecast, Alerts, Explain and ATT&CK pages. Every value comes from the AnalysisResult."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import appstate as A
import charts as CH
import components as C
import theme as T
from src.analysis import workflow as W
from src.product.knowledge import feature_label


def _res():
    return st.session_state["analysis"]


def _scope_note(res) -> None:
    st.caption(f"{res.name} · {res.n:,} forecast windows · alert threshold {res.thr:.0%} · {A.model_label(res.fc)}")


# ---------------------------------------------------------------------------------------------- Overview
def overview() -> None:
    T.header("Network Attack Forecast", "Replay-based analysis of network traffic and future attack risk.", [T.chip(_res().name), T.chip(A.model_label(_res().fc))])
    C.live_page(_overview_body)


def _overview_body(res) -> None:
    k = st.session_state["k"]; cur = W.current_summary(res, k); fr = W.forecast_frame(res, k); span = W.forecast_span(fr)
    col = T.STATE_COLOR[cur["state"]]; horizon = res.fc.H * res.fc.wsec
    c = st.columns(4)
    c[0].markdown(T.stat("Current network state", cur["state"], "forecast risk vs the alert threshold", col), unsafe_allow_html=True)
    c[1].markdown(T.stat("Current risk", f'{cur["risk"]:.0%}', f"peak forecast probability over the next {horizon}s", col), unsafe_allow_html=True)
    tid = res.fc.meta.get("attack_tactics", {}).get(cur["stage"], "")
    c[2].markdown(T.stat("Predicted attack stage", cur["stage"], f"ATT&CK tactic {tid}" if tid else "no step above threshold"), unsafe_allow_html=True)
    c[3].markdown(T.stat("Forecast window", f"{span[0]} → {span[1]}" if span else "none", "steps at/above the alert threshold" if span else "no step reaches the alert threshold"), unsafe_allow_html=True)
    T.section("Forecast horizon")
    tiles = "".join(f'<div class="t" style="--c:{T.STATE_COLOR[r["state"]]}"><div class="h">{r["label"]}</div><div class="p">{r["risk"]:.0%}</div>'
                    f'<div class="g">{T.esc(r["stage"] if r["stage"] != "None" else r["state"])}</div></div>' for _, r in fr.iterrows())
    st.markdown(f'<div class="hz">{tiles}</div>', unsafe_allow_html=True)
    T.section("Why this forecast matters")
    with st.container(border=True):
        st.write(W.risk_text(res, k, fr))
        al = W.alert_open_at(res, k)
        if al:
            st.markdown(f'{T.chip("ALERT OPEN", T.STATE_COLOR["HIGH RISK"])} &nbsp; {al["stage"]} forecast since {al["opened_time"]:%H:%M:%S}', unsafe_allow_html=True)
        else:
            st.caption("No alert is open at this window.")
    b = st.columns([1, 1, 1, 4])
    if b[0].button("View Forecast", width="stretch"):
        A.goto_rerun("forecast", k=k)
    if b[1].button("Explain", width="stretch"):
        A.goto_rerun("explain", k=k, sel_step=A.default_step(res, k))
    if b[2].button("ATT&CK", width="stretch"):
        A.goto_rerun("attack", k=k, sel_step=A.default_step(res, k))
    _scope_note(res)


# ---------------------------------------------------------------------------------------------- Traffic
def traffic() -> None:
    res = _res(); s = res.stats; st_ = res.states
    T.header("Traffic", "Network telemetry derived from the uploaded traffic.", [T.chip(res.name)])
    c = st.columns(5)
    c[0].markdown(T.stat("Records", T.num(s.get("records")), "flow records" if s.get("kind") != "state-table" else "flows in pre-aggregated windows"), unsafe_allow_html=True)
    c[1].markdown(T.stat("Time range", f'<span style="font-size:1.05rem">{s["start"]:%H:%M:%S} → {s["end"]:%H:%M:%S}</span>', f'{s["start"]:%Y-%m-%d}'), unsafe_allow_html=True)
    c[2].markdown(T.stat("Duration", T.secs(s.get("duration_s"))), unsafe_allow_html=True)
    active = int(((st_["flow_count"] > 0) & (~st_["filled"])).sum())
    c[3].markdown(T.stat("Active windows", f"{active:,}", f'of {s["windows"]:,} total'), unsafe_allow_html=True)
    c[4].markdown(T.stat("Window size", f'{s["window_seconds"]} s'), unsafe_allow_html=True)
    flows = float(st_["flow_count"].sum()); dur = float((st_["duration_mean"] * st_["flow_count"]).sum() / flows) if flows > 0 else None
    c = st.columns(4)
    for col, (lab, v, h) in zip(c, (("Flows", T.num(flows), "summed over all windows"), ("Packets", T.num(st_["packets_total"].sum()), "summed over all windows"),
                                    ("Bytes", T.num(st_["bytes_total"].sum()), "summed over all windows"), ("Mean flow duration", T.secs(dur), "flow-weighted"))):
        col.markdown(T.stat(lab, v, h), unsafe_allow_html=True)
    mix = CH.traffic_mix(st_, res.labelled)
    if mix:
        T.section("Traffic composition")
        keys = [k for k in ("protocol", "ports", "labels") if k in mix]
        for col, k in zip(st.columns(len(keys)), keys):
            col.plotly_chart(mix[k], width="stretch", key=f"mix_{k}")
        if not res.labelled:
            st.caption("No ground-truth labels in this file, so no attack / benign distribution is shown.")
    T.section("Activity over time")
    C.live_page(_traffic_body)


def _traffic_body(res) -> None:
    k = st.session_state["k"]; cur = W.current_summary(res, k); figs = CH.traffic_charts(res.states, cur["time"])
    a, b = st.columns(2); a.plotly_chart(figs["flows"], width="stretch", key="tc_flows"); b.plotly_chart(figs["volume"], width="stretch", key="tc_vol")
    a, b = st.columns(2); a.plotly_chart(figs["ports"], width="stretch", key="tc_ports"); b.plotly_chart(figs["pattern"], width="stretch", key="tc_pat")
    st.caption("The dotted line marks the replay position (the CURRENT window).")
    S = res.states.iloc[cur["row"]]
    st.markdown(f"### Current window &nbsp; <small style='color:#6b6a68'>{cur['time']:%H:%M:%S}</small>", unsafe_allow_html=True)
    c = st.columns(4)
    for col, (lab, v) in zip(c, (("Flows", T.num(S["flow_count"])), ("Packets", T.num(S["packets_total"])), ("Bytes", T.num(S["bytes_total"])), ("Mean duration", T.secs(S["duration_mean"])))):
        col.markdown(T.stat(lab, v), unsafe_allow_html=True)
    pc = [x for x in res.states.columns if x.startswith("dst_port_")]
    ports = S[pc].astype(float).sort_values(ascending=False); ports = ports[ports > 0].head(5)
    T.kv([("Distinct destination ports", f'{S["unique_dst_ports"]:,.0f}'),
          ("Main destination ports", ", ".join(f'{i.replace("dst_port_", "")} ({v:.0%})' for i, v in ports.items()) or "none"),
          ("TCP / UDP share", f'{S["protocol_tcp"]:.0%} / {S["protocol_udp"]:.0%}'), ("SYN · RST · FIN flags", f'{S["syn_count"]:,.0f} · {S["rst_count"]:,.0f} · {S["fin_count"]:,.0f}'),
          ("Unanswered-flow share", f'{S["failed_flow_ratio"]:.0%}'), ("Scan pattern score", f'{S["scan_score"]:.2f}')])
    un = sorted(res.info.get("feature_report", {}).get("unavailable_in_input", []))
    if un:
        st.caption("Fields not present in this file (empty for the model): " + ", ".join(un[:8]) + (" ..." if len(un) > 8 else ""))


# ---------------------------------------------------------------------------------------------- Forecast
def forecast() -> None:
    T.header("Forecast", "The model's view of the next few windows, issued at the current window.", [T.chip(_res().name)])
    C.live_page(_forecast_body)


def _forecast_body(res) -> None:
    k = st.session_state["k"]
    C.flow_strip(res)
    cur = W.current_summary(res, k); fr = W.forecast_frame(res, k); obs = W.observed_frame(res, k)
    step = C.step_selector(res, "step_forecast")
    span = W.forecast_span(fr); peak = fr.loc[fr["risk"].idxmax()]
    if span:
        T.banner("Forecast crosses the alert threshold", f"Forecast probability is at or above {res.thr:.0%} from {span[0]} to {span[1]}; the peak is {peak['risk']:.0%} at {peak['label']}.", T.STATE_COLOR["HIGH RISK"])
    else:
        T.banner("Forecast stays below the alert threshold", f"The highest forecast probability is {peak['risk']:.0%} at {peak['label']}, below the {res.thr:.0%} threshold.", T.STATE_COLOR["NORMAL"])
    r = fr.iloc[step - 1]
    left, right = st.columns([1.35, 1], gap="large")
    left.plotly_chart(CH.horizon_bars(fr, res.thr, step), width="stretch", key="fc_bars")
    with right:
        st.markdown(f"#### Forecast step +{int(r['seconds'])} seconds")
        a, b = st.columns(2)
        a.markdown(T.stat("Expected state", r["state"], "", T.STATE_COLOR[r["state"]]), unsafe_allow_html=True)
        b.markdown(T.stat("Risk", f'{r["risk"]:.0%}', f"alert threshold {res.thr:.0%}", T.STATE_COLOR[r["state"]]), unsafe_allow_html=True)
        st.markdown("")
        a, b = st.columns(2)
        a.markdown(T.stat("Predicted stage", r["stage"], "" if r["stage"] != "None" else f'if an attack occurs: {r["likely_stage"]} ({r["stage_conf"]:.0%})'), unsafe_allow_html=True)
        if res.labelled and r["actual_attack"] is not None:
            b.markdown(T.stat("Actual outcome", "Attack" if r["actual_attack"] else "Normal", f'{r["actual_stage"]} at {r["target_time"]:%H:%M:%S}' if r["actual_attack"] else f'at {r["target_time"]:%H:%M:%S}'), unsafe_allow_html=True)
        else:
            b.markdown(T.stat("Actual outcome", "N/A", "ground truth unavailable for this file" if not res.labelled else "outside the file"), unsafe_allow_html=True)
    st.plotly_chart(CH.forecast_timeline(obs, fr, res.thr, step, res.labelled), width="stretch", key="fc_timeline")
    b = st.columns([1, 1, 4])
    if b[0].button("Explain this step", key="fc_explain"):
        A.goto_rerun("explain")
    if b[1].button("ATT&CK for this step", key="fc_attack"):
        A.goto_rerun("attack")
    st.markdown("### K-step forecast")
    ex = W.expected_now(res, k)
    rows = [{"Step": "Current", "Time": "NOW", "Risk": ex["risk"], "State": ex["state"] or "—", "Predicted stage": ex["stage"] or "—"}]
    rows += [{"Step": f"+{int(x['step'])}", "Time": f"+{int(x['seconds'])}s", "Risk": x["risk"], "State": x["state"], "Predicted stage": x["stage"]} for _, x in fr.iterrows()]
    tdf = pd.DataFrame(rows)
    if res.labelled:
        act = [("Attack: " + cur["actual_stage"]) if cur["actual_attack"] else "Normal"]
        act += ["—" if x["actual_attack"] is None else (("Attack: " + x["actual_stage"]) if x["actual_attack"] else "Normal") for _, x in fr.iterrows()]
        tdf["Actual (dataset label)"] = act
    st.dataframe(tdf, hide_index=True, width="stretch", column_config={"Risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    st.caption("Current row: the forecast the model issued one window earlier for this window (n/a for the first window). Future rows: this window's K-step forecast from the trained model's output heads. Forecast ≠ ground truth.")


# ---------------------------------------------------------------------------------------------- Alerts
def alerts() -> None:
    res = _res()
    T.header("Alerts", "Alerts raised by the warning engine over the analysed traffic.", [T.chip(res.name)])
    recs = W.alert_records(res); w = res.cfg["warning"]
    crit = res.thr + float(w["critical_margin"]) * (1 - res.thr)
    if not recs:
        st.success("✓ No forecast alerts")
        st.write("No configured alert condition has been triggered for the analyzed traffic.")
        st.caption(f"Alert condition: forecast risk ≥ {res.thr:.0%} for {w['persistence_windows']} consecutive windows.")
        _engine_rules(res, w, crit); return
    esc_n = {}
    for e in res.rs.events:
        if e.get("type") == "escalated":
            esc_n[e["alert_id"]] = esc_n.get(e["alert_id"], 0) + 1
    c = st.columns(4)
    c[0].markdown(T.stat("Alerts", str(len(recs))), unsafe_allow_html=True)
    c[1].markdown(T.stat("Open at end of replay", str(sum(1 for r in recs if r["status"] == "OPEN"))), unsafe_allow_html=True)
    hi = max((r["severity"] for r in recs), key=lambda s: W.STATE_RANK.get(s, 0))
    c[2].markdown(T.stat("Highest severity", hi, "", T.STATE_COLOR.get(hi)), unsafe_allow_html=True)
    c[3].markdown(T.stat("Peak risk", f'{max(r["peak_risk"] for r in recs):.0%}'), unsafe_allow_html=True)
    T.section("Alert history")
    rows = [[f'ALR-{r["n"]:03d}', f'{r["opened_time"]:%Y-%m-%d %H:%M:%S}', T.state_chip(r["severity"]), f'{r["peak_risk"]:.0%}', f'+{r["seconds_ahead"]}s', T.esc(r["stage"]),
             T.chip(r["status"], T.STATE_COLOR["HIGH RISK"] if r["status"] == "OPEN" else None), f'{r["windows"]}', f'{esc_n.get(r["n"], 0)}'] for r in recs]
    T.table(["Alert ID", "Time opened", "Severity", "Risk score", "Forecast horizon", "Attack stage", "Status", "Windows", "Escalations"], rows, mono=(0,))
    T.section("Alert detail")
    labels = [f'ALR-{r["n"]:03d} · {r["opened_time"]:%H:%M:%S} · {r["stage"]} · {r["severity"]}' for r in recs]
    ck = st.session_state["k"]
    default = next((i for i, r in enumerate(recs) if r["k_open"] <= ck and (r["k_close"] is None or ck < r["k_close"])), 0)
    a = recs[labels.index(st.selectbox("Alert", labels, index=default, label_visibility="collapsed"))]
    C.alert_card(res, a, key=f"alert_{a['n']}")
    df = pd.DataFrame([{"Alert ID": f'ALR-{r["n"]:03d}', "Opened": r["opened_time"], "Closed": r["closed_time"] if r["closed_time"] is not None else pd.NaT, "Stage progression": " > ".join(r["stages"]),
                        "Severity": r["severity"], "Peak risk": r["peak_risk"], "Forecast horizon (s)": r["seconds_ahead"], "Windows": r["windows"], "Escalations": esc_n.get(r["n"], 0),
                        "Status": r["status"]} for r in recs])
    _engine_rules(res, w, crit)
    st.download_button("Download alerts (CSV)", df.to_csv(index=False).encode(), f"alerts_{res.id}.csv", "text/csv")


def _engine_rules(res, w, crit) -> None:
    T.section("Warning-engine rules (from configs/config.yaml)")
    T.kv([("Persistence", f"opens after {w['persistence_windows']} consecutive windows with forecast risk ≥ {res.thr:.0%}"),
          ("Hysteresis", f"closes only after {w['clear_windows']} windows below {w['clear_ratio']:.0%} of the threshold ({res.thr * w['clear_ratio']:.0%})"),
          ("Cooldown", f"{w['cooldown_windows']} windows before the same stage can re-open an alert"),
          ("De-duplication", "further above-threshold windows update the open alert instead of creating a new one"),
          ("Escalation", f"WARNING → HIGH RISK when forecast risk ≥ {crit:.0%}; WATCH from {w['watch_ratio'] * res.thr:.0%}")])


# ---------------------------------------------------------------------------------------------- Explain
def _explanation(res, k: int, step: int) -> dict:
    cache = st.session_state["cache_ex"]; key = (res.id, k, step, "ex")
    if key not in cache:
        if len(cache) > 200:
            cache.clear()
        cache[key] = W.explain_step(res, k, step)
    return cache[key]


def explain() -> None:
    T.header("Explain", "Why did the model predict this? Explanations follow the forecast step you select.", [T.chip(_res().name)])
    C.live_page(_explain_body)


def _evidence_table(ev: pd.DataFrame, limit: int | None = None) -> None:
    if ev is None or not len(ev):
        st.caption("No raw evidence available for these features."); return
    t = pd.DataFrame({"Signal": [feature_label(f) for f in ev["feature"]], "Current window": ev["current"], "Earlier windows (mean)": ev["earlier_mean"], "Change": ev["change"]})
    st.dataframe(t.head(limit) if limit else t, hide_index=True, width="stretch", column_config={"Change": st.column_config.NumberColumn(format="percent")})


def _explain_body(res) -> None:
    ss = st.session_state; k = ss["k"]
    step = C.step_selector(res, "step_explain"); fr = W.forecast_frame(res, k); r = fr.iloc[step - 1]
    st.markdown(f"### Why was +{int(r['seconds'])}s forecast as {r['state']}?")
    c = st.columns(3)
    c[0].markdown(T.stat("Forecast", r["stage"] if r["stage"] != "None" else "No attack stage", f'most likely stage if attacked: {r["likely_stage"]}' if r["stage"] == "None" else "", T.STATE_COLOR[r["state"]]), unsafe_allow_html=True)
    c[1].markdown(T.stat("Risk", f'{r["risk"]:.0%}', f"threshold {res.thr:.0%}", T.STATE_COLOR[r["state"]]), unsafe_allow_html=True)
    c[2].markdown(T.stat("Forecast issued at", f'<span style="font-size:1.1rem">{W.window_time(res, k):%H:%M:%S}</span>', f"window {k + 1:,} · {res.fc.L} windows of history"), unsafe_allow_html=True)
    st.markdown("&nbsp;")
    try:
        ex = _explanation(res, k, step)
    except Exception as e:
        st.error(f"The explanation could not be computed: {e}"); return
    T.section("Top contributing features")
    try:
        from src.product.knowledge import driver_sentences
        sents = driver_sentences(ex["top_features"], 4)
    except Exception:
        sents = []
    if sents:
        st.markdown("\n".join(f"- {x}" for x in sents))
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Feature importance · Gradient × input")
        st.plotly_chart(CH.contributions(ex["top_features"], "attribution", "Gradient × input attribution (orange raises this step's risk)"), width="stretch", key="ex_attr")
        st.caption(f"Local to this forecast step. Each bar sums a feature's attribution over all {res.fc.L} input windows.")
    with right:
        st.markdown("#### Temporal importance")
        st.plotly_chart(CH.temporal(ex["temporal_attention"], ex["temporal_attribution"], ex["attention_available"]), width="stretch", key="ex_temp")
        st.caption("Bars: attention weight of the last window over the history (heads averaged; the same for every forecast step). Line: share of total attribution per window (specific to this step)."
                   if ex["attention_available"] else "This model type does not expose attention weights; only per-window attribution is shown.")
    with st.expander("SHAP (exact SHAP values for this forecast step)"):
        st.caption("Computed on demand with SHAP GradientExplainer against background sequences saved at training time; values are summed over the input windows.")
        skey = (res.id, k, step, "shap")
        if st.button("Compute SHAP for this step", key="btn_shap"):
            with st.spinner("Computing SHAP values ..."):
                ss["cache_ex"][skey] = W.shap_step(res, k, step)
        sh = ss["cache_ex"].get(skey)
        if sh is None:
            st.caption("Not computed yet. SHAP runs on demand because it is slow; use the button above.")
        elif not sh.get("available"):
            st.warning(f'SHAP explanation unavailable for this analysis. {sh.get("reason") or ""}')
        else:
            st.plotly_chart(CH.contributions(sh["top_features"], "shap_value", f'SHAP value per feature (summed over {res.fc.L} windows; {sh["n_background"]} background sequences)'), width="stretch", key="ex_shap")
    st.markdown("#### Traffic evidence")
    _evidence_table(ex["evidence"])
    st.caption("Observed (raw-unit) values of the features that contributed most, for the same input the model saw. Change = current vs the mean of the earlier input windows.")


# ---------------------------------------------------------------------------------------------- ATT&CK
def attack() -> None:
    T.header("MITRE ATT&CK", "Predicted attack stage and its ATT&CK tactic for the selected forecast step.", [T.chip(_res().name)])
    C.live_page(_attack_body)


def _attack_body(res) -> None:
    k = st.session_state["k"]
    step = C.step_selector(res, "step_attack"); fr = W.forecast_frame(res, k); r = fr.iloc[step - 1]; meta = A.metadata()
    stage = r["stage"]; hypothetical = stage == "None"
    if hypothetical:
        st.info(f'The forecast risk at +{int(r["seconds"])}s ({r["risk"]:.0%}) is below the alert threshold ({res.thr:.0%}), so no attack stage is predicted. '
                f'If an attack were to occur, the model considers **{r["likely_stage"]}** most likely ({r["stage_conf"]:.0%} of the stage probability).')
        stage = r["likely_stage"]
    m = W.attack_mapping(meta, res.cfg, stage)
    col = T.STAGE_COLOR.get(stage, T.ACCENT)
    T.section("Current forecast" if not hypothetical else "Most likely stage (not predicted)")
    big, side = st.columns([1.5, 1], gap="large")
    big.markdown(f'<div class="bigstage" style="--c:{col}"><div class="lbl">{"Predicted attack stage" if not hypothetical else "Most likely stage if an attack occurs"}</div>'
                 f'<div class="nm">{T.esc(stage)}</div>'
                 f'<div class="hint">ATT&amp;CK tactic {T.esc(m["tactic_id"] or "not mapped")}' + (f' · <a href="{m["url"]}" target="_blank">attack.mitre.org</a>' if m["tactic_id"] else "") + '</div></div>', unsafe_allow_html=True)
    with side:
        T.kv([("Model confidence", f'<b>{r["stage_conf"]:.0%}</b> of the stage probability'), ("Forecast horizon", f'<b>+{int(r["seconds"])} seconds</b>'),
              ("Forecast risk", f'<b style="color:{T.STATE_COLOR[r["state"]]}">{r["risk"]:.0%}</b> · {r["state"]}')])
    T.section("Kill-chain stages known to the model")
    cells = ""
    for sname in [x for x in res.fc.stages if x != "Benign"]:
        tid = W.attack_mapping(meta, res.cfg, sname)["tactic_id"] or ""
        on = " on" if sname == stage and not hypothetical else ""
        cells += f'<div class="st{on}" style="--c:{T.STAGE_COLOR.get(sname, T.ACCENT)}"><span class="id">{T.esc(tid)}</span><b>{T.esc(sname)}</b></div>'
    st.markdown(f'<div class="ladder">{cells}</div>', unsafe_allow_html=True)
    if m["dataset_labels"] or m["label_patterns"]:
        st.caption("Configured mapping (configs/config.yaml): dataset attack labels " + ", ".join(f"`{x}`" for x in m["dataset_labels"] + m["label_patterns"]) + f" are mapped to **{stage}**. "
                   "This project maps forecasts to ATT&CK **tactics (stages)**; technique IDs are not modelled, so none are shown.")
    else:
        st.caption("This project maps forecasts to ATT&CK **tactics (stages)**. Technique IDs are not modelled, so none are shown.")
    st.plotly_chart(CH.stage_probs(res.fc.stages, res.rs.stage_prob[k][step - 1], f"Stage probabilities at +{int(r['seconds'])}s (model stage head)"), width="stretch", key="at_stage")
    st.markdown("#### Evidence")
    try:
        _evidence_table(_explanation(res, k, step)["evidence"], 6)
        st.caption("Observed traffic signals that contributed most to this step's forecast (same attribution as the Explain page).")
    except Exception as e:
        st.caption(f"Evidence unavailable: {e}")
    if res.labelled and r["actual_attack"] is not None:
        st.markdown("#### Actual outcome (dataset label)")
        st.write(f'At {r["target_time"]:%H:%M:%S}: **{"Attack - " + r["actual_stage"] if r["actual_attack"] else "Normal traffic"}**. Ground truth from the file, shown separately from the forecast.')
