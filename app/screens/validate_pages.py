"""Forecast Validation (this run, only with ground truth) and Evaluation (saved held-out artifact, clearly separate)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import appstate as A
import charts as CH
import theme as T
from src.analysis.validation import validate_forecast


def forecast_validation() -> None:
    res = st.session_state["analysis"]
    T.header("Forecast Validation", "Prediction vs actual outcome in THIS uploaded file. The model was not retrained.", [T.chip(res.name)])
    cache = st.session_state["validation_cache"]
    if res.id not in cache:
        with st.spinner("Comparing forecasts with the labelled outcomes ..."):
            cache[res.id] = validate_forecast(res)
    v = cache[res.id]
    if not v["available"]:
        st.info(v["reason"])
        st.button("Upload another traffic file", type="primary", on_click=A.start_new)
        return
    f, o = v["forecast"], v["onset"]
    st.caption(f'{res.name} · {v["n_origins"]:,} forecast origins with a full {res.fc.H}-step future · alert threshold {v["threshold"]:.0%} · computed from this run.')
    if not v["has_attacks"]:
        st.warning("This file has labels but contains no attack windows, so recall and precision are undefined (shown as n/a). Only false alarms can be assessed.")
    T.section("Forecast metrics · every forecast step, every origin")
    c = st.columns(6)
    for col, (lab, val) in zip(c, (("Precision", T.f3(f["precision"])), ("Recall", T.f3(f["recall"])), ("F1", T.f3(f["f1"])), ("False-positive rate", T.f3(f["false_positive_rate"])),
                                    ("PR-AUC", T.f3(f.get("pr_auc"))), ("Samples", f'{f["support_total"]:,}'))):
        col.markdown(T.stat(lab, val), unsafe_allow_html=True)
    if o:
        T.section("Early-warning metrics · benign now, attack starts within the horizon")
        c = st.columns(5)
        for col, (lab, val) in zip(c, (("Precision", T.f3(o["precision"])), ("Recall", T.f3(o["recall"])), ("F1", T.f3(o["f1"])), ("False-positive rate", T.f3(o["false_positive_rate"])),
                                        ("Onsets", f'{o["support_positive"]:,}'))):
            col.markdown(T.stat(lab, val), unsafe_allow_html=True)
    lt = v.get("lead_time") or {}
    T.section("Warning lead time · early warnings, late detections, missed detections")
    if lt.get("available"):
        c = st.columns(5)
        for col, (lab, val) in zip(c, (("Attack episodes", str(lt["attack_episodes"])), ("Warned early", str(lt["warned_early"])), ("Detected late", str(lt["detected_late"])),
                                        ("Missed", str(lt["missed"])), ("Mean lead time", T.secs(lt["mean_lead_seconds"])))):
            col.markdown(T.stat(lab, val), unsafe_allow_html=True)
        st.caption(f'False alerts: {lt["false_alerts"]} ({lt["false_alerts_per_hour"]:.2f}/hour). {lt["definition"]} '
                   f'{lt["not_forecastable_episodes"]} episode(s) begin before a full input history exists and are not scored.')
    else:
        st.caption("Lead time could not be computed for this file.")
    T.section("Prediction vs actual · per window")
    out = v["outcomes"]
    if out is not None:
        show = out.rename(columns={"time": "Window time", "forecast_risk": "Forecast risk", "predicted_attack": "Predicted attack", "predicted_stage": "Predicted stage",
                                   "actual_attack": "Actual attack", "actual_stage": "Actual stage", "correct": "Correct"}).drop(columns=["k"])
        only_wrong = st.toggle("Show only incorrect forecasts", value=False)
        if only_wrong:
            show = show[~show["Correct"]]
        st.dataframe(show, hide_index=True, width="stretch", height=380, column_config={"Forecast risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        st.caption("Predicted = the peak forecast over the next steps reached the alert threshold. Actual = an attack-labelled window occurs within those steps.")
        st.download_button("Download forecast-vs-actual (CSV)", show.to_csv(index=False).encode(), f"validation_{res.id}.csv", "text/csv")
    st.button("Upload another traffic file", on_click=A.start_new)


def evaluation() -> None:
    meta, ev = A.metadata(), A.evaluation()
    T.header("Evaluation", "Model Evaluation — Held-out Evaluation Dataset", [T.chip("ADMIN", T.ACCENT)])
    st.warning("These are saved results of the trained model on its held-out evaluation data. They are NOT predictions or scores for any uploaded file, "
               "and they do not mean the current traffic has these probabilities.")
    if meta.get("synthetic_demo_data"):
        st.error("The saved model was trained on synthetic demo data: these numbers are not research results.")
    if not ev or "forecast" not in ev:
        st.info("No evaluation artifact found (models/evaluation.json). Run `python scripts/train.py ...` or `python scripts/evaluate.py ...`."); return
    f, o, lt, sm, sf = ev["forecast"], ev["onset"], ev["lead_time"], ev["stage"], ev["state_forecast"]
    T.kv([("Model", f'{A.model_label(A.forecaster())} · v{T.esc(str(meta.get("project_version", "N/A")))}'), ("Dataset", T.esc(A.dataset_label(meta))), ("Split", T.esc(str(ev.get("split", "N/A")))),
          ("Evaluated", T.esc(str(ev.get("evaluated_at", ""))[:19].replace("T", " "))), ("Alert threshold", f'{ev["threshold"]:.3f}'), ("Held-out samples", f'{ev["n_samples"]:,}')])
    T.section("Forecast metrics · all forecast steps")
    c = st.columns(5)
    for col, (lab, k) in zip(c, (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("False-positive rate", "false_positive_rate"), ("PR-AUC", "pr_auc"))):
        col.markdown(T.stat(lab, T.f3(f.get(k)), f'early-warning: {T.f3(o.get(k))}'), unsafe_allow_html=True)
    st.caption("Large number = all forecast steps. Hint line = early-warning subset. " + o.get("definition", ""))
    T.section("State forecasting error and warning performance")
    c = st.columns(4)
    c[0].markdown(T.stat("MAE", T.f3(sf["mae"]), f'persistence baseline {T.f3(sf["persistence_mae"])}'), unsafe_allow_html=True)
    c[1].markdown(T.stat("RMSE", T.f3(sf["rmse"]), f'persistence baseline {T.f3(sf["persistence_rmse"])}'), unsafe_allow_html=True)
    c[2].markdown(T.stat("Mean lead time", T.secs(lt.get("mean_lead_seconds")), f'{lt["warned_early"]} of {lt["forecastable_episodes"]} episodes warned early'), unsafe_allow_html=True)
    c[3].markdown(T.stat("False alerts / hour", f'{lt["false_alerts_per_hour"]:.2f}'), unsafe_allow_html=True)
    st.caption(f'Space: {sf.get("space", "")}. Detected late: {lt["detected_late"]} · missed: {lt["missed"]} · not forecastable: {lt["not_forecastable_episodes"]}.')
    st.plotly_chart(CH.per_horizon(pd.DataFrame(ev["per_horizon"])), width="stretch", key="ev_horizon")
    T.section("Stage support")
    sup = pd.DataFrame([{"Stage": k, "Held-out samples": v, "Scored": v > 0} for k, v in sm["support"].items()])
    st.dataframe(sup, hide_index=True, width="stretch")
    st.caption(f'Stage accuracy {sm["accuracy"]:.3f} is computed only over stages with support: {", ".join(sm["classes_with_support"])}. '
               f'Not scored (no support): {", ".join(sm["classes_without_support"]) or "none"}. It is not nine-class accuracy.')
    b = ev.get("baselines", {}).get("logistic_sequence", {})
    if b and "forecast" in b:
        T.section("Baseline comparison")
        st.dataframe(pd.DataFrame({"Model": [A.model_label(A.forecaster()), "Logistic regression (sequence)"], "Forecast F1": [f["f1"], b["forecast"]["f1"]], "Forecast recall": [f["recall"], b["forecast"]["recall"]],
                                   "Early-warning F1": [o["f1"], b["onset"]["f1"]], "Early-warning recall": [o["recall"], b["onset"]["recall"]]}).round(3), hide_index=True, width="stretch")
