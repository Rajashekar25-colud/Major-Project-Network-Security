"""Analysis workflow against the REAL torch model (trained on the synthetic test fixture). Requires torch."""
import numpy as np
import pytest

pytest.importorskip("torch")

from src.analysis import workflow as W  # noqa: E402
from src.analysis.integrity import fingerprint, unchanged  # noqa: E402
from src.analysis.ingest import inspect_traffic  # noqa: E402
from src.analysis.validation import validate_forecast  # noqa: E402
from src.models.inference import Forecaster  # noqa: E402


def _run(trained, states, cfg, thr=None):
    out, _ = trained
    fc = Forecaster.load(out)
    return fc, out, W.run_analysis(states.iloc[:1200].reset_index(drop=True), "t.csv", fc, cfg, thr, info={"kind": "state-table", "labelled": True}, models_dir=out)


def test_analysis_never_changes_model(trained, states, cfg):
    out, _ = trained
    before = fingerprint(out)
    fc, _, res = _run(trained, states, cfg)
    W.explain_step(res, res.n // 2, 3)                        # gradients flow to the input, never into the weights
    assert unchanged(before, fingerprint(out)) and unchanged(res.integrity_before, res.integrity_after)


def test_forecast_steps_come_from_model_heads(trained, states, cfg):
    fc, _, res = _run(trained, states, cfg)
    k = res.n // 2
    fr = W.forecast_frame(res, k)
    direct = fc.predict_batch(fc.scale(res.states), np.array([res.rs.ends[k]]))["attack_prob"][0]
    np.testing.assert_allclose(fr["risk"].to_numpy(), direct, atol=1e-6)
    assert len(fr) == fc.H


def test_explanation_follows_selected_step(trained, states, cfg):
    fc, _, res = _run(trained, states, cfg)
    k = res.n // 2
    e1, e2 = W.explain_step(res, k, 1), W.explain_step(res, k, fc.H)
    assert e1["top_features"] and e2["top_features"]
    a1 = np.array([f["attribution"] for f in e1["top_features"]]); a2 = np.array([f["attribution"] for f in e2["top_features"]])
    assert not np.allclose(a1, a2) or [f["feature"] for f in e1["top_features"]] != [f["feature"] for f in e2["top_features"]]
    assert abs(sum(e1["temporal_attention"]) - 1) < 1e-3 and len(e1["evidence"]) > 0


def test_shap_per_step_is_real_or_reported_unavailable(trained, states, cfg):
    fc, out, res = _run(trained, states, cfg)
    import src.config as C
    C.MODELS_DIR = out                                         # background sequences live next to the trained fixture model
    sh = W.shap_step(res, res.n // 2, 2, 5)
    assert sh["available"] is False and sh.get("reason") or (sh["available"] and len(sh["top_features"]) == 5 and sh["shap_matrix"].shape == (fc.L, len(fc.names)))


def test_labelled_validation_with_real_model(trained, states, cfg):
    _, _, res = _run(trained, states, cfg)
    v = validate_forecast(res)
    assert v["available"] and v["forecast"]["tp"] + v["forecast"]["fp"] + v["forecast"]["tn"] + v["forecast"]["fn"] == v["n_origins"] * res.fc.H
