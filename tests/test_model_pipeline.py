import json

import numpy as np
import torch

from src.explain.explainer import explain_window
from src.models.inference import Forecaster
from src.replay import ReplaySession
from src.features.preprocess import Preprocessor


def test_artifacts_and_metadata_complete(trained):
    out, r = trained
    for f in ("world_model.pt", "metadata.json", "evaluation.json", "background_sequences.npy"):
        assert (out / f).exists()
    m = json.loads((out / "metadata.json").read_text())
    required = ["preprocessing_version", "trained_at", "evaluated_at", "model", "optimizer", "seed", "dataset", "feature_names",
                "stage_names", "warning_threshold", "calibration", "window", "environment", "preprocessing"]
    assert not [k for k in required if k not in m]
    assert {"hidden_size", "layers", "dropout", "learning_rate", "batch_size", "epochs"} <= set(m["model"])
    assert m["dataset"]["sources"] and m["dataset"]["split_summary"]


def test_model_beats_trivial_and_outputs_are_horizon_dependent(trained, states, cfg):
    out, r = trained
    ev = r["evaluation"]
    assert ev["forecast"]["pr_auc"] > ev["forecast"]["support_positive"] / ev["forecast"]["support_total"]   # better than prevalence
    assert ev["baselines"]["persistence_oracle"]["onset"]["recall"] == 0.0
    fc = Forecaster.load(out)
    X = fc.scale(states)
    p = fc.predict_batch(X, np.arange(fc.L, fc.L + 200))["attack_prob"]
    assert p.shape == (200, fc.H) and p.std(axis=1).mean() > 0           # not the same value at every horizon


def test_rollout_and_explain(trained, states):
    out, _ = trained
    fc = Forecaster.load(out)
    X = fc.scale(states); seq = X[100:100 + fc.L]
    ro = fc.model.rollout(torch.from_numpy(seq).unsqueeze(0), 8)
    assert ro["attack_prob"].shape == (1, 8)
    ex = explain_window(fc, seq, 5)
    assert len(ex["top_features"]) == 5 and abs(sum(ex["temporal_attention"]) - 1) < 1e-3


def test_replay_controls_and_alerts(trained, states, cfg):
    out, _ = trained
    fc = Forecaster.load(out)
    rs = ReplaySession(states, fc, cfg)
    assert rs.current() is None
    rs.step(5); assert rs.cursor == rs.lo + 4 and len(rs.history()) == 5
    rs.reset(); assert rs.cursor == rs.lo - 1 and not rs.engine.alerts
    rs.set_range(100, 400); assert rs.n_total == 301
    assert rs.step(10_000) == 301 and rs.finished
    assert len(rs.alerts_df()) >= 1
    rs.set_threshold(0.99); rs.step(500); assert len(rs.engine.alerts) <= 1


def test_feature_mismatch_validation(trained, states):
    out, _ = trained
    fc = Forecaster.load(out)
    assert fc.validate(states)["ok"]
    bad = states.drop(columns=["syn_count", "scan_score"])
    rep = fc.validate(bad)
    assert not rep["ok"] and set(rep["missing"]) == {"syn_count", "scan_score"}


def test_preprocessor_fit_on_train_only_neutralises_constant(states, cfg):
    from src.data.schema import feature_names
    names = feature_names(cfg)
    tr = states.iloc[:500]
    pre = Preprocessor(names).fit(tr)
    assert not pre.active[names.index("ttl_mean")]                          # flow CSV has no TTL
    z = pre.transform(states.assign(ttl_mean=200.0))
    assert (z[:, names.index("ttl_mean")] == 0).all()
