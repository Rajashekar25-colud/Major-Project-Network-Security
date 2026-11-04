import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.world_model import WorldModel
from src.product import store
from src.product.knowledge import driver_sentences, feature_label, playbook
from src.product.learning import _expand_stage_head, drift_report
from src.product.report import build_report
from src.product.analysis import run_analysis
from src.models.inference import Forecaster


def test_store_idempotent_keeps_triage(tmp_path):
    db = tmp_path / "t.db"
    df = pd.DataFrame([dict(id="INC-1", source="s", opened_time=pd.Timestamp("2018-01-01"), closed_time=pd.NaT, stage="Impact", progression="Impact",
                            severity="High", peak_risk=.9, eta_seconds=20, windows=3, confirmed=1)])
    store.upsert_incidents(df, db); store.update_incident("INC-1", "Resolved", "done", db); store.upsert_incidents(df, db)
    r = store.list_incidents(db).iloc[0]
    assert r["status"] == "Resolved" and r["note"] == "done" and len(store.list_incidents(db)) == 1
    with pytest.raises(ValueError):
        store.update_incident("INC-1", "bogus", "", db)


def test_feature_labels_and_playbooks():
    assert feature_label("dst_port_22") == "share of traffic to port 22 (SSH)"
    assert "scanning" in feature_label("scan_score")
    assert all(playbook(s)["actions"] for s in ("Impact", "Credential Access", "Command and Control", "Initial Access"))
    assert driver_sentences([{"feature": "syn_ratio", "attribution": 0.3}])[0].startswith("Raises")


def test_stage_head_expansion_preserves_outputs():
    m = WorldModel(10, 4, 6, 32, 2, 0.2, 4).eval(); x = torch.randn(4, 12, 10)
    n = _expand_stage_head(m, 6, dict(hidden_size=32, layers=2, dropout=0.2, attention_heads=4)).eval()
    assert torch.allclose(m(x)["stage_logits"], n(x)["stage_logits"][:, :, :4], atol=1e-5)


def test_analysis_incidents_and_report(trained, states, cfg):
    out, _ = trained
    fc = Forecaster.load(out)
    a = run_analysis(states.iloc[:1500].reset_index(drop=True), "t.csv", fc, cfg, fc.threshold)
    assert a.summary["windows"] == 1500 and {"id", "severity", "stage", "k_open"} <= set(a.incidents.columns)
    assert a.incidents["id"].is_unique
    html = build_report({"product_name": "X", "organisation": "Org"}, "t.csv", a.summary, a.incidents, a.session.dec["risk"], a.threshold, "v000")
    assert "<svg" in html and "Org" in html and "Recommended response" in html


def test_drift_detects_shift(trained, states, monkeypatch):
    out, _ = trained
    import src.product.learning as L
    monkeypatch.setattr(L, "MODELS_DIR", out)
    fc = Forecaster.load(out)
    same = drift_report(fc, states.iloc[:800])
    shifted = states.iloc[:800].copy()
    shifted["flow_count"] = shifted["flow_count"] * 40 + 500
    shifted["bytes_total"] = shifted["bytes_total"] * 30
    assert drift_report(fc, shifted)["significant_share"] > same["significant_share"]
