"""Tests that tie the implementation to the PPT / synopsis requirements."""
import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data.flows import UNSW_COLUMNS
from src.data.windows import build_states
from src.models.inference import Forecaster
from src.models.world_model import build_model
from src.product.counterfactual import compare, rollout


def test_synopsis_hyperparameters_are_the_defaults():
    m = load_config()["model"]
    assert m["batch_size"] == 32 and m["learning_rate"] == 0.001 and m["optimizer"] == "adam"
    assert 20 <= m["epochs"] <= 50 and 0.2 <= m["dropout"] <= 0.5


def test_transformer_and_lstm_share_interface():
    hp = dict(hidden_size=32, layers=2, dropout=0.2, attention_heads=4)
    for kind in ("lstm", "transformer"):
        m = build_model(kind, 60, 5, 6, hp)
        o = m(torch.randn(3, 12, 60), True)
        assert o["state"].shape == (3, 6, 60) and o["stage_logits"].shape == (3, 6, 5) and o["attention"].shape[0] == 3
        assert m.rollout(torch.randn(2, 12, 60), 3)["attack_prob"].shape == (2, 3)


def test_transformer_trains_end_to_end(states, cfg, tmp_path):
    from src.models.train import train_from_states
    c = {**cfg, "model": {**cfg["model"], "type": "transformer", "epochs": 2, "hidden_size": 32, "max_train_samples_per_epoch": 1500}}
    r = train_from_states(states, c, tmp_path, synthetic=True, log=lambda *_: None, with_baselines=False)
    assert r["meta"]["model"]["type"] == "transformer"
    fc = Forecaster.load(tmp_path)                       # loader rebuilds the right architecture
    assert type(fc.model).__name__ == "TransformerWorldModel"
    assert "accuracy" in r["evaluation"]["forecast"]


def test_counterfactual_changes_generated_states(trained, states):
    out, _ = trained
    fc = Forecaster.load(out)
    X = fc.scale(states); seq = X[200:200 + fc.L]
    sc = {"cut": {"label": "Cut", "description": "", "scale": {"flow_count": 0.1, "scan_score": 0.0}}}
    res = compare(fc, seq, 8, sc, fc.threshold)
    base, cut = res["runs"]["no_action"], res["runs"]["cut"]
    i = fc.names.index("flow_count")
    assert (cut["raw_states"][1:, i] < base["raw_states"][1:, i] * 0.5).all()      # counterfactual traffic really is lower
    assert cut["raw_states"][:, fc.names.index("scan_score")].max() == 0
    assert list(res["summary"]["key"]) == ["no_action", "cut"] and len(base["attack_prob"]) == 8
    assert np.allclose(rollout(fc, seq, 4)["attack_prob"], base["attack_prob"][:4])   # deterministic baseline


def test_unsw_nb15_raw_and_headed(tmp_path):
    cfg = load_config()
    rng = np.random.default_rng(1); n = 12000
    t0 = 1_421_927_414; st = t0 + np.sort(rng.random(n) * 1800)
    cat = np.array([""] * n, dtype=object)
    cat[(st > t0 + 300) & (st < t0 + 500)] = "DoS"; cat[(st > t0 + 900) & (st < t0 + 1000)] = " Fuzzers"; cat[(st > t0 + 1300) & (st < t0 + 1400)] = "Backdoors"
    df = pd.DataFrame(0, index=range(n), columns=UNSW_COLUMNS)
    df["srcip"] = "59.166.0.1"; df["dstip"] = "149.171.126.1"; df["sport"] = rng.integers(1024, 65000, n); df["dsport"] = rng.choice([80, 53, 21], n)
    df["proto"] = rng.choice(["tcp", "udp", "unas"], n); df["dur"] = rng.exponential(0.5, n); df["Spkts"] = 5; df["Dpkts"] = 4; df["sbytes"] = 500; df["dbytes"] = 800
    df["Stime"] = st; df["Sintpkt"] = rng.exponential(30, n); df["attack_cat"] = cat; df["Label"] = (cat != "").astype(int)
    df.to_csv(tmp_path / "UNSW-NB15_1.csv", header=False, index=False)       # raw files are header-less
    df.to_csv(tmp_path / "headed.csv", index=False)
    a, _ = build_states([tmp_path / "UNSW-NB15_1.csv"], cfg, progress=None)
    b, _ = build_states([tmp_path / "headed.csv"], cfg, progress=None)
    assert set(a.loc[a.is_attack, "stage"]) == {"Impact", "Initial Access", "Command and Control"}
    assert a["stage"].tolist() == b["stage"].tolist()
    assert 0.3 < a["duration_mean"].median() < 0.7 and a["iat_mean"].median() < 0.1      # seconds / milliseconds converted
    assert a["protocol_other"].max() > 0                                                   # 'unas' kept, not dropped


def test_flagged_traffic_columns_exist(states):
    for c in ("flow_count", "unique_dst_ports", "syn_ratio", "failed_flow_ratio", "scan_score", "dst_port_80"):
        assert c in states.columns
