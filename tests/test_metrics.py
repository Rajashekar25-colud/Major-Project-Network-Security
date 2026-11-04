import numpy as np
import pandas as pd

from src.evaluation.metrics import choose_threshold, episodes, lead_time_report, stage_metrics

W = dict(watch_ratio=0.7, critical_margin=0.5, persistence_windows=2, clear_ratio=0.8, clear_windows=3, cooldown_windows=5)


def _mk(n, attack_ranges, risk_fn):
    is_atk = np.zeros(n, bool)
    for a, b in attack_ranges:
        is_atk[a:b + 1] = True
    st = pd.DataFrame({"timestamp": pd.date_range("2018-01-01", periods=n, freq="10s"), "is_attack": is_atk, "segment": 0})
    risk = np.array([risk_fn(i) for i in range(n)])
    tl = pd.DataFrame({"idx": np.arange(n), "timestamp": st["timestamp"], "risk": risk, "headline_stage": "Impact"})
    for k in range(6):
        tl[f"p_t{k + 1}"] = risk
    return st, tl


def test_episodes_merge_small_gaps():
    a = np.zeros(30, bool); a[5:8] = True; a[9:12] = True; a[20:22] = True
    assert episodes(a, np.zeros(30, int)) == [(5, 11), (20, 21)]


def test_lead_time_early_late_missed():
    # episodes at 30-35 (warned from 26), 60-65 (detected late at 62), 100-105 (never)
    def risk(i):
        return 0.9 if (26 <= i <= 36) or (62 <= i <= 66) else 0.05
    st, tl = _mk(150, [(30, 35), (60, 65), (100, 105)], risk)
    r = lead_time_report(tl, st, 0.5, W, 10, 6, 12)
    assert (r["warned_early"], r["detected_late"], r["missed"]) == (1, 1, 1)
    assert r["warning_coverage"] == 1 / 3 and r["min_lead_seconds"] == 30.0   # opens at window 27 (persistence=2)


def test_false_alert_counted():
    st, tl = _mk(150, [(100, 105)], lambda i: 0.9 if 20 <= i <= 25 else 0.05)
    r = lead_time_report(tl, st, 0.5, W, 10, 6, 12)
    assert r["false_alerts"] == 1 and r["missed"] == 1


def test_stage_metrics_reports_missing_support():
    vocab = ["Benign", "Initial Access", "Impact"]
    m = stage_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), vocab)
    assert m["classes_without_support"] == ["Impact"] and "Impact" not in m["per_class"]


def test_threshold_respects_target_fpr():
    rng = np.random.default_rng(0)
    quiet = np.ones(2000, bool); pos = rng.random(2000) < 0.2
    risk = np.where(pos, rng.beta(5, 2, 2000), rng.beta(2, 6, 2000))
    t, info = choose_threshold(risk, quiet, pos, 0.05, 0.5)
    assert info["val_onset_fpr"] <= 0.05 + 1e-9 and 0.05 < t < 1
