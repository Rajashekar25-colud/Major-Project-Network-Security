"""Workflow tests that do not need torch/streamlit (FakeForecaster stands in for the model). The real trained model is
covered in test_analysis_real_model.py."""
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from fake_forecaster import FakeForecaster  # noqa: E402
from src.analysis import history as H  # noqa: E402
from src.analysis import workflow as W  # noqa: E402
from src.analysis.ingest import inspect_traffic  # noqa: E402
from src.analysis.integrity import fingerprint, unchanged  # noqa: E402
from src.analysis.validation import NO_TRUTH, validate_forecast  # noqa: E402
from src.config import load_config  # noqa: E402

CFG = load_config()
FLOW_CSV = ROOT / "data" / "sample" / "test_flows_cic_format.csv"
PCAP = ROOT / "data" / "sample" / "sample_traffic.pcap"
_cache: dict = {}


def fc():
    if "fc" not in _cache:
        ref = pd.read_csv(ROOT / "data" / "states" / "demo_states.csv", parse_dates=["timestamp"])
        _cache["fc"] = FakeForecaster(CFG, ref)
    return _cache["fc"]


def csv_report():
    if "rep" not in _cache:
        _cache["rep"] = inspect_traffic("t.csv", FLOW_CSV.read_bytes(), fc(), CFG)
    return _cache["rep"]


def csv_result(thr=None):
    rep = csv_report()
    return W.run_analysis(rep.states, "t.csv", fc(), CFG, thr, info=rep.info, stats=rep.stats)


# ---------------------------------------------------------------- ingest / validation
def test_valid_csv_reports_real_statistics():
    rep = csv_report()
    assert rep.ok and all(c.status != "fail" for c in rep.checks)
    raw_rows = len(pd.read_csv(FLOW_CSV))
    assert rep.stats["records"] == raw_rows                                   # record count is the file's, not invented
    assert rep.stats["windows"] == len(rep.states) and rep.stats["window_seconds"] == fc().wsec
    assert rep.stats["start"] == rep.states["timestamp"].min()
    assert rep.stats["labelled"] is True
    assert any(c.name == "No training performed" for c in rep.checks)


def test_invalid_csv_is_rejected_with_reason():
    r = inspect_traffic("x.csv", b"a,b\n1,2\n", fc(), CFG)
    assert not r.ok and "required field" in r.error


def test_unsupported_and_empty_files():
    assert "Unsupported" in inspect_traffic("x.txt", b"hi", fc(), CFG).error
    assert "empty" in inspect_traffic("x.csv", b"", fc(), CFG).error


def test_insufficient_temporal_history():
    ref = pd.read_csv(ROOT / "data" / "states" / "demo_states.csv", parse_dates=["timestamp"]).iloc[:8]
    b = io.BytesIO(); ref.to_csv(b, index=False)
    r = inspect_traffic("tiny.csv", b.getvalue(), fc(), CFG)
    assert not r.ok and "Insufficient temporal history" in r.error and "8 window" in r.error


def test_incompatible_schema_lists_missing_features():
    ref = pd.read_csv(ROOT / "data" / "states" / "demo_states.csv", parse_dates=["timestamp"]).iloc[:200].drop(columns=["syn_count", "scan_score", "bytes_total"])
    b = io.BytesIO(); ref.to_csv(b, index=False)
    r = inspect_traffic("bad.csv", b.getvalue(), fc(), CFG)
    assert not r.ok and "3 required model feature" in r.error and "syn_count" in r.error


def test_pcap_is_analysable_and_unlabelled():
    r = inspect_traffic("s.pcap", PCAP.read_bytes(), fc(), CFG)
    assert r.ok and r.info["kind"] == "pcap" and r.stats["labelled"] is False
    res = W.run_analysis(r.states, "s.pcap", fc(), CFG, info=r.info, stats=r.stats)
    assert res.n > 0 and validate_forecast(res) == {"available": False, "reason": NO_TRUTH}


# ---------------------------------------------------------------- forecast / alerts
def test_forecast_frame_is_exactly_the_model_output():
    res = csv_result(); f = fc()
    for k in (0, res.n // 2, res.n - 1):
        fr = W.forecast_frame(res, k)
        assert len(fr) == f.H and list(fr["seconds"]) == [(i + 1) * f.wsec for i in range(f.H)]      # horizon comes from the model
        np.testing.assert_allclose(fr["risk"].to_numpy(), res.rs.attack_prob[k])
        assert all(fr["above"] == (fr["risk"] >= res.thr))
        assert all((fr["stage"] == "None") == ~fr["above"])


def test_alert_threshold_controls_alerts():
    quiet = csv_result(0.999)
    assert W.alert_records(quiet) == []
    res = csv_result(0.5); recs = W.alert_records(res)
    assert recs and all(r["risk_at_step"] >= res.thr for r in recs)
    assert all(r["k_open"] < r["k_last"] + 1 for r in recs) and [r["n"] for r in recs] == list(range(1, len(recs) + 1))
    W.apply_threshold(res, 0.999)
    assert W.alert_records(res) == [] and res.thr == 0.999


def test_alert_requires_persistence():
    res = csv_result(0.5); p = CFG["warning"]["persistence_windows"]
    for a in W.alert_records(res):
        assert a["windows"] >= p
        assert W.alert_open_at(res, a["k_open"]) is not None


def test_expected_now_uses_earlier_forecast():
    res = csv_result()
    assert W.expected_now(res, 0)["risk"] is None
    assert W.expected_now(res, 5)["risk"] == float(res.rs.attack_prob[4, 0])


def test_observed_frame_covers_history():
    res = csv_result(); o = W.observed_frame(res, 20)
    assert len(o) == fc().L and o["offset"].iloc[-1] == 0 and o["offset"].iloc[0] == -(fc().L - 1)


def test_attack_mapping_comes_from_config():
    m = W.attack_mapping(fc().meta, CFG, "Credential Access")
    assert m["tactic_id"] == CFG["attack_tactics"]["Credential Access"] and "ssh-bruteforce" in m["dataset_labels"] and m["techniques_modelled"] is False


def test_evidence_uses_real_values():
    res = csv_result(); k = 30
    ev = W.traffic_evidence(res, k, ["flow_count", "not_a_feature"])
    row = int(res.rs.ends[k]); L = fc().L
    assert list(ev["feature"]) == ["flow_count"]
    assert ev["current"].iloc[0] == res.states["flow_count"].iat[row]
    assert abs(ev["earlier_mean"].iloc[0] - res.states["flow_count"].to_numpy()[row - L + 1:row].mean()) < 1e-9


# ---------------------------------------------------------------- validation
def test_labelled_validation_matches_independent_count():
    res = csv_result(0.5); v = validate_forecast(res)
    assert v["available"] and v["onset"] is not None
    f = v["forecast"]; st, fcst = res.states, fc()
    from src.models.sequences import sample_ends
    ends = sample_ends(st, fcst.L, fcst.H); atk = st["is_attack"].to_numpy(); tp = fp = fn = tn = 0
    for e in ends:
        pos = res.rs.pos_of[int(e)]
        for j in range(fcst.H):
            pred = res.rs.attack_prob[pos, j] >= 0.5; act = bool(atk[e + 1 + j])
            tp += pred and act; fp += pred and not act; fn += (not pred) and act; tn += (not pred) and (not act)
    assert (f["tp"], f["fp"], f["fn"], f["tn"]) == (tp, fp, fn, tn)
    out = v["outcomes"]
    assert len(out) == len(ends) and out["correct"].equals(out["predicted_attack"] == out["actual_attack"])
    assert v["lead_time"]["attack_episodes"] >= 1


def test_undefined_metrics_are_none_not_zero():
    rep = csv_report(); st = rep.states.copy(); st["is_attack"] = False; st["stage"] = "Benign"
    res = W.run_analysis(st, "t.csv", fc(), CFG, 0.5, info={**rep.info, "labelled": True}, stats=rep.stats)
    v = validate_forecast(res)
    assert v["available"] and v["forecast"]["recall"] is None and v["forecast"]["f1"] is None and not v["has_attacks"]


# ---------------------------------------------------------------- history & integrity & guards
def test_history_roundtrip(tmp_path):
    res = csv_result(0.5); H.save_analysis(res, tmp_path)
    items = H.list_history(tmp_path)
    assert len(items) == 1 and items[0]["filename"] == "t.csv" and items[0]["alerts"] == len(W.alert_records(res)) and items[0]["labelled"] is True
    back = H.load_states(res.id, tmp_path)
    assert len(back) == len(res.states) and back["is_attack"].sum() == res.states["is_attack"].sum()
    H.delete_analysis(res.id, tmp_path)
    assert H.list_history(tmp_path) == []


def test_fingerprint_detects_any_change(tmp_path):
    (tmp_path / "world_model.pt").write_bytes(b"weights"); (tmp_path / "metadata.json").write_text("{}")
    a = fingerprint(tmp_path); assert unchanged(a, fingerprint(tmp_path))
    (tmp_path / "world_model.pt").write_bytes(b"weights2")
    assert not unchanged(a, fingerprint(tmp_path))


def test_run_analysis_reports_unchanged_model(tmp_path):
    (tmp_path / "world_model.pt").write_bytes(b"w"); (tmp_path / "metadata.json").write_text("{}")
    rep = csv_report()
    res = W.run_analysis(rep.states, "t.csv", fc(), CFG, info=rep.info, stats=rep.stats, models_dir=tmp_path)
    assert unchanged(res.integrity_before, res.integrity_after)


def test_progress_is_real():
    rep = csv_report(); seen = []
    W.run_analysis(rep.states, "t.csv", fc(), CFG, info=rep.info, stats=rep.stats, progress=lambda s, d, t: seen.append((s, d, t)))
    n = len(seen and [x for x in seen if x[0] == "forecast"])
    assert n >= 1 and seen[-1] == ("risk", csv_result().n, csv_result().n)
    assert [x for x in seen if x[0] == "forecast"][-1][1] == csv_result().n


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_no_training_or_weight_writes_in_inference_code():
    for p in list((ROOT / "src" / "analysis").glob("*.py")) + list((ROOT / "app").rglob("*.py")):
        s = p.read_text(encoding="utf-8")
        assert not re.search(r"models\.train|train_from_states|fine_?tune|\.backward\(|optim\.|torch\.save|\.fit\(", s), p
        assert "random" not in re.findall(r"^\s*(?:import|from)\s+(\w+)", s, re.M), p


def test_no_hardcoded_predictions_in_ui():
    for p in (ROOT / "app").rglob("*.py"):
        s = p.read_text(encoding="utf-8")
        assert not re.search(r"\b(risk|prob\w*|attack_prob)\s*=\s*0?\.\d+\b", s), p           # no literal probabilities
        assert not re.search(r">\s*\d{1,3}%\s*<", s), p                                         # no literal "72%" tiles


def test_evaluation_page_never_uses_current_analysis():
    s = _src("app/screens/validate_pages.py"); body = s[s.index("def evaluation"):]
    assert 'session_state["analysis"]' not in body and "rs.attack_prob" not in body and "Held-out Evaluation Dataset" in body


def test_sidebar_is_state_aware():
    s = _src("app/main.py")
    assert "if A.has_analysis():" in s and s.index("if A.has_analysis():") < s.index('"Overview"') < s.index('"History"')
    assert s.index('_nav("History"') > s.index('"Forecast Validation"')
