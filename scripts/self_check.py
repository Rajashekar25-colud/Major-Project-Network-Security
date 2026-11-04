"""One-command health check: is the whole system working?   python scripts/self_check.py

Prints PASS / WARN / FAIL per check. Exit code 1 if anything FAILs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _common import ROOT

from src.config import load_config
from src.models.inference import Forecaster, artifacts_present
from src.models.sequences import split_labels

RES = []


def rec(status, name, detail=""):
    RES.append(status)
    icon = {"PASS": "[ PASS ]", "WARN": "[ WARN ]", "FAIL": "[ FAIL ]"}[status]
    print(f"{icon} {name}" + (f"  - {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=None)
    a = ap.parse_args()
    cfg = load_config()
    models = ROOT / "models"

    # 1. artifacts ------------------------------------------------------------------------
    if not artifacts_present(models):
        rec("FAIL", "model artifacts", "models/world_model.pt or metadata.json missing -> run scripts/train.py"); return finish()
    meta = json.loads((models / "metadata.json").read_text(encoding="utf-8"))
    fc = Forecaster.load(models)
    rec("PASS", "model loads", f"{meta['model']['parameters']:,} params, trained {meta['trained_at']}")
    if meta.get("synthetic_demo_data"):
        rec("WARN", "model trained on", "SYNTHETIC demo data - retrain on the real dataset: python scripts/train.py --states data/states/cicids2018_states.csv")
    else:
        rec("PASS", "model trained on", f"{len(meta['dataset'].get('sources', []))} source file(s), {meta['dataset'].get('windows_total', 0):,} windows")
    missing = [k for k in ("preprocessing_version", "trained_at", "evaluated_at", "optimizer", "calibration", "split") if k not in meta]
    rec("PASS" if not missing else "WARN", "metadata complete", f"missing {missing}" if missing else "")
    rec("PASS" if 0.05 <= meta["warning_threshold"] <= 0.9 else "WARN", "warning threshold",
        f"{meta['warning_threshold']:.3f}" + ("" if meta["warning_threshold"] <= 0.9 else " (very high: probabilities poorly calibrated, retrain)"))
    rec("PASS" if "Initial Access" in meta["stage_names"] or meta.get("synthetic_demo_data") else "WARN", "stage vocabulary", ", ".join(meta["stage_names"]))

    # 2. states -----------------------------------------------------------------------------
    from src.service import load_states_csv
    cand = [a.states] if a.states else [ROOT / "data/states/cicids2018_states.csv", ROOT / "data/states/demo_states.csv"]
    sp = next((Path(c) for c in cand if c and Path(c).exists()), None)
    if sp is None:
        rec("FAIL", "state table", "none found -> run scripts/prepare_states.py"); return finish()
    st = load_states_csv(sp, cfg)
    rec("PASS", "state table", f"{sp.name}: {len(st):,} windows, {st['segment'].nunique()} segments")
    yr = st["timestamp"].dt.year
    rec("PASS" if yr.between(2000, 2035).all() else "FAIL", "timestamps sane", f"{yr.min()}-{yr.max()}")
    rep = fc.validate(st)
    rec("PASS" if rep["ok"] else "FAIL", "feature schema matches model", f"missing {rep['missing'][:5]}" if not rep["ok"] else f"{len(fc.names)} features")
    if not rep["ok"]:
        return finish()
    neg = (st[["duration_mean", "flow_count", "packets_total", "bytes_total"]] < 0).any().any()
    rec("FAIL" if neg else "PASS", "no negative durations/counts")
    rec("PASS" if st["is_attack"].any() else "FAIL", "attack windows present", f"{int(st['is_attack'].sum()):,}")

    # 3. forecasting behaves like a K-step forecaster --------------------------------------------
    X = fc.scale(st)
    from src.models.sequences import inference_ends
    ends = inference_ends(st, fc.L)
    sel = ends[:: max(len(ends) // 3000, 1)]
    pr = fc.predict_batch(X, sel)
    ap_ = pr["attack_prob"]
    rec("PASS" if np.isfinite(ap_).all() and ap_.min() >= 0 and ap_.max() <= 1 else "FAIL", "probabilities valid", f"range {ap_.min():.3f}-{ap_.max():.3f}")
    rec("PASS" if ap_.std(axis=1).mean() > 1e-4 else "FAIL", "horizon-dependent forecast", f"mean spread across horizons {ap_.std(axis=1).mean():.4f}")
    y = st["is_attack"].to_numpy()[sel]
    sep = ap_.max(axis=1)[y].mean() - ap_.max(axis=1)[~y].mean() if y.any() and (~y).any() else float("nan")
    rec("PASS" if sep > 0.1 else "WARN", "risk higher during attacks than normal traffic", f"difference {sep:+.3f}")

    # 4. stored evaluation ------------------------------------------------------------------
    evp = models / "evaluation.json"
    if evp.exists():
        ev = json.loads(evp.read_text(encoding="utf-8"))
        o, f = ev["onset"], ev["forecast"]
        prev = f["support_positive"] / max(f["support_total"], 1)
        rec("PASS" if f.get("pr_auc", 0) > prev * 1.5 else "WARN", "forecast beats chance (PR-AUC vs prevalence)", f"PR-AUC {f.get('pr_auc', float('nan')):.3f} vs {prev:.3f}")
        rec("PASS" if o["recall"] > 0.2 else "WARN", "early-warning (onset) recall", f"recall {o['recall']:.2f}, precision {o['precision']:.2f}, FPR {o['false_positive_rate']:.3f}")
        lt = ev["lead_time"]
        rec("PASS" if lt["warned_early"] > 0 else "WARN", "warns before attacks", f"{lt['warned_early']} early / {lt['detected_late']} late / {lt['missed']} missed of {lt['forecastable_episodes']} episodes; mean lead {lt['mean_lead_seconds']}")
        sm = ev["stage"]
        rec("PASS", "stage evaluation scope", f"scored classes: {', '.join(sm['classes_with_support'])}")
    else:
        rec("WARN", "evaluation.json", "missing -> run scripts/evaluate.py")

    # 5. replay + warning engine on hold-out attack data -----------------------------------------------
    from src.replay import ReplaySession
    c2 = load_config(); c2["window"] = fc.meta["window"]; c2["split"] = fc.meta["split"]
    part = split_labels(st, c2)
    tmask = (part == "test")
    from src.models.sequences import holdout_blocks
    blocks = holdout_blocks(st, part, "test", fc.L + fc.H + 5)
    if blocks:
        ho = pd.concat([st.iloc[i].assign(segment=n) for n, i in enumerate(blocks)], ignore_index=True)
        rs = ReplaySession(ho, fc, cfg)
        n = rs.step(10 ** 7)
        al = rs.alerts_df()
        atk = int(ho["is_attack"].sum())
        rec("PASS" if len(al) > 0 else "WARN", "replay over HOLD-OUT blocks raises alerts",
            f"{n:,} windows ({atk} labelled attack windows) -> {len(al)} alerts" + (f", severities {al['peak_severity'].value_counts().to_dict()}" if len(al) else ""))
        if len(al):
            rec("PASS" if al["opened_time"].is_monotonic_increasing else "WARN", "alerts ordered in time", f"{len(al)} alerts")
    else:
        rec("WARN", "hold-out replay", "no hold-out blocks found")
    rs = ReplaySession(st, fc, cfg)
    rs.set_range(0, min(200, len(rs.ends) - 1)); rs.step(5); rs.reset()
    rec("PASS" if rs.cursor == rs.lo - 1 and not rs.engine.alerts else "FAIL", "replay start/step/reset", "")

    # 6. upload paths ------------------------------------------------------------------------------
    from src.service import states_from_upload
    for name in ("test_flows_cic_format.csv", "sample_traffic.pcap"):
        p = ROOT / "data" / "sample" / name
        if not p.exists():
            rec("WARN", f"upload test {name}", "file not found"); continue
        try:
            s2, info = states_from_upload(p.name, p.read_bytes(), cfg)
            r = fc.validate(s2)
            ok = r["ok"] and len(s2) >= fc.L + 2
            rec("PASS" if ok else "WARN", f"upload path: {name}", f"{info['kind']}, {len(s2)} windows" + ("" if ok else f" (need >= {fc.L + 2})"))
        except Exception as e:
            rec("FAIL", f"upload path: {name}", str(e)[:160])
    finish()


def finish():
    print("\n" + "=" * 60)
    print(f"RESULT: {RES.count('PASS')} passed, {RES.count('WARN')} warnings, {RES.count('FAIL')} failed")
    print("WORKING" if "FAIL" not in RES else "NOT WORKING - fix the FAIL lines above")
    sys.exit(1 if "FAIL" in RES else 0)


if __name__ == "__main__":
    main()
