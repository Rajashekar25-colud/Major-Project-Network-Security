"""Cross-dataset test: score the trained model on a DIFFERENT dataset (e.g. UNSW-NB15) it never saw.

    python scripts/prepare_states.py --data <folder with UNSW-NB15_1.csv ...> --out data/states/unsw_states.csv
    python scripts/cross_dataset_eval.py --states data/states/unsw_states.csv
Every window of the other dataset is treated as test data. Stages the model never learned are reported as 'unseen'.
"""
import argparse
import json

import numpy as np
from _common import ROOT

from src.config import load_config
from src.evaluation.evaluate import evaluate_model
from src.models.inference import Forecaster
from src.product.learning import drift_report
from src.service import load_states_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--models", default=str(ROOT / "models"))
    a = ap.parse_args()
    cfg = load_config()
    fc = Forecaster.load(a.models)
    cfg["window"] = fc.meta["window"]
    st = load_states_csv(a.states, cfg)
    rep = fc.validate(st)
    if not rep["ok"]:
        raise SystemExit(f"Feature mismatch: {rep['missing'][:6]}")
    part = np.array(["test"] * len(st), dtype=object)
    ev = evaluate_model(fc, st, part, fc.scale(st), cfg, "test", with_baselines=False)
    dr = drift_report(fc, st)
    out = {"dataset": a.states, "model": fc.meta.get("trained_at"), "evaluation": ev, "drift": dr.get("overall"), "drift_advice": dr.get("advice")}
    (ROOT / "models" / "cross_dataset_eval.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    f, o, lt = ev["forecast"], ev["onset"], ev["lead_time"]
    print(f"windows={len(st):,} samples={ev['n_samples']:,} drift={dr.get('overall')}")
    print(f"forecast  acc={f['accuracy']:.3f} P={f['precision']:.3f} R={f['recall']:.3f} F1={f['f1']:.3f} FPR={f['false_positive_rate']:.3f}")
    print(f"onset     P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f}   lead: early={lt['warned_early']} late={lt['detected_late']} missed={lt['missed']}")
    print(f"stages with support: {ev['stage'].get('classes_with_support')}  unseen-stage samples: {ev['stage'].get('unseen_stage_samples')}")
    print("saved models/cross_dataset_eval.json")


if __name__ == "__main__":
    main()
