"""Re-evaluate a trained model on the chronological per-day hold-out.

    python scripts/evaluate.py --states data/states/cicids2018_states.csv
"""
import argparse
import json
from _common import ROOT

from src.config import load_config
from src.evaluation.evaluate import evaluate_model
from src.models.inference import Forecaster
from src.models.sequences import split_labels
from src.service import load_states_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    fc = Forecaster.load(a.models)
    cfg["window"] = fc.meta["window"]; cfg["split"] = fc.meta["split"]      # evaluate exactly as trained
    states = load_states_csv(a.states, cfg)
    rep = fc.validate(states)
    if not rep["ok"]:
        raise SystemExit(f"Feature mismatch: missing {rep['missing'][:8]}")
    part = split_labels(states, cfg)
    ev = evaluate_model(fc, states, part, fc.scale(states), cfg, a.split)
    out = ROOT / "models" / "evaluation.json" if a.split == "test" else ROOT / "models" / f"evaluation_{a.split}.json"
    out.write_text(json.dumps(ev, indent=2, default=str), encoding="utf-8")
    f, o, lt = ev["forecast"], ev["onset"], ev["lead_time"]
    print(f"forecast  P={f['precision']:.3f} R={f['recall']:.3f} F1={f['f1']:.3f} FPR={f['false_positive_rate']:.3f}")
    print(f"onset     P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f} FPR={o['false_positive_rate']:.3f}")
    print(f"lead time episodes={lt['attack_episodes']} early={lt['warned_early']} late={lt['detected_late']} missed={lt['missed']} mean={lt['mean_lead_seconds']}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
