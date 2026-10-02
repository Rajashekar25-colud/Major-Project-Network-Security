"""Train LSTM and Transformer on the same states and compare them (PPT 'Model Upgrade').

    python scripts/compare_models.py --states data/states/cicids2018_states.csv --epochs 20
Writes models/model_comparison.json (shown in the dashboard under Model & Learning). Does NOT touch the live model.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

from _common import ROOT

from src.config import load_config
from src.models.train import train_from_states
from src.service import load_states_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=str(ROOT / "data" / "states" / "cicids2018_states.csv"))
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()
    rows = []
    for kind in ("lstm", "transformer"):
        cfg = load_config(); cfg["model"]["type"] = kind; cfg["model"]["epochs"] = a.epochs
        states = load_states_csv(a.states, cfg)
        out = Path(tempfile.mkdtemp(prefix=f"cmp_{kind}_"))
        t0 = time.time()
        r = train_from_states(states, cfg, out, log=lambda *_: None, with_baselines=False)
        ev, meta = r["evaluation"], r["meta"]
        lt = ev["lead_time"]
        rows.append({"Model": "LSTM + attention" if kind == "lstm" else "Transformer encoder", "Parameters": meta["model"]["parameters"],
                     "Epochs run": meta["epochs_run"], "Train seconds": round(time.time() - t0),
                     "Accuracy": ev["forecast"]["accuracy"], "Precision": ev["forecast"]["precision"], "Recall": ev["forecast"]["recall"],
                     "F1": ev["forecast"]["f1"], "False-positive rate": ev["forecast"]["false_positive_rate"],
                     "Onset F1": ev["onset"]["f1"], "Warned early": lt["warned_early"], "Detected late": lt["detected_late"], "Missed": lt["missed"]})
        print(f"[{kind}] F1={rows[-1]['F1']:.3f} onset F1={rows[-1]['Onset F1']:.3f} early={lt['warned_early']} params={rows[-1]['Parameters']:,}")
    (ROOT / "models" / "model_comparison.json").write_text(json.dumps({"rows": rows, "states": a.states}, indent=2), encoding="utf-8")
    print("saved models/model_comparison.json")


if __name__ == "__main__":
    main()
