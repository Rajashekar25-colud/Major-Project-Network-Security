"""Train the temporal world model, calibrate the warning threshold and evaluate on the hold-out.

From raw flow CSVs (chunked):   python scripts/train.py --data <folder>
From prepared states:           python scripts/train.py --states data/states/cicids2018_states.csv
"""
import argparse
from _common import ROOT, find_csvs

from src.config import load_config
from src.data.windows import build_states
from src.models.train import train_from_states
from src.service import load_states_csv, save_states


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="Folder or CSV of raw flow files")
    src.add_argument("--states", help="Prepared state table CSV (scripts/prepare_states.py)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-baselines", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.epochs:
        cfg["model"]["epochs"] = a.epochs
    info = {}
    if a.data:
        files = find_csvs(a.data)
        states, infos = build_states(files, cfg, a.chunksize)
        save_states(states, ROOT / "data" / "states" / "cicids2018_states.csv")
        info = {"raw_files": infos, "input": str(a.data)}
    else:
        states = load_states_csv(a.states, cfg)
        info = {"input": str(a.states)}
    train_from_states(states, cfg, a.models, dataset_info=info, with_baselines=not a.no_baselines)


if __name__ == "__main__":
    main()
