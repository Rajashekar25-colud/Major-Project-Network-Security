"""Create the synthetic CIC-IDS2018-format demo dataset + a demo model.

The demo model exists ONLY so the dashboard runs before the real dataset has been
processed. Its metadata is flagged synthetic_demo_data=true and the dashboard shows
a banner. NEVER report demo metrics as results.
"""
import argparse
import tempfile
from pathlib import Path

from _common import ROOT, find_csvs

from src.config import load_config
from src.data.synthetic import make_demo_dataset
from src.data.windows import build_states
from src.models.train import train_from_states
from src.service import save_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--windows", type=int, default=1500)
    ap.add_argument("--models", default=str(ROOT / "models"))
    a = ap.parse_args()
    cfg = load_config(); cfg["model"]["epochs"] = a.epochs
    with tempfile.TemporaryDirectory() as tmp:
        infos = make_demo_dataset(tmp, 6, a.windows)
        states, finfo = build_states(find_csvs(tmp), cfg, progress=print)
    save_states(states, ROOT / "data" / "states" / "demo_states.csv")
    train_from_states(states, cfg, a.models, dataset_info={"files": finfo, "note": "SYNTHETIC demo data"}, synthetic=True)


if __name__ == "__main__":
    main()
