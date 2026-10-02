"""Cut a replay test file out of the chronological HOLD-OUT of your prepared states.

The result (data/states/test_replay.csv) contains only windows the model never trained on,
chosen so that every attack stage that exists in the hold-out is represented. Load it in the
dashboard (sidebar -> Replay source -> test_replay.csv) and press Start: you should see
alerts open before/at attacks.

    python scripts/make_test_file.py
"""
import argparse

import numpy as np
import pandas as pd
from _common import ROOT

from src.config import load_config
from src.models.inference import Forecaster
from src.models.sequences import holdout_blocks, split_labels
from src.service import load_states_csv, save_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=str(ROOT / "data" / "states" / "cicids2018_states.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "states" / "test_replay.csv"))
    ap.add_argument("--runs", type=int, default=10, help="max number of hold-out blocks to include")
    a = ap.parse_args()
    cfg = load_config()
    fc = Forecaster.load(ROOT / "models")
    cfg["window"] = fc.meta["window"]; cfg["split"] = fc.meta["split"]
    st = load_states_csv(a.states, cfg)
    part = split_labels(st, cfg)
    runs = []
    for idx in holdout_blocks(st, part, "test", fc.L + fc.H + 5):
        sub = st.iloc[idx]
        runs.append((idx, set(sub.loc[sub["is_attack"], "stage"]), int(sub["is_attack"].sum())))
    if not runs:
        raise SystemExit("No hold-out blocks found. Train first (python scripts/train.py ...).")
    chosen, covered = [], set()
    for idx, stages, n_atk in sorted(runs, key=lambda r: -len(r[1] - covered) * 1000 - r[2]):
        if len(chosen) >= a.runs:
            break
        if n_atk == 0 and len(chosen) >= a.runs - 2:
            continue
        chosen.append(idx); covered |= stages
    chosen.sort(key=lambda i: st["timestamp"].iat[i[0]])
    parts = []
    for n, idx in enumerate(chosen):
        d = st.iloc[idx].copy(); d["segment"] = n; d["source"] = f"holdout-{n + 1:02d}-" + str(st['source'].iat[idx[0]])[:20]
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    save_states(out, a.out)
    print(f"[TEST FILE] {len(out):,} hold-out windows in {len(chosen)} blocks -> {a.out}")
    print(f"            attack windows: {int(out['is_attack'].sum())}; stages covered: {sorted(covered) or 'none'}")
    print("            Load it in the dashboard: Replay source -> test_replay.csv")


if __name__ == "__main__":
    main()
