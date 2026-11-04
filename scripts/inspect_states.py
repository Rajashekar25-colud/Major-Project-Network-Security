"""Diagnostics for a prepared state table: per-day coverage, segments, attack stages, split support.

    python scripts/inspect_states.py --states data\\states\\cicids2018_states.csv
"""
import argparse
import numpy as np
import pandas as pd
from _common import ROOT

from src.config import load_config
from src.models.sequences import sample_ends, split_labels
from src.service import load_states_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=str(ROOT / "data" / "states" / "cicids2018_states.csv"))
    a = ap.parse_args()
    cfg = load_config()
    st = load_states_csv(a.states, cfg)
    L, H = cfg["window"]["sequence_length"], cfg["window"]["forecast_horizon"]
    part = split_labels(st, cfg)
    print(f"{len(st):,} windows, {st['segment'].nunique()} segments, strategy={cfg['split']['strategy']}\n")
    print("=== per source file ===")
    for s, g in st.groupby("source", sort=False):
        sizes = g.groupby("segment").size()
        span_h = (g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 3600
        print(f"{s[:42]:44s} windows={len(g):5d} segs={len(sizes):3d} longest={sizes.max():5d} "
              f"span={span_h:5.1f}h flows/win(med)={g['flow_count'].median():7.0f} attack_win={int(g['is_attack'].sum()):5d}")
        stg = g.loc[g["is_attack"], "stage"].value_counts()
        if len(stg):
            print("      " + ", ".join(f"{k}={v}" for k, v in stg.items()))
    print("\n=== forecast samples per split and stage (stage of the window right after the sample) ===")
    for lab in ("train", "val", "test"):
        e = sample_ends(st, L, H, part, lab)
        nxt = st["stage"].where(st["is_attack"], "Benign").to_numpy()[e + 1] if len(e) else []
        vc = pd.Series(nxt).value_counts()
        print(f"{lab:5s} samples={len(e):6d}  " + ", ".join(f"{k}={v}" for k, v in vc.items()))
    print("\n=== 10 shortest and 5 longest segments (windows) ===")
    sz = st.groupby("segment").size().sort_values()
    print(sz.head(10).to_dict(), sz.tail(5).to_dict())


if __name__ == "__main__":
    main()
