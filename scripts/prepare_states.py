"""Stream CSE-CIC-IDS2018 (or any flow CSV) files into a network-state table.

Chunked: never loads a whole file. Output: data/states/<name>.csv
    python scripts/prepare_states.py --data "D:\\CSE-CIC-IDS2018\\Processed Traffic Data for ML Algorithms"
"""
import argparse
import json
from _common import ROOT, find_csvs  # noqa: F401

from src.config import load_config
from src.data.windows import build_states
from src.service import save_states


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Folder (searched recursively) or a single flow CSV")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "states" / "cicids2018_states.csv"))
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--pattern", default="*.csv")
    a = ap.parse_args()
    cfg = load_config(a.config)
    files = find_csvs(a.data, a.pattern)
    print(f"Found {len(files)} file(s):"); [print("  -", f.name) for f in files]
    states, infos = build_states(files, cfg, a.chunksize)
    save_states(states, a.out)
    info_path = str(a.out).replace(".csv", "_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(infos, f, indent=2, default=str)
    print(f"\n[STATES] {len(states):,} windows ({states['segment'].nunique()} segments) -> {a.out}")
    print(states.loc[states["is_attack"], "stage"].value_counts().to_string() or "  (no attack windows)")


if __name__ == "__main__":
    main()
