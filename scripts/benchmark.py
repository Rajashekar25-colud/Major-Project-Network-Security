"""Measure computational cost and scalability (synopsis: 'reduced computational complexity / improved scalability').

    python scripts/benchmark.py
Reports flow-ingestion throughput (chunked, constant memory), forecast throughput and peak memory.
"""
import argparse
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from _common import ROOT

from src.config import load_config
from src.data.synthetic import make_demo_dataset
from src.data.windows import build_states
from src.models.inference import Forecaster
from src.models.sequences import inference_ends


def rss_mb():
    try:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--windows", type=int, default=1500)
    a = ap.parse_args()
    cfg = load_config()
    with tempfile.TemporaryDirectory() as tmp:
        make_demo_dataset(tmp, a.days, a.windows)
        files = sorted(Path(tmp).glob("*.csv"))
        size = sum(f.stat().st_size for f in files) / 1e6
        t0 = time.time(); st, infos = build_states(files, cfg, chunksize=50_000, progress=None); dt = time.time() - t0
        rows = sum(i["flow_rows"] for i in infos if "flow_rows" in i)
    print(f"[INGEST]   {rows:,} flow rows ({size:.0f} MB) -> {len(st):,} windows in {dt:.1f}s = {rows / dt:,.0f} rows/s, {size / dt:.0f} MB/s (chunked, peak RSS {rss_mb():.0f} MB)")
    try:
        fc = Forecaster.load(ROOT / "models")
    except Exception as e:
        print("[FORECAST] no trained model:", e); return
    X = fc.scale(st); ends = inference_ends(st, fc.L)
    t0 = time.time(); fc.predict_batch(X, ends); dt = time.time() - t0
    print(f"[FORECAST] {len(ends):,} windows forecast in {dt:.2f}s = {len(ends) / dt:,.0f} windows/s "
          f"(one window = {fc.wsec}s of traffic -> {len(ends) / dt * fc.wsec / 3600:,.0f} hours of traffic per second of compute)")
    print(f"[MODEL]    {sum(p.numel() for p in fc.model.parameters()):,} parameters, peak RSS {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
