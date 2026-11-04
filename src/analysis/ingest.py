"""Upload validation: new traffic file -> network states + an honest report of what was / was not possible.
Never touches model weights or the scaler (the model is used read-only to check feature compatibility)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..models.sequences import inference_ends
from ..service import states_from_upload

SUPPORTED = (".csv", ".pcap", ".cap", ".pcapng")


@dataclass
class Check:
    name: str
    status: str            # ok | warn | fail
    detail: str = ""


@dataclass
class TrafficReport:
    filename: str
    ok: bool
    checks: list[Check] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    states: pd.DataFrame | None = None
    error: str | None = None


def _fail(rep: TrafficReport, name: str, detail: str) -> TrafficReport:
    rep.checks.append(Check(name, "fail", detail)); rep.ok = False; rep.error = detail
    return rep


def traffic_stats(states: pd.DataFrame, info: dict[str, Any], fc) -> dict[str, Any]:
    wsec = int(fc.wsec)
    ts = states["timestamp"]
    start, end = ts.min(), ts.max() + pd.Timedelta(seconds=wsec)
    seg_len = states.groupby("segment").size()
    return {"records": int(info.get("records", 0)), "start": start, "end": end, "duration_s": float((end - start).total_seconds()),
            "windows": int(len(states)), "window_seconds": wsec, "segments": int(states["segment"].nunique()),
            "gap_filled_windows": int(states["filled"].sum()), "forecastable_windows": int(len(inference_ends(states, fc.L))),
            "longest_segment_windows": int(seg_len.max()) if len(seg_len) else 0, "history_windows": int(fc.L),
            "horizon_windows": int(fc.H), "labelled": bool(info.get("labelled", False)), "kind": info.get("kind")}


def inspect_traffic(name: str, data: bytes, fc, cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> TrafficReport:
    rep = TrafficReport(filename=name, ok=True)
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        return _fail(rep, "File format recognized", f"Unsupported file type '{suffix or 'none'}'. Upload a CSV, PCAP or PCAPNG file.")
    if not data:
        return _fail(rep, "File format recognized", "The file is empty.")
    try:
        states, info = states_from_upload(name, data, cfg, progress)
    except Exception as e:
        return _fail(rep, "File format recognized", f"The file could not be read as network traffic: {e}")
    if states is None or len(states) == 0:
        return _fail(rep, "File format recognized", "No usable traffic records were found in the file.")
    rep.info, rep.states = info, states
    kind = {"flow-csv": "Flow CSV", "pcap": "Packet capture", "state-table": "Prepared network-state table"}.get(info["kind"], info["kind"])
    rep.checks.append(Check("File format recognized", "ok", kind))

    ts = states["timestamp"]
    if ts.isna().any() or not np.issubdtype(ts.dtype, np.datetime64):
        return _fail(rep, "Timestamp detected", "Timestamps could not be parsed for all records.")
    rep.checks.append(Check("Timestamp detected", "ok", f"{ts.min():%Y-%m-%d %H:%M:%S} to {ts.max():%Y-%m-%d %H:%M:%S}"))

    if info["kind"] == "flow-csv":
        opt = info.get("optional_fields", [])
        rep.checks.append(Check("Required traffic fields detected", "ok", "timestamp, destination port, protocol found; optional fields present: " + (", ".join(opt) if opt else "none")))
    elif info["kind"] == "pcap":
        rep.checks.append(Check("Required traffic fields detected", "ok", "derived from packet headers (ports, protocol, flags, TTL, window, payload, timing)"))
    else:
        rep.checks.append(Check("Required traffic fields detected", "ok", "pre-aggregated network-state table"))

    bad = 0
    for _, g in states.groupby("segment"):
        bad += int((np.diff(g["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64")) <= 0).sum())
    if bad:
        return _fail(rep, "Temporal ordering verified", f"{bad} time windows are out of order or duplicated.")
    rep.checks.append(Check("Temporal ordering verified", "ok", f"windows are chronological in {int(states['segment'].nunique())} continuous segment(s)"))

    r = fc.validate(states)
    if not r["ok"]:
        m = r["missing"]
        return _fail(rep, "Compatible feature representation",
                     f"Unable to analyze this file because {len(m)} required model feature(s) could not be derived: {', '.join(m[:10])}" + (" ..." if len(m) > 10 else ""))
    notes = []
    if r["unavailable_in_input"]:
        u = r["unavailable_in_input"]
        notes.append(f"{len(u)} feature(s) the model uses are empty in this file ({', '.join(u[:6])}" + (" ..." if len(u) > 6 else "") + ")")
    if r["novel_in_input"]:
        notes.append(f"{len(r['novel_in_input'])} packet-level feature(s) are present but ignored because the model was trained without them")
    rep.checks.append(Check("Compatible feature representation", "warn" if notes else "ok", "; ".join(notes) if notes else f"all {len(fc.names)} model features were derived"))
    rep.info["feature_report"] = r

    rep.stats = traffic_stats(states, info, fc)
    if rep.stats["forecastable_windows"] == 0:
        return _fail(rep, "Sufficient temporal history",
                     f"Insufficient temporal history: the model needs {fc.L} consecutive windows ({fc.L * fc.wsec} s) before it can forecast, "
                     f"but the longest continuous segment in this file has only {rep.stats['longest_segment_windows']} window(s).")
    rep.checks.append(Check("Sufficient temporal history", "ok", f"{rep.stats['forecastable_windows']:,} windows have the {fc.L} windows of history the model requires"))
    rep.checks.append(Check("No training performed", "ok", "this file is inference data only; model weights and scaler are not modified"))
    return rep
