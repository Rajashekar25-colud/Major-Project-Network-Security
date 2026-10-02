"""Chunked, schema-tolerant flow-CSV ingestion.

Works with the processed CSE-CIC-IDS2018 CICFlowMeter files (with or without
Flow ID / Src IP / Src Port / Dst IP), other CICFlowMeter exports and generic
NetFlow-style CSVs. Only timestamp, destination port and protocol are mandatory.

Never loads a whole file: files are read in chunks with ``usecols`` so that only
the needed columns are parsed (multi-GB files fit a 16 GB laptop).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from .schema import BENIGN, LabelMapper, check_required, resolve_columns

PROTO_NAMES = {"tcp": 6, "udp": 17, "icmp": 1, "icmpv6": 58, "sctp": 132, "gre": 47}
# Standard UNSW-NB15 column order, used for the dataset's headerless raw files.
UNSW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts",
    "swin", "dwin", "stcpb", "dtcpb", " smeansz", "dmeansz", "trans_depth",
    "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt",
    "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
    "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
    "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "Label",
]
_TS_FORMATS = [
    "%d/%m/%Y %H:%M:%S",      # CSE-CIC-IDS2018 (day first)
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %I:%M:%S %p",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p",
    "%Y/%m/%d %H:%M:%S",
]

CANON_NUMERIC = ["dst_port", "protocol", "dur", "fp", "bp", "fb", "bb", "iatm", "iats", "iatx",
                 "win", "ttl", "retr", "plm", "pls", "syn", "ack", "rst", "fin", "psh", "urg"]


class TimestampParser:
    """Parses a timestamp column; remembers the format that worked for a file."""

    def __init__(self, forced_format: str | None = None):
        self.fmt = forced_format

    def __call__(self, s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors="coerce")
            unit = "ms" if v.dropna().abs().median() > 1e11 else "s"
            return pd.to_datetime(v, unit=unit, errors="coerce")
        s = s.astype(str).str.strip()
        if self.fmt:
            return pd.to_datetime(s, format=self.fmt, errors="coerce")
        best, best_ok = None, -1.0
        sample = s.iloc[: min(len(s), 2000)]
        for f in _TS_FORMATS:
            ok = pd.to_datetime(sample, format=f, errors="coerce").notna().mean()
            if ok > best_ok:
                best, best_ok = f, ok
            if ok > 0.98:
                break
        if best_ok >= 0.5:
            self.fmt = best
            return pd.to_datetime(s, format=best, errors="coerce")
        return pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")


def _num(df: pd.DataFrame, col: str | None, n: int, default: float = 0.0) -> pd.Series:
    if col is None:
        return pd.Series(default, index=df.index, dtype="float64")
    v = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return v.replace([np.inf, -np.inf], np.nan)


def canonicalize_chunk(raw: pd.DataFrame, colmap: dict[str, str], config: dict[str, Any],
                       ts_parser: TimestampParser, mapper: LabelMapper) -> pd.DataFrame:
    """Raw CSV chunk -> canonical flow frame (documented, unit-normalised columns)."""
    n = len(raw)
    units = config["units"]
    out = pd.DataFrame(index=raw.index)
    out["timestamp"] = ts_parser(raw[colmap["timestamp"]])
    out["dst_port"] = _num(raw, colmap["dst_port"], n)

    proto = raw[colmap["protocol"]]
    pnum = pd.to_numeric(proto, errors="coerce")
    if pnum.isna().mean() > 0.5:      # textual protocols (tcp/udp/icmp)
        pnum = proto.astype(str).str.strip().str.lower().map(PROTO_NAMES).fillna(0)
    out["protocol"] = pnum.astype("float64").fillna(0)

    def col(name, default=0.0):
        return _num(raw, colmap.get(name), n, default)

    out["dur"] = (col("flow_duration").clip(lower=0) / units["duration_divisor"])
    out["fp"] = col("fwd_packets").clip(lower=0)
    out["bp"] = col("bwd_packets").clip(lower=0)
    out["fb"] = col("fwd_bytes").clip(lower=0)
    out["bb"] = col("bwd_bytes").clip(lower=0)
    out["iatm"] = col("iat_mean").clip(lower=0) / units["iat_divisor"]
    out["iats"] = col("iat_std").clip(lower=0) / units["iat_divisor"]
    out["iatx"] = col("iat_max").clip(lower=0) / units["iat_divisor"]
    # -1 means "not applicable" in CICFlowMeter window fields -> NaN (excluded from stats)
    w = col("win_fwd")
    out["win"] = w.where(w >= 0)
    out["ttl"] = col("ttl", np.nan) if "ttl" in colmap else np.nan
    out["retr"] = col("retransmissions", np.nan) if "retransmissions" in colmap else np.nan
    out["plm"] = col("pktlen_mean").clip(lower=0)
    out["pls"] = col("pktlen_std").clip(lower=0)
    for f in ("syn", "ack", "rst", "fin", "psh", "urg"):
        out[f] = col(f).clip(lower=0)

    for f, name in (("src_ip", "src_ip"), ("dst_ip", "dst_ip")):
        out[name] = raw[colmap[f]].astype(str) if f in colmap else None
    out["src_port"] = _num(raw, colmap["src_port"], n) if "src_port" in colmap else np.nan

    # Drop unusable rows FIRST (this also removes repeated header rows that some CIC files
    # contain, whose timestamp cannot be parsed) so their text never reaches the label mapper.
    keep = out["timestamp"].notna() & out["dst_port"].notna() & out["protocol"].notna()
    if "label" in colmap:
        rawlab_all = raw[colmap["label"]].astype(str).str.strip()
        keep &= rawlab_all.str.lower() != "label"
    out = out[keep].copy()
    if "label" in colmap:
        rawlab = rawlab_all[keep]
        out["raw_label"] = rawlab
        lut = {u: mapper(u) for u in rawlab.unique()}
        out["stage"] = rawlab.map(lut)
    else:
        out["raw_label"] = "unlabeled"
        out["stage"] = BENIGN
    return out


def iter_flow_chunks(path: str | Path, config: dict[str, Any], mapper: LabelMapper | None = None,
                     chunksize: int = 200_000,
                     progress: Callable[[int], None] | None = None) -> Iterator[pd.DataFrame]:
    """Yield canonical flow chunks from a CSV without loading it whole."""
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    headerless = len(header) == len(UNSW_COLUMNS) and not resolve_columns(header, config).get("timestamp")
    if headerless:
        header = UNSW_COLUMNS
    colmap = resolve_columns(header, config)
    if "attack_cat" in header:
        colmap["label"] = "attack_cat"
    if any(n in colmap for n in ("timestamp", "dst_port")) and (
        colmap.get("timestamp") == "Stime" or colmap.get("dst_port") == "dsport"
    ):
        config = {**config, "units": {**config["units"], "duration_divisor": 1.0, "iat_divisor": 1000.0}}
    check_required(colmap, path.name)
    mapper = mapper or LabelMapper(config)
    ts_parser = TimestampParser(config.get("timestamp_format"))
    usecols = list(dict.fromkeys(colmap.values()))
    total = 0
    reader = pd.read_csv(path, names=UNSW_COLUMNS if headerless else None,
                         header=None if headerless else "infer", usecols=usecols, chunksize=chunksize,
                         low_memory=False,
                         on_bad_lines="skip", encoding_errors="replace")
    for raw in reader:
        total += len(raw)
        chunk = canonicalize_chunk(raw, colmap, config, ts_parser, mapper)
        if progress:
            progress(total)
        if len(chunk):
            yield chunk


def describe_csv(path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Cheap header inspection used for data-health validation (no full read)."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    colmap = resolve_columns(header, config)
    missing_required = [f for f in ("timestamp", "dst_port", "protocol") if f not in colmap]
    optional_present = [f for f in ("src_ip", "dst_ip", "src_port", "ttl", "retransmissions", "label") if f in colmap]
    return {"columns": len(header), "resolved": colmap, "missing_required": missing_required,
            "optional_present": optional_present}
