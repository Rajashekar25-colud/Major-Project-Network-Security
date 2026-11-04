"""Time-window network-state construction (chunk-safe, mergeable statistics).

Flows are aggregated into fixed windows. All per-window statistics are kept as
mergeable sums / maxima / sets, so a window that spans several CSV chunks is
aggregated exactly once and no raw flow is ever kept in memory.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .flows import iter_flow_chunks
from .schema import BENIGN, LabelMapper, feature_names

SUM_COLS = [
    "n", "fp", "bp", "fb", "bb", "dur", "dur2", "iatm", "iats", "win", "win2", "win_n",
    "ttl", "ttl2", "ttl_n", "retr", "plm", "pls", "pay", "pay2",
    "syn", "ack", "rst", "fin", "psh", "urg", "failed", "small", "wk",
    "tcp", "udp", "icmp", "oth", "prate", "brate", "atk",
]
_PORT_BUCKETS_PREFIX = "pb_"


class WindowAccumulator:
    def __init__(self, config: dict[str, Any], source: str = "stream"):
        self.cfg = config
        self.source = source
        self.wsec = int(config["window"]["seconds"])
        self.buckets = list(config["features"]["port_buckets"])
        self.wk_ports = set(config["features"]["well_known_ports"])
        self.small_max = float(config["features"]["small_flow_max_packets"])
        self.rate_clip = float(config["features"]["rate_clip"])
        self.sum_cols = SUM_COLS + [f"{_PORT_BUCKETS_PREFIX}{p}" for p in self.buckets] + [f"{_PORT_BUCKETS_PREFIX}other"]
        self.sums: dict[pd.Timestamp, np.ndarray] = {}
        self.imax: dict[pd.Timestamp, float] = {}
        self.sets: dict[pd.Timestamp, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        self.stages: dict[pd.Timestamp, Counter] = defaultdict(Counter)
        self.flows_seen = 0
        self.has = {"ip": False, "sport": False, "ttl": False, "retr": False}

    # ------------------------------------------------------------------
    def add(self, c: pd.DataFrame) -> None:
        if c.empty:
            return
        self.flows_seen += len(c)
        d = pd.DataFrame(index=c.index)
        d["window"] = c["timestamp"].dt.floor(f"{self.wsec}s")
        d["n"] = 1.0
        for k in ("fp", "bp", "fb", "bb", "dur", "iatm", "iats", "syn", "ack", "rst", "fin", "psh", "urg", "plm", "pls"):
            d[k] = c[k].fillna(0.0)
        d["dur2"] = d["dur"] ** 2
        wv = c["win"]
        d["win_n"] = wv.notna().astype(float)
        d["win"] = wv.fillna(0.0)
        d["win2"] = d["win"] ** 2
        tv = c["ttl"]
        d["ttl_n"] = tv.notna().astype(float)
        d["ttl"] = tv.fillna(0.0)
        d["ttl2"] = d["ttl"] ** 2
        d["retr"] = c["retr"].fillna(0.0)
        d["pay"] = d["fb"] + d["bb"]
        d["pay2"] = d["pay"] ** 2
        pk = d["fp"] + d["bp"]
        d["failed"] = (d["bp"] == 0).astype(float)
        d["small"] = (pk <= self.small_max).astype(float)
        port = c["dst_port"]
        d["wk"] = port.isin(self.wk_ports).astype(float)
        proto = c["protocol"]
        d["tcp"] = (proto == 6).astype(float)
        d["udp"] = (proto == 17).astype(float)
        d["icmp"] = (proto == 1).astype(float)
        d["oth"] = (~proto.isin([1, 6, 17])).astype(float)
        dsec = d["dur"].clip(lower=1e-3)
        d["prate"] = (pk / dsec).clip(upper=self.rate_clip)
        d["brate"] = (d["pay"] / dsec).clip(upper=self.rate_clip)
        atk = (c["stage"] != BENIGN)
        d["atk"] = atk.astype(float)
        for p in self.buckets:
            d[f"{_PORT_BUCKETS_PREFIX}{p}"] = (port == p).astype(float)
        d[f"{_PORT_BUCKETS_PREFIX}other"] = (~port.isin(self.buckets)).astype(float)

        g = d.groupby("window", sort=False)
        sums = g[self.sum_cols].sum()
        maxs = c.assign(window=d["window"]).groupby("window", sort=False)["iatx"].max()
        for ts, row in zip(sums.index, sums.to_numpy()):
            if ts in self.sums:
                self.sums[ts] += row
            else:
                self.sums[ts] = row.copy()
        for ts, v in maxs.items():
            self.imax[ts] = max(self.imax.get(ts, 0.0), float(v) if pd.notna(v) else 0.0)

        # unique-value sets (mergeable across chunks)
        u = pd.DataFrame({"window": d["window"], "v": port.astype("int64")}).drop_duplicates()
        for ts, v in zip(u["window"], u["v"]):
            self.sets[ts]["dport"].add(v)
        if c["src_port"].notna().any():
            self.has["sport"] = True
            u = pd.DataFrame({"window": d["window"], "v": c["src_port"]}).dropna().drop_duplicates()
            for ts, v in zip(u["window"], u["v"]):
                self.sets[ts]["sport"].add(int(v))
        if c["src_ip"].notna().any() and c["dst_ip"].notna().any():
            self.has["ip"] = True
            u = pd.DataFrame({"window": d["window"], "s": c["src_ip"], "d": c["dst_ip"]}).dropna().drop_duplicates()
            for ts, s, dd in zip(u["window"], u["s"], u["d"]):
                st = self.sets[ts]
                st["sip"].add(s); st["dip"].add(dd); st["pair"].add((s, dd))
        if c["ttl"].notna().any():
            self.has["ttl"] = True
        if c["retr"].notna().any():
            self.has["retr"] = True
        if atk.any():
            cnt = pd.DataFrame({"window": d["window"][atk], "stage": c["stage"][atk]}).value_counts()
            for (ts, stg), k in cnt.items():
                self.stages[ts][stg] += int(k)

    # ------------------------------------------------------------------
    def finalize(self) -> pd.DataFrame:
        cfg = self.cfg
        lab = cfg["labeling"]
        names = feature_names(cfg)
        if not self.sums:
            return pd.DataFrame(columns=["timestamp"] + names + ["attack_flows", "attack_fraction", "stage", "is_attack", "source", "filled"])
        wins = sorted(self.sums)
        S = pd.DataFrame(np.vstack([self.sums[w] for w in wins]), columns=self.sum_cols)
        n = S["n"].to_numpy()
        n1 = np.maximum(n, 1.0)

        def div(a, b):
            b = np.asarray(b, dtype=float)
            return np.divide(np.asarray(a, dtype=float), b, out=np.zeros(len(b)), where=b > 0)

        def var(s1, s2, cnt):
            m = div(s1, cnt)
            return np.maximum(div(s2, cnt) - m * m, 0.0)

        F = pd.DataFrame({"timestamp": wins})
        F["flow_count"] = n
        F["unique_src_ips"] = [len(self.sets[w].get("sip", ())) for w in wins]
        F["unique_dst_ips"] = [len(self.sets[w].get("dip", ())) for w in wins]
        F["unique_src_ports"] = [len(self.sets[w].get("sport", ())) for w in wins]
        F["unique_dst_ports"] = [len(self.sets[w].get("dport", ())) for w in wins]
        F["packets_total"] = S["fp"] + S["bp"]
        F["bytes_total"] = S["fb"] + S["bb"]
        F["pkt_rate_mean"] = div(S["prate"], n)
        F["byte_rate_mean"] = div(S["brate"], n)
        F["duration_mean"] = div(S["dur"], n)
        F["duration_std"] = np.sqrt(var(S["dur"], S["dur2"], n))
        F["iat_mean"] = div(S["iatm"], n)
        F["iat_std"] = div(S["iats"], n)
        F["iat_max"] = [self.imax.get(w, 0.0) for w in wins]
        F["ttl_mean"] = div(S["ttl"], S["ttl_n"])
        F["ttl_var"] = var(S["ttl"], S["ttl2"], S["ttl_n"])
        F["tcp_window_mean"] = div(S["win"], S["win_n"])
        F["tcp_window_var"] = var(S["win"], S["win2"], S["win_n"])
        F["pkt_len_mean"] = div(S["plm"], n)
        F["pkt_len_std"] = div(S["pls"], n)
        F["payload_mean"] = div(S["pay"], n)
        F["payload_std"] = np.sqrt(var(S["pay"], S["pay2"], n))
        F["retransmissions"] = S["retr"]
        for k in ("syn", "ack", "rst", "fin", "psh", "urg"):
            F[f"{k}_count"] = S[k]
        F["syn_ratio"] = div(S["syn"], n)
        F["rst_ratio"] = div(S["rst"], n)
        F["inbound_outbound_byte_ratio"] = div(S["bb"], S["fb"])
        F["inbound_outbound_packet_ratio"] = div(S["bp"], S["fp"])
        nsrc = F["unique_src_ips"].to_numpy(dtype=float)
        F["distinct_dst_per_src_mean"] = [len(self.sets[w].get("pair", ())) / max(len(self.sets[w].get("sip", ())), 1) if self.has["ip"] else 0.0 for w in wins]
        F["failed_flow_ratio"] = div(S["failed"], n)
        F["small_flow_ratio"] = div(S["small"], n)
        breadth = F["unique_dst_ports"].to_numpy() / (n + 10.0)
        F["scan_score"] = np.clip(breadth * (0.5 + 0.5 * F["failed_flow_ratio"].to_numpy()) * F["small_flow_ratio"].to_numpy() * 2.0, 0, 1)
        F["well_known_dst_port_ratio"] = div(S["wk"], n)
        F["protocol_tcp"] = div(S["tcp"], n)
        F["protocol_udp"] = div(S["udp"], n)
        F["protocol_icmp"] = div(S["icmp"], n)
        F["protocol_other"] = div(S["oth"], n)
        for p in self.buckets:
            F[f"dst_port_{p}"] = div(S[f"{_PORT_BUCKETS_PREFIX}{p}"], n)
        F["dst_port_other"] = div(S[f"{_PORT_BUCKETS_PREFIX}other"], n)

        atk = S["atk"].to_numpy()
        frac = div(atk, n)
        is_attack = (atk >= lab["min_attack_flows"]) & (frac >= lab["min_attack_fraction"])
        stage = []
        for w, ia in zip(wins, is_attack):
            c = self.stages.get(w)
            stage.append(c.most_common(1)[0][0] if (ia and c) else BENIGN)
        F["attack_flows"] = atk
        F["attack_fraction"] = frac
        F["stage"] = stage
        F["is_attack"] = is_attack
        F["source"] = self.source
        F["filled"] = False
        F[names] = F[names].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float64")
        return F


# ----------------------------------------------------------------------
def states_from_flow_csv(path: str | Path, config: dict[str, Any], chunksize: int = 200_000,
                         mapper: LabelMapper | None = None,
                         progress: Callable[[str], None] | None = print) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream one flow CSV into window states. Returns (states, info)."""
    path = Path(path)
    t0 = time.time()
    mapper = mapper or LabelMapper(config)
    acc = WindowAccumulator(config, source=path.name)
    last = {"t": t0}

    def _p(total):
        if progress and time.time() - last["t"] > 5:
            last["t"] = time.time()
            progress(f"    {path.name}: {total:,} rows, {len(acc.sums):,} windows so far")
    for chunk in iter_flow_chunks(path, config, mapper, chunksize, _p):
        acc.add(chunk)
    states = acc.finalize()
    info = {"file": path.name, "size_bytes": path.stat().st_size, "flow_rows": int(acc.flows_seen),
            "windows": int(len(states)), "seconds": round(time.time() - t0, 1),
            "has_ip": acc.has["ip"], "has_src_port": acc.has["sport"], "has_ttl": acc.has["ttl"],
            "has_retransmissions": acc.has["retr"]}
    if len(states):
        info["start"] = str(states["timestamp"].min()); info["end"] = str(states["timestamp"].max())
    return states, info


def states_from_flows(flows: pd.DataFrame, config: dict[str, Any], source: str = "flows") -> pd.DataFrame:
    """Canonical flow frame (e.g. from PCAP) -> states."""
    acc = WindowAccumulator(config, source=source)
    acc.add(flows)
    return acc.finalize()


# ----------------------------------------------------------------------
def regularize(states: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Make each source's window grid regular and assign contiguous ``segment`` ids.

    * Missing windows inside a gap <= max_gap_windows become explicit all-zero
      windows (a quiet network is information, not missing data).
    * Longer gaps, and different sources (capture days), start a new segment, so a
      sequence never spans an unrelated time discontinuity.
    """
    names = feature_names(config)
    wsec = pd.Timedelta(seconds=int(config["window"]["seconds"]))
    max_gap = int(config["window"]["max_gap_windows"])
    parts, seg_base = [], 0
    for source, g in states.groupby("source", sort=False):
        g = g.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        steps = ((g["timestamp"].diff() / wsec).round().fillna(1)).astype(int)
        seg = (steps > max_gap + 1).cumsum() + seg_base
        g["segment"] = seg.to_numpy()
        for sid, sg in g.groupby("segment", sort=True):
            grid = pd.date_range(sg["timestamp"].iloc[0], sg["timestamp"].iloc[-1], freq=wsec)
            full = sg.set_index("timestamp").reindex(grid)
            miss = full["segment"].isna()
            full.loc[miss, names] = 0.0
            full.loc[miss, ["attack_flows", "attack_fraction"]] = 0.0
            full.loc[miss, "stage"] = BENIGN
            full.loc[miss, "is_attack"] = False
            full["source"] = source
            full["segment"] = sid
            full["filled"] = miss.to_numpy()
            full.index.name = "timestamp"
            parts.append(full.reset_index())
        seg_base = int(seg.max()) + 1
    out = pd.concat(parts, ignore_index=True)
    out["is_attack"] = out["is_attack"].astype(bool)
    out["filled"] = out["filled"].astype(bool)
    out["segment"] = out["segment"].astype(int)
    return out


def prune_short_segments(states: pd.DataFrame, min_len: int) -> tuple[pd.DataFrame, int]:
    """Drop isolated fragments (e.g. stray timestamps) that are too short to form a sequence."""
    size = states.groupby("segment")["segment"].transform("size")
    keep = size >= min_len
    out = states[keep].copy()
    out["segment"] = pd.factorize(out["segment"])[0]
    return out.reset_index(drop=True), int((~keep).sum())


def build_states(paths: Iterable[str | Path], config: dict[str, Any], chunksize: int = 200_000,
                 progress: Callable[[str], None] | None = print) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Process several CSV files (one per capture day) -> regularised states + per-file info."""
    mapper = LabelMapper(config)
    parts, infos = [], []
    for p in paths:
        if progress:
            progress(f"[FILE] {Path(p).name}")
        s, info = states_from_flow_csv(p, config, chunksize, mapper, progress)
        if progress:
            progress(f"    -> {info['flow_rows']:,} flows -> {info['windows']:,} windows in {info['seconds']}s")
        if len(s):
            parts.append(s); infos.append(info)
    if not parts:
        raise ValueError("No usable rows were found in the provided file(s).")
    parts.sort(key=lambda s: s["timestamp"].min())
    states = regularize(pd.concat(parts, ignore_index=True), config)
    w = config["window"]
    msw = w.get("min_segment_windows", "auto")
    min_len = int(w["sequence_length"]) + int(w["forecast_horizon"]) + 1 if msw == "auto" else int(msw)
    nseg_before = int(states["segment"].nunique())
    states, dropped = prune_short_segments(states, min_len)
    if progress:
        progress(f"[SEGMENTS] {nseg_before} -> {states['segment'].nunique()} (dropped {dropped} windows in fragments shorter than {min_len})")
    if mapper.unmapped and progress:
        progress(f"[WARN] {sum(mapper.unmapped.values()):,} flows had unmapped labels -> "
                 f"'{config['labeling']['unknown_attack_stage']}': {dict(list(mapper.unmapped.items())[:8])}")
    infos.append({"unmapped_labels": mapper.unmapped, "segments_before_pruning": nseg_before, "windows_dropped_in_short_fragments": dropped})
    return states, infos
