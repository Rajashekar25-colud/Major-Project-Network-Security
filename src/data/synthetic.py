"""Synthetic CSE-CIC-IDS2018-*format* traffic generator.

PURPOSE: unit tests, CI, and an out-of-the-box demo model so the dashboard runs
before the real 6.4 GiB dataset is processed. It is NOT real network data and
any metric computed from it must never be reported as a research result.

Days contain benign background traffic plus multi-stage attack campaigns. Many
campaigns are preceded by an unlabeled reconnaissance-like probing ramp (exactly
how public datasets behave: the dataset labels only the attack itself), and there
are decoy probing blips that are NOT followed by attacks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CIC_COLUMNS = [
    "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt",
    "URG Flag Cnt", "Pkt Len Mean", "Pkt Len Std", "Init Fwd Win Byts", "Init Bwd Win Byts", "Label",
]

# kind -> (label, dst_port, protocol, flows/window range, fwd pkts, bwd pkts, bytes/pkt, dur_s)
ATTACKS = {
    "ssh_brute": ("SSH-Bruteforce", 22, 6, (25, 60), (3, 6), (2, 4), 90, (0.2, 2.0)),
    "ftp_brute": ("FTP-BruteForce", 21, 6, (25, 60), (3, 5), (2, 3), 70, (0.2, 1.5)),
    "dos_hulk": ("DoS attacks-Hulk", 80, 6, (150, 400), (4, 9), (0, 2), 400, (0.5, 8.0)),
    "ddos_loic": ("DDOS attack-LOIC-UDP", 80, 17, (200, 500), (2, 4), (0, 0), 512, (0.01, 0.5)),
    "bot": ("Bot", 8080, 6, (4, 10), (6, 12), (5, 10), 180, (20.0, 60.0)),
    "infil": ("Infilteration", 445, 6, (4, 12), (10, 40), (8, 30), 900, (5.0, 40.0)),
    "sqli": ("SQL Injection", 80, 6, (10, 25), (6, 14), (4, 10), 700, (0.5, 4.0)),
    "xss": ("Brute Force -XSS", 80, 6, (10, 25), (5, 12), (4, 9), 650, (0.5, 4.0)),
}


def _benign(rng, n, t0, wsec):
    ports = rng.choice([443, 80, 53, 22, 445, 25, 3389, 8080, 123, 1024 + rng.integers(0, 40000)],
                       size=n, p=[.38, .2, .16, .03, .04, .03, .02, .06, .05, .03])
    tcp = rng.random(n) < 0.78
    proto = np.where(ports == 53, 17, np.where(tcp, 6, 17))
    dur = rng.lognormal(-1.0, 1.6, n).clip(0, 110)
    fp = 1 + rng.geometric(0.25, n); bp = np.where(proto == 6, 1 + rng.geometric(0.3, n), rng.integers(0, 3, n))
    bpp = rng.lognormal(5.2, 0.8, n)
    return dict(ts=t0 + rng.random(n) * wsec, port=ports, proto=proto, dur=dur, fp=fp, bp=bp,
                fb=fp * bpp * 0.6, bb=bp * bpp, syn=(proto == 6) * 1, rst=(rng.random(n) < 0.03) * 1,
                fin=(proto == 6) * (rng.random(n) < 0.5) * 1, psh=(proto == 6) * (rng.random(n) < 0.5) * 1,
                ack=(proto == 6) * 1, label=np.array(["Benign"] * n, dtype=object),
                win=np.where(proto == 6, rng.choice([8192, 29200, 65535, 64240], n), -1))


def _probe(rng, n, t0, wsec):
    """Unlabeled reconnaissance-like probing: many tiny unanswered flows to varied ports."""
    ports = rng.integers(1, 10000, n)
    return dict(ts=t0 + rng.random(n) * wsec, port=ports, proto=np.full(n, 6), dur=rng.random(n) * 0.05,
                fp=np.ones(n), bp=np.zeros(n), fb=np.full(n, 44.0), bb=np.zeros(n), syn=np.ones(n),
                rst=(rng.random(n) < 0.5) * 1, fin=np.zeros(n), psh=np.zeros(n), ack=np.zeros(n),
                label=np.array(["Benign"] * n, dtype=object), win=np.full(n, 1024))


def _attack(rng, kind, n, t0, wsec):
    label, port, proto, _, fpr, bpr, bpp, dr = ATTACKS[kind]
    fp = rng.integers(fpr[0], fpr[1] + 1, n); bp = rng.integers(bpr[0], bpr[1] + 1, n) if bpr[1] else np.zeros(n)
    dur = rng.uniform(dr[0], dr[1], n)
    return dict(ts=t0 + rng.random(n) * wsec, port=np.full(n, port), proto=np.full(n, proto), dur=dur, fp=fp, bp=bp,
                fb=fp * bpp * rng.uniform(.6, 1.2, n), bb=bp * bpp * rng.uniform(.5, 1.5, n),
                syn=np.full(n, 1 if proto == 6 else 0), rst=(rng.random(n) < (0.4 if "brute" in kind else 0.1)) * 1,
                fin=(rng.random(n) < 0.3) * 1, psh=(rng.random(n) < 0.6) * 1, ack=np.full(n, 1 if proto == 6 else 0),
                label=np.array([label] * n, dtype=object), win=np.full(n, 29200 if proto == 6 else -1))


def make_day(path: str | Path, seed: int, date: str = "14/02/2018", n_windows: int = 900,
             wsec: int = 10, with_ips: bool = False, messy: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp(pd.to_datetime(date, format="%d/%m/%Y")) + pd.Timedelta(hours=8)
    t_sec = np.arange(n_windows) * wsec
    pieces: list[dict] = []
    # attack campaigns
    plan = []
    w = int(rng.integers(30, 60))
    kinds = list(ATTACKS)
    while w < n_windows - 60:
        kind = kinds[int(rng.integers(0, len(kinds)))]
        length = int(rng.integers(8, 26)) if kind not in ("bot", "infil") else int(rng.integers(12, 30))
        prelude = int(rng.integers(4, 9)) if rng.random() < 0.7 else 0
        plan.append((w, length, kind, prelude))
        w += prelude + length + int(rng.integers(40, 110))
    # decoy probing blips (not followed by an attack)
    decoys = [int(x) for x in rng.integers(20, n_windows - 20, 12)]
    atk_windows = set()
    for (s, l, k, p) in plan:
        atk_windows.update(range(s - p, s + l + 2))
    decoys = [d for d in decoys if all(abs(d - x) > 4 for x in atk_windows)]
    for i in range(n_windows):
        t0 = float(t_sec[i])
        rate = 22 * (1 + 0.35 * np.sin(i / 60.0)) + rng.normal(0, 3)
        pieces.append(_benign(rng, max(int(rng.poisson(max(rate, 4))), 2), t0, wsec))
    for (s, l, k, p) in plan:
        for j in range(p):                    # probing ramp -> intensity rises
            n = int(8 + 60 * (j + 1) / max(p, 1) * rng.uniform(.7, 1.3))
            pieces.append(_probe(rng, n, float(t_sec[s - p + j]), wsec))
        lo, hi = ATTACKS[k][3]
        for j in range(l):
            ramp = min(1.0, 0.45 + 0.3 * j)
            n = int(rng.integers(lo, hi + 1) * ramp)
            if n > 0:
                pieces.append(_attack(rng, k, n, float(t_sec[s + j]), wsec))
    for d in decoys:
        for j in range(int(rng.integers(1, 4))):
            pieces.append(_probe(rng, int(rng.integers(15, 45)), float(t_sec[d + j]), wsec))

    cols = {k: np.concatenate([np.asarray(p[k]) for p in pieces]) for k in pieces[0]}
    df = pd.DataFrame(cols).sort_values("ts").reset_index(drop=True)
    n = len(df)
    ts = base + pd.to_timedelta(df["ts"], unit="s")
    dur_us = (df["dur"] * 1e6).round()
    npk = (df["fp"] + df["bp"]).clip(lower=1)
    iat_mean = (dur_us / npk).round()
    out = pd.DataFrame({
        "Dst Port": df["port"].astype(int), "Protocol": df["proto"].astype(int),
        "Timestamp": ts.dt.strftime("%d/%m/%Y %H:%M:%S"),
        "Flow Duration": dur_us.astype("int64"), "Tot Fwd Pkts": df["fp"].astype(int), "Tot Bwd Pkts": df["bp"].astype(int),
        "TotLen Fwd Pkts": df["fb"].round().astype(int), "TotLen Bwd Pkts": df["bb"].round().astype(int),
        "Flow Byts/s": ((df["fb"] + df["bb"]) / df["dur"].clip(lower=1e-6)).round(2),
        "Flow Pkts/s": (npk / df["dur"].clip(lower=1e-6)).round(2),
        "Flow IAT Mean": iat_mean, "Flow IAT Std": (iat_mean * rng.uniform(.2, 1.4, n)).round(),
        "Flow IAT Max": (dur_us * rng.uniform(.4, 1.0, n)).round(),
        "FIN Flag Cnt": df["fin"].astype(int), "SYN Flag Cnt": df["syn"].astype(int), "RST Flag Cnt": df["rst"].astype(int),
        "PSH Flag Cnt": df["psh"].astype(int), "ACK Flag Cnt": df["ack"].astype(int), "URG Flag Cnt": 0,
        "Pkt Len Mean": ((df["fb"] + df["bb"]) / npk).round(2), "Pkt Len Std": ((df["fb"] + df["bb"]) / npk * .4).round(2),
        "Init Fwd Win Byts": df["win"].astype(int), "Init Bwd Win Byts": np.where(df["proto"] == 6, 219, -1),
        "Label": df["label"],
    })
    if with_ips:  # like Thuesday-20-02-2018: Flow ID / Src IP / Src Port / Dst IP present
        srcs = rng.integers(1, 60, n)
        attacker = out["Label"] != "Benign"
        src_ip = np.where(attacker, "172.31.70.4", "172.31.69." + pd.Series(srcs).astype(str))
        out.insert(0, "Flow ID", [f"f{i}" for i in range(n)])
        out.insert(1, "Src IP", src_ip)
        out.insert(2, "Src Port", rng.integers(1024, 65535, n))
        out.insert(3, "Dst IP", "172.31.69." + pd.Series(rng.integers(1, 30, n)).astype(str))
        out = out.rename(columns={"Dst Port": "Dst Port"})
    if messy:     # embedded header row, Infinity, negative durations (all seen in the real files)
        out = out.astype(object)
        out.loc[rng.integers(0, n, 20), "Flow Byts/s"] = "Infinity"
        out.loc[rng.integers(0, n, 20), "Flow Duration"] = -1187300000
        hdr = pd.DataFrame([list(out.columns)], columns=out.columns)
        mid = n // 2
        out = pd.concat([out.iloc[:mid], hdr, out.iloc[mid:]], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return {"file": str(path), "rows": int(len(out)), "campaigns": [(int(s), int(l), k, int(p)) for s, l, k, p in plan]}


DAY_NAMES = [("Wednesday-14-02-2018", "14/02/2018"), ("Thursday-15-02-2018", "15/02/2018"),
             ("Friday-16-02-2018", "16/02/2018"), ("Thuesday-20-02-2018", "20/02/2018"),
             ("Wednesday-21-02-2018", "21/02/2018"), ("Thursday-22-02-2018", "22/02/2018")]


def make_demo_dataset(out_dir: str | Path, n_days: int = 6, n_windows: int = 900, seed: int = 7) -> list[dict]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    infos = []
    for i, (name, date) in enumerate(DAY_NAMES[:n_days]):
        infos.append(make_day(out / f"{name}_TrafficForML_CICFlowMeter.csv", seed + i, date, n_windows,
                              with_ips=(name.startswith("Thuesday")), messy=(i % 3 == 1)))
    return infos
