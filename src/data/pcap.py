"""PCAP ingestion (pure Python, no scapy needed for classic .pcap) -> canonical flows.

Extracts the packet-level telemetry named in the project abstract:
TTL, TCP window size, payload size, packet inter-arrival time, retransmissions
and TCP flags, plus 5-tuple flow statistics. Flows are bidirectional; the sender
of the first packet is the forward direction.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .schema import BENIGN

_MAGIC = {b"\xd4\xc3\xb2\xa1": ("<", 1e-6), b"\xa1\xb2\xc3\xd4": (">", 1e-6),
          b"\x4d\x3c\xb2\xa1": ("<", 1e-9), b"\xa1\xb2\x3c\x4d": (">", 1e-9)}
_PCAPNG = b"\x0a\x0d\x0d\x0a"


def _iter_pcap(path: Path) -> Iterator[tuple[float, bytes, int]]:
    with open(path, "rb") as f:
        head = f.read(24)
        if len(head) < 24 or head[:4] not in _MAGIC:
            raise ValueError("Not a classic .pcap file")
        end, res = _MAGIC[head[:4]]
        linktype = struct.unpack(end + "I", head[20:24])[0] & 0x0FFFFFFF
        while True:
            h = f.read(16)
            if len(h) < 16:
                return
            sec, frac, caplen, _ = struct.unpack(end + "IIII", h)
            yield sec + frac * res, f.read(caplen), linktype


def _iter_pcapng(path: Path) -> Iterator[tuple[float, bytes, int]]:
    """Minimal pcapng reader (SHB / IDB / EPB / SPB) - enough for Wireshark / tcpdump captures."""
    end = "<"
    ifaces: list[tuple[int, float]] = []
    with open(path, "rb") as f:
        while True:
            h = f.read(8)
            if len(h) < 8:
                return
            btype = struct.unpack(end + "I", h[:4])[0]
            if h[:4] == _PCAPNG:                         # section header: learn byte order
                blen_raw = h[4:8]
                bom = f.read(4)
                end = "<" if struct.unpack("<I", bom)[0] == 0x1A2B3C4D else ">"
                blen = struct.unpack(end + "I", blen_raw)[0]
                f.read(blen - 12); ifaces = []
                continue
            blen = struct.unpack(end + "I", h[4:8])[0]
            body = f.read(max(blen - 12, 0)); f.read(4)
            if btype == 1 and len(body) >= 8:            # interface description
                lt = struct.unpack(end + "H", body[:2])[0]; res = 1e-6
                o = 8
                while o + 4 <= len(body):
                    code, ln = struct.unpack(end + "HH", body[o:o + 4])
                    if code == 0:
                        break
                    if code == 9 and ln >= 1:            # if_tsresol
                        v = body[o + 4]
                        res = 2.0 ** -(v & 0x7F) if v & 0x80 else 10.0 ** -(v & 0x7F)
                    o += 4 + ((ln + 3) & ~3)
                ifaces.append((lt, res))
            elif btype == 6 and len(body) >= 20:         # enhanced packet
                iid, th, tl, cap, _ = struct.unpack(end + "IIIII", body[:20])
                lt, res = ifaces[iid] if iid < len(ifaces) else (1, 1e-6)
                yield ((th << 32) | tl) * res, body[20:20 + cap], lt
            elif btype == 3 and len(body) >= 4:          # simple packet
                lt, _ = ifaces[0] if ifaces else (1, 1e-6)
                yield 0.0, body[4:], lt


def _iter_scapy(path: Path) -> Iterator[tuple[float, bytes, int]]:  # pragma: no cover
    from scapy.utils import PcapReader
    with PcapReader(str(path)) as r:
        for p in r:
            yield float(p.time), bytes(p), 1


def _ip_offset(buf: bytes, lt: int) -> int | None:
    """Offset of the IP header for a link type (None = not IP / unknown)."""
    if lt == 1:                                         # Ethernet (+ optional 802.1Q / QinQ)
        if len(buf) < 14:
            return None
        et = struct.unpack("!H", buf[12:14])[0]; off = 14
        while et in (0x8100, 0x88A8) and len(buf) >= off + 4:
            et = struct.unpack("!H", buf[off + 2:off + 4])[0]; off += 4
        return off if et in (0x0800, 0x86DD) else None
    if lt == 0:                                         # BSD loopback / null
        return 4
    if lt == 113:                                       # Linux cooked v1
        return 16
    if lt == 276:                                       # Linux cooked v2
        return 20
    if lt in (101, 12, 14, 228, 229):                   # raw IP
        return 0
    for off in (0, 14, 4, 16, 20):                      # unknown link type: sniff
        if len(buf) > off and (buf[off] >> 4) in (4, 6) and (
                (buf[off] >> 4 == 4 and (buf[off] & 15) >= 5) or (buf[off] >> 4 == 6 and len(buf) >= off + 40)):
            return off
    return None


def _ipv6_ext(ip: bytes) -> tuple[int, int]:
    nh, off = ip[6], 40
    for _ in range(8):
        if nh in (0, 43, 60) and len(ip) >= off + 2:
            nh, off = ip[off], off + (ip[off + 1] + 1) * 8
        elif nh == 44 and len(ip) >= off + 8:
            nh, off = ip[off], off + 8
        elif nh == 51 and len(ip) >= off + 2:
            nh, off = ip[off], off + (ip[off + 1] + 2) * 4
        else:
            break
    return nh, off


def parse_packet(buf: bytes, linktype: int):
    """-> (proto, src, dst, sport, dport, ttl, win, flags, payload_len, seq, ip_len) or None."""
    off = _ip_offset(buf, linktype)
    if off is None:
        return None
    ip = buf[off:]
    if not ip:
        return None
    ver = ip[0] >> 4
    if ver == 4:
        if len(ip) < 20:
            return None
        ihl = (ip[0] & 0x0F) * 4
        total = struct.unpack("!H", ip[2:4])[0] or len(ip)
        ttl, proto = ip[8], ip[9]
        src = ".".join(map(str, ip[12:16])); dst = ".".join(map(str, ip[16:20]))
        l4 = ip[ihl:]
    elif ver == 6:
        if len(ip) < 40:
            return None
        import socket
        plen = struct.unpack("!H", ip[4:6])[0]
        ttl = ip[7]
        proto, l4off = _ipv6_ext(ip)
        src = socket.inet_ntop(socket.AF_INET6, bytes(ip[8:24])); dst = socket.inet_ntop(socket.AF_INET6, bytes(ip[24:40]))
        total = 40 + plen; ihl = l4off
        l4 = ip[l4off:]
    else:
        return None
    sport = dport = win = flags = seq = 0
    if proto == 6 and len(l4) >= 20:
        sport, dport, seq = struct.unpack("!HHI", l4[:8])
        doff = (l4[12] >> 4) * 4
        flags = l4[13]; win = struct.unpack("!H", l4[14:16])[0]
        payload = max(total - ihl - doff, 0)
    elif proto == 17 and len(l4) >= 8:
        sport, dport = struct.unpack("!HH", l4[:4]); payload = max(total - ihl - 8, 0)
    elif proto in (1, 58):
        payload = max(total - ihl - 8, 0)
    else:
        payload = max(total - ihl, 0)
    return proto, src, dst, sport, dport, ttl, win, flags, payload, seq, total


def open_capture(path: Path):
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic in _MAGIC:
        return _iter_pcap(path)
    if magic == _PCAPNG:
        return _iter_pcapng(path)
    try:
        return _iter_scapy(path)
    except ImportError:
        raise ValueError(f"{path.name}: unrecognised capture format (magic bytes {magic.hex()}). Expected .pcap or .pcapng.")


class _Flow:
    __slots__ = ("t0", "last", "src", "dst", "sport", "dport", "proto", "fp", "bp", "fb", "bb", "times", "ttls",
                 "win", "flags", "seen", "retr", "lens")

    def __init__(self, t, src, dst, sport, dport, proto):
        self.t0 = self.last = t; self.src, self.dst, self.sport, self.dport, self.proto = src, dst, sport, dport, proto
        self.fp = self.bp = self.fb = self.bb = 0
        self.times = [t]; self.ttls = []; self.win = None; self.flags = np.zeros(6); self.seen = set(); self.retr = 0
        self.lens = []


def pcap_to_flows(path: str | Path, config: dict[str, Any], idle_timeout: float = 120.0,
                  active_timeout: float = 600.0, max_packets: int | None = None) -> pd.DataFrame:
    path = Path(path)
    gen = open_capture(path)
    stats = {"packets": 0, "parsed": 0, "linktypes": set()}

    flows: dict[tuple, _Flow] = {}
    done: list[_Flow] = []
    n = 0
    for ts, buf, lt in gen:
        n += 1
        stats["packets"] += 1; stats["linktypes"].add(lt)
        if max_packets and n > max_packets:
            break
        p = parse_packet(buf, lt)
        if p is None:
            continue
        stats["parsed"] += 1
        proto, src, dst, sport, dport, ttl, win, flags, payload, seq, total = p
        fwd_key = (proto, src, sport, dst, dport); rev_key = (proto, dst, dport, src, sport)
        fl = flows.get(fwd_key); direction = 0
        if fl is None:
            fl = flows.get(rev_key); direction = 1
        if fl is not None and (ts - fl.last > idle_timeout or ts - fl.t0 > active_timeout):
            done.append(fl); flows.pop(fwd_key, None); flows.pop(rev_key, None); fl = None
        if fl is None:
            fl = _Flow(ts, src, dst, sport, dport, proto); flows[fwd_key] = fl; direction = 0
        fl.last = ts
        if len(fl.times) < 5000:
            fl.times.append(ts)
        fl.lens.append(total)
        if direction == 0:
            fl.fp += 1; fl.fb += payload; fl.ttls.append(ttl)
            if proto == 6:
                if fl.win is None or (flags & 0x02):
                    fl.win = win if fl.win is None else fl.win
                key = (seq, payload, flags & 0x07)
                if payload > 0 and key in fl.seen:
                    fl.retr += 1
                fl.seen.add(key)
        else:
            fl.bp += 1; fl.bb += payload
        if proto == 6:
            for j, bit in enumerate((0x02, 0x10, 0x04, 0x01, 0x08, 0x20)):   # syn ack rst fin psh urg
                if flags & bit:
                    fl.flags[j] += 1
    done.extend(flows.values())
    if not done:
        raise ValueError(
            f"No IP packets could be parsed from {path.name}: {stats['packets']} packet(s) read, link type(s) {sorted(stats['linktypes'])}. "
            "Supported: Ethernet(+VLAN), raw IP, loopback, Linux-cooked; IPv4 and IPv6; classic pcap and pcapng. "
            "If the packets are ARP/STP/other non-IP traffic only, there is nothing to analyse.")
    rows = []
    for fl in done:
        t = np.asarray(fl.times); iat = np.diff(t) if len(t) > 1 else np.zeros(1)
        L = np.asarray(fl.lens, dtype=float)
        rows.append({
            "timestamp": pd.Timestamp(fl.t0, unit="s"), "dst_port": fl.dport, "protocol": fl.proto,
            "src_ip": fl.src, "dst_ip": fl.dst, "src_port": float(fl.sport),
            "dur": float(fl.last - fl.t0), "fp": float(fl.fp), "bp": float(fl.bp), "fb": float(fl.fb), "bb": float(fl.bb),
            "iatm": float(iat.mean()), "iats": float(iat.std()), "iatx": float(iat.max()),
            "win": float(fl.win) if (fl.win is not None and fl.proto == 6) else np.nan,
            "ttl": float(np.mean(fl.ttls)) if fl.ttls else np.nan, "retr": float(fl.retr),
            "plm": float(L.mean()), "pls": float(L.std()),
            "syn": fl.flags[0], "ack": fl.flags[1], "rst": fl.flags[2], "fin": fl.flags[3], "psh": fl.flags[4], "urg": fl.flags[5],
            "raw_label": "unlabeled", "stage": BENIGN})
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


# ----------------------------------------------------------------------
def _frame(ts, src, dst, sport, dport, proto, ttl=64, flags=0x18, win=64240, payload=0, seq=1):
    ip_src = bytes(int(x) for x in src.split(".")); ip_dst = bytes(int(x) for x in dst.split("."))
    if proto == 6:
        l4 = struct.pack("!HHIIBBHHH", sport, dport, seq, 0, 5 << 4, flags, win, 0, 0) + b"\x00" * payload
    elif proto == 17:
        l4 = struct.pack("!HHHH", sport, dport, 8 + payload, 0) + b"\x00" * payload
    else:
        l4 = struct.pack("!BBHI", 8, 0, 0, 0) + b"\x00" * payload
    total = 20 + len(l4)
    ip = struct.pack("!BBHHHBBH", 0x45, 0, total, 0, 0, ttl, proto, 0) + ip_src + ip_dst
    eth = b"\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\x08\x00"
    return ts, eth + ip + l4


def write_pcap(path: str | Path, frames: list[tuple[float, bytes]]) -> None:
    frames = sorted(frames, key=lambda x: x[0])
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for ts, b in frames:
            f.write(struct.pack("<IIII", int(ts), int((ts % 1) * 1e6), len(b), len(b)) + b)


def make_sample_pcap(path: str | Path, seed: int = 3, minutes: float = 4.0) -> None:
    """Small synthetic capture: normal web/DNS traffic, then a port scan and an SSH brute force."""
    rng = np.random.default_rng(seed)
    fr: list[tuple[float, bytes]] = []
    t0 = 1_700_000_000.0
    clients = [f"10.0.0.{i}" for i in range(10, 30)]
    server, dns = "10.0.0.5", "10.0.0.2"
    for i in range(int(minutes * 60)):
        for _ in range(int(rng.poisson(3))):
            c = clients[int(rng.integers(0, len(clients)))]; t = t0 + i + rng.random(); sp = int(rng.integers(20000, 60000))
            dp = int(rng.choice([443, 80, 443, 8080]))
            pl = int(rng.integers(200, 1200))
            fr += [_frame(t, c, server, sp, dp, 6, 64, 0x02, 64240), _frame(t + .01, server, c, dp, sp, 6, 60, 0x12, 29200),
                   _frame(t + .02, c, server, sp, dp, 6, 64, 0x10, 64240),
                   _frame(t + .05, c, server, sp, dp, 6, 64, 0x18, 64240, payload=pl, seq=100),
                   _frame(t + .06, server, c, dp, sp, 6, 60, 0x18, 29200, payload=int(rng.integers(500, 1400)), seq=900),
                   _frame(t + .30, c, server, sp, dp, 6, 64, 0x11, 64240)]
            if rng.random() < 0.03:    # retransmission of the data segment
                fr.append(_frame(t + .35, c, server, sp, dp, 6, 64, 0x18, 64240, payload=pl, seq=100))
        if i % 5 == 0:
            c = clients[int(rng.integers(0, len(clients)))]
            fr += [_frame(t0 + i, c, dns, int(rng.integers(30000, 60000)), 53, 17, 64, payload=40),
                   _frame(t0 + i + .01, dns, c, 53, 30000, 17, 63, payload=90)]
    atk = "10.0.0.66"
    ts = t0 + 90                                  # reconnaissance: SYN scan across many ports (unanswered)
    for j, port in enumerate(rng.permutation(np.arange(1, 1500))[:700]):
        fr.append(_frame(ts + j * 0.03, atk, server, 41000 + j % 20, int(port), 6, 47, 0x02, 1024))
        if port in (22, 80, 443):
            fr.append(_frame(ts + j * 0.03 + .005, server, atk, int(port), 41000 + j % 20, 6, 60, 0x12, 29200))
    ts = t0 + 140                                 # credential access: SSH brute force, many short sessions
    for j in range(180):
        sp = 50000 + j; t = ts + j * 0.25
        fr += [_frame(t, atk, server, sp, 22, 6, 47, 0x02, 29200), _frame(t + .01, server, atk, 22, sp, 6, 60, 0x12, 29200),
               _frame(t + .02, atk, server, sp, 22, 6, 47, 0x18, 29200, payload=64, seq=5),
               _frame(t + .06, server, atk, 22, sp, 6, 60, 0x04, 0)]
    write_pcap(path, fr)
