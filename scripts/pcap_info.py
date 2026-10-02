"""Show what is inside a capture file (format, link types, protocols) - use when a PCAP will not load.

    python scripts/pcap_info.py --pcap path\\to\\file.pcap
"""
import argparse
from collections import Counter
from pathlib import Path

from _common import ROOT  # noqa: F401

from src.data.pcap import open_capture, parse_packet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    a = ap.parse_args()
    p = Path(a.pcap)
    print(f"{p.name}: {p.stat().st_size:,} bytes, magic {p.read_bytes()[:4].hex()}")
    lts, protos, n, ok = Counter(), Counter(), 0, 0
    first = None
    for ts, buf, lt in open_capture(p):
        n += 1; lts[lt] += 1
        if first is None:
            first = buf[:32].hex()
        r = parse_packet(buf, lt)
        if r:
            ok += 1; protos[r[0]] += 1
    print(f"packets={n} parsed_as_ip={ok} link_types={dict(lts)} ip_protocols={dict(protos)}")
    print("first packet bytes:", first)


if __name__ == "__main__":
    main()
