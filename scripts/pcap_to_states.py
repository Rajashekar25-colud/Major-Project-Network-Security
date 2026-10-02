"""Convert a PCAP into a packet-level state table (TTL, TCP window, retransmissions ...).

    python scripts/pcap_to_states.py --pcap data/sample/sample_traffic.pcap
"""
import argparse
from _common import ROOT

from src.config import load_config
from src.data.pcap import pcap_to_flows
from src.data.windows import regularize, states_from_flows
from src.service import save_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config()
    fl = pcap_to_flows(a.pcap, cfg)
    st = regularize(states_from_flows(fl, cfg, a.pcap.split("/")[-1]), cfg)
    out = a.out or str(ROOT / "data" / "states" / "pcap_states.csv")
    save_states(st, out)
    print(f"{len(fl):,} flows -> {len(st):,} windows -> {out}")


if __name__ == "__main__":
    main()
