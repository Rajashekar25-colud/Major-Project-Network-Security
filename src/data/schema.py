"""Column alias resolution, label -> ATT&CK stage mapping and feature schema."""
from __future__ import annotations

import re
from typing import Any, Iterable

BENIGN = "Benign"


def norm(s: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def resolve_columns(header: Iterable[str], config: dict[str, Any]) -> dict[str, str]:
    """Map canonical field -> actual column name in the file (first alias wins)."""
    lookup: dict[str, str] = {}
    for c in header:
        lookup.setdefault(norm(c), c)
    found: dict[str, str] = {}
    for canonical, aliases in config["columns"].items():
        for a in aliases:
            if norm(a) in lookup:
                found[canonical] = lookup[norm(a)]
                break
    return found


REQUIRED_FIELDS = ["timestamp", "dst_port", "protocol"]


def check_required(found: dict[str, str], source: str = "CSV") -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in found]
    if missing:
        raise ValueError(
            f"{source}: required field(s) {missing} not found. Only these are mandatory: "
            f"{REQUIRED_FIELDS}. IP addresses and source ports are optional."
        )


class LabelMapper:
    """Dataset label -> ATT&CK stage using the YAML mapping (exact, then regex)."""

    def __init__(self, config: dict[str, Any]):
        m = config["stage_mapping"]
        self.benign = {norm(x) for x in m.get("benign_labels", [])}
        self.exact = {norm(k): v for k, v in m.get("exact", {}).items()}
        self.regex = [(re.compile(p, re.I), v) for p, v in m.get("regex", [])]
        self.unknown_stage = config["labeling"]["unknown_attack_stage"]
        self.unmapped: dict[str, int] = {}

    def __call__(self, raw: object) -> str:
        key = norm(raw)
        if key in self.benign or key in ("", "nan", "none"):
            return BENIGN
        if key in self.exact:
            return self.exact[key]
        s = str(raw)
        for rx, stage in self.regex:
            if rx.search(s):
                return stage
        self.unmapped[s] = self.unmapped.get(s, 0) + 1
        return self.unknown_stage


# --------------------------------------------------------------------------
# Feature schema (order matters: it is the model input order)
# --------------------------------------------------------------------------
def feature_names(config: dict[str, Any]) -> list[str]:
    base = [
        "flow_count", "unique_src_ips", "unique_dst_ips", "unique_src_ports", "unique_dst_ports",
        "packets_total", "bytes_total", "pkt_rate_mean", "byte_rate_mean",
        "duration_mean", "duration_std", "iat_mean", "iat_std", "iat_max",
        "ttl_mean", "ttl_var", "tcp_window_mean", "tcp_window_var",
        "pkt_len_mean", "pkt_len_std", "payload_mean", "payload_std", "retransmissions",
        "syn_count", "ack_count", "rst_count", "fin_count", "psh_count", "urg_count",
        "syn_ratio", "rst_ratio",
        "inbound_outbound_byte_ratio", "inbound_outbound_packet_ratio",
        "distinct_dst_per_src_mean", "failed_flow_ratio", "small_flow_ratio", "scan_score",
        "well_known_dst_port_ratio",
        "protocol_tcp", "protocol_udp", "protocol_icmp", "protocol_other",
    ]
    ports = [f"dst_port_{p}" for p in config["features"]["port_buckets"]] + ["dst_port_other"]
    return base + ports


# Features that depend on packet-level telemetry / IP fields and are therefore only
# populated when the source provides them (PCAP or NetFlow with IPs).
PACKET_LEVEL_FEATURES = ["ttl_mean", "ttl_var", "retransmissions"]
IP_LEVEL_FEATURES = ["unique_src_ips", "unique_dst_ips", "unique_src_ports", "distinct_dst_per_src_mean"]

# Heavy-tailed features get a signed log1p before standardisation.
LOG_FEATURES = [
    "flow_count", "unique_src_ips", "unique_dst_ips", "unique_src_ports", "unique_dst_ports",
    "packets_total", "bytes_total", "pkt_rate_mean", "byte_rate_mean",
    "duration_mean", "duration_std", "iat_mean", "iat_std", "iat_max",
    "tcp_window_mean", "tcp_window_var", "pkt_len_mean", "pkt_len_std",
    "payload_mean", "payload_std", "retransmissions",
    "syn_count", "ack_count", "rst_count", "fin_count", "psh_count", "urg_count",
    "inbound_outbound_byte_ratio", "inbound_outbound_packet_ratio", "distinct_dst_per_src_mean",
    "ttl_var",
]
