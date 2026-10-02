"""Plain-language knowledge: feature descriptions and response playbooks per ATT&CK tactic."""
from __future__ import annotations

PORT_SERVICES = {20: "FTP data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP mail", 53: "DNS", 80: "web (HTTP)", 110: "POP3 mail",
                 123: "NTP time", 135: "Windows RPC", 139: "NetBIOS", 143: "IMAP mail", 443: "web (HTTPS)", 445: "SMB file sharing",
                 3389: "Remote Desktop", 8080: "alternate web", 8443: "alternate HTTPS"}

FEATURE_LABELS = {
    "flow_count": "connection volume", "unique_src_ips": "number of distinct sources", "unique_dst_ips": "number of distinct targets",
    "unique_src_ports": "variety of source ports", "unique_dst_ports": "number of different ports contacted",
    "packets_total": "packet volume", "bytes_total": "data volume", "pkt_rate_mean": "packet rate", "byte_rate_mean": "data rate",
    "duration_mean": "average connection length", "duration_std": "variation in connection length", "iat_mean": "average gap between packets",
    "iat_std": "irregularity of packet timing", "iat_max": "longest pause between packets", "ttl_mean": "packet hop limit (TTL)",
    "ttl_var": "variation in TTL", "tcp_window_mean": "TCP window size", "tcp_window_var": "variation in TCP window size",
    "pkt_len_mean": "average packet size", "pkt_len_std": "variation in packet size", "payload_mean": "average payload per connection",
    "payload_std": "variation in payload size", "retransmissions": "packet retransmissions", "syn_count": "connection attempts (SYN)",
    "ack_count": "acknowledgements (ACK)", "rst_count": "connection resets (RST)", "fin_count": "connection closes (FIN)",
    "psh_count": "push flags (PSH)", "urg_count": "urgent flags", "syn_ratio": "share of new-connection attempts",
    "rst_ratio": "share of reset connections", "inbound_outbound_byte_ratio": "reply-to-request data ratio",
    "inbound_outbound_packet_ratio": "reply-to-request packet ratio", "distinct_dst_per_src_mean": "targets contacted per source",
    "failed_flow_ratio": "share of unanswered connections", "small_flow_ratio": "share of very small connections",
    "scan_score": "port-scanning pattern", "well_known_dst_port_ratio": "share of traffic to well-known service ports",
    "protocol_tcp": "TCP share of traffic", "protocol_udp": "UDP share of traffic", "protocol_icmp": "ICMP share of traffic",
    "protocol_other": "share of other protocols"}


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name == "dst_port_other":
        return "share of traffic to uncommon ports"
    if name.startswith("dst_port_"):
        try:
            p = int(name.split("_")[-1])
            return f"share of traffic to port {p}" + (f" ({PORT_SERVICES[p]})" if p in PORT_SERVICES else "")
        except ValueError:
            pass
    return name.replace("_", " ")


def driver_sentences(top_features: list[dict], k: int = 4) -> list[str]:
    """Top attribution entries -> short readable sentences."""
    out = []
    for f in top_features[:k]:
        up = f.get("attribution", f.get("shap_value", 0)) > 0
        out.append(f"{'Raises' if up else 'Lowers'} the risk: {feature_label(f['feature'])}")
    return out


PLAYBOOKS = {
    "Reconnaissance": {"id": "TA0043", "meaning": "Someone is mapping hosts, ports and services before an attack (scanning, probing).",
        "actions": ["Identify the scanning source(s) and confirm whether they are authorised scanners.", "Block or rate-limit the source at the perimeter firewall.",
                    "Review exposed services and close ports that do not need to be reachable.", "Raise logging detail on the targeted hosts for the next 24 hours."]},
    "Initial Access": {"id": "TA0001", "meaning": "Exploitation of a public-facing service or web application (for example injection or cross-site scripting).",
        "actions": ["Check web/application logs for injected payloads and unusual request patterns.", "Enable or tighten WAF rules for the affected application.",
                    "Patch the affected service and validate recent deployments.", "Isolate the host if compromise is suspected and preserve logs."]},
    "Credential Access": {"id": "TA0006", "meaning": "Repeated authentication attempts - a brute-force or password-guessing campaign.",
        "actions": ["Enforce account lockout / rate limiting on the targeted service (SSH, FTP, web login).", "Require multi-factor authentication for exposed logins.",
                    "Search authentication logs for any successful login from the same source.", "Block the offending source and force password resets on targeted accounts."]},
    "Discovery": {"id": "TA0007", "meaning": "An already-present actor is enumerating the internal network and accounts.",
        "actions": ["Identify the internal host generating the enumeration traffic.", "Review recent logins to that host.", "Restrict its access to internal segments until verified."]},
    "Lateral Movement": {"id": "TA0008", "meaning": "Traffic patterns consistent with an actor moving between internal systems after getting a foothold.",
        "actions": ["Isolate the source host from internal network segments.", "Review use of remote admin protocols (SMB, RDP, SSH) between internal hosts.",
                    "Reset credentials used on the affected hosts and check for new accounts or services.", "Hunt for the initial foothold on the source host (endpoint logs, EDR)."]},
    "Command and Control": {"id": "TA0011", "meaning": "A compromised host appears to be communicating with an external controller (botnet-like beaconing).",
        "actions": ["Identify the internal host(s) producing the beaconing traffic.", "Block the destination addresses/domains at the firewall and DNS.",
                    "Isolate and reimage the affected host after collecting forensic evidence.", "Search other hosts for the same destination or pattern."]},
    "Exfiltration": {"id": "TA0010", "meaning": "Unusual outbound data transfer that may indicate data theft.",
        "actions": ["Identify the source host and the destination of the large outbound transfer.", "Cut the connection or block the destination immediately.",
                    "Determine which data was accessible to the host and assess exposure.", "Engage the incident-response process and notify the data owner."]},
    "Impact": {"id": "TA0040", "meaning": "Traffic floods consistent with denial-of-service: the aim is to exhaust capacity and disrupt availability.",
        "actions": ["Enable upstream rate limiting / DDoS protection and block the dominant source ranges.", "Scale or fail over the targeted service if possible.",
                    "Confirm service health and capacity from the monitoring side.", "Keep packet captures of the flood for analysis and reporting."]},
}
DEFAULT_PLAYBOOK = {"id": "-", "meaning": "Anomalous traffic progression was forecast.", "actions": ["Review the flagged traffic in detail.", "Escalate according to your incident-response policy."]}


def playbook(stage: str) -> dict:
    return PLAYBOOKS.get(stage, DEFAULT_PLAYBOOK)
