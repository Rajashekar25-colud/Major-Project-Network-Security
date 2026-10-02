from src.data.pcap import make_sample_pcap, pcap_to_flows
from src.data.windows import regularize, states_from_flows


def test_pcap_packet_level_features(tmp_path, cfg):
    p = tmp_path / "s.pcap"; make_sample_pcap(p)
    fl = pcap_to_flows(p, cfg)
    assert len(fl) > 500
    assert fl["ttl"].notna().all() and fl["win"].notna().any() and fl["retr"].sum() > 0
    st = regularize(states_from_flows(fl, cfg, "s.pcap"), cfg)
    assert st["ttl_mean"].max() > 30 and st["retransmissions"].sum() > 0
    assert st["unique_src_ips"].max() > 1 and st["scan_score"].max() > 0.3     # scan window visible


import socket
import struct

import pytest

from src.data.pcap import _frame


def _classic(path, frames, strip, lt):
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, lt))
        for ts, b in frames:
            b = strip(b); f.write(struct.pack("<IIII", int(ts), int((ts % 1) * 1e6), len(b), len(b)) + b)


@pytest.mark.parametrize("name,strip,lt", [
    ("raw", lambda b: b[14:], 101), ("null", lambda b: b"\x02\x00\x00\x00" + b[14:], 0),
    ("sll", lambda b: b"\x00" * 14 + b"\x08\x00" + b[14:], 113), ("unknown-linktype", lambda b: b[14:], 999)])
def test_link_types(tmp_path, cfg, name, strip, lt):
    fr = [_frame(1_700_000_000 + i * 0.1, "10.0.0.1", "10.0.0.2", 1000 + i, 80, 6, 64, 0x02) for i in range(30)]
    _classic(tmp_path / "x.pcap", fr, strip, lt)
    assert len(pcap_to_flows(tmp_path / "x.pcap", cfg)) == 30


def test_pcapng_and_ipv6(tmp_path, cfg):
    def blk(t, body):
        body += b"\0" * ((-len(body)) % 4); n = 12 + len(body)
        return struct.pack("<II", t, n) + body + struct.pack("<I", n)
    ng = blk(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1)) + blk(1, struct.pack("<HHI", 1, 0, 65535))
    for i in range(20):
        _, b = _frame(1_700_000_000 + i, "10.0.0.1", "10.0.0.2", 1000 + i, 80, 6)
        t = int((1_700_000_000 + i) * 1e6)
        ng += blk(6, struct.pack("<IIIII", 0, t >> 32, t & 0xFFFFFFFF, len(b), len(b)) + b)
    (tmp_path / "c.pcapng").write_bytes(ng)
    assert len(pcap_to_flows(tmp_path / "c.pcapng", cfg)) == 20
    tcp = struct.pack("!HHIIBBHHH", 2000, 443, 1, 0, 5 << 4, 0x02, 1000, 0, 0)
    ip = struct.pack("!IHBB", 0x60000000, len(tcp), 6, 64) + socket.inet_pton(socket.AF_INET6, "fe80::1") + socket.inet_pton(socket.AF_INET6, "fe80::2")
    with open(tmp_path / "v6.pcap", "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        b = b"\0" * 12 + b"\x86\xdd" + ip + tcp
        for i in range(10):
            f.write(struct.pack("<IIII", 1_700_000_000 + i, 0, len(b), len(b)) + b)
    assert pcap_to_flows(tmp_path / "v6.pcap", cfg)["dst_port"].iloc[0] == 443


def test_non_ip_capture_gives_clear_error(tmp_path, cfg):
    with open(tmp_path / "arp.pcap", "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1) + struct.pack("<IIII", 1, 0, 60, 60) + b"\xff" * 12 + b"\x08\x06" + b"\0" * 46)
    with pytest.raises(ValueError, match="packet\\(s\\) read"):
        pcap_to_flows(tmp_path / "arp.pcap", cfg)
