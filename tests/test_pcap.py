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
