import numpy as np
import pandas as pd
import pytest

from src.data.flows import TimestampParser, iter_flow_chunks
from src.data.schema import LabelMapper, check_required, resolve_columns
from src.data.windows import build_states, regularize, states_from_flow_csv
from src.data.synthetic import make_day


def test_columns_resolve_without_ip(cfg):
    m = resolve_columns(["Dst Port", "Protocol", "Timestamp", "Label", "Flow Duration"], cfg)
    assert {"dst_port", "protocol", "timestamp", "label"} <= set(m)
    assert "src_ip" not in m
    check_required(m)                     # IP / source port are optional


def test_missing_required_raises(cfg):
    with pytest.raises(ValueError):
        check_required(resolve_columns(["Foo", "Bar"], cfg))


def test_label_mapping_real_cic_labels(cfg):
    lm = LabelMapper(cfg)
    assert lm("Benign") == "Benign"
    assert lm("DoS attacks-Hulk") == "Impact" and lm("DDOS attack-HOIC") == "Impact"
    assert lm("Infilteration") == "Lateral Movement" and lm("Bot") == "Command and Control"
    assert lm("SSH-Bruteforce") == "Credential Access" and lm("SQL Injection") == "Initial Access"
    assert lm("Totally New Attack") == cfg["labeling"]["unknown_attack_stage"] and lm.unmapped


def test_timestamp_day_first(cfg):
    p = TimestampParser()
    out = p(pd.Series(["02/03/2018 08:47:38", "14/02/2018 10:00:00"]))
    assert out.iloc[0].month == 3 and out.iloc[1].day == 14


def test_chunked_equals_whole(tmp_path, cfg):
    f = tmp_path / "d.csv"; make_day(f, 3, n_windows=200, with_ips=True, messy=True)
    a, _ = states_from_flow_csv(f, cfg, chunksize=500, progress=None)
    b, _ = states_from_flow_csv(f, cfg, chunksize=10_000_000, progress=None)
    cols = [c for c in a.columns if a[c].dtype.kind == "f"]
    np.testing.assert_allclose(a[cols].to_numpy(), b[cols].to_numpy(), rtol=1e-9, atol=1e-9)


def test_actual_cic_schema_without_ip_loads(tmp_path, cfg):
    f = tmp_path / "x.csv"; make_day(f, 1, n_windows=120, with_ips=False, messy=True)
    chunks = list(iter_flow_chunks(f, cfg, chunksize=1000))
    assert sum(len(c) for c in chunks) > 1000
    assert not any(c["raw_label"].str.lower().eq("label").any() for c in chunks)   # embedded header row dropped


def test_regularize_fills_small_gaps_and_splits_big(cfg):
    from src.data.schema import feature_names
    n = feature_names(cfg)
    ts = pd.to_datetime(["2018-02-14 08:00:00", "2018-02-14 08:00:10", "2018-02-14 08:00:40", "2018-02-14 09:30:00"])
    df = pd.DataFrame({"timestamp": ts, "attack_flows": 0.0, "attack_fraction": 0.0, "stage": "Benign",
                       "is_attack": False, "source": "a", "filled": False})
    for c in n:
        df[c] = 1.0
    r = regularize(df, cfg)
    assert r["filled"].sum() == 2                      # 20 s gap -> 2 idle windows
    assert r["segment"].nunique() == 2                 # 90-min gap starts a new segment
    assert (r.loc[r["filled"], n] == 0).all().all()


def test_state_features_sane(states):
    assert states["timestamp"].dt.year.min() == 2018
    assert states["flow_count"].min() >= 0 and states["is_attack"].any()
    assert states["syn_ratio"].between(0, 1.0001).all()
