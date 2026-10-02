import numpy as np

from src.models.sequences import inference_ends, sample_ends, split_labels


def test_no_leakage_between_parts(states, cfg):
    L, H = cfg["window"]["sequence_length"], cfg["window"]["forecast_horizon"]
    part = split_labels(states, cfg)
    for lab in ("train", "val", "test"):
        e = sample_ends(states, L, H, part, lab)
        assert len(e)
        for i in e[:: max(len(e) // 50, 1)]:
            assert (part[i - L + 1:i + H + 1] == lab).all()
            assert states["segment"].iat[i - L + 1] == states["segment"].iat[i + H]
    assert set(part[states["source"].eq(states["source"].iat[0]).to_numpy()]) >= {"train", "val", "test"}


def test_vocab_is_train_only(states, cfg):
    from src.models.sequences import stage_ids, stage_vocab
    part = split_labels(states, cfg)
    v = stage_vocab(states, part, cfg)
    tr_stages = set(states.loc[(part == "train") & states["is_attack"], "stage"])
    assert set(v[1:]) == tr_stages and v[0] == "Benign"
    assert (stage_ids(states, v)[states["is_attack"].to_numpy() & ~states["stage"].isin(v).to_numpy()] == -1).all()


def test_every_day_has_holdout(states, cfg):
    part = split_labels(states, cfg)
    for s in states["source"].unique():
        assert (part[states["source"].eq(s).to_numpy()] == "test").any()


def test_inference_ends_respect_segments(states, cfg):
    L = cfg["window"]["sequence_length"]
    e = inference_ends(states, L)
    assert (states["segment"].to_numpy()[e - L + 1] == states["segment"].to_numpy()[e]).all()
