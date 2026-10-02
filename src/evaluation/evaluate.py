"""Hold-out evaluation of the world model against comparable forecasting baselines."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..models.inference import Forecaster
from ..models.sequences import future_targets, gather, sample_ends, stage_ids
from .metrics import (binary_metrics, choose_threshold, lead_time_report, onset_masks, stage_metrics)


def _timeline(states, ends, attack_prob, headline):
    df = pd.DataFrame({"idx": ends, "timestamp": states["timestamp"].to_numpy()[ends],
                       "risk": attack_prob.max(axis=1), "headline_stage": headline})
    for k in range(attack_prob.shape[1]):
        df[f"p_t{k + 1}"] = attack_prob[:, k]
    return df


def _flat(X, ends, L):
    return X[ends[:, None] + np.arange(-L + 1, 1)[None, :]].reshape(len(ends), -1)


def _fit_lr_baseline(X, states, part, sid_attack, L, H, cfg, rng):
    """Direct forecasting baseline: one logistic regression per horizon on the flattened
    last-L windows (same inputs, same targets, same split as the world model)."""
    tr = sample_ends(states, L, H, part, "train")
    if len(tr) > 40000:
        tr = rng.choice(tr, 40000, replace=False)
    _, y = future_targets(sid_attack[0], sid_attack[1], tr, H)
    Xtr = _flat(X, tr, L)
    models = []
    for k in range(H):
        if y[:, k].sum() < 2 or (1 - y[:, k]).sum() < 2:
            models.append(None); continue
        m = LogisticRegression(max_iter=300, class_weight="balanced", C=0.05)
        m.fit(Xtr, y[:, k]); models.append(m)

    def predict(ends):
        Z = _flat(X, ends, L)
        out = np.zeros((len(ends), H))
        for k, m in enumerate(models):
            out[:, k] = 0.0 if m is None else m.predict_proba(Z)[:, 1]
        return out
    return predict


def evaluate_model(fc: Forecaster, states: pd.DataFrame, part: np.ndarray, X: np.ndarray, config: dict[str, Any],
                   label: str = "test", with_baselines: bool = True, log=print) -> dict[str, Any]:
    t0 = time.time()
    L, H, wsec = fc.L, fc.H, fc.wsec
    thr = fc.threshold
    sid = stage_ids(states, fc.stages)
    is_atk = states["is_attack"].to_numpy()
    ends = sample_ends(states, L, H, part, label)
    res: dict[str, Any] = {"split": label, "n_samples": int(len(ends)), "threshold": thr,
                           "evaluated_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    if len(ends) == 0:
        res["error"] = f"No '{label}' samples available (too little data per capture day)."
        return res
    pred = fc.predict_batch(X, ends, want_state=True)
    dec = fc.decode(pred["attack_prob"], pred["stage_prob"])
    y_stage, y_atk = future_targets(sid, is_atk, ends, H)
    ap = pred["attack_prob"]

    # ---- attack-forecast metrics (all horizons pooled + per horizon) ----
    res["forecast"] = binary_metrics(y_atk, ap, thr)
    res["per_horizon"] = [{"horizon": k + 1, "seconds_ahead": (k + 1) * wsec, **binary_metrics(y_atk[:, k], ap[:, k], thr)} for k in range(H)]
    quiet, pos = onset_masks(is_atk, ends, y_atk)
    onset_y = (pos & quiet)[quiet].astype(int)
    res["onset"] = binary_metrics(onset_y, dec["risk"][quiet], thr)
    res["onset"]["definition"] = ("Samples whose current window is benign; positive = an attack starts within the next "
                                  f"{H} windows. This isolates true early warning from 'the attack is already happening'.")

    # ---- stage forecast ----
    pred_stage = dec["stage_idx"]
    res["stage"] = stage_metrics(y_stage, pred_stage, fc.stages)
    att = y_stage > 0
    if att.any():
        res["stage"]["accuracy_given_attack"] = float((pred_stage[att] == y_stage[att]).mean())

    # ---- state forecasting (learned dynamics) vs naive persistence ----
    _, y_state = gather(X, ends, L, H)
    act = fc.pre.active
    err = pred["state"][:, :, act] - y_state[:, :, act]
    last = X[ends][:, None, :][:, :, act]
    per_err = np.abs(last - y_state[:, :, act])
    res["state_forecast"] = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
                             "persistence_mae": float(per_err.mean()), "persistence_rmse": float(np.sqrt((per_err ** 2).mean())),
                             "mae_per_horizon": [float(np.abs(err[:, k]).mean()) for k in range(H)],
                             "space": "standardised active features"}

    # ---- warning lead time (through the real warning engine) ----
    part_mask = part == label
    tl = _timeline(states, ends, ap, dec["headline_stage"])
    res["lead_time"] = lead_time_report(tl, states, thr, config["warning"], wsec, H, L, eval_mask=part_mask)

    # ---- comparable future-forecasting baselines ----
    if with_baselines:
        base: dict[str, Any] = {}
        # (a) persistence: "an attack now continues" (oracle knowledge of the CURRENT label)
        cur = np.repeat(is_atk[ends][:, None].astype(float), H, axis=1)
        base["persistence_oracle"] = {"forecast": binary_metrics(y_atk, cur, 0.5),
                                      "onset": binary_metrics(onset_y, cur.max(axis=1)[quiet], 0.5),
                                      "note": "Uses the true current label; it can never warn before onset (onset recall is 0 by construction)."}
        # (b) logistic regression on the flattened sequence, threshold chosen on validation like the world model
        try:
            rng = np.random.default_rng(int(config["seed"]))
            lr = _fit_lr_baseline(X, states, part, (sid, is_atk), L, H, config, rng)
            vend = sample_ends(states, L, H, part, "val")
            vy_stage, vy = future_targets(sid, is_atk, vend, H)
            vq, vp = onset_masks(is_atk, vend, vy)
            lthr, _ = choose_threshold(lr(vend).max(axis=1), vq, vp, config["calibration"]["target_false_positive_rate"],
                                       config["calibration"]["fallback_threshold"])
            lp = lr(ends)
            base["logistic_sequence"] = {"forecast": binary_metrics(y_atk, lp, lthr),
                                         "onset": binary_metrics(onset_y, lp.max(axis=1)[quiet], lthr), "threshold": lthr}
            ltl = _timeline(states, ends, lp, np.where(lp.max(axis=1) >= lthr, "Attack", "Benign"))
            base["logistic_sequence"]["lead_time"] = lead_time_report(ltl, states, lthr, config["warning"], wsec, H, L, eval_mask=part_mask)
            base["logistic_sequence"]["note"] = "Same inputs / targets / split / threshold rule as the world model (direct multi-horizon forecasting)."
        except Exception as e:  # keep evaluation robust
            base["logistic_sequence"] = {"error": str(e)}
        res["baselines"] = base
    res["seconds"] = round(time.time() - t0, 1)
    return res
