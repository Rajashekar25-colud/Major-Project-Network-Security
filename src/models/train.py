"""Training, threshold calibration and artifact writing."""
from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..data.schema import feature_names
from ..evaluation.evaluate import evaluate_model
from ..evaluation.metrics import choose_threshold, onset_masks
from ..features.preprocess import Preprocessor
from .inference import Forecaster
from .sequences import (future_targets, gather, sample_ends, split_labels, stage_ids, stage_vocab)
from .world_model import WorldModel, multitask_loss


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batches(ends, bs, rng, shuffle):
    order = rng.permutation(len(ends)) if shuffle else np.arange(len(ends))
    for i in range(0, len(order), bs):
        yield ends[order[i:i + bs]]


def _split_summary(states, part):
    out = {}
    for lab in ("train", "val", "test", "gap"):
        m = part == lab
        sub = states[m]
        out[lab] = {"windows": int(m.sum()), "attack_windows": int(sub["is_attack"].sum()),
                    "stages": {k: int(v) for k, v in sub.loc[sub["is_attack"], "stage"].value_counts().items()}}
    return out


def train_from_states(states: pd.DataFrame, config: dict[str, Any], out_dir: str | Path = "models",
                      dataset_info: dict[str, Any] | None = None, synthetic: bool = False,
                      log=print, with_baselines: bool = True) -> dict[str, Any]:
    t_start = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg_w, cfg_m = config["window"], config["model"]
    L, H = int(cfg_w["sequence_length"]), int(cfg_w["forecast_horizon"])
    seed_everything(int(config["seed"]))
    rng = np.random.default_rng(int(config["seed"]))
    states = states.reset_index(drop=True)
    if len(states) < int(cfg_w["min_windows"]):
        raise SystemExit(f"Only {len(states)} windows; need >= window.min_windows={cfg_w['min_windows']}.")

    part = split_labels(states, config)
    vocab = stage_vocab(states, part, config)
    sid = stage_ids(states, vocab)
    is_atk = states["is_attack"].to_numpy()
    names = feature_names(config)
    pre = Preprocessor(names, cfg_m["feature_clip"]).fit(states[part == "train"])
    X = pre.transform(states)
    summ = _split_summary(states, part)
    log(f"[SPLIT] " + ", ".join(f"{k}={v['windows']:,} (attack {v['attack_windows']:,})" for k, v in summ.items()))
    log(f"[STAGES] model vocabulary: {vocab}")
    log(f"[FEATURES] {int(pre.active.sum())}/{len(names)} active (constant-in-train features are neutralised)")

    tr = sample_ends(states, L, H, part, "train"); va = sample_ends(states, L, H, part, "val")
    if len(tr) < 50 or len(va) < 10:
        raise SystemExit(f"Too few samples (train={len(tr)}, val={len(va)}). Provide more data or lower window sizes.")
    ytr_stage, ytr_atk = future_targets(sid, is_atk, tr, H)
    yva_stage, yva_atk = future_targets(sid, is_atk, va, H)
    if ytr_atk.sum() == 0:
        raise SystemExit("The training split contains no attack windows in the forecast horizon; nothing to learn.")

    S = len(vocab)
    cnt = np.array([(ytr_stage == c).sum() for c in range(S)], dtype=np.float64)
    cw = np.where(cnt > 0, (cnt.sum() / (S * np.maximum(cnt, 1))) ** 0.5, 0.0)
    cw = cw / max(cw[cnt > 0].mean(), 1e-9)
    pos = float(ytr_atk.sum()); neg = float(ytr_atk.size - pos)
    pos_w = float(np.clip(neg / max(pos, 1.0), 1.0, 10.0))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = WorldModel(len(names), S, H, cfg_m["hidden_size"], cfg_m["layers"], cfg_m["dropout"], cfg_m["attention_heads"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg_m["learning_rate"], weight_decay=cfg_m["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    act = torch.tensor(pre.active, dtype=torch.float32, device=dev)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=dev); pw_t = torch.tensor(pos_w, device=dev)
    noise = float(cfg_m.get("input_noise", 0.0))
    lw = (cfg_m["state_loss_weight"], cfg_m["stage_loss_weight"], cfg_m["attack_loss_weight"])

    def run_batch(e, ystage_all, yatk_all, idx_map, train: bool):
        xin, ys = gather(X, e, L, H)
        pos_in = idx_map(e)
        xb = torch.from_numpy(xin).to(dev); ysb = torch.from_numpy(ys).to(dev)
        yst = torch.from_numpy(ystage_all[pos_in]).to(dev); yat = torch.from_numpy(yatk_all[pos_in]).to(dev)
        if train and noise > 0:
            xb = xb + noise * torch.randn_like(xb) * act        # light input-noise augmentation (active features only)
        o = model(xb)
        return multitask_loss(o, ysb, yst, yat, act, cw_t, pw_t, *lw)

    tr_map = {int(e): i for i, e in enumerate(tr)}; va_map = {int(e): i for i, e in enumerate(va)}
    tr_idx = lambda e: np.fromiter((tr_map[int(x)] for x in e), dtype=np.int64, count=len(e))
    va_idx = lambda e: np.fromiter((va_map[int(x)] for x in e), dtype=np.int64, count=len(e))

    best, bad, hist = float("inf"), 0, []
    best_state = None
    cap = int(cfg_m["max_train_samples_per_epoch"])
    for epoch in range(1, int(cfg_m["epochs"]) + 1):
        model.train(); tl = 0.0; nb = 0
        ep_ends = tr if len(tr) <= cap else rng.choice(tr, cap, replace=False)
        for e in _batches(ep_ends, int(cfg_m["batch_size"]), rng, True):
            loss, _ = run_batch(e, ytr_stage, ytr_atk, tr_idx, True)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg_m["grad_clip"]))
            opt.step(); tl += loss.item(); nb += 1
        model.eval(); vl = 0.0; vb = 0
        with torch.no_grad():
            for e in _batches(va, 512, rng, False):
                loss, _ = run_batch(e, yva_stage, yva_atk, va_idx, False)
                vl += loss.item(); vb += 1
        tl /= max(nb, 1); vl /= max(vb, 1); sched.step(vl)
        hist.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "lr": opt.param_groups[0]["lr"]})
        log(f"[EPOCH {epoch:02d}] train={tl:.4f} val={vl:.4f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg_m["early_stopping_patience"]):
                log("[EARLY STOP]"); break
    model.load_state_dict(best_state); model.cpu()
    torch.save(model.state_dict(), out / "world_model.pt")

    # ---- calibrate warning threshold on the validation split ----
    meta_stub = {"window": cfg_w, "feature_names": names, "stage_names": vocab, "model": {
        "hidden_size": cfg_m["hidden_size"], "layers": cfg_m["layers"], "dropout": cfg_m["dropout"],
        "attention_heads": cfg_m["attention_heads"]}, "preprocessing": pre.to_dict(), "warning_threshold": 0.5}
    fc = Forecaster(model, meta_stub)
    vp = fc.predict_batch(X, va)
    vquiet, vpos = onset_masks(is_atk, va, yva_atk)
    thr, cal = choose_threshold(vp["attack_prob"].max(axis=1), vquiet, vpos,
                                config["calibration"]["target_false_positive_rate"], config["calibration"]["fallback_threshold"])
    fc.threshold = thr
    log(f"[CALIBRATION] warning threshold = {thr:.3f} ({cal['method']})")

    meta: dict[str, Any] = {
        "schema_version": 2,
        "project": config["project"]["name"], "project_version": config["project"]["version"],
        "preprocessing_version": config["project"]["preprocessing_version"],
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "training_seconds": round(time.time() - t_start, 1),
        "synthetic_demo_data": bool(synthetic),
        "feature_names": names, "n_features": len(names),
        "active_features": [n for n, a in zip(names, pre.active) if a],
        "inactive_features": [n for n, a in zip(names, pre.active) if not a],
        "stage_names": vocab, "stage_order": config["stage_order"], "attack_tactics": config["attack_tactics"],
        "window": cfg_w, "labeling": config["labeling"], "split": config["split"],
        "model": {**cfg_m, "architecture": "Linear-LayerNorm encoder -> 2-layer LSTM -> multi-head temporal self-attention -> "
                  "K-step heads (state / ATT&CK stage / attack probability)", "parameters": int(sum(p.numel() for p in model.parameters())),
                  "positive_class_weight": pos_w, "class_weights": {v: float(w) for v, w in zip(vocab, cw)}},
        "optimizer": {"name": "AdamW", "lr": cfg_m["learning_rate"], "weight_decay": cfg_m["weight_decay"], "scheduler": "ReduceLROnPlateau"},
        "seed": config["seed"], "device": dev, "best_val_loss": best, "epochs_run": len(hist), "history": hist,
        "preprocessing": pre.to_dict(),
        "warning_threshold": thr, "calibration": {**cal, **config["calibration"]}, "warning": config["warning"],
        "dataset": {**(dataset_info or {}), "windows_total": int(len(states)), "segments": int(states["segment"].nunique()),
                    "start": str(states["timestamp"].min()), "end": str(states["timestamp"].max()),
                    "sources": sorted(map(str, states["source"].unique())), "split_summary": summ,
                    "stage_window_counts": {k: int(v) for k, v in states.loc[states["is_attack"], "stage"].value_counts().items()},
                    "benign_windows": int((~states["is_attack"]).sum())},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__, "platform": platform.platform()},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    fc.meta = meta
    log("[EVAL] evaluating on the chronological per-day hold-out ...")
    ev = evaluate_model(fc, states, part, X, config, "test", with_baselines, log)
    (out / "evaluation.json").write_text(json.dumps(ev, indent=2, default=str), encoding="utf-8")
    meta["evaluated_at"] = ev.get("evaluated_at")
    meta["evaluation_summary"] = {k: ev.get(k) for k in ("n_samples",)} | {
        "onset_f1": ev.get("onset", {}).get("f1"), "onset_recall": ev.get("onset", {}).get("recall"),
        "onset_fpr": ev.get("onset", {}).get("false_positive_rate")}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    # background sequences (train, scaled) used by SHAP explanations in the dashboard
    bg_e = rng.choice(tr, size=min(100, len(tr)), replace=False)
    np.save(out / "background_sequences.npy", gather(X, np.sort(bg_e), L, H)[0])
    log(f"[DONE] artifacts written to {out.resolve()} in {time.time() - t_start:.0f}s")
    return {"meta": meta, "evaluation": ev}
