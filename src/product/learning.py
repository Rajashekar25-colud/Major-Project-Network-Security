"""Learning from new, unseen data: drift detection, safe fine-tuning, model versions, promote / rollback.

Fine-tuning never overwrites the live model. It creates a CANDIDATE version, scores it against the
current model on held-out blocks of the new data AND on the original data (forgetting check), and only
replaces the live model when the user promotes it. Every version can be restored.
"""
from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from ..config import MODELS_DIR, ROOT, load_config
from ..evaluation.evaluate import evaluate_model
from ..evaluation.metrics import choose_threshold, onset_masks
from ..models.inference import Forecaster
from ..models.sequences import future_targets, gather, sample_ends, split_labels, stage_ids
from ..models.world_model import build_model, multitask_loss
from . import store

VERSIONS = MODELS_DIR / "versions"
FILES = ("world_model.pt", "metadata.json", "evaluation.json", "background_sequences.npy")


# ------------------------------------------------------------------------------------------- drift
def drift_report(fc: Forecaster, states: pd.DataFrame) -> dict[str, Any]:
    """Population-stability index of every active feature: new traffic vs what the model was trained on."""
    bgp = MODELS_DIR / "background_sequences.npy"
    if not bgp.exists():
        return {"available": False, "reason": "reference sample not found (retrain to create it)"}
    ref = np.load(bgp).reshape(-1, len(fc.names))
    new = fc.scale(states)
    rows = []
    for j, n in enumerate(fc.names):
        if not fc.pre.active[j]:
            continue
        r, x = ref[:, j], new[:, j]
        edges = np.unique(np.quantile(r, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            psi = 0.0 if abs(x.mean() - r.mean()) < 0.5 else 1.0
        else:
            edges[0], edges[-1] = -np.inf, np.inf
            pr = np.histogram(r, edges)[0] / len(r); px = np.histogram(x, edges)[0] / len(x)
            pr, px = np.clip(pr, 1e-4, None), np.clip(px, 1e-4, None)
            psi = float(((px - pr) * np.log(px / pr)).sum())
        rows.append({"feature": n, "psi": psi, "mean_shift": float(x.mean() - r.mean())})
    df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    df["level"] = np.where(df["psi"] >= 0.25, "significant", np.where(df["psi"] >= 0.10, "moderate", "stable"))
    share = float((df["psi"] >= 0.25).mean())
    overall = "significant" if share > 0.25 else "moderate" if share > 0.08 or (df["psi"] >= 0.10).mean() > 0.3 else "stable"
    return {"available": True, "table": df, "overall": overall, "significant_share": share,
            "advice": {"stable": "Traffic looks like the data the model was trained on.",
                       "moderate": "Some traffic characteristics have shifted. Monitor detection quality; consider learning from recent labelled data.",
                       "significant": "Traffic differs substantially from the training data. Learning from recent labelled traffic is recommended."}[overall]}


# ------------------------------------------------------------------------------------------- versions
def _index() -> list[dict[str, Any]]:
    p = VERSIONS / "index.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _save_index(idx: list[dict[str, Any]]) -> None:
    VERSIONS.mkdir(parents=True, exist_ok=True)
    (VERSIONS / "index.json").write_text(json.dumps(idx, indent=2, default=str), encoding="utf-8")


def _copy_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)


def ensure_baseline_version() -> list[dict[str, Any]]:
    """Snapshot the model that is live right now as 'v000' the first time versions are used."""
    idx = _index()
    if not idx and (MODELS_DIR / "world_model.pt").exists():
        meta = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
        d = VERSIONS / "v000_original"
        _copy_files(MODELS_DIR, d)
        idx = [{"id": "v000_original", "created": meta.get("trained_at"), "status": "active", "parent": None,
                "note": "Original trained model", "stages": meta.get("stage_names"), "threshold": meta.get("warning_threshold")}]
        _save_index(idx)
    return idx


def list_versions() -> list[dict[str, Any]]:
    return ensure_baseline_version()


def promote(version_id: str) -> None:
    idx = ensure_baseline_version()
    tgt = next((v for v in idx if v["id"] == version_id), None)
    if tgt is None:
        raise ValueError(f"unknown version {version_id}")
    _copy_files(VERSIONS / version_id, MODELS_DIR)
    for v in idx:
        if v["status"] == "active":
            v["status"] = "archived"
    tgt["status"] = "active"; tgt["promoted"] = pd.Timestamp.now().isoformat(timespec="seconds")
    _save_index(idx)
    store.audit("model_promote", version_id)


def active_version() -> str:
    for v in ensure_baseline_version():
        if v["status"] == "active":
            return v["id"]
    return "v000_original"


# ------------------------------------------------------------------------------------------- fine-tune
def _expand_stage_head(old: WorldModel, new_stages: int, hp: dict[str, Any]) -> WorldModel:
    H, F, S = old.horizon, old.input_size, old.num_stages
    m = build_model(hp.get("type", "lstm"), F, new_stages, H, hp)
    sd = {k: v for k, v in old.state_dict().items() if not k.startswith("stage_head.2")}
    m.load_state_dict(sd, strict=False)
    ow, ob = old.stage_head[2].weight.data, old.stage_head[2].bias.data
    nw, nb = m.stage_head[2].weight.data, m.stage_head[2].bias.data
    nw.mul_(0.1); nb.fill_(float(ob.min()) - 1.0)
    for h in range(H):
        nw[h * new_stages:h * new_stages + S] = ow[h * S:(h + 1) * S]
        nb[h * new_stages:h * new_stages + S] = ob[h * S:(h + 1) * S]
    return m


def _arrays(X, states, ends, vocab, L, H):
    xin, ys = gather(X, ends, L, H)
    sid = stage_ids(states, vocab)
    yst, yat = future_targets(sid, states["is_attack"].to_numpy(), ends, H)
    return xin.astype(np.float32), ys.astype(np.float32), yst, yat


def fine_tune(new_states: pd.DataFrame, source_name: str, log: Callable[[str], None] = print,
              base_states_path: Path | None = None, epochs: int = 6, lr: float = 2e-4, anchor: float = 1e-3,
              min_windows: int = 600) -> dict[str, Any]:
    t0 = time.time()
    ensure_baseline_version()
    fc = Forecaster.load(MODELS_DIR)
    cfg = load_config(); cfg["window"] = fc.meta["window"]; cfg["split"] = fc.meta["split"]
    L, H = fc.L, fc.H
    st = new_states.reset_index(drop=True)
    if not st["is_attack"].any():
        raise ValueError("The new data contains no labelled attack windows, so there is nothing to learn from. "
                         "Use a labelled flow CSV (with a Label column).")
    if len(st) < min_windows:
        raise ValueError(f"Need at least {min_windows} labelled windows ({min_windows * fc.wsec / 60:.0f} minutes of traffic); got {len(st)}.")
    part = split_labels(st, cfg)
    X = fc.pre.transform(st)                               # frozen scaler: the live model's view of the world
    tr, va, te = (sample_ends(st, L, H, part, k) for k in ("train", "val", "test"))
    if len(tr) < 50 or len(va) < 10 or len(te) < 10:
        raise ValueError(f"Not enough usable samples (train {len(tr)}, validation {len(va)}, test {len(te)}). Provide a longer capture.")
    old_vocab = list(fc.stages)
    new_stage = set(st.loc[(part == "train") & st["is_attack"].to_numpy(), "stage"]) - set(old_vocab)
    order = cfg["stage_order"]
    vocab = old_vocab + [s for s in order if s in new_stage] + sorted(new_stage - set(order))
    log(f"[LEARN] new samples train/val/test = {len(tr)}/{len(va)}/{len(te)}; new tactics learned: {sorted(new_stage) or 'none'}")

    xin, ys, yst, yat = _arrays(X, st, tr, vocab, L, H)
    vin, vys, vst, vat = _arrays(X, st, va, vocab, L, H)
    n_new = len(xin)
    base_path = base_states_path or (ROOT / "data" / "states" / "cicids2018_states.csv")
    base = None
    if Path(base_path).exists():
        from ..service import load_states_csv
        base = load_states_csv(base_path, cfg)
        bpart = split_labels(base, cfg)
        btr = sample_ends(base, L, H, bpart, "train")
        rng = np.random.default_rng(0)
        pick = rng.choice(btr, min(len(btr), max(n_new, 2000), 20000), replace=False)
        bx, bys, bst, bat = _arrays(fc.pre.transform(base), base, pick, vocab, L, H)
        xin, ys, yst, yat = (np.concatenate([a, b]) for a, b in ((xin, bx), (ys, bys), (yst, bst), (yat, bat)))
        log(f"[LEARN] replaying {len(pick):,} original samples to prevent forgetting")
    else:
        log("[LEARN] original training data not found - using weight anchoring only")

    hp = fc.meta["model"]
    model = _expand_stage_head(fc.model, len(vocab), hp) if len(vocab) != len(old_vocab) else copy.deepcopy(fc.model)
    ref = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    act = torch.tensor(fc.pre.active, dtype=torch.float32)
    S = len(vocab)
    cnt = np.array([(yst == c).sum() for c in range(S)], dtype=np.float64)
    cw = np.where(cnt > 0, (cnt.sum() / (S * np.maximum(cnt, 1))) ** 0.5, 0.0); cw = cw / max(cw[cnt > 0].mean(), 1e-9)
    pos = float(yat.sum()); pw = torch.tensor(float(np.clip((yat.size - pos) / max(pos, 1), 1, 10)))
    cwt = torch.tensor(cw, dtype=torch.float32)
    lw = (hp["state_loss_weight"], hp["stage_loss_weight"], hp["attack_loss_weight"])
    best, best_sd, bad = float("inf"), None, 0
    rng = np.random.default_rng(1)
    for ep in range(1, epochs + 1):
        model.train(); order_ = rng.permutation(len(xin)); tl = 0.0; nb = 0
        for i in range(0, len(order_), 128):
            b = order_[i:i + 128]
            o = model(torch.from_numpy(xin[b]))
            loss, _ = multitask_loss(o, torch.from_numpy(ys[b]), torch.from_numpy(yst[b]), torch.from_numpy(yat[b]), act, cwt, pw, *lw)
            reg = sum(((p - ref[n]) ** 2).sum() for n, p in model.named_parameters() if n in ref and p.shape == ref[n].shape)
            loss = loss + anchor * reg
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vl = 0.0; vb = 0
            for i in range(0, len(vin), 512):
                o = model(torch.from_numpy(vin[i:i + 512]))
                l_, _ = multitask_loss(o, torch.from_numpy(vys[i:i + 512]), torch.from_numpy(vst[i:i + 512]), torch.from_numpy(vat[i:i + 512]), act, cwt, pw, *lw)
                vl += l_.item(); vb += 1
        vl /= max(vb, 1)
        log(f"[LEARN] epoch {ep}/{epochs}  train {tl / max(nb, 1):.4f}  validation {vl:.4f}")
        if vl < best - 1e-4:
            best, bad = vl, 0; best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(best_sd); model.eval()

    # candidate forecaster + validation-calibrated threshold on the NEW data
    meta = copy.deepcopy(fc.meta)
    meta["stage_names"] = vocab
    cand = Forecaster(model, meta)
    vp = cand.predict_batch(X, va)["attack_prob"].max(axis=1)
    q, p_ = onset_masks(st["is_attack"].to_numpy(), va, vat)
    thr, cal = choose_threshold(vp, q, p_, cfg["calibration"]["target_false_positive_rate"], fc.threshold)
    thr = float(np.clip(0.5 * thr + 0.5 * fc.threshold, 0.05, 0.95)) if cal["method"].startswith("fallback") is False else fc.threshold
    cand.threshold = thr; meta["warning_threshold"] = thr

    log("[LEARN] scoring current vs candidate on held-out blocks of the new data ...")
    before = evaluate_model(fc, st, part, X, cfg, "test", False, lambda *_: None)
    after = evaluate_model(cand, st, part, X, cfg, "test", False, lambda *_: None)
    forget = None
    if base is not None:
        bX = fc.pre.transform(base)
        b_before = evaluate_model(fc, base, bpart, bX, cfg, "test", False, lambda *_: None)
        b_after = evaluate_model(cand, base, bpart, bX, cfg, "test", False, lambda *_: None)
        forget = {"before": b_before["forecast"], "after": b_after["forecast"], "onset_before": b_before["onset"], "onset_after": b_after["onset"]}

    idx = _index()
    vid = f"v{len(idx):03d}_{time.strftime('%Y%m%d_%H%M%S')}"
    d = VERSIONS / vid
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "world_model.pt")
    meta["trained_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    meta["model"]["parameters"] = int(sum(p.numel() for p in model.parameters()))
    meta["lineage"] = {"parent": active_version(), "learned_from": source_name, "new_windows": int(len(st)), "epochs": epochs,
                       "new_tactics": sorted(new_stage), "learning_rate": lr, "anchor_strength": anchor}
    (d / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (d / "evaluation.json").write_text(json.dumps(after, indent=2, default=str), encoding="utf-8")
    if (MODELS_DIR / "background_sequences.npy").exists():
        shutil.copy2(MODELS_DIR / "background_sequences.npy", d / "background_sequences.npy")
    idx.append({"id": vid, "created": meta["trained_at"], "status": "candidate", "parent": active_version(),
                "note": f"Learned from {source_name} ({len(st):,} windows)", "stages": vocab, "threshold": thr})
    _save_index(idx)
    store.audit("model_fine_tune", f"{vid} from {source_name}")
    res = {"version": vid, "before": before, "after": after, "forgetting": forget, "new_tactics": sorted(new_stage),
           "threshold": thr, "seconds": round(time.time() - t0, 1), "windows": int(len(st))}
    (d / "learning_report.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    log(f"[LEARN] candidate {vid} ready in {res['seconds']}s")
    return res
