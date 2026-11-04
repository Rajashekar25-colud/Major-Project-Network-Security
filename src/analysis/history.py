"""Analysis history persisted locally in data/history: the processed windows of each analysed file + a small metadata
record. Never model artifacts and never anything used for training."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ROOT

HISTORY_DIR = ROOT / "data" / "history"


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", Path(name).stem).strip("-").lower()[:30] or "traffic"


def new_id(name: str, when: datetime | None = None) -> str:
    return f"{(when or datetime.now()):%Y%m%d-%H%M%S}-{_slug(name)}"


def save_analysis(res, directory: Path | None = None, owner: str | None = None) -> None:
    import numpy as np

    from .workflow import alert_records, level_label
    d = Path(directory or HISTORY_DIR); d.mkdir(parents=True, exist_ok=True)
    risk = np.asarray(res.rs.dec["risk"], dtype=float) if res.n else np.array([])
    peak = float(np.nanmax(risk)) if risk.size else None
    meta = {"id": res.id, "owner": owner, "peak_risk": peak, "state": level_label(res, peak) if peak is not None else None, "filename": res.name, "created": res.created.isoformat(timespec="seconds"), "kind": res.info.get("kind"),
            "records": res.stats.get("records"), "windows": res.stats.get("windows"), "forecast_windows": res.n, "alerts": len(alert_records(res)),
            "labelled": bool(res.labelled), "threshold": res.thr, "start": str(res.stats.get("start")), "end": str(res.stats.get("end")),
            "model_fingerprint": (res.integrity_after or {}).get("combined"),
            "info": {k: v for k, v in res.info.items() if k in ("kind", "records", "labelled", "windows", "optional_fields")}}
    res.states.to_csv(d / f"{res.id}.states.csv.gz", index=False)
    (d / f"{res.id}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def list_history(directory: Path | None = None, owner: str | None = None) -> list[dict[str, Any]]:
    """All saved analyses, newest first. ``owner`` restricts the list to that account (None = every account, for administrators)."""
    d = Path(directory or HISTORY_DIR); out = []
    for p in (sorted(d.glob("*.json"), reverse=True) if d.exists() else []):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if owner is not None and str(m.get("owner") or "").lower() != owner.lower():
            continue
        if (d / f"{m.get('id')}.states.csv.gz").exists():
            out.append(m)
    return sorted(out, key=lambda m: m.get("created", ""), reverse=True)


def load_states(analysis_id: str, directory: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(Path(directory or HISTORY_DIR) / f"{analysis_id}.states.csv.gz", parse_dates=["timestamp"])
    for c in ("is_attack", "filled"):
        df[c] = df[c].astype(bool)
    df["segment"] = df["segment"].astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


def delete_analysis(analysis_id: str, directory: Path | None = None) -> None:
    d = Path(directory or HISTORY_DIR)
    for suf in (".json", ".states.csv.gz"):
        (d / f"{analysis_id}{suf}").unlink(missing_ok=True)
