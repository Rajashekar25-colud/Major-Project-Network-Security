"""Model integrity: fingerprint of the trained artifacts, used to show that analysis never changes the model."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_FILES = ("world_model.pt", "metadata.json")     # weights, and metadata (which holds the scaler statistics)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(models_dir: str | Path) -> dict[str, Any]:
    d = Path(models_dir)
    parts = {f: _sha(d / f) for f in _FILES if (d / f).exists()}
    combined = hashlib.sha256("|".join(f"{k}:{v}" for k, v in sorted(parts.items())).encode()).hexdigest()
    return {"files": parts, "combined": combined, "short": combined[:12]}


def unchanged(before: dict[str, Any] | None, after: dict[str, Any] | None) -> bool:
    return bool(before) and bool(after) and before["combined"] == after["combined"]
