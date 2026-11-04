"""Local incident store (SQLite): triage status, notes and an audit trail. No external services."""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ROOT

DB_PATH = ROOT / "data" / "app.db"
STATUSES = ["New", "Investigating", "Resolved", "False positive"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents(
  id TEXT PRIMARY KEY, source TEXT, opened_time TEXT, closed_time TEXT, stage TEXT, progression TEXT,
  severity TEXT, peak_risk REAL, eta_seconds REAL, windows INTEGER, confirmed INTEGER DEFAULT -1,
  status TEXT DEFAULT 'New', note TEXT DEFAULT '', first_seen TEXT DEFAULT CURRENT_TIMESTAMP, updated TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS audit(ts TEXT DEFAULT CURRENT_TIMESTAMP, action TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS ix_inc_time ON incidents(opened_time);
"""


@contextmanager
def _db(path: Path | None = None):
    p = Path(path or DB_PATH); p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def incident_id(source: str, opened_time: Any, stage: str) -> str:
    h = hashlib.sha1(f"{source}|{opened_time}|{stage}".encode()).hexdigest()[:6].upper()
    return f"INC-{h}"


def upsert_incidents(df: pd.DataFrame, path: Path | None = None) -> int:
    """Insert new incidents; keep triage status/notes of existing ones (idempotent)."""
    if df is None or df.empty:
        return 0
    n = 0
    with _db(path) as con:
        for r in df.itertuples(index=False):
            cur = con.execute(
                "INSERT INTO incidents(id,source,opened_time,closed_time,stage,progression,severity,peak_risk,eta_seconds,windows,confirmed) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET closed_time=excluded.closed_time, severity=excluded.severity, "
                "peak_risk=excluded.peak_risk, windows=excluded.windows, progression=excluded.progression, confirmed=excluded.confirmed",
                (r.id, r.source, str(r.opened_time), None if pd.isna(r.closed_time) else str(r.closed_time), r.stage, r.progression,
                 r.severity, float(r.peak_risk), float(r.eta_seconds), int(r.windows), int(r.confirmed)))
            n += cur.rowcount
    return n


def list_incidents(path: Path | None = None, source: str | None = None) -> pd.DataFrame:
    with _db(path) as con:
        q = "SELECT * FROM incidents" + (" WHERE source=?" if source else "") + " ORDER BY opened_time DESC"
        rows = con.execute(q, (source,) if source else ()).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame(columns=[
        "id", "source", "opened_time", "closed_time", "stage", "progression", "severity", "peak_risk", "eta_seconds", "windows",
        "confirmed", "status", "note", "first_seen", "updated"])


def update_incident(inc_id: str, status: str, note: str, path: Path | None = None) -> None:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    with _db(path) as con:
        con.execute("UPDATE incidents SET status=?, note=?, updated=CURRENT_TIMESTAMP WHERE id=?", (status, note, inc_id))
        con.execute("INSERT INTO audit(action,detail) VALUES(?,?)", ("incident_update", f"{inc_id} -> {status}"))


def audit(action: str, detail: str, path: Path | None = None) -> None:
    with _db(path) as con:
        con.execute("INSERT INTO audit(action,detail) VALUES(?,?)", (action, detail))


def audit_log(limit: int = 100, path: Path | None = None) -> pd.DataFrame:
    with _db(path) as con:
        rows = con.execute("SELECT ts, action, detail FROM audit ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])
