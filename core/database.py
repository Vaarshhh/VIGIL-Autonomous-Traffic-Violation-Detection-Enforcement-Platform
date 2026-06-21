"""
core/database.py
─────────────────
SQLite persistence for violation history. Stdlib only — no extra dependency,
keeps the prototype runnable on a plain laptop with no DB server setup.

Schema
------
violations(
    id              INTEGER PRIMARY KEY,
    challan_id      TEXT,           -- groups multiple violation rows from the same frame
    frame_id        INTEGER,
    violation_type  TEXT,
    confidence      REAL,
    severity        REAL,
    fine_inr        INTEGER,
    section         TEXT,
    risk_level      TEXT,
    plate_number    TEXT,
    plate_confidence REAL,
    camera_id       TEXT,
    location        TEXT,
    timestamp       TEXT,           -- ISO 8601, also used for date-range search
    evidence_path   TEXT            -- path to the annotated evidence image on disk
)
"""

from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "violations.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    challan_id       TEXT NOT NULL,
    frame_id         INTEGER,
    violation_type   TEXT NOT NULL,
    confidence       REAL,
    severity         REAL,
    fine_inr         INTEGER,
    section          TEXT,
    risk_level       TEXT,
    plate_number     TEXT,
    plate_confidence REAL,
    camera_id        TEXT,
    location         TEXT,
    timestamp        TEXT NOT NULL,
    evidence_path    TEXT
);
CREATE INDEX IF NOT EXISTS idx_violations_plate ON violations(plate_number);
CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp);
CREATE INDEX IF NOT EXISTS idx_violations_type ON violations(violation_type);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def insert_violation(record: dict) -> int:
    """record keys: challan_id, frame_id, violation_type, confidence, severity,
    fine_inr, section, risk_level, plate_number, plate_confidence, camera_id,
    location, timestamp, evidence_path"""
    fields = [
        "challan_id", "frame_id", "violation_type", "confidence", "severity",
        "fine_inr", "section", "risk_level", "plate_number", "plate_confidence",
        "camera_id", "location", "timestamp", "evidence_path",
    ]
    values = [record.get(f) for f in fields]
    placeholders = ", ".join("?" for _ in fields)
    with _connect() as conn:
        cur = conn.execute(
            f"INSERT INTO violations ({', '.join(fields)}) VALUES ({placeholders})",
            values,
        )
        return cur.lastrowid


def search(
    plate: Optional[str] = None,
    date_from: Optional[str] = None,   # "YYYY-MM-DD"
    date_to: Optional[str] = None,     # "YYYY-MM-DD"
    violation_type: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    clauses, params = [], []
    if plate:
        clauses.append("plate_number LIKE ?")
        params.append(f"%{plate.upper()}%")
    if date_from:
        clauses.append("timestamp >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("timestamp <= ?")
        params.append(date_to + " 23:59:59")
    if violation_type:
        clauses.append("violation_type = ?")
        params.append(violation_type)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"SELECT * FROM violations {where} ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_by_challan_id(challan_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM violations WHERE challan_id = ? ORDER BY id", (challan_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_summary() -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM violations").fetchone()["c"]
        total_fine = conn.execute("SELECT COALESCE(SUM(fine_inr), 0) AS s FROM violations").fetchone()["s"]
        unique_plates = conn.execute(
            "SELECT COUNT(DISTINCT plate_number) AS c FROM violations WHERE plate_number IS NOT NULL"
        ).fetchone()["c"]

        by_type_rows = conn.execute(
            "SELECT violation_type, COUNT(*) AS c FROM violations GROUP BY violation_type ORDER BY c DESC"
        ).fetchall()
        by_risk_rows = conn.execute(
            "SELECT risk_level, COUNT(*) AS c FROM violations GROUP BY risk_level ORDER BY c DESC"
        ).fetchall()

    return {
        "total_violations": total,
        "total_fine_inr": total_fine,
        "unique_plates": unique_plates,
        "by_type": {r["violation_type"]: r["c"] for r in by_type_rows},
        "by_risk": {r["risk_level"]: r["c"] for r in by_risk_rows},
    }


def get_trend(days: int = 30) -> list[dict]:
    """Daily violation count + fine total for the last N days, oldest first."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT substr(timestamp, 1, 10) AS day,
                   COUNT(*) AS count,
                   COALESCE(SUM(fine_inr), 0) AS fine
            FROM violations
            WHERE timestamp >= date('now', ?)
            GROUP BY day
            ORDER BY day ASC
            """,
            (f"-{days} days",),
        ).fetchall()
        return [{"date": r["day"], "count": r["count"], "fine": r["fine"]} for r in rows]


def get_recent(limit: int = 20) -> list[dict]:
    return search(limit=limit)
