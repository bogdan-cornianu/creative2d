"""SQLite-backed job records."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    status TEXT NOT NULL,          -- queued | running | done | failed | cancelled
    progress REAL NOT NULL DEFAULT 0,
    stage TEXT NOT NULL DEFAULT '',
    message TEXT NOT NULL DEFAULT '',
    spec TEXT NOT NULL,
    result TEXT,
    error TEXT
);
"""

FINAL = {"done", "failed", "cancelled"}


class JobStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)
            # Jobs interrupted by a restart can not resume.
            self._db.execute(
                "UPDATE jobs SET status='failed', error='server restarted' WHERE status IN ('queued','running')"
            )
            self._db.commit()

    def create(self, spec: dict) -> str:
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs (id, created, updated, status, spec) VALUES (?, ?, ?, 'queued', ?)",
                (job_id, now, now, json.dumps(spec)),
            )
            self._db.commit()
        return job_id

    def update(self, job_id: str, **fields) -> None:
        if "result" in fields and fields["result"] is not None:
            fields["result"] = json.dumps(fields["result"])
        fields["updated"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))
            self._db.commit()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r, include_result=False) for r in rows]

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            self._db.commit()

    @staticmethod
    def _row(row: sqlite3.Row, include_result: bool = True) -> dict:
        d = dict(row)
        d["spec"] = json.loads(d["spec"])
        result = json.loads(d["result"]) if d["result"] else None
        if include_result:
            d["result"] = result
        else:
            d.pop("result", None)
            d["thumb"] = _thumb(result)
        return d


def _thumb(manifest: dict | None) -> str | None:
    """Path under the job's assets/ of one image that represents it in the job list."""
    if not manifest:
        return None
    r = manifest.get("result", {})
    if r.get("kind") == "background":
        layers = [e["texture"] for e in r.get("entries", []) if e.get("loader") == "image" and e.get("texture")]
        return layers[0] if layers else None
    frames = r.get("frames") or []
    return f"frames/{frames[0]}.png" if frames else None
