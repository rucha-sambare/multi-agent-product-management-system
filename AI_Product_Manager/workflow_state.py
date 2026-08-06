"""Persistent workflow state and human-review decisions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from AI_Product_Manager.config import settings


class WorkflowStore:
    def __init__(self, db_path=None):
        settings.ensure_directories()
        self.db_path = str(db_path or settings.state_db)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, app_name TEXT NOT NULL,
                    app_id TEXT, status TEXT NOT NULL, current_step TEXT NOT NULL,
                    payload TEXT NOT NULL, error TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                )"""
            )

    def create(self, app_name, app_id=None):
        run_id = uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, app_name, app_id, "running", "created", "{}", None, now, now),
            )
        return run_id

    def update(self, run_id, *, status=None, step=None, payload=None, error=None):
        current = self.get(run_id)
        if not current:
            raise KeyError(f"Unknown run: {run_id}")
        merged = current["payload"]
        if payload:
            merged.update(payload)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute(
                """UPDATE runs SET status=?, current_step=?, payload=?, error=?,
                   updated_at=? WHERE run_id=?""",
                (
                    status or current["status"],
                    step or current["current_step"],
                    json.dumps(merged, default=str),
                    error,
                    now,
                    run_id,
                ),
            )

    def get(self, run_id):
        with self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        return result

    def list(self, limit=50):
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

