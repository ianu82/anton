from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class ServiceStore:
    """SQLite-backed store for session/run/audit state."""

    def __init__(self, db_path: Path, *, audit_log_path: Path | None = None) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_log_path = audit_log_path
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                      id TEXT PRIMARY KEY,
                      created_at REAL NOT NULL,
                      workspace_path TEXT NOT NULL,
                      metadata TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS runs (
                      id TEXT PRIMARY KEY,
                      session_id TEXT NOT NULL,
                      created_at REAL NOT NULL,
                      started_at REAL NOT NULL,
                      completed_at REAL,
                      status TEXT NOT NULL,
                      prompt TEXT NOT NULL,
                      response TEXT,
                      error TEXT,
                      input_tokens INTEGER NOT NULL DEFAULT 0,
                      output_tokens INTEGER NOT NULL DEFAULT 0,
                      cancel_requested INTEGER NOT NULL DEFAULT 0,
                      worker_mode TEXT NOT NULL DEFAULT 'local',
                      pending_approval_ids TEXT NOT NULL DEFAULT '[]',
                      FOREIGN KEY(session_id) REFERENCES sessions(id)
                    );

                    CREATE TABLE IF NOT EXISTS run_idempotency (
                      session_id TEXT NOT NULL,
                      idempotency_key TEXT NOT NULL,
                      run_id TEXT NOT NULL,
                      created_at REAL NOT NULL,
                      PRIMARY KEY(session_id, idempotency_key)
                    );

                    CREATE TABLE IF NOT EXISTS run_events (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id TEXT NOT NULL,
                      run_id TEXT NOT NULL,
                      ts REAL NOT NULL,
                      event_type TEXT NOT NULL,
                      payload TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_run_events_session_id_id ON run_events(session_id, id);
                    CREATE INDEX IF NOT EXISTS idx_run_events_run_id ON run_events(run_id);
                    CREATE INDEX IF NOT EXISTS idx_runs_session_id ON runs(session_id, created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status, created_at DESC);

                    CREATE TABLE IF NOT EXISTS approvals (
                      id TEXT PRIMARY KEY,
                      session_id TEXT NOT NULL,
                      run_id TEXT NOT NULL,
                      tool_name TEXT NOT NULL,
                      tool_input TEXT NOT NULL,
                      fingerprint TEXT NOT NULL,
                      reason TEXT NOT NULL,
                      status TEXT NOT NULL,
                      created_at REAL NOT NULL,
                      decided_at REAL,
                      decision_note TEXT
                    );

                    CREATE INDEX IF NOT EXISTS idx_approvals_session_fingerprint ON approvals(session_id, fingerprint, created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status, created_at DESC);

                    CREATE TABLE IF NOT EXISTS artifacts (
                      id TEXT PRIMARY KEY,
                      run_id TEXT NOT NULL,
                      kind TEXT NOT NULL,
                      path TEXT NOT NULL,
                      created_at REAL NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON artifacts(run_id);

                    CREATE TABLE IF NOT EXISTS skills (
                      id TEXT PRIMARY KEY,
                      name TEXT NOT NULL,
                      description TEXT NOT NULL,
                      latest_version INTEGER NOT NULL,
                      created_at REAL NOT NULL,
                      updated_at REAL NOT NULL,
                      metadata TEXT NOT NULL
                    );

                    CREATE UNIQUE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
                    CREATE INDEX IF NOT EXISTS idx_skills_updated_at ON skills(updated_at DESC);

                    CREATE TABLE IF NOT EXISTS skill_versions (
                      id TEXT PRIMARY KEY,
                      skill_id TEXT NOT NULL,
                      version INTEGER NOT NULL,
                      prompt_template TEXT NOT NULL,
                      required_params TEXT NOT NULL,
                      created_at REAL NOT NULL,
                      FOREIGN KEY(skill_id) REFERENCES skills(id)
                    );

                    CREATE UNIQUE INDEX IF NOT EXISTS idx_skill_versions_skill_version
                    ON skill_versions(skill_id, version);
                    CREATE INDEX IF NOT EXISTS idx_skill_versions_skill_id
                    ON skill_versions(skill_id, created_at DESC);

                    CREATE TABLE IF NOT EXISTS scheduled_runs (
                      id TEXT PRIMARY KEY,
                      name TEXT NOT NULL,
                      session_id TEXT NOT NULL,
                      skill_id TEXT NOT NULL,
                      skill_version INTEGER,
                      params TEXT NOT NULL,
                      interval_seconds INTEGER NOT NULL,
                      next_run_at REAL NOT NULL,
                      status TEXT NOT NULL,
                      last_run_id TEXT,
                      last_run_at REAL,
                      created_at REAL NOT NULL,
                      updated_at REAL NOT NULL,
                      FOREIGN KEY(session_id) REFERENCES sessions(id),
                      FOREIGN KEY(skill_id) REFERENCES skills(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_scheduled_runs_status_next
                    ON scheduled_runs(status, next_run_at);
                    CREATE INDEX IF NOT EXISTS idx_scheduled_runs_session_id
                    ON scheduled_runs(session_id, created_at DESC);

                    CREATE TABLE IF NOT EXISTS usage_ledger (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id TEXT NOT NULL,
                      run_id TEXT NOT NULL,
                      ts REAL NOT NULL,
                      metric TEXT NOT NULL,
                      amount REAL NOT NULL,
                      metadata TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_usage_run_id ON usage_ledger(run_id);
                    """
                )
                self._ensure_runs_columns(conn)
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _ensure_runs_columns(conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(runs)").fetchall()
        existing = {str(row[1]) for row in rows}

        if "cancel_requested" not in existing:
            conn.execute("ALTER TABLE runs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0")
        if "worker_mode" not in existing:
            conn.execute("ALTER TABLE runs ADD COLUMN worker_mode TEXT NOT NULL DEFAULT 'local'")

    def create_session(
        self,
        *,
        session_id: str,
        workspace_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        payload = json.dumps(metadata or {}, sort_keys=True)

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO sessions(id, created_at, workspace_path, metadata) VALUES(?, ?, ?, ?)",
                    (session_id, now, workspace_path, payload),
                )
                conn.commit()
            finally:
                conn.close()

        return {
            "id": session_id,
            "created_at": now,
            "workspace_path": workspace_path,
            "metadata": metadata or {},
        }

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id, created_at, workspace_path, metadata FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            return None
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "workspace_path": row["workspace_path"],
            "metadata": json.loads(row["metadata"]),
        }

    def create_run(
        self,
        *,
        run_id: str,
        session_id: str,
        prompt: str,
        worker_mode: str = "local",
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO runs(
                      id, session_id, created_at, started_at, status, prompt,
                      worker_mode, pending_approval_ids
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, '[]')
                    """,
                    (run_id, session_id, now, now, "running", prompt, worker_mode),
                )
                conn.commit()
            finally:
                conn.close()

        return {
            "id": run_id,
            "session_id": session_id,
            "created_at": now,
            "started_at": now,
            "status": "running",
            "prompt": prompt,
            "worker_mode": worker_mode,
        }

    def reserve_idempotency_key(
        self,
        *,
        session_id: str,
        idempotency_key: str,
        run_id: str,
    ) -> str:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                try:
                    conn.execute(
                        """
                        INSERT INTO run_idempotency(session_id, idempotency_key, run_id, created_at)
                        VALUES(?, ?, ?, ?)
                        """,
                        (session_id, idempotency_key, run_id, now),
                    )
                    conn.commit()
                    return run_id
                except sqlite3.IntegrityError:
                    row = conn.execute(
                        """
                        SELECT run_id FROM run_idempotency
                        WHERE session_id = ? AND idempotency_key = ?
                        """,
                        (session_id, idempotency_key),
                    ).fetchone()
                    if row is None:
                        raise
                    return str(row["run_id"])
            finally:
                conn.close()

    def run_id_for_idempotency_key(
        self,
        *,
        session_id: str,
        idempotency_key: str,
    ) -> str | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT run_id FROM run_idempotency
                    WHERE session_id = ? AND idempotency_key = ?
                    """,
                    (session_id, idempotency_key),
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return None
        return str(row["run_id"])

    def complete_run(
        self,
        *,
        run_id: str,
        status: str,
        response: str,
        error: str | None,
        input_tokens: int,
        output_tokens: int,
        pending_approval_ids: list[str],
    ) -> None:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE runs
                    SET completed_at = ?, status = ?, response = ?, error = ?,
                        input_tokens = ?, output_tokens = ?, pending_approval_ids = ?
                    WHERE id = ?
                    """,
                    (
                        now,
                        status,
                        response,
                        error,
                        input_tokens,
                        output_tokens,
                        json.dumps(pending_approval_ids),
                        run_id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def set_run_status(self, *, run_id: str, status: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE runs SET status = ? WHERE id = ?",
                    (status, run_id),
                )
                conn.commit()
            finally:
                conn.close()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM runs WHERE id = ?",
                    (run_id,),
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            return None
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "status": row["status"],
            "prompt": row["prompt"],
            "response": row["response"],
            "error": row["error"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "cancel_requested": bool(row["cancel_requested"]),
            "worker_mode": row["worker_mode"],
            "pending_approval_ids": json.loads(row["pending_approval_ids"]),
        }

    def request_run_cancel(self, run_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE runs
                    SET cancel_requested = 1
                    WHERE id = ? AND completed_at IS NULL
                    """,
                    (run_id,),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def is_run_cancel_requested(self, run_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT cancel_requested FROM runs WHERE id = ?",
                    (run_id,),
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            return False
        return bool(row["cancel_requested"])

    def list_runs(self, *, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, created_at, completed_at, status, prompt, error,
                           input_tokens, output_tokens, cancel_requested, worker_mode
                    FROM runs
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (session_id, max(1, limit)),
                ).fetchall()
            finally:
                conn.close()

        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "status": row["status"],
                "prompt": row["prompt"],
                "error": row["error"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "cancel_requested": bool(row["cancel_requested"]),
                "worker_mode": row["worker_mode"],
            }
            for row in rows
        ]

    def append_event(
        self,
        *,
        session_id: str,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> int:
        now = time.time()
        payload_json = json.dumps(payload, sort_keys=True)

        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO run_events(session_id, run_id, ts, event_type, payload)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (session_id, run_id, now, event_type, payload_json),
                )
                event_id = int(cur.lastrowid)
                conn.commit()
            finally:
                conn.close()

        if self._audit_log_path is not None:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(
                {
                    "id": event_id,
                    "ts": now,
                    "session_id": session_id,
                    "run_id": run_id,
                    "event_type": event_type,
                    "payload": payload,
                },
                sort_keys=True,
            )
            with open(self._audit_log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

        return event_id

    def list_events(self, *, session_id: str, since_id: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, run_id, ts, event_type, payload
                    FROM run_events
                    WHERE session_id = ? AND id > ?
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (session_id, since_id, max(1, limit)),
                ).fetchall()
            finally:
                conn.close()

        return [
            {
                "id": int(row["id"]),
                "run_id": row["run_id"],
                "ts": row["ts"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload"]),
            }
            for row in rows
        ]

    def list_run_events(self, *, run_id: str, limit: int = 2000) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, session_id, ts, event_type, payload
                    FROM run_events
                    WHERE run_id = ?
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (run_id, max(1, limit)),
                ).fetchall()
            finally:
                conn.close()

        return [
            {
                "id": int(row["id"]),
                "session_id": row["session_id"],
                "ts": row["ts"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload"]),
            }
            for row in rows
        ]

    def create_approval(
        self,
        *,
        session_id: str,
        run_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        fingerprint: str,
        reason: str,
    ) -> str:
        approval_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO approvals(
                        id, session_id, run_id, tool_name, tool_input, fingerprint,
                        reason, status, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        approval_id,
                        session_id,
                        run_id,
                        tool_name,
                        json.dumps(tool_input, sort_keys=True),
                        fingerprint,
                        reason,
                        "pending",
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return approval_id

    def find_approval_decision(self, *, session_id: str, fingerprint: str) -> str | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT status FROM approvals
                    WHERE session_id = ? AND fingerprint = ? AND status IN ('approved', 'rejected')
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (session_id, fingerprint),
                ).fetchone()
            finally:
                conn.close()
        return None if row is None else str(row["status"])

    def list_approvals(
        self,
        *,
        session_id: str | None = None,
        status: str = "pending",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        where = ["status = ?"]
        params: list[Any] = [status]
        if session_id is not None:
            where.append("session_id = ?")
            params.append(session_id)

        sql = (
            "SELECT id, session_id, run_id, tool_name, tool_input, reason, status, "
            "created_at, decided_at, decision_note "
            "FROM approvals WHERE " + " AND ".join(where) + " ORDER BY created_at DESC LIMIT ?"
        )
        params.append(max(1, limit))

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, tuple(params)).fetchall()
            finally:
                conn.close()

        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "run_id": row["run_id"],
                "tool_name": row["tool_name"],
                "tool_input": json.loads(row["tool_input"]),
                "reason": row["reason"],
                "status": row["status"],
                "created_at": row["created_at"],
                "decided_at": row["decided_at"],
                "decision_note": row["decision_note"],
            }
            for row in rows
        ]

    def set_approval_decision(self, *, approval_id: str, approved: bool, note: str | None = None) -> dict[str, Any] | None:
        status = "approved" if approved else "rejected"
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE approvals
                    SET status = ?, decided_at = ?, decision_note = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (status, now, note or "", approval_id),
                )
                conn.commit()
                if cur.rowcount == 0:
                    row = conn.execute(
                        """
                        SELECT id, session_id, run_id, tool_name, tool_input, reason,
                               status, created_at, decided_at, decision_note
                        FROM approvals WHERE id = ?
                        """,
                        (approval_id,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        """
                        SELECT id, session_id, run_id, tool_name, tool_input, reason,
                               status, created_at, decided_at, decision_note
                        FROM approvals WHERE id = ?
                        """,
                        (approval_id,),
                    ).fetchone()
            finally:
                conn.close()

        if row is None:
            return None
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "run_id": row["run_id"],
            "tool_name": row["tool_name"],
            "tool_input": json.loads(row["tool_input"]),
            "reason": row["reason"],
            "status": row["status"],
            "created_at": row["created_at"],
            "decided_at": row["decided_at"],
            "decision_note": row["decision_note"],
        }

    def add_artifact(self, *, run_id: str, kind: str, path: str) -> str:
        artifact_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO artifacts(id, run_id, kind, path, created_at) VALUES(?, ?, ?, ?, ?)",
                    (artifact_id, run_id, kind, path, now),
                )
                conn.commit()
            finally:
                conn.close()
        return artifact_id

    def list_artifacts(self, *, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT id, kind, path, created_at FROM artifacts WHERE run_id = ? ORDER BY created_at ASC",
                    (run_id,),
                ).fetchall()
            finally:
                conn.close()
        return [
            {
                "id": row["id"],
                "kind": row["kind"],
                "path": row["path"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def create_skill(
        self,
        *,
        name: str,
        description: str,
        prompt_template: str,
        required_params: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        skill_id = uuid.uuid4().hex[:12]
        version_id = uuid.uuid4().hex[:12]
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO skills(
                      id, name, description, latest_version, created_at, updated_at, metadata
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        skill_id,
                        name,
                        description,
                        1,
                        now,
                        now,
                        json.dumps(metadata or {}, sort_keys=True),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO skill_versions(
                      id, skill_id, version, prompt_template, required_params, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version_id,
                        skill_id,
                        1,
                        prompt_template,
                        json.dumps(sorted(set(required_params))),
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        skill = self.get_skill(skill_id)
        if skill is None:  # pragma: no cover - defensive
            raise RuntimeError(f"Skill '{skill_id}' was created but could not be loaded.")
        return skill

    def add_skill_version(
        self,
        *,
        skill_id: str,
        prompt_template: str,
        required_params: list[str],
    ) -> dict[str, Any] | None:
        now = time.time()
        version_id = uuid.uuid4().hex[:12]
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT latest_version FROM skills WHERE id = ?",
                    (skill_id,),
                ).fetchone()
                if row is None:
                    return None
                next_version = int(row["latest_version"]) + 1
                conn.execute(
                    """
                    INSERT INTO skill_versions(
                      id, skill_id, version, prompt_template, required_params, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version_id,
                        skill_id,
                        next_version,
                        prompt_template,
                        json.dumps(sorted(set(required_params))),
                        now,
                    ),
                )
                conn.execute(
                    "UPDATE skills SET latest_version = ?, updated_at = ? WHERE id = ?",
                    (next_version, now, skill_id),
                )
                conn.commit()
            finally:
                conn.close()
        return self.get_skill(skill_id)

    def list_skills(self, *, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, name, description, latest_version, created_at, updated_at, metadata
                    FROM skills
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (max(1, limit),),
                ).fetchall()
            finally:
                conn.close()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "latest_version": int(row["latest_version"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]

    def get_skill(self, skill_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                skill_row = conn.execute(
                    """
                    SELECT id, name, description, latest_version, created_at, updated_at, metadata
                    FROM skills
                    WHERE id = ?
                    """,
                    (skill_id,),
                ).fetchone()
                if skill_row is None:
                    return None
                version_row = conn.execute(
                    """
                    SELECT version, prompt_template, required_params, created_at
                    FROM skill_versions
                    WHERE skill_id = ? AND version = ?
                    """,
                    (skill_id, int(skill_row["latest_version"])),
                ).fetchone()
            finally:
                conn.close()
        if version_row is None:
            return None
        return {
            "id": skill_row["id"],
            "name": skill_row["name"],
            "description": skill_row["description"],
            "latest_version": int(skill_row["latest_version"]),
            "created_at": skill_row["created_at"],
            "updated_at": skill_row["updated_at"],
            "metadata": json.loads(skill_row["metadata"]),
            "latest_prompt_template": version_row["prompt_template"],
            "latest_required_params": json.loads(version_row["required_params"]),
            "latest_version_created_at": version_row["created_at"],
        }

    def get_skill_template(self, *, skill_id: str, version: int | None = None) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                skill_row = conn.execute(
                    """
                    SELECT id, name, description, latest_version
                    FROM skills
                    WHERE id = ?
                    """,
                    (skill_id,),
                ).fetchone()
                if skill_row is None:
                    return None
                resolved_version = int(version or skill_row["latest_version"])
                version_row = conn.execute(
                    """
                    SELECT version, prompt_template, required_params, created_at
                    FROM skill_versions
                    WHERE skill_id = ? AND version = ?
                    """,
                    (skill_id, resolved_version),
                ).fetchone()
            finally:
                conn.close()
        if version_row is None:
            return None
        return {
            "skill_id": skill_row["id"],
            "name": skill_row["name"],
            "description": skill_row["description"],
            "version": int(version_row["version"]),
            "prompt_template": version_row["prompt_template"],
            "required_params": json.loads(version_row["required_params"]),
            "created_at": version_row["created_at"],
        }

    def create_schedule(
        self,
        *,
        name: str,
        session_id: str,
        skill_id: str,
        skill_version: int | None,
        params: dict[str, str],
        interval_seconds: int,
        start_at: float,
        status: str = "active",
    ) -> dict[str, Any]:
        schedule_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO scheduled_runs(
                      id, name, session_id, skill_id, skill_version, params,
                      interval_seconds, next_run_at, status, last_run_id, last_run_at,
                      created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                    """,
                    (
                        schedule_id,
                        name,
                        session_id,
                        skill_id,
                        skill_version,
                        json.dumps(params, sort_keys=True),
                        int(interval_seconds),
                        float(start_at),
                        status,
                        now,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        schedule = self.get_schedule(schedule_id)
        if schedule is None:  # pragma: no cover - defensive
            raise RuntimeError(f"Schedule '{schedule_id}' was created but could not be loaded.")
        return schedule

    def get_schedule(self, schedule_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT id, name, session_id, skill_id, skill_version, params,
                           interval_seconds, next_run_at, status, last_run_id, last_run_at,
                           created_at, updated_at
                    FROM scheduled_runs
                    WHERE id = ?
                    """,
                    (schedule_id,),
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "session_id": row["session_id"],
            "skill_id": row["skill_id"],
            "skill_version": row["skill_version"],
            "params": json.loads(row["params"]),
            "interval_seconds": int(row["interval_seconds"]),
            "next_run_at": row["next_run_at"],
            "status": row["status"],
            "last_run_id": row["last_run_id"],
            "last_run_at": row["last_run_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_schedules(self, *, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        where = ""
        params: tuple[Any, ...]
        if status is None:
            params = (max(1, limit),)
        else:
            where = "WHERE status = ? "
            params = (status, max(1, limit))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, name, session_id, skill_id, skill_version, params,
                           interval_seconds, next_run_at, status, last_run_id, last_run_at,
                           created_at, updated_at
                    FROM scheduled_runs
                    """
                    + where
                    + "ORDER BY created_at DESC LIMIT ?",
                    params,
                ).fetchall()
            finally:
                conn.close()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "session_id": row["session_id"],
                "skill_id": row["skill_id"],
                "skill_version": row["skill_version"],
                "params": json.loads(row["params"]),
                "interval_seconds": int(row["interval_seconds"]),
                "next_run_at": row["next_run_at"],
                "status": row["status"],
                "last_run_id": row["last_run_id"],
                "last_run_at": row["last_run_at"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def record_schedule_trigger(self, *, schedule_id: str, run_id: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT interval_seconds
                    FROM scheduled_runs
                    WHERE id = ?
                    """,
                    (schedule_id,),
                ).fetchone()
                if row is None:
                    return None
                next_run = now + int(row["interval_seconds"])
                conn.execute(
                    """
                    UPDATE scheduled_runs
                    SET last_run_id = ?, last_run_at = ?, next_run_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (run_id, now, next_run, now, schedule_id),
                )
                conn.commit()
            finally:
                conn.close()
        return self.get_schedule(schedule_id)

    def set_schedule_status(self, *, schedule_id: str, status: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE scheduled_runs
                    SET status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, now, schedule_id),
                )
                conn.commit()
                if cur.rowcount == 0:
                    return None
            finally:
                conn.close()
        return self.get_schedule(schedule_id)

    def append_usage(
        self,
        *,
        session_id: str,
        run_id: str,
        metric: str,
        amount: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO usage_ledger(session_id, run_id, ts, metric, amount, metadata)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        run_id,
                        time.time(),
                        metric,
                        float(amount),
                        json.dumps(metadata or {}, sort_keys=True),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def metrics_summary(self) -> dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                completed_rows = conn.execute(
                    "SELECT status, started_at, completed_at FROM runs WHERE completed_at IS NOT NULL"
                ).fetchall()
                status_counts = conn.execute(
                    "SELECT status, COUNT(*) AS count FROM runs GROUP BY status"
                ).fetchall()
            finally:
                conn.close()

        latencies: list[float] = []
        successes = 0
        failures = 0
        approvals = 0
        cancelled = 0

        for row in completed_rows:
            status = row["status"]
            if status == "completed":
                successes += 1
            elif status == "approval_required":
                approvals += 1
            elif status == "cancelled":
                cancelled += 1
            else:
                failures += 1
            latency = float(row["completed_at"] - row["started_at"])
            if latency >= 0:
                latencies.append(latency)

        latencies.sort()

        def pct(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            idx = int((len(values) - 1) * p)
            return values[idx]

        by_status = {str(row["status"]): int(row["count"]) for row in status_counts}

        return {
            "run_count": sum(by_status.values()),
            "completed": successes,
            "failed": failures,
            "approval_required": approvals,
            "cancelled": cancelled,
            "running": by_status.get("running", 0),
            "queued": by_status.get("queued", 0),
            "status_counts": by_status,
            "latency_p50": pct(latencies, 0.5),
            "latency_p95": pct(latencies, 0.95),
            "success_rate": (successes / len(completed_rows)) if completed_rows else 0.0,
        }
