from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class MinionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class MinionInfo:
    """Represents a tracked minion process."""

    id: str
    task: str
    folder: str
    pid: int | None = None
    status: MinionStatus = MinionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    error: str | None = None
    parent_session_id: str | None = None

    # Scheduling fields
    cron_expr: str | None = None  # e.g. "*/5 * * * *" for cron-style scheduling
    every: str | None = None  # human-friendly frequency: "5m", "1h", "30s"
    start_at: datetime | None = None  # when to start (None = immediately)
    end_at: datetime | None = None  # when to stop (None = no end)
    max_runs: int | None = None  # max number of executions (None = unlimited)
    run_count: int = 0  # how many times this minion has executed

    @staticmethod
    def make_id() -> str:
        return uuid.uuid4().hex[:12]

    @property
    def minion_dir(self) -> Path:
        """Return the minion's dedicated working directory: .anton/minions/<id>/"""
        return Path(self.folder) / ".anton" / "minions" / self.id

    def ensure_dir(self) -> Path:
        """Create the minion directory if it doesn't exist and return its path."""
        d = self.minion_dir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def has_runs_remaining(self) -> bool:
        """Check if the minion can still run based on max_runs."""
        if self.max_runs is None:
            return True
        return self.run_count < self.max_runs

    def is_within_schedule(self, now: datetime | None = None) -> bool:
        """Check if the current time is within the start_at/end_at window."""
        now = now or datetime.now()
        if self.start_at is not None and now < self.start_at:
            return False
        if self.end_at is not None and now > self.end_at:
            return False
        return True

    def record_run(self) -> None:
        """Increment run_count after a successful execution."""
        self.run_count += 1

    def save_status(self) -> None:
        """Persist minion status to its directory as status.json."""
        d = self.ensure_dir()
        data = {
            "id": self.id,
            "task": self.task,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "run_count": self.run_count,
            "max_runs": self.max_runs,
            "every": self.every,
            "cron_expr": self.cron_expr,
            "start_at": self.start_at.isoformat() if self.start_at else None,
            "end_at": self.end_at.isoformat() if self.end_at else None,
        }
        (d / "status.json").write_text(json.dumps(data, indent=2) + "\n")


class MinionRegistry:
    """In-memory registry for tracking minion processes."""

    def __init__(self) -> None:
        self._minions: dict[str, MinionInfo] = {}

    def register(self, minion: MinionInfo) -> None:
        self._minions[minion.id] = minion

    def get(self, minion_id: str) -> MinionInfo | None:
        return self._minions.get(minion_id)

    def list_all(self) -> list[MinionInfo]:
        return list(self._minions.values())

    def list_running(self) -> list[MinionInfo]:
        return [m for m in self._minions.values() if m.status == MinionStatus.RUNNING]

    def list_scheduled(self) -> list[MinionInfo]:
        """List minions that have a schedule (cron or every)."""
        return [
            m for m in self._minions.values()
            if m.cron_expr is not None or m.every is not None
        ]

    def update_status(
        self,
        minion_id: str,
        status: MinionStatus,
        *,
        error: str | None = None,
    ) -> bool:
        """Update a minion's status. Returns True if found."""
        minion = self._minions.get(minion_id)
        if minion is None:
            return False

        minion.status = status
        if error is not None:
            minion.error = error
        if status in (MinionStatus.COMPLETED, MinionStatus.FAILED, MinionStatus.KILLED):
            minion.completed_at = datetime.now()
            # If killed, clear all scheduling so it won't be rescheduled
            if status == MinionStatus.KILLED:
                minion.cron_expr = None
                minion.every = None

        return True

    def remove(self, minion_id: str) -> bool:
        """Remove a minion from the registry. Returns True if found."""
        return self._minions.pop(minion_id, None) is not None
