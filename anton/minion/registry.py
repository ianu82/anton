from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


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
    cron_expr: str | None = None  # e.g. "*/5 * * * *" for scheduled minions

    @staticmethod
    def make_id() -> str:
        return uuid.uuid4().hex[:12]


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
        """List minions that have a cron schedule."""
        return [m for m in self._minions.values() if m.cron_expr is not None]

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
            # If killed, clear the cron schedule so it won't be rescheduled
            if status == MinionStatus.KILLED:
                minion.cron_expr = None

        return True

    def remove(self, minion_id: str) -> bool:
        """Remove a minion from the registry. Returns True if found."""
        return self._minions.pop(minion_id, None) is not None
