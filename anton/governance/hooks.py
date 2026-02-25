from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from anton.llm.provider import Usage


@dataclass(slots=True)
class ToolGateResult:
    allow: bool
    message: str = ""
    pending_approval_id: str | None = None


ToolGate = Callable[[str, dict[str, Any]], ToolGateResult | Awaitable[ToolGateResult]]
UsageHook = Callable[[Usage], None]
AuditHook = Callable[[str, dict[str, Any]], None]
