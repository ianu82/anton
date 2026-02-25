from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PolicyDecision:
    allow: bool
    reason: str = ""
    requires_approval: bool = False


@dataclass(slots=True)
class PolicyConfig:
    blocked_packages: set[str] = field(default_factory=lambda: {"torch-nightly", "tensorflow-gpu"})
    allow_connector_writes: bool = True
    require_approval_for_connector_writes: bool = True
    max_estimated_seconds_without_approval: int = 90
    connector_max_query_limit: int = 10_000


class PolicyEngine:
    def __init__(self, config: PolicyConfig) -> None:
        self._config = config

    def evaluate(self, tool_name: str, tool_input: dict) -> PolicyDecision:
        if tool_name == "scratchpad":
            return self._evaluate_scratchpad(tool_input)
        if tool_name == "connector":
            return self._evaluate_connector(tool_input)
        return PolicyDecision(allow=True)

    def _evaluate_scratchpad(self, tool_input: dict) -> PolicyDecision:
        action = str(tool_input.get("action", "")).lower()
        if action == "install":
            packages = tool_input.get("packages", [])
            if isinstance(packages, list):
                blocked = [pkg for pkg in packages if str(pkg).lower() in self._config.blocked_packages]
                if blocked:
                    joined = ", ".join(blocked)
                    return PolicyDecision(
                        allow=False,
                        reason=f"Package(s) blocked by policy: {joined}",
                    )

        if action == "exec":
            estimated_seconds = tool_input.get("estimated_execution_time_seconds", 0)
            try:
                estimate = int(estimated_seconds)
            except (TypeError, ValueError):
                estimate = 0
            if estimate > self._config.max_estimated_seconds_without_approval:
                return PolicyDecision(
                    allow=False,
                    reason=(
                        "Long-running scratchpad execution requires approval "
                        f"(estimate={estimate}s, threshold={self._config.max_estimated_seconds_without_approval}s)."
                    ),
                    requires_approval=True,
                )

        return PolicyDecision(allow=True)

    def _evaluate_connector(self, tool_input: dict) -> PolicyDecision:
        action = str(tool_input.get("action", "")).lower()

        if action == "query":
            limit = tool_input.get("limit", 1000)
            try:
                parsed_limit = int(limit)
            except (TypeError, ValueError):
                parsed_limit = 1000
            if parsed_limit > self._config.connector_max_query_limit:
                return PolicyDecision(
                    allow=False,
                    reason=(
                        f"Query limit {parsed_limit} exceeds policy max "
                        f"{self._config.connector_max_query_limit}."
                    ),
                )

        if action == "write":
            if self._config.allow_connector_writes:
                if self._config.require_approval_for_connector_writes:
                    return PolicyDecision(
                        allow=False,
                        reason="Connector write requires approval.",
                        requires_approval=True,
                    )
                return PolicyDecision(allow=True)

            return PolicyDecision(
                allow=False,
                reason="Connector write operations are disabled by policy.",
            )

        return PolicyDecision(allow=True)
