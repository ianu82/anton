from __future__ import annotations

from dataclasses import dataclass

from anton.llm.provider import Usage


@dataclass(slots=True)
class BudgetConfig:
    max_tokens_per_run: int = 100_000
    max_tool_calls_per_run: int = 60


class BudgetTracker:
    def __init__(self, config: BudgetConfig) -> None:
        self._config = config
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._tool_calls = 0

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

    def record_usage(self, usage: Usage) -> None:
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens

    def record_tool_call(self) -> None:
        self._tool_calls += 1

    def check(self) -> str | None:
        if self.total_tokens > self._config.max_tokens_per_run:
            return (
                "Run token budget exceeded "
                f"({self.total_tokens} > {self._config.max_tokens_per_run})."
            )
        if self._tool_calls > self._config.max_tool_calls_per_run:
            return (
                "Run tool-call budget exceeded "
                f"({self._tool_calls} > {self._config.max_tool_calls_per_run})."
            )
        return None
