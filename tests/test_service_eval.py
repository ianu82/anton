from __future__ import annotations

import json
from pathlib import Path

from anton.config.settings import AntonSettings
from anton.llm.provider import LLMResponse, StreamComplete, StreamTextDelta, Usage
from anton.service.eval import run_eval_sync


class EvalLLM:
    coding_model = "dummy-coder"

    async def plan_stream(self, **kwargs):
        response = LLMResponse(
            content="analysis complete",
            tool_calls=[],
            usage=Usage(input_tokens=4, output_tokens=6),
            stop_reason="end_turn",
        )
        yield StreamTextDelta(text="analysis complete")
        yield StreamComplete(response=response)

    async def plan(self, **kwargs):
        return LLMResponse(
            content="analysis complete",
            tool_calls=[],
            usage=Usage(input_tokens=4, output_tokens=6),
            stop_reason="end_turn",
        )


def test_eval_harness_runs_tasks(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: EvalLLM())

    settings = AntonSettings()
    settings.resolve_workspace(str(tmp_path))

    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            [
                {"name": "t1", "prompt": "hello", "expect_contains": "analysis"},
                {"name": "t2", "prompt": "world", "expect_contains": "analysis"},
            ]
        )
    )

    output_path = tmp_path / "summary.json"

    summary = run_eval_sync(
        settings=settings,
        tasks_path=tasks_path,
        output_path=output_path,
    )

    assert summary["run_count"] == 2
    assert summary["passed_count"] == 2
    assert output_path.exists()
