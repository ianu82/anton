from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from anton.events.bus import EventBus
from anton.events.types import Phase, StatusUpdate
from anton.llm.provider import LLMResponse, Usage
from anton.skill.builder import SkillBuilder, _extract_code
from anton.skill.registry import SkillRegistry
from anton.skill.spec import SkillSpec

VALID_SKILL_CODE = '''\
```python
from anton.skill.base import SkillResult, skill


@skill("count_lines", "Count lines in a file")
async def count_lines(path: str) -> SkillResult:
    with open(path) as f:
        lines = f.readlines()
    return SkillResult(output=len(lines), metadata={"path": path})
```
'''

BROKEN_SKILL_CODE = '''\
```python
from anton.skill.base import SkillResult, skill

@skill("count_lines", "Count lines in a file")
async def count_lines(path: str) -> SkillResult:
    # missing return
    x = 1 / 0
```
'''


def _make_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


@pytest.fixture()
def spec() -> SkillSpec:
    return SkillSpec(
        name="count_lines",
        description="Count lines in a file",
        parameters={"path": "str"},
    )


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


class TestBuildSuccess:
    async def test_build_succeeds_first_attempt(self, tmp_path: Path, spec: SkillSpec, bus: EventBus):
        mock_llm = AsyncMock()
        mock_llm.code = AsyncMock(return_value=_make_response(VALID_SKILL_CODE))
        registry = SkillRegistry()

        builder = SkillBuilder(
            llm_client=mock_llm,
            registry=registry,
            user_skills_dir=tmp_path,
            bus=bus,
        )
        result = await builder.build(spec)

        assert result is not None
        assert result.name == "count_lines"
        assert registry.get("count_lines") is not None
        mock_llm.code.assert_awaited_once()

    async def test_build_succeeds_on_retry(self, tmp_path: Path, spec: SkillSpec, bus: EventBus):
        mock_llm = AsyncMock()
        mock_llm.code = AsyncMock(
            side_effect=[
                _make_response(BROKEN_SKILL_CODE),
                _make_response(VALID_SKILL_CODE),
            ]
        )
        registry = SkillRegistry()

        # Need test_inputs to trigger execution failure on broken code
        spec_with_test = SkillSpec(
            name="count_lines",
            description="Count lines in a file",
            parameters={"path": "str"},
            test_inputs={"path": __file__},  # use this test file as input
        )

        builder = SkillBuilder(
            llm_client=mock_llm,
            registry=registry,
            user_skills_dir=tmp_path,
            bus=bus,
        )
        result = await builder.build(spec_with_test)

        assert result is not None
        assert result.name == "count_lines"
        assert mock_llm.code.await_count == 2


class TestBuildFailure:
    async def test_build_fails_all_attempts(self, tmp_path: Path, spec: SkillSpec, bus: EventBus):
        mock_llm = AsyncMock()
        # Return code with syntax error every time
        bad_response = _make_response("```python\ndef broken(\n```")
        mock_llm.code = AsyncMock(return_value=bad_response)
        registry = SkillRegistry()

        builder = SkillBuilder(
            llm_client=mock_llm,
            registry=registry,
            user_skills_dir=tmp_path,
            bus=bus,
        )
        result = await builder.build(spec)

        assert result is None
        assert mock_llm.code.await_count == 3
        assert registry.get("count_lines") is None


class TestAttemptFilePattern:
    async def test_attempt_file_cleaned_on_success(self, tmp_path: Path, spec: SkillSpec, bus: EventBus):
        mock_llm = AsyncMock()
        mock_llm.code = AsyncMock(return_value=_make_response(VALID_SKILL_CODE))
        registry = SkillRegistry()

        builder = SkillBuilder(
            llm_client=mock_llm, registry=registry, user_skills_dir=tmp_path, bus=bus,
        )
        await builder.build(spec)

        attempt_path = tmp_path / spec.name / f".{spec.name}.attempt.py"
        assert not attempt_path.exists()
        # skill.py should exist (promoted from attempt)
        assert (tmp_path / spec.name / "skill.py").exists()

    async def test_attempt_file_cleaned_on_failure(self, tmp_path: Path, spec: SkillSpec, bus: EventBus):
        mock_llm = AsyncMock()
        bad_response = _make_response("```python\ndef broken(\n```")
        mock_llm.code = AsyncMock(return_value=bad_response)
        registry = SkillRegistry()

        builder = SkillBuilder(
            llm_client=mock_llm, registry=registry, user_skills_dir=tmp_path, bus=bus,
        )
        await builder.build(spec)

        attempt_path = tmp_path / spec.name / f".{spec.name}.attempt.py"
        assert not attempt_path.exists()

    async def test_skill_py_not_overwritten_on_failure(self, tmp_path: Path, spec: SkillSpec, bus: EventBus):
        """A broken build should never overwrite an existing skill.py."""
        skill_dir = tmp_path / spec.name
        skill_dir.mkdir(parents=True)
        skill_path = skill_dir / "skill.py"
        skill_path.write_text("# original working code\n", encoding="utf-8")

        mock_llm = AsyncMock()
        bad_response = _make_response("```python\ndef broken(\n```")
        mock_llm.code = AsyncMock(return_value=bad_response)
        registry = SkillRegistry()

        builder = SkillBuilder(
            llm_client=mock_llm, registry=registry, user_skills_dir=tmp_path, bus=bus,
        )
        await builder.build(spec)

        assert skill_path.read_text(encoding="utf-8") == "# original working code\n"


class TestErrorContextInjection:
    async def test_error_file_fed_into_build_prompt(self, tmp_path: Path, spec: SkillSpec, bus: EventBus):
        """When skill.error exists, its content is injected into the LLM messages."""
        skill_dir = tmp_path / spec.name
        skill_dir.mkdir(parents=True)
        error_path = skill_dir / "skill.error"
        error_path.write_text("RuntimeError: boom\n", encoding="utf-8")
        skill_path = skill_dir / "skill.py"
        skill_path.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

        mock_llm = AsyncMock()
        mock_llm.code = AsyncMock(return_value=_make_response(VALID_SKILL_CODE))
        registry = SkillRegistry()

        builder = SkillBuilder(
            llm_client=mock_llm, registry=registry, user_skills_dir=tmp_path, bus=bus,
        )
        await builder.build(spec)

        # Check that the LLM received the error context
        call_kwargs = mock_llm.code.call_args
        messages = call_kwargs.kwargs["messages"]
        error_messages = [m for m in messages if "failed to load" in m.get("content", "").lower()]
        assert len(error_messages) == 1
        assert "RuntimeError: boom" in error_messages[0]["content"]
        assert "raise RuntimeError('boom')" in error_messages[0]["content"]

    async def test_error_file_deleted_on_success(self, tmp_path: Path, spec: SkillSpec, bus: EventBus):
        skill_dir = tmp_path / spec.name
        skill_dir.mkdir(parents=True)
        error_path = skill_dir / "skill.error"
        error_path.write_text("old error\n", encoding="utf-8")

        mock_llm = AsyncMock()
        mock_llm.code = AsyncMock(return_value=_make_response(VALID_SKILL_CODE))
        registry = SkillRegistry()

        builder = SkillBuilder(
            llm_client=mock_llm, registry=registry, user_skills_dir=tmp_path, bus=bus,
        )
        await builder.build(spec)

        assert not error_path.exists()


class TestExtractCode:
    def test_python_fences(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        assert _extract_code(text) == "print('hello')\n"

    def test_plain_fences(self):
        text = "```\nprint('hello')\n```"
        assert _extract_code(text) == "print('hello')\n"

    def test_no_fences(self):
        text = "print('hello')"
        assert _extract_code(text) == "print('hello')\n"


class TestStatusEvents:
    async def test_events_published_during_build(self, tmp_path: Path, spec: SkillSpec, bus: EventBus):
        mock_llm = AsyncMock()
        mock_llm.code = AsyncMock(return_value=_make_response(VALID_SKILL_CODE))
        registry = SkillRegistry()
        queue = bus.subscribe()

        builder = SkillBuilder(
            llm_client=mock_llm,
            registry=registry,
            user_skills_dir=tmp_path,
            bus=bus,
        )
        await builder.build(spec)

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        status_events = [e for e in events if isinstance(e, StatusUpdate)]
        assert len(status_events) >= 2  # building + success
        assert all(e.phase == Phase.SKILL_BUILDING for e in status_events)
