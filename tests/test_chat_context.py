from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.chat import (
    ChatSession,
    _build_runtime_context,
    _chat_execution_mode_line,
    _handle_setup_execution_mode,
)
from anton.config.settings import AntonSettings
from anton.tools import MEMORIZE_TOOL
from anton.context.self_awareness import SelfAwarenessContext
from anton.execution_policy import ExecutionMode
from anton.llm.provider import LLMResponse, ToolCall, Usage
from anton.workspace import Workspace
from anton.memory.cortex import Cortex
from anton.memory.hippocampus import Hippocampus


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


def _memorize_response(
    text: str, entries: list[dict], tool_id: str = "tc_mem_1"
) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[
            ToolCall(
                id=tool_id,
                name="memorize",
                input={"entries": entries},
            ),
        ],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


@pytest.fixture()
def ctx_dir(tmp_path):
    d = tmp_path / "context"
    d.mkdir()
    return d


@pytest.fixture()
def sa(ctx_dir):
    return SelfAwarenessContext(ctx_dir)


@pytest.fixture()
def ws(tmp_path):
    w = Workspace(tmp_path)
    w.initialize()
    return w


@pytest.fixture()
def memory_dirs(tmp_path):
    global_dir = tmp_path / "global_memory"
    project_dir = tmp_path / "project_memory"
    global_dir.mkdir()
    project_dir.mkdir()
    return global_dir, project_dir


@pytest.fixture()
def cortex(memory_dirs):
    global_dir, project_dir = memory_dirs
    return Cortex(global_dir=global_dir, project_dir=project_dir, mode="autopilot")


class TestMemorizeTool:
    def test_tool_definition_structure(self):
        assert MEMORIZE_TOOL["name"] == "memorize"
        props = MEMORIZE_TOOL["input_schema"]["properties"]
        assert "entries" in props

    async def test_memorize_creates_rule(self, cortex, memory_dirs):
        """When LLM calls memorize, a rule is created in memory."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _memorize_response(
                    "I'll remember that.",
                    [{"text": "Use httpx instead of requests", "kind": "always", "scope": "project"}],
                ),
                _text_response("Got it, I've noted that for future reference."),
            ]
        )

        session = ChatSession(mock_llm, cortex=cortex)
        reply = await session.turn("always use httpx instead of requests")
        await asyncio.sleep(0)  # Let fire-and-forget encode task run

        assert "noted" in reply.lower() or "reference" in reply.lower()

        # Verify the rule was written
        global_dir, project_dir = memory_dirs
        rules_path = project_dir / "rules.md"
        assert rules_path.exists()
        content = rules_path.read_text()
        assert "Use httpx instead of requests" in content

    async def test_memorize_creates_lesson(self, cortex, memory_dirs):
        """When LLM calls memorize with kind=lesson, a lesson is created."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _memorize_response(
                    "Noting that.",
                    [{"text": "CoinGecko rate-limits at 50/min", "kind": "lesson", "scope": "global", "topic": "api-coingecko"}],
                ),
                _text_response("Done."),
            ]
        )

        session = ChatSession(mock_llm, cortex=cortex)
        await session.turn("coingecko rate limits at 50 per minute")
        await asyncio.sleep(0)  # Let fire-and-forget encode task run

        global_dir, project_dir = memory_dirs
        lessons_path = global_dir / "lessons.md"
        assert lessons_path.exists()
        content = lessons_path.read_text()
        assert "CoinGecko rate-limits at 50/min" in content

    async def test_memory_injected_into_system_prompt(self, cortex, memory_dirs):
        """Memory context is injected into the system prompt."""
        global_dir, project_dir = memory_dirs

        # Pre-populate some memory
        hc = Hippocampus(project_dir)
        hc.encode_rule("Use httpx instead of requests", kind="always", confidence="high", source="user")

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))

        session = ChatSession(mock_llm, cortex=cortex)
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "Project Rules" in system_prompt
        assert "Use httpx instead of requests" in system_prompt

    async def test_no_cortex_excludes_memorize_tool(self):
        """Without cortex or self_awareness, memorize tool is not offered."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))

        session = ChatSession(mock_llm, self_awareness=None, cortex=None)
        await session.turn("hello")

        call_kwargs = mock_llm.plan.call_args
        tools = call_kwargs.kwargs.get("tools", [])
        tool_names = [t["name"] for t in tools]
        assert "memorize" not in tool_names
        assert "patch" in tool_names
        assert "scratchpad" in tool_names

    async def test_cortex_includes_memorize_tool(self, cortex):
        """With cortex, memorize tool is included."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))

        session = ChatSession(mock_llm, cortex=cortex)
        await session.turn("hello")

        call_kwargs = mock_llm.plan.call_args
        tools = call_kwargs.kwargs.get("tools", [])
        tool_names = [t["name"] for t in tools]
        assert "memorize" in tool_names
        assert "patch" in tool_names
        assert "scratchpad" in tool_names

    async def test_tool_result_in_history(self, cortex, memory_dirs):
        """memorize tool result appears in conversation history."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _memorize_response(
                    "Noting.",
                    [{"text": "Test memory", "kind": "lesson", "scope": "project"}],
                ),
                _text_response("Done."),
            ]
        )

        session = ChatSession(mock_llm, cortex=cortex)
        await session.turn("note this")

        tool_result_msgs = [
            m for m in session.history
            if m["role"] == "user" and isinstance(m["content"], list)
        ]
        assert len(tool_result_msgs) == 1
        result_content = tool_result_msgs[0]["content"][0]["content"]
        assert "Memory updated" in result_content


class TestAntonMdInjection:
    async def test_anton_md_injected_into_system_prompt(self, ws, cortex):
        """anton.md content is injected into the system prompt."""
        ws.anton_md_path.write_text("This project uses Django and PostgreSQL")

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))

        session = ChatSession(
            mock_llm,
            cortex=cortex,
            workspace=ws,
        )
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "Project Context" in system_prompt
        assert "Django and PostgreSQL" in system_prompt

    async def test_empty_anton_md_no_section(self, ws, cortex):
        """Empty anton.md doesn't add a section to the prompt."""
        ws.anton_md_path.write_text("")

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))

        session = ChatSession(
            mock_llm,
            cortex=cortex,
            workspace=ws,
        )
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "Project Context" not in system_prompt


class TestRuntimeContext:
    async def test_runtime_context_injected_into_system_prompt(self):
        """Runtime context (provider/model) appears in the system prompt."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))

        session = ChatSession(
            mock_llm,
            runtime_context="- Provider: anthropic\n- Planning model: claude-sonnet-4-6\n- Coding model: claude-opus-4-6",
        )
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "Provider: anthropic" in system_prompt
        assert "claude-sonnet-4-6" in system_prompt
        assert "claude-opus-4-6" in system_prompt

    async def test_system_prompt_warns_not_to_ask_about_llm(self):
        """System prompt includes instruction to never ask which LLM to use."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))

        session = ChatSession(
            mock_llm,
            runtime_context="- Provider: anthropic",
        )
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "NEVER ask the user which" in system_prompt

    async def test_conversation_discipline_in_prompt(self):
        """System prompt includes conversation discipline rules."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))

        session = ChatSession(mock_llm, runtime_context="")
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "WAIT for their reply" in system_prompt

    async def test_system_prompt_describes_local_privileged_scratchpad(self):
        """System prompt should not describe the scratchpad as a security sandbox."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))

        session = ChatSession(mock_llm, runtime_context="")
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "local Python subprocess" in system_prompt
        assert "not a security sandbox" in system_prompt
        assert "Execution Policy" in system_prompt
        assert "full_trust" in system_prompt

    def test_scratchpad_tool_description_warns_about_privileges(self):
        """Tool description should explain the current trust model honestly."""
        session = ChatSession(AsyncMock(), runtime_context="")
        scratchpad_tool = next(tool for tool in session._build_tools() if tool["name"] == "scratchpad")
        description = scratchpad_tool["description"]
        assert "not a security sandbox" in description
        assert "may be available in os.environ" in description
        assert "Execution policy:" in description

    async def test_read_only_system_prompt_reports_helpers_unavailable(self):
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))

        session = ChatSession(
            mock_llm,
            runtime_context="",
            execution_mode=ExecutionMode.READ_ONLY,
        )
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "get_llm()" in system_prompt
        assert "unavailable in this execution mode" in system_prompt

    def test_read_only_tool_description_reports_helpers_unavailable(self):
        session = ChatSession(AsyncMock(), runtime_context="", execution_mode=ExecutionMode.READ_ONLY)
        scratchpad_tool = next(tool for tool in session._build_tools() if tool["name"] == "scratchpad")
        description = scratchpad_tool["description"]
        assert "Helper capabilities:" in description
        assert "query_minds_data()" in description
        assert "unavailable in this execution mode" in description

    def test_runtime_context_hides_query_minds_data_in_read_only(self):
        settings = AntonSettings(
            minds_api_key="minds-key",
            minds_datasource="warehouse",
            minds_datasource_engine="postgres",
            execution_mode=ExecutionMode.READ_ONLY,
        )

        context = _build_runtime_context(settings)

        assert "CONNECTED DATASOURCE" in context
        assert "query_minds_data()" in context
        assert "unavailable in execution mode `read_only`" in context
        assert "pre-loaded in the scratchpad namespace" not in context

    def test_runtime_context_keeps_query_minds_data_in_full_trust(self):
        settings = AntonSettings(
            minds_api_key="minds-key",
            minds_datasource="warehouse",
            minds_datasource_engine="postgres",
            execution_mode=ExecutionMode.FULL_TRUST,
        )

        context = _build_runtime_context(settings)

        assert "query_minds_data()" in context
        assert "pre-loaded in the scratchpad namespace" in context


class TestExecutionModeSetup:
    def test_chat_execution_mode_line(self):
        settings = AntonSettings(execution_mode=ExecutionMode.WORKSPACE_WRITE)

        assert _chat_execution_mode_line(settings) == "workspace_write mode. To change this, type /setup"

    async def test_setup_execution_mode_persists_default_and_restarts_when_changed(
        self,
        tmp_path,
        monkeypatch,
    ):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *args, **kwargs: "3")
        monkeypatch.setattr(
            "anton.sandbox.ensure_execution_mode_supported",
            lambda mode: ExecutionMode.coerce(mode),
        )

        restarted_session = object()
        restart_mock = AsyncMock(return_value=restarted_session)
        monkeypatch.setattr("anton.chat._restart_session_with_history", restart_mock)

        workspace_base = tmp_path / "workspace"
        workspace_base.mkdir()
        workspace = Workspace(workspace_base)
        workspace.initialize()
        settings = AntonSettings(execution_mode=ExecutionMode.FULL_TRUST)
        console = MagicMock()
        original_session = object()

        result = await _handle_setup_execution_mode(
            console,
            settings,
            workspace,
            state={},
            self_awareness=None,
            cortex=None,
            session=original_session,
        )

        assert result is restarted_session
        assert settings.execution_mode is ExecutionMode.READ_ONLY
        assert "ANTON_EXECUTION_MODE=read_only" in (home / ".anton" / ".env").read_text()
        restart_mock.assert_awaited_once()
        printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
        assert "must restart the current chat session" in printed

    async def test_setup_execution_mode_skips_restart_when_mode_is_unchanged(
        self,
        tmp_path,
        monkeypatch,
    ):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *args, **kwargs: "2")
        monkeypatch.setattr(
            "anton.sandbox.ensure_execution_mode_supported",
            lambda mode: ExecutionMode.coerce(mode),
        )

        restart_mock = AsyncMock()
        monkeypatch.setattr("anton.chat._restart_session_with_history", restart_mock)

        workspace_base = tmp_path / "workspace"
        workspace_base.mkdir()
        workspace = Workspace(workspace_base)
        workspace.initialize()
        settings = AntonSettings(execution_mode=ExecutionMode.WORKSPACE_WRITE)
        console = MagicMock()
        original_session = object()

        result = await _handle_setup_execution_mode(
            console,
            settings,
            workspace,
            state={},
            self_awareness=None,
            cortex=None,
            session=original_session,
        )

        assert result is original_session
        assert settings.execution_mode is ExecutionMode.WORKSPACE_WRITE
        assert "ANTON_EXECUTION_MODE=workspace_write" in (home / ".anton" / ".env").read_text()
        restart_mock.assert_not_awaited()
        printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
        assert "Current chat already uses this mode." in printed
