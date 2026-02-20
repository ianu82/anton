from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from anton.chat import SCRATCHPAD_TOOL, ChatSession
from anton.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


def _scratchpad_response(
    text: str, action: str, name: str, code: str = "", tool_id: str = "tc_sp_1"
) -> LLMResponse:
    tc_input = {"action": action, "name": name}
    if code:
        tc_input["code"] = code
    return LLMResponse(
        content=text,
        tool_calls=[
            ToolCall(id=tool_id, name="scratchpad", input=tc_input),
        ],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


class TestScratchpadToolDefinition:
    def test_tool_definition_structure(self):
        assert SCRATCHPAD_TOOL["name"] == "scratchpad"
        props = SCRATCHPAD_TOOL["input_schema"]["properties"]
        assert "action" in props
        assert "name" in props
        assert "code" in props
        assert SCRATCHPAD_TOOL["input_schema"]["required"] == ["action", "name"]

    async def test_scratchpad_tool_in_tools(self):
        """scratchpad should always be in _build_tools() output."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        try:
            await session.turn("hello")

            call_kwargs = mock_llm.plan.call_args
            tools = call_kwargs.kwargs.get("tools", [])
            tool_names = [t["name"] for t in tools]
            assert "scratchpad" in tool_names
        finally:
            await session.close()


class TestScratchpadExecViaChat:
    async def test_scratchpad_exec_via_chat(self):
        """exec action flows through and returns output."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _scratchpad_response("Let me compute.", "exec", "main", "print(7 * 6)"),
                _text_response("The answer is 42."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        try:
            reply = await session.turn("what is 7 * 6?")

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            assert len(tool_result_msgs) == 1
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "42" in result_content
        finally:
            await session.close()


class TestScratchpadViewViaChat:
    async def test_scratchpad_view_via_chat(self):
        """view action returns cell history."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _scratchpad_response("Running code.", "exec", "analysis", "x = 10\nprint(x)"),
                _scratchpad_response("Let me check history.", "view", "analysis"),
                _text_response("Here's the history."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        try:
            await session.turn("run and show")

            # Find the view result (second tool result)
            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            assert len(tool_result_msgs) == 2
            view_content = tool_result_msgs[1]["content"][0]["content"]
            assert "Cell 1" in view_content
            assert "10" in view_content
        finally:
            await session.close()


class TestScratchpadRemoveViaChat:
    async def test_scratchpad_remove_via_chat(self):
        """remove action cleans up the scratchpad."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _scratchpad_response("Creating.", "exec", "tmp", "print('hi')"),
                _scratchpad_response("Removing.", "remove", "tmp"),
                _text_response("Cleaned up."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        try:
            await session.turn("create and remove")

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            remove_content = tool_result_msgs[1]["content"][0]["content"]
            assert "removed" in remove_content.lower()
        finally:
            await session.close()


class TestScratchpadDumpViaChat:
    async def test_scratchpad_dump_via_chat(self):
        """dump action flows through chat, returns markdown with code fences."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                # First: exec some code
                _scratchpad_response("Running.", "exec", "main", "print(42)"),
                # Second: dump the scratchpad
                _scratchpad_response("Here's your work.", "dump", "main"),
                # Final text reply
                _text_response("Done!"),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        try:
            await session.turn("show me my work")

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            assert len(tool_result_msgs) == 2
            dump_content = tool_result_msgs[1]["content"][0]["content"]
            assert "```python" in dump_content
            assert "## Scratchpad: main" in dump_content
            assert "42" in dump_content
        finally:
            await session.close()


class _FakeAsyncIter:
    """Wraps items into an async iterator for mocking plan_stream."""

    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class TestScratchpadStreaming:
    async def test_scratchpad_in_streaming_path(self):
        """scratchpad exec should work in turn_stream() too."""
        tool_response = _scratchpad_response("Computing.", "exec", "s", "print(99)")
        final_response = _text_response("Got 99.")

        mock_llm = AsyncMock()

        call_count = 0

        def fake_plan_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _FakeAsyncIter([StreamComplete(response=tool_response)])
            return _FakeAsyncIter([StreamComplete(response=final_response)])

        mock_llm.plan_stream = fake_plan_stream
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        try:
            events = []
            async for event in session.turn_stream("compute 99"):
                events.append(event)

            assert any(isinstance(e, StreamComplete) for e in events)

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            assert len(tool_result_msgs) == 1
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "99" in result_content
        finally:
            await session.close()
