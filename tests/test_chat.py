from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from anton.chat import ChatSession
from anton.llm.provider import LLMResponse, ToolCall, Usage


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


def _tool_response(text: str, task: str, tool_id: str = "tc_1") -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[
            ToolCall(id=tool_id, name="execute_task", input={"task": task}),
        ],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


class TestChatSession:
    async def test_conversational_turn(self):
        """Text-only response for casual conversation."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hey! How can I help?"))
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        reply = await session.turn("hi")

        assert reply == "Hey! How can I help?"
        mock_run.assert_not_awaited()
        assert len(session.history) == 2  # user + assistant

    async def test_tool_call_delegates_to_agent(self):
        """When LLM calls execute_task, it delegates to the run callback."""
        mock_llm = AsyncMock()
        # First call returns tool_use, second call (after tool result) returns text
        mock_llm.plan = AsyncMock(
            side_effect=[
                _tool_response("Let me do that.", "list all Python files"),
                _text_response("Done! Found 12 Python files."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        reply = await session.turn("list all python files")

        assert reply == "Done! Found 12 Python files."
        mock_run.assert_awaited_once_with("list all Python files")
        # user, assistant(tool_use), user(tool_result), assistant(text)
        assert len(session.history) == 4

    async def test_tool_call_failure_reported(self):
        """When the task raises an exception, the error is fed back to the LLM."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _tool_response("On it.", "do something risky"),
                _text_response("That didn't work. Want me to try a different approach?"),
            ]
        )
        mock_run = AsyncMock(side_effect=RuntimeError("skill not found"))

        session = ChatSession(mock_llm, mock_run)
        reply = await session.turn("do something risky")

        assert "different approach" in reply
        mock_run.assert_awaited_once()

    async def test_history_grows_across_turns(self):
        """Multiple turns accumulate in history."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _text_response("Hi there!"),
                _text_response("Sure, what repo?"),
                _text_response("Got it, I'll look into that."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        await session.turn("hello")
        await session.turn("can you check something")
        await session.turn("the anton repo")

        # 3 user messages + 3 assistant messages
        assert len(session.history) == 6
        assert session.history[0]["role"] == "user"
        assert session.history[1]["role"] == "assistant"

    async def test_tool_result_format(self):
        """Tool result messages follow the Anthropic protocol format."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _tool_response("Running.", "test task", tool_id="tc_abc"),
                _text_response("All done."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        await session.turn("do it")

        # Check the tool_result message
        tool_result_msg = session.history[2]
        assert tool_result_msg["role"] == "user"
        assert isinstance(tool_result_msg["content"], list)
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "tc_abc"

    async def test_assistant_tool_use_message_format(self):
        """Assistant messages with tool calls use the content-blocks format."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _tool_response("Thinking...", "analyze code", tool_id="tc_99"),
                _text_response("Analysis complete."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        await session.turn("analyze the code")

        # Check the assistant tool_use message
        assistant_msg = session.history[1]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        blocks = assistant_msg["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["id"] == "tc_99"
        assert blocks[1]["name"] == "execute_task"

    async def test_empty_content_with_tool_call(self):
        """Tool call with no accompanying text still works."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="tc_1", name="execute_task", input={"task": "run tests"}),
                    ],
                    usage=Usage(),
                    stop_reason="tool_use",
                ),
                _text_response("Tests passed!"),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        reply = await session.turn("run the tests")

        assert reply == "Tests passed!"
        mock_run.assert_awaited_once_with("run tests")
        # assistant content blocks should only have tool_use (no empty text block)
        assistant_msg = session.history[1]
        assert len(assistant_msg["content"]) == 1
        assert assistant_msg["content"][0]["type"] == "tool_use"
