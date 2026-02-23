from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from anton.chat import ChatSession
from anton.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTextDelta,
    ToolCall,
    Usage,
)


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


class TestChatSession:
    async def test_conversational_turn(self):
        """Text-only response for casual conversation."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hey! How can I help?"))

        session = ChatSession(mock_llm)
        reply = await session.turn("hi")

        assert reply == "Hey! How can I help?"
        assert len(session.history) == 2  # user + assistant

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

        session = ChatSession(mock_llm)
        await session.turn("hello")
        await session.turn("can you check something")
        await session.turn("the anton repo")

        # 3 user messages + 3 assistant messages
        assert len(session.history) == 6
        assert session.history[0]["role"] == "user"
        assert session.history[1]["role"] == "assistant"


# --- Helpers for streaming tests ---

async def _fake_plan_stream(events):
    """Return an async generator factory that yields events from a list of event sequences."""
    call_count = 0

    async def _gen(**kwargs):
        nonlocal call_count
        for ev in events[call_count]:
            yield ev
        call_count += 1

    return _gen


class TestChatSessionStreaming:
    async def test_turn_stream_yields_text_deltas(self):
        """Streaming turn yields text deltas and updates history."""
        mock_llm = AsyncMock()

        async def _stream(**kwargs):
            yield StreamTextDelta(text="Hello ")
            yield StreamTextDelta(text="world!")
            yield StreamComplete(response=_text_response("Hello world!"))

        mock_llm.plan_stream = _stream

        session = ChatSession(mock_llm)
        events = []
        async for event in session.turn_stream("hi"):
            events.append(event)

        # Should have 2 text deltas + 1 complete
        text_deltas = [e for e in events if isinstance(e, StreamTextDelta)]
        completes = [e for e in events if isinstance(e, StreamComplete)]
        assert len(text_deltas) == 2
        assert text_deltas[0].text == "Hello "
        assert text_deltas[1].text == "world!"
        assert len(completes) == 1

        # History: user + assistant
        assert len(session.history) == 2
        assert session.history[1]["content"] == "Hello world!"
