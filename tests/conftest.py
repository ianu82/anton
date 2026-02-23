from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from anton.channel.base import Channel
from anton.llm.provider import LLMResponse, ToolCall, Usage


@pytest.fixture()
def mock_channel() -> AsyncMock:
    ch = AsyncMock(spec=Channel)
    ch.emit = AsyncMock()
    ch.prompt = AsyncMock(return_value="yes")
    ch.close = AsyncMock()
    return ch


@pytest.fixture()
def make_llm_response():
    def _factory(
        content: str = "",
        tool_calls: list[ToolCall] | None = None,
        input_tokens: int = 10,
        output_tokens: int = 20,
        stop_reason: str | None = "end_turn",
    ) -> LLMResponse:
        return LLMResponse(
            content=content,
            tool_calls=tool_calls or [],
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            stop_reason=stop_reason,
        )

    return _factory
