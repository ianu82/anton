from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from anton.llm.provider import LLMResponse, ToolCall, Usage
from anton.skill.context import SkillLLM, get_llm, set_skill_llm


class TestSkillLLM:
    async def test_complete_delegates_to_provider(self):
        provider = AsyncMock()
        provider.complete = AsyncMock(
            return_value=LLMResponse(
                content="hello",
                tool_calls=[],
                usage=Usage(input_tokens=5, output_tokens=3),
                stop_reason="end_turn",
            )
        )
        llm = SkillLLM(provider, "test-model")

        response = await llm.complete(
            system="sys", messages=[{"role": "user", "content": "hi"}]
        )

        assert response.content == "hello"
        provider.complete.assert_called_once_with(
            model="test-model",
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            tool_choice=None,
            max_tokens=4096,
        )

    async def test_model_property(self):
        provider = AsyncMock()
        llm = SkillLLM(provider, "claude-opus-4-6")
        assert llm.model == "claude-opus-4-6"

    async def test_complete_passes_tools(self):
        provider = AsyncMock()
        provider.complete = AsyncMock(
            return_value=LLMResponse(content="ok", stop_reason="end_turn")
        )
        llm = SkillLLM(provider, "m")

        tools = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
        await llm.complete(system="s", messages=[], tools=tools, max_tokens=1024)

        provider.complete.assert_called_once_with(
            model="m", system="s", messages=[], tools=tools, tool_choice=None, max_tokens=1024
        )


    async def test_generate_object_single_model(self):
        """generate_object() should parse a tool call into a Pydantic model."""
        from pydantic import BaseModel

        class Sentiment(BaseModel):
            label: str
            confidence: float

        provider = AsyncMock()
        provider.complete = AsyncMock(
            return_value=LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        name="Sentiment",
                        input={"label": "positive", "confidence": 0.95},
                    )
                ],
                usage=Usage(input_tokens=10, output_tokens=20),
                stop_reason="tool_use",
            )
        )
        llm = SkillLLM(provider, "test-model")

        result = await llm.generate_object(
            Sentiment,
            system="Classify sentiment.",
            messages=[{"role": "user", "content": "I love this!"}],
        )

        assert isinstance(result, Sentiment)
        assert result.label == "positive"
        assert result.confidence == 0.95
        # Verify tool_choice was forced
        call_kwargs = provider.complete.call_args[1]
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "Sentiment"}
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["name"] == "Sentiment"

    async def test_generate_object_list_model(self):
        """generate_object(list[Model]) should return a list of instances."""
        from pydantic import BaseModel

        class Tag(BaseModel):
            name: str
            score: float

        provider = AsyncMock()
        provider.complete = AsyncMock(
            return_value=LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        name="Tag_array",
                        input={
                            "items": [
                                {"name": "python", "score": 0.9},
                                {"name": "async", "score": 0.7},
                            ]
                        },
                    )
                ],
                usage=Usage(input_tokens=10, output_tokens=20),
                stop_reason="tool_use",
            )
        )
        llm = SkillLLM(provider, "test-model")

        result = await llm.generate_object(
            list[Tag],
            system="Extract tags.",
            messages=[{"role": "user", "content": "Python async code"}],
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].name == "python"
        assert result[1].score == 0.7

    async def test_generate_object_raises_on_no_tool_calls(self):
        """generate_object() should raise if LLM returns no tool calls."""
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str

        provider = AsyncMock()
        provider.complete = AsyncMock(
            return_value=LLMResponse(
                content="I can't do that",
                tool_calls=[],
                usage=Usage(input_tokens=5, output_tokens=10),
                stop_reason="end_turn",
            )
        )
        llm = SkillLLM(provider, "test-model")

        with pytest.raises(ValueError, match="LLM did not return structured output"):
            await llm.generate_object(
                Item,
                system="Extract.",
                messages=[{"role": "user", "content": "test"}],
            )


class TestGetLLM:
    def test_raises_without_context(self):
        # Reset context to ensure clean state
        from anton.skill.context import _current_llm
        _current_llm.set(None)

        with pytest.raises(RuntimeError, match="No LLM available"):
            get_llm()

    def test_returns_llm_after_set(self):
        provider = AsyncMock()
        set_skill_llm(provider, "test-model")

        llm = get_llm()
        assert llm.model == "test-model"

        # Clean up
        from anton.skill.context import _current_llm
        _current_llm.set(None)
