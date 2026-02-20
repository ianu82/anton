"""LLM access for skills.

Skills that need LLM capabilities call ``get_llm()`` to get a pre-configured
client. Credentials and model selection are handled by Anton's runtime — skills
never need API keys.

Usage inside a skill::

    from anton.skill.context import get_llm

    llm = get_llm()
    response = await llm.complete(
        system="You are a helpful classifier.",
        messages=[{"role": "user", "content": text}],
    )
    answer = response.content

Structured output::

    from pydantic import BaseModel
    from anton.skill.context import get_llm

    class Sentiment(BaseModel):
        label: str
        confidence: float

    llm = get_llm()
    result = await llm.generate_object(
        Sentiment,
        system="Classify the sentiment of the text.",
        messages=[{"role": "user", "content": "I love this!"}],
    )
    # result is a Sentiment instance
"""

from __future__ import annotations

import json as _json
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anton.llm.provider import LLMProvider, LLMResponse


class SkillLLM:
    """LLM access pre-configured with Anton's credentials and model.

    Skills receive this via ``get_llm()`` — they never instantiate it directly.
    """

    def __init__(self, provider: LLMProvider, model: str) -> None:
        self._provider = provider
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call the LLM. Same interface as LLMProvider.complete but model is pre-set."""
        return await self._provider.complete(
            model=self._model,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )

    async def generate_object(
        self,
        schema_class,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> Any:
        """Generate a structured object matching a Pydantic model.

        Uses tool_choice to force the LLM to return structured data matching
        the schema. Supports single models and ``list[Model]``.

        Args:
            schema_class: A Pydantic BaseModel subclass, or ``list[Model]``.
            system: System prompt.
            messages: Conversation messages.
            max_tokens: Max tokens for the LLM call.

        Returns:
            An instance of schema_class (or a list of instances).
        """
        from pydantic import BaseModel

        # Detect list[Model] vs single Model
        is_list = hasattr(schema_class, "__origin__") and schema_class.__origin__ is list
        if is_list:
            inner_class = schema_class.__args__[0]

            class _ArrayWrapper(BaseModel):
                items: list[inner_class]  # type: ignore[valid-type]

            schema = _ArrayWrapper.model_json_schema()
            tool_name = f"{inner_class.__name__}_array"
        else:
            schema = schema_class.model_json_schema()
            tool_name = schema_class.__name__

        tool = {
            "name": tool_name,
            "description": f"Generate structured output matching the {tool_name} schema.",
            "input_schema": schema,
        }

        response = await self.complete(
            system=system,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
            max_tokens=max_tokens,
        )

        if not response.tool_calls:
            raise ValueError("LLM did not return structured output.")

        raw = response.tool_calls[0].input

        if is_list:
            wrapper = _ArrayWrapper.model_validate(raw)
            return wrapper.items
        return schema_class.model_validate(raw)


_current_llm: ContextVar[SkillLLM | None] = ContextVar("_current_llm", default=None)


def get_llm() -> SkillLLM:
    """Get the LLM for the current skill execution.

    No credentials or model selection needed — Anton provides both.

    Raises:
        RuntimeError: If called outside of Anton's skill execution context.
    """
    llm = _current_llm.get()
    if llm is None:
        raise RuntimeError(
            "No LLM available. This function must be called from within a skill "
            "executed by Anton's runtime."
        )
    return llm


def set_skill_llm(provider: LLMProvider, model: str) -> None:
    """Set the LLM for skill execution. Called by the Agent before running skills."""
    _current_llm.set(SkillLLM(provider, model))
