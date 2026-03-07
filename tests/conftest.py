from __future__ import annotations

import os
import site
import sysconfig
from unittest.mock import AsyncMock

import pytest

from anton.llm.provider import LLMResponse, ToolCall, Usage


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


@pytest.fixture()
def scratchpad_runtime_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTON_RUNTIME_HOME", str(tmp_path / "runtimes"))
    monkeypatch.setenv("ANTON_SCRATCHPAD_BASE", str(tmp_path / "scratchpad-venvs"))
    site_paths = [sysconfig.get_paths()["purelib"], site.getusersitepackages()]
    monkeypatch.setenv("ANTON_RUNTIME_TEST_SITE_PACKAGES", os.pathsep.join(site_paths))
