from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ExecutionMode(str, Enum):
    FULL_TRUST = "full_trust"
    WORKSPACE_WRITE = "workspace_write"
    READ_ONLY = "read_only"


@dataclass(frozen=True)
class ScratchpadPolicyDecision:
    allow: bool
    reason: str = ""


class ScratchpadExecutionPolicy:
    def __init__(self, mode: ExecutionMode = ExecutionMode.FULL_TRUST) -> None:
        self._mode = mode

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    def authorize(self, action: str, *, packages: list[str] | None = None) -> ScratchpadPolicyDecision:
        normalized = action.strip().lower()
        if self._mode == ExecutionMode.FULL_TRUST:
            return ScratchpadPolicyDecision(allow=True)

        if normalized == "install":
            return ScratchpadPolicyDecision(
                allow=False,
                reason=(
                    f"Scratchpad package installs are disabled in execution mode "
                    f"'{self._mode.value}'."
                ),
            )

        return ScratchpadPolicyDecision(allow=True)


_BASE_ENV_KEYS = {
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONPATH",
    "SHELL",
    "TERM",
    "TMP",
    "TMPDIR",
    "TEMP",
    "USER",
}


def parse_execution_mode(raw: str | ExecutionMode | None) -> ExecutionMode:
    if isinstance(raw, ExecutionMode):
        return raw
    text = str(raw or ExecutionMode.FULL_TRUST.value).strip().lower()
    for mode in ExecutionMode:
        if mode.value == text:
            return mode
    raise ValueError(f"Unsupported execution mode: {raw}")


def build_scratchpad_env(
    *,
    mode: ExecutionMode,
    workspace_path: Path | None,
    coding_provider: str,
    coding_model: str,
    coding_api_key: str,
    runtime_profile: str,
    uv_path: str | None,
) -> dict[str, str]:
    if mode == ExecutionMode.FULL_TRUST:
        env = os.environ.copy()
    else:
        env = {key: value for key, value in os.environ.items() if key in _BASE_ENV_KEYS}

    if coding_model:
        env["ANTON_SCRATCHPAD_MODEL"] = coding_model
    if coding_provider:
        env["ANTON_SCRATCHPAD_PROVIDER"] = coding_provider
    env["ANTON_RUNTIME_PROFILE"] = runtime_profile
    env["ANTON_SCRATCHPAD_EXECUTION_MODE"] = mode.value

    if workspace_path is not None:
        env["ANTON_WORKSPACE_PATH"] = str(workspace_path)

    if mode == ExecutionMode.FULL_TRUST:
        if "ANTHROPIC_API_KEY" not in env and "ANTON_ANTHROPIC_API_KEY" in env:
            env["ANTHROPIC_API_KEY"] = env["ANTON_ANTHROPIC_API_KEY"]
        if "OPENAI_API_KEY" not in env and "ANTON_OPENAI_API_KEY" in env:
            env["OPENAI_API_KEY"] = env["ANTON_OPENAI_API_KEY"]
        if coding_api_key:
            sdk_key = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
            }.get(coding_provider, "")
            if sdk_key and sdk_key not in env:
                env[sdk_key] = coding_api_key
        if uv_path:
            env["ANTON_UV_PATH"] = uv_path
    else:
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("OPENAI_API_KEY", None)
        env.pop("ANTON_ANTHROPIC_API_KEY", None)
        env.pop("ANTON_OPENAI_API_KEY", None)
        env.pop("ANTON_UV_PATH", None)

    return env
