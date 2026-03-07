from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ExecutionMode(str, Enum):
    FULL_TRUST = "full_trust"
    WORKSPACE_WRITE = "workspace_write"
    READ_ONLY = "read_only"

    @classmethod
    def coerce(cls, value: str | "ExecutionMode" | None) -> "ExecutionMode":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.FULL_TRUST
        normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unknown execution mode: {value}")


_BASE_ENV_ALLOWLIST = {
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONIOENCODING",
    "REQUESTS_CA_BUNDLE",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "USER",
    "USERNAME",
    "WINDIR",
}


@dataclass
class ScratchpadExecutionPolicy:
    mode: ExecutionMode = ExecutionMode.FULL_TRUST
    granted_env_vars: set[str] = field(default_factory=set)
    granted_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mode = ExecutionMode.coerce(self.mode)

    def authorize(self, action: str, *, packages: list[str] | None = None) -> str | None:
        if self.mode is ExecutionMode.FULL_TRUST:
            return None

        if action == "install":
            return (
                f"Rejected: execution mode '{self.mode.value}' does not allow generic package installs. "
                "Use a managed runtime profile instead."
            )
        if packages:
            return (
                f"Rejected: execution mode '{self.mode.value}' does not allow exec-time package installs. "
                "Switch to a managed runtime profile or rerun in full_trust."
            )
        return None

    def llm_helpers_available(self) -> bool:
        return self.mode is ExecutionMode.FULL_TRUST

    def minds_query_available(self) -> bool:
        return self.mode is ExecutionMode.FULL_TRUST

    def helper_prompt_note(self) -> str:
        if self.mode is ExecutionMode.FULL_TRUST:
            return (
                "Secret-backed scratchpad helpers can be available in this session. "
                "`get_llm()` and `agentic_loop()` are available when Anton has a coding model "
                "configured for the scratchpad, and `query_minds_data()` is available when a "
                "Minds datasource is configured."
            )
        return (
            "Secret-backed scratchpad helpers like `get_llm()`, `agentic_loop()`, and "
            "`query_minds_data()` are unavailable in this execution mode unless Anton "
            "explicitly grants them."
        )

    def build_subprocess_env(
        self,
        *,
        base_env: dict[str, str] | None = None,
        coding_provider: str,
        coding_model: str,
        coding_api_key: str,
        runtime_profile: str,
        scratchpad_name: str,
        workspace_path: Path | None,
        anton_root: Path,
        uv_path: str | None,
    ) -> dict[str, str]:
        source_env = dict(base_env or os.environ)

        if self.mode is ExecutionMode.FULL_TRUST:
            env = source_env.copy()
        else:
            env = {
                key: value
                for key, value in source_env.items()
                if key in _BASE_ENV_ALLOWLIST
            }
            for key in self.granted_env_vars:
                if key in source_env:
                    env[key] = source_env[key]
            env.update(self.granted_env)
            env["PYTHONDONTWRITEBYTECODE"] = "1"

        if self.llm_helpers_available():
            if coding_model:
                env["ANTON_SCRATCHPAD_MODEL"] = coding_model
            if coding_provider:
                env["ANTON_SCRATCHPAD_PROVIDER"] = coding_provider
        env["ANTON_RUNTIME_PROFILE"] = runtime_profile
        env["ANTON_SCRATCHPAD_NAME"] = scratchpad_name
        env["ANTON_EXECUTION_MODE_INTERNAL"] = self.mode.value
        if workspace_path is not None:
            env["ANTON_WORKSPACE_PATH"] = str(workspace_path)
        if uv_path:
            env["ANTON_UV_PATH"] = uv_path

        anton_root_str = str(anton_root)
        python_path = env.get("PYTHONPATH", "")
        current_entries = [entry for entry in python_path.split(os.pathsep) if entry]
        if anton_root_str not in current_entries:
            env["PYTHONPATH"] = anton_root_str + (os.pathsep + python_path if python_path else "")

        if self.mode is ExecutionMode.FULL_TRUST:
            if "ANTHROPIC_API_KEY" not in env and "ANTON_ANTHROPIC_API_KEY" in source_env:
                env["ANTHROPIC_API_KEY"] = source_env["ANTON_ANTHROPIC_API_KEY"]
            if "OPENAI_API_KEY" not in env and "ANTON_OPENAI_API_KEY" in source_env:
                env["OPENAI_API_KEY"] = source_env["ANTON_OPENAI_API_KEY"]
            if "OPENAI_BASE_URL" not in env and "ANTON_OPENAI_BASE_URL" in source_env:
                env["OPENAI_BASE_URL"] = source_env["ANTON_OPENAI_BASE_URL"]
            if "ANTON_MINDS_API_KEY" not in env and "MINDS_API_KEY" in source_env:
                env["ANTON_MINDS_API_KEY"] = source_env["MINDS_API_KEY"]

            sdk_key = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "openai-compatible": "OPENAI_API_KEY",
            }.get(coding_provider, "")
            if sdk_key and coding_api_key and sdk_key not in env:
                env[sdk_key] = coding_api_key

        return env

    def prompt_note(self) -> str:
        if self.mode is ExecutionMode.FULL_TRUST:
            return (
                "Execution mode is full_trust. Scratchpads inherit Anton's normal local process "
                "privileges and broad ambient environment access. "
                + self.helper_prompt_note()
            )
        workspace_access = (
            "workspace read/write access"
            if self.mode is ExecutionMode.WORKSPACE_WRITE
            else "workspace read-only access"
        )
        return (
            f"Execution mode is {self.mode.value}. Scratchpads run with {workspace_access}, "
            "no network, no ambient secrets, and no generic package installs. Adapt to policy "
            "feedback instead of assuming full machine access. "
            + self.helper_prompt_note()
        )
