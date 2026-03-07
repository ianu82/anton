from __future__ import annotations

from pathlib import Path

from anton.execution_policy import ExecutionMode, ScratchpadExecutionPolicy
from anton.scratchpad import ScratchpadManager


class TestScratchpadExecutionPolicy:
    def test_full_trust_env_inherits_ambient_vars_and_bridges_sdk_keys(self, tmp_path):
        policy = ScratchpadExecutionPolicy(mode=ExecutionMode.FULL_TRUST)
        env = policy.build_subprocess_env(
            base_env={
                "CUSTOM_ENV": "present",
                "ANTON_ANTHROPIC_API_KEY": "ant-key",
                "ANTON_OPENAI_API_KEY": "openai-key",
                "ANTON_OPENAI_BASE_URL": "https://example.test/v1",
                "MINDS_API_KEY": "minds-key",
            },
            coding_provider="anthropic",
            coding_model="claude-test",
            coding_api_key="fallback-key",
            runtime_profile="base",
            scratchpad_name="main",
            workspace_path=tmp_path,
            anton_root=Path("/repo/anton"),
            uv_path="/usr/bin/uv",
        )

        assert env["CUSTOM_ENV"] == "present"
        assert env["ANTHROPIC_API_KEY"] == "ant-key"
        assert env["OPENAI_API_KEY"] == "openai-key"
        assert env["OPENAI_BASE_URL"] == "https://example.test/v1"
        assert env["ANTON_MINDS_API_KEY"] == "minds-key"
        assert env["ANTON_RUNTIME_PROFILE"] == "base"
        assert env["ANTON_SCRATCHPAD_NAME"] == "main"

    def test_workspace_write_env_strips_ambient_secrets(self, tmp_path):
        policy = ScratchpadExecutionPolicy(mode=ExecutionMode.WORKSPACE_WRITE)
        env = policy.build_subprocess_env(
            base_env={
                "PATH": "/usr/bin",
                "HOME": "/tmp/home",
                "CUSTOM_ENV": "present",
                "ANTON_ANTHROPIC_API_KEY": "ant-key",
                "ANTHROPIC_API_KEY": "direct-key",
            },
            coding_provider="anthropic",
            coding_model="claude-test",
            coding_api_key="fallback-key",
            runtime_profile="ml",
            scratchpad_name="analysis",
            workspace_path=tmp_path,
            anton_root=Path("/repo/anton"),
            uv_path=None,
        )

        assert env["PATH"] == "/usr/bin"
        assert env["HOME"] == "/tmp/home"
        assert env["ANTON_RUNTIME_PROFILE"] == "ml"
        assert env["ANTON_SCRATCHPAD_NAME"] == "analysis"
        assert env["ANTON_EXECUTION_MODE_INTERNAL"] == "workspace_write"
        assert "CUSTOM_ENV" not in env
        assert "ANTON_ANTHROPIC_API_KEY" not in env
        assert "ANTHROPIC_API_KEY" not in env

    def test_restricted_modes_reject_generic_installs(self):
        policy = ScratchpadExecutionPolicy(mode=ExecutionMode.READ_ONLY)

        assert "does not allow generic package installs" in policy.authorize("install")
        assert "does not allow exec-time package installs" in policy.authorize(
            "exec",
            packages=["cowsay"],
        )


class TestScratchpadManagerPolicy:
    def test_manager_exposes_policy_note(self):
        manager = ScratchpadManager(execution_mode=ExecutionMode.WORKSPACE_WRITE)

        assert manager.execution_mode is ExecutionMode.WORKSPACE_WRITE
        assert "workspace_write" in manager.policy_prompt_note()
