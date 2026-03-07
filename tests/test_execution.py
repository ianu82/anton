from __future__ import annotations

from pathlib import Path

from anton.execution import (
    ExecutionMode,
    ScratchpadExecutionPolicy,
    build_scratchpad_env,
    parse_execution_mode,
)


def test_parse_execution_mode_defaults_to_full_trust():
    assert parse_execution_mode(None) == ExecutionMode.FULL_TRUST
    assert parse_execution_mode("read_only") == ExecutionMode.READ_ONLY


def test_full_trust_env_inherits_ambient_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTON_ANTHROPIC_API_KEY", "secret-123")
    monkeypatch.setenv("CUSTOM_FLAG", "1")

    env = build_scratchpad_env(
        mode=ExecutionMode.FULL_TRUST,
        workspace_path=tmp_path,
        coding_provider="anthropic",
        coding_model="claude-test",
        coding_api_key="",
        runtime_profile="base",
        uv_path="/tmp/uv",
    )

    assert env["ANTHROPIC_API_KEY"] == "secret-123"
    assert env["CUSTOM_FLAG"] == "1"
    assert env["ANTON_UV_PATH"] == "/tmp/uv"


def test_safe_mode_env_redacts_ambient_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTON_ANTHROPIC_API_KEY", "secret-123")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai")
    monkeypatch.setenv("CUSTOM_FLAG", "1")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = build_scratchpad_env(
        mode=ExecutionMode.READ_ONLY,
        workspace_path=tmp_path,
        coding_provider="anthropic",
        coding_model="claude-test",
        coding_api_key="secret-123",
        runtime_profile="base",
        uv_path="/tmp/uv",
    )

    assert env["PATH"] == "/usr/bin"
    assert "CUSTOM_FLAG" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "ANTON_UV_PATH" not in env


def test_scratchpad_policy_blocks_installs_outside_full_trust():
    policy = ScratchpadExecutionPolicy(ExecutionMode.WORKSPACE_WRITE)
    decision = policy.authorize("install", packages=["cowsay"])
    assert decision.allow is False
    assert "disabled" in decision.reason


def test_scratchpad_policy_allows_exec_outside_full_trust():
    policy = ScratchpadExecutionPolicy(ExecutionMode.READ_ONLY)
    decision = policy.authorize("exec")
    assert decision.allow is True
