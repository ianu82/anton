from __future__ import annotations

import os
from pathlib import Path

import pytest

from anton.config.settings import AntonSettings


class TestAntonSettingsDefaults:
    def test_default_planning_provider(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.planning_provider == "anthropic"

    def test_default_planning_model(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.planning_model == "claude-sonnet-4-6"

    def test_default_coding_provider(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.coding_provider == "anthropic"

    def test_default_coding_model(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.coding_model == "claude-haiku-4-5-20251001"

    def test_default_skills_dir(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.skills_dir == "skills"

    def test_default_user_skills_dir(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.user_skills_dir == ".anton/skills"

    def test_default_memory_dir(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.memory_dir == ".anton"

    def test_default_context_dir(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.context_dir == ".anton/context"

    def test_default_api_key_is_none(self):
        s = AntonSettings(_env_file=None)
        assert s.anthropic_api_key is None


class TestAntonSettingsEnvOverride:
    def test_env_overrides_planning_model(self, monkeypatch):
        monkeypatch.setenv("ANTON_PLANNING_MODEL", "custom-model")
        s = AntonSettings(_env_file=None)
        assert s.planning_model == "custom-model"

    def test_env_overrides_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTON_ANTHROPIC_API_KEY", "sk-test-key")
        s = AntonSettings(_env_file=None)
        assert s.anthropic_api_key == "sk-test-key"

    def test_env_overrides_skills_dir(self, monkeypatch):
        monkeypatch.setenv("ANTON_SKILLS_DIR", "/custom/skills")
        s = AntonSettings(_env_file=None)
        assert s.skills_dir == "/custom/skills"


class TestWorkspaceResolution:
    def test_resolve_workspace_defaults_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = AntonSettings(anthropic_api_key="test", _env_file=None)
        s.resolve_workspace()

        assert s.workspace_path == tmp_path
        assert Path(s.memory_dir) == tmp_path / ".anton"
        assert Path(s.user_skills_dir) == tmp_path / ".anton" / "skills"
        assert Path(s.context_dir) == tmp_path / ".anton" / "context"
        assert (tmp_path / ".anton").is_dir()

    def test_resolve_workspace_with_explicit_folder(self, tmp_path):
        s = AntonSettings(anthropic_api_key="test", _env_file=None)
        s.resolve_workspace(str(tmp_path))

        assert s.workspace_path == tmp_path
        assert Path(s.memory_dir) == tmp_path / ".anton"
        assert Path(s.user_skills_dir) == tmp_path / ".anton" / "skills"
        assert Path(s.context_dir) == tmp_path / ".anton" / "context"
        assert (tmp_path / ".anton").is_dir()

    def test_resolve_workspace_creates_anton_dir(self, tmp_path):
        s = AntonSettings(anthropic_api_key="test", _env_file=None)
        s.resolve_workspace(str(tmp_path))

        assert (tmp_path / ".anton").exists()
        assert (tmp_path / ".anton").is_dir()

    def test_resolve_workspace_preserves_absolute_paths(self, tmp_path):
        s = AntonSettings(
            anthropic_api_key="test",
            memory_dir="/absolute/path",
            _env_file=None,
        )
        s.resolve_workspace(str(tmp_path))

        # Absolute path should not be changed
        assert s.memory_dir == "/absolute/path"

    def test_workspace_path_before_resolve(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = AntonSettings(anthropic_api_key="test", _env_file=None)
        # Before resolve, workspace_path falls back to cwd
        assert s.workspace_path == tmp_path
