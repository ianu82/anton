from __future__ import annotations

from pathlib import Path

from anton.execution_policy import ExecutionMode
from anton.windows_sandbox import WindowsSandboxConfig, windows_launcher_command


class TestWindowsSandboxConfig:
    def test_round_trip_serialization(self, tmp_path):
        config = WindowsSandboxConfig.build(
            mode=ExecutionMode.WORKSPACE_WRITE,
            executable=tmp_path / "overlay" / "Scripts" / "python.exe",
            args=["script.py", "--flag"],
            workspace_path=tmp_path / "workspace",
            overlay_dir=tmp_path / "overlay",
            runtime_path=tmp_path / "runtime",
            anton_root=tmp_path / "repo",
            python_roots=[tmp_path / "python-home"],
            extra_write_paths=[tmp_path / "boot"],
        )

        decoded = WindowsSandboxConfig.from_base64(config.to_base64())

        assert decoded == config

    def test_access_grants_for_read_only(self, tmp_path):
        config = WindowsSandboxConfig.build(
            mode=ExecutionMode.READ_ONLY,
            executable=tmp_path / "overlay" / "Scripts" / "python.exe",
            args=["script.py"],
            workspace_path=tmp_path / "workspace",
            overlay_dir=tmp_path / "overlay",
            excluded_paths=[tmp_path / "workspace" / ".venv"],
            runtime_path=tmp_path / "runtime",
            anton_root=tmp_path / "repo",
            python_roots=[tmp_path / "python-home"],
            extra_write_paths=[tmp_path / "boot"],
        )

        grants = {(grant.path, grant.access, grant.effect) for grant in config.access_grants()}

        assert (str((tmp_path / "overlay").resolve()), "modify", "grant") in grants
        assert (str((tmp_path / "boot").resolve()), "modify", "grant") in grants
        assert (str((tmp_path / "workspace").resolve()), "read", "grant") in grants
        assert (str((tmp_path / "workspace" / ".venv").resolve()), "full", "deny") in grants
        assert (str((tmp_path / "runtime").resolve()), "read", "grant") in grants
        assert (str((tmp_path / "repo").resolve()), "read", "grant") in grants
        assert (str((tmp_path / "python-home").resolve()), "read", "grant") in grants

    def test_access_grants_for_workspace_write(self, tmp_path):
        config = WindowsSandboxConfig.build(
            mode=ExecutionMode.WORKSPACE_WRITE,
            executable=tmp_path / "overlay" / "Scripts" / "python.exe",
            args=["script.py"],
            workspace_path=tmp_path / "workspace",
            overlay_dir=tmp_path / "overlay",
            excluded_paths=[tmp_path / "workspace" / ".venv"],
        )

        grants = {(grant.path, grant.access, grant.effect) for grant in config.access_grants()}

        assert (str((tmp_path / "workspace").resolve()), "modify", "grant") in grants
        assert (str((tmp_path / "workspace" / ".venv").resolve()), "full", "deny") in grants
        assert (str((tmp_path / "overlay").resolve()), "modify", "grant") in grants

    def test_launcher_command_uses_module_entrypoint(self, tmp_path):
        config = WindowsSandboxConfig.build(
            mode=ExecutionMode.READ_ONLY,
            executable=tmp_path / "overlay" / "Scripts" / "python.exe",
            args=["script.py"],
            workspace_path=tmp_path / "workspace",
            overlay_dir=tmp_path / "overlay",
        )

        argv = windows_launcher_command(config)

        assert argv[1:3] == ("-m", "anton.windows_launcher")
        assert WindowsSandboxConfig.from_base64(argv[3]) == config
