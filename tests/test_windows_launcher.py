from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import anton.windows_launcher as windows_launcher
from anton.execution_policy import ExecutionMode
from anton.windows_sandbox import WindowsSandboxConfig


class TestWindowsLauncher:
    def test_launch_config_uses_job_object(self, monkeypatch, tmp_path):
        monkeypatch.setattr(windows_launcher.sys, "platform", "win32")
        monkeypatch.setattr(windows_launcher, "create_kill_on_close_job", lambda: 77)

        assigned: list[tuple[int, int]] = []
        monkeypatch.setattr(
            windows_launcher,
            "assign_process_to_job",
            lambda job, proc: assigned.append((job, proc)),
        )

        closed: list[int] = []
        monkeypatch.setattr(windows_launcher, "close_handle", lambda handle: closed.append(handle))
        resumed: list[int] = []
        monkeypatch.setattr(windows_launcher, "resume_thread", lambda handle: resumed.append(handle))

        proc = MagicMock()
        proc._handle = 123
        proc._thread = 456
        proc.wait.return_value = 9
        popen_kwargs: list[dict] = []

        def fake_popen(*args, **kwargs):
            popen_kwargs.append(kwargs)
            return proc

        monkeypatch.setattr(windows_launcher.subprocess, "Popen", fake_popen)

        config = WindowsSandboxConfig.build(
            mode=ExecutionMode.READ_ONLY,
            executable=tmp_path / "overlay" / "Scripts" / "python.exe",
            args=["script.py"],
            workspace_path=tmp_path / "workspace",
            overlay_dir=tmp_path / "overlay",
        )

        exit_code = windows_launcher.launch_config(config)

        assert exit_code == 9
        assert assigned == [(77, 123)]
        assert resumed == [456]
        assert closed == [77]
        assert popen_kwargs[0]["creationflags"] == windows_launcher.CREATE_SUSPENDED
        proc.kill.assert_not_called()

    def test_launch_config_kills_process_if_assignment_fails(self, monkeypatch, tmp_path):
        monkeypatch.setattr(windows_launcher.sys, "platform", "win32")
        monkeypatch.setattr(windows_launcher, "create_kill_on_close_job", lambda: 88)
        monkeypatch.setattr(
            windows_launcher,
            "assign_process_to_job",
            lambda job, proc: (_ for _ in ()).throw(RuntimeError("assign failed")),
        )

        closed: list[int] = []
        monkeypatch.setattr(windows_launcher, "close_handle", lambda handle: closed.append(handle))
        monkeypatch.setattr(windows_launcher, "resume_thread", lambda handle: None)

        proc = MagicMock()
        proc._handle = 321
        proc._thread = 654
        proc.poll.return_value = None
        monkeypatch.setattr(windows_launcher.subprocess, "Popen", lambda *args, **kwargs: proc)

        config = WindowsSandboxConfig.build(
            mode=ExecutionMode.WORKSPACE_WRITE,
            executable=tmp_path / "overlay" / "Scripts" / "python.exe",
            args=["script.py"],
            workspace_path=tmp_path / "workspace",
            overlay_dir=tmp_path / "overlay",
        )

        with pytest.raises(RuntimeError, match="assign failed"):
            windows_launcher.launch_config(config)

        proc.kill.assert_called_once()
        proc.wait.assert_called()
        assert closed == [88]
