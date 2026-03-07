from __future__ import annotations

import pytest

import anton.windows_launcher as windows_launcher
from anton.execution_policy import ExecutionMode
from anton.windows_identity import AppContainerProfile, WindowsProcessHandles
from anton.windows_sandbox import WindowsSandboxConfig


class TestWindowsLauncher:
    def test_launch_config_grants_access_and_runs_under_job_object(self, monkeypatch, tmp_path):
        monkeypatch.setattr(windows_launcher.sys, "platform", "win32")
        monkeypatch.setattr(
            windows_launcher,
            "ensure_appcontainer_profile",
            lambda mode: AppContainerProfile(
                name="AntonScratchpadReadOnly",
                sid_ptr=444,
                sid_string="S-1-15-2-444",
            ),
        )

        granted: list[tuple[str, list]] = []
        monkeypatch.setattr(
            windows_launcher,
            "apply_access_grants",
            lambda sid, grants: granted.append((sid, grants)),
        )
        monkeypatch.setattr(windows_launcher, "create_kill_on_close_job", lambda: 77)

        assigned: list[tuple[int, int]] = []
        monkeypatch.setattr(
            windows_launcher,
            "assign_process_to_job",
            lambda job, proc: assigned.append((job, proc)),
        )

        resumed: list[int] = []
        monkeypatch.setattr(windows_launcher, "resume_thread", lambda handle: resumed.append(handle))
        monkeypatch.setattr(windows_launcher, "wait_for_process_exit", lambda handle: 9)

        launched: list[tuple[AppContainerProfile, str, tuple[str, ...], str, dict[str, str]]] = []
        monkeypatch.setattr(
            windows_launcher,
            "launch_appcontainer_process",
            lambda **kwargs: launched.append(
                (
                    kwargs["profile"],
                    kwargs["executable"],
                    kwargs["args"],
                    kwargs["cwd"],
                    kwargs["env"],
                )
            )
            or WindowsProcessHandles(process_handle=123, thread_handle=456, pid=789),
        )

        freed: list[int] = []
        monkeypatch.setattr(windows_launcher, "free_sid", lambda sid: freed.append(sid))

        closed: list[int] = []
        monkeypatch.setattr(windows_launcher, "close_handle", lambda handle: closed.append(handle))

        config = WindowsSandboxConfig.build(
            mode=ExecutionMode.READ_ONLY,
            executable=tmp_path / "overlay" / "Scripts" / "python.exe",
            args=["script.py"],
            workspace_path=tmp_path / "workspace",
            overlay_dir=tmp_path / "overlay",
        )

        exit_code = windows_launcher.launch_config(config)

        assert exit_code == 9
        assert granted[0][0] == "S-1-15-2-444"
        assert assigned == [(77, 123)]
        assert resumed == [456]
        assert launched[0][1] == config.executable
        assert launched[0][2] == config.args
        assert launched[0][3] == config.workspace_path
        assert "PATH" in launched[0][4]
        assert closed == [456, 123, 77]
        assert freed == [444]

    def test_launch_config_closes_handles_when_process_launch_fails(self, monkeypatch, tmp_path):
        monkeypatch.setattr(windows_launcher.sys, "platform", "win32")
        monkeypatch.setattr(
            windows_launcher,
            "ensure_appcontainer_profile",
            lambda mode: AppContainerProfile(
                name="AntonScratchpadWorkspaceWrite",
                sid_ptr=555,
                sid_string="S-1-15-2-555",
            ),
        )
        monkeypatch.setattr(windows_launcher, "apply_access_grants", lambda sid, grants: None)
        monkeypatch.setattr(windows_launcher, "create_kill_on_close_job", lambda: 88)
        monkeypatch.setattr(
            windows_launcher,
            "launch_appcontainer_process",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("launch failed")),
        )

        freed: list[int] = []
        monkeypatch.setattr(windows_launcher, "free_sid", lambda sid: freed.append(sid))

        closed: list[int] = []
        monkeypatch.setattr(windows_launcher, "close_handle", lambda handle: closed.append(handle))

        with pytest.raises(RuntimeError, match="launch failed"):
            windows_launcher.launch_config(
                WindowsSandboxConfig.build(
                    mode=ExecutionMode.WORKSPACE_WRITE,
                    executable=tmp_path / "overlay" / "Scripts" / "python.exe",
                    args=["script.py"],
                    workspace_path=tmp_path / "workspace",
                    overlay_dir=tmp_path / "overlay",
                )
            )

        assert closed == [88]
        assert freed == [555]
