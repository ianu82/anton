from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import anton.sandbox as sandbox_module
from anton.execution_policy import ExecutionMode
from anton.sandbox import build_sandbox_launch, ensure_execution_mode_supported


class TestSandboxSupport:
    def test_windows_supports_safe_modes(self, monkeypatch):
        monkeypatch.setattr(sandbox_module.sys, "platform", "win32")
        assert ensure_execution_mode_supported(ExecutionMode.READ_ONLY) is ExecutionMode.READ_ONLY
        assert ensure_execution_mode_supported(ExecutionMode.WORKSPACE_WRITE) is ExecutionMode.WORKSPACE_WRITE

    def test_windows_launch_spec_uses_launcher_module(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sandbox_module.sys, "platform", "win32")
        monkeypatch.setattr(sandbox_module, "ensure_execution_mode_supported", lambda mode: ExecutionMode.coerce(mode))

        workspace = tmp_path / "workspace"
        overlay = tmp_path / "overlay"
        runtime = tmp_path / "runtime"
        workspace.mkdir()
        overlay.mkdir()
        runtime.mkdir()

        spec = build_sandbox_launch(
            ExecutionMode.READ_ONLY,
            executable=str(overlay / "Scripts" / "python.exe"),
            args=["script.py"],
            workspace_path=workspace,
            overlay_dir=overlay,
            runtime_path=runtime,
            anton_root=tmp_path / "repo",
            python_roots=[tmp_path / "python-home"],
            extra_write_paths=[tmp_path / "boot"],
        )

        assert spec.runner == "windows_read_only"
        assert spec.argv[1:3] == ("-m", "anton.windows_launcher")

    def test_read_only_rejects_hard_linked_workspace_files(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sandbox_module.sys, "platform", "win32")
        monkeypatch.setattr(sandbox_module, "ensure_execution_mode_supported", lambda mode: ExecutionMode.coerce(mode))

        outside = tmp_path / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        workspace = tmp_path / "workspace"
        overlay = tmp_path / "overlay"
        workspace.mkdir()
        overlay.mkdir()
        os.link(outside, workspace / "linked.txt")

        with pytest.raises(RuntimeError, match="hard-linked file"):
            build_sandbox_launch(
                ExecutionMode.READ_ONLY,
                executable=str(overlay / "Scripts" / "python.exe"),
                args=["script.py"],
                workspace_path=workspace,
                overlay_dir=overlay,
            )

    def test_darwin_launch_contains_network_deny(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sandbox_module.sys, "platform", "darwin")
        monkeypatch.setattr(sandbox_module, "_find_sandbox_exec", lambda: "/usr/bin/sandbox-exec")

        workspace = tmp_path / "workspace"
        overlay = tmp_path / "overlay"
        workspace.mkdir()
        overlay.mkdir()

        spec = build_sandbox_launch(
            ExecutionMode.WORKSPACE_WRITE,
            executable="/usr/bin/python3",
            args=["-c", "print('ok')"],
            workspace_path=workspace,
            overlay_dir=overlay,
        )

        assert spec.argv[0] == "/usr/bin/sandbox-exec"
        assert "(deny network*)" in spec.profile_text
        assert "(allow file-read-metadata)" in spec.profile_text
        assert "(allow file-read-data\n" in spec.profile_text
        assert '(subpath "' in spec.profile_text
        assert f'(subpath "{workspace.resolve()}")' in spec.profile_text

    def test_linux_launch_uses_bwrap(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sandbox_module.sys, "platform", "linux")
        monkeypatch.setattr(sandbox_module, "_find_bwrap", lambda: "/usr/bin/bwrap")

        workspace = tmp_path / "workspace"
        overlay = tmp_path / "overlay"
        workspace.mkdir()
        overlay.mkdir()

        spec = build_sandbox_launch(
            ExecutionMode.READ_ONLY,
            executable="/usr/bin/python3",
            args=["-c", "print('ok')"],
            workspace_path=workspace,
            overlay_dir=overlay,
            runtime_path=tmp_path / "runtime",
        )

        assert spec.argv[0] == "/usr/bin/bwrap"
        assert "--unshare-net" in spec.argv
        assert "--ro-bind" in spec.argv


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="sandbox-exec integration is only available on macOS hosts with sandbox-exec",
)
class TestDarwinSandboxIntegration:
    def test_read_only_blocks_outside_workspace_reads(self, tmp_path):
        workspace = tmp_path / "workspace"
        overlay = tmp_path / "overlay"
        workspace.mkdir()
        overlay.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("top-secret", encoding="utf-8")

        code = (
            "from pathlib import Path; import sys\n"
            "target = Path(sys.argv[1])\n"
            "try:\n"
            "    print(target.read_text())\n"
            "except Exception as exc:\n"
            "    print(type(exc).__name__)\n"
        )
        spec = build_sandbox_launch(
            ExecutionMode.READ_ONLY,
            executable=sys.executable,
            args=["-c", code, str(outside)],
            workspace_path=workspace,
            overlay_dir=overlay,
        )
        result = subprocess.run(
            spec.argv,
            capture_output=True,
            text=True,
            cwd=str(workspace),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
        )

        assert "top-secret" not in result.stdout
        assert "PermissionError" in result.stdout or result.returncode != 0

    def test_read_only_blocks_workspace_writes(self, tmp_path):
        workspace = tmp_path / "workspace"
        overlay = tmp_path / "overlay"
        workspace.mkdir()
        overlay.mkdir()
        target = workspace / "blocked.txt"

        code = (
            "from pathlib import Path; import sys\n"
            "target = Path(sys.argv[1])\n"
            "try:\n"
            "    target.write_text('blocked')\n"
            "    print('write-ok')\n"
            "except Exception as exc:\n"
            "    print(type(exc).__name__)\n"
        )
        spec = build_sandbox_launch(
            ExecutionMode.READ_ONLY,
            executable=sys.executable,
            args=["-c", code, str(target)],
            workspace_path=workspace,
            overlay_dir=overlay,
        )
        result = subprocess.run(
            spec.argv,
            capture_output=True,
            text=True,
            cwd=str(workspace),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
        )

        assert not target.exists()
        assert "PermissionError" in result.stdout or result.returncode != 0

    def test_workspace_write_allows_workspace_writes_but_blocks_network(self, tmp_path):
        workspace = tmp_path / "workspace"
        overlay = tmp_path / "overlay"
        workspace.mkdir()
        overlay.mkdir()
        target = workspace / "allowed.txt"

        code = (
            "from pathlib import Path; import socket, sys\n"
            "target = Path(sys.argv[1])\n"
            "target.write_text('ok')\n"
            "print('write-ok')\n"
            "sock = socket.socket()\n"
            "sock.settimeout(1)\n"
            "try:\n"
            "    sock.connect(('1.1.1.1', 80))\n"
            "    print('network-ok')\n"
            "except Exception as exc:\n"
            "    print(type(exc).__name__)\n"
        )
        spec = build_sandbox_launch(
            ExecutionMode.WORKSPACE_WRITE,
            executable=sys.executable,
            args=["-c", code, str(target)],
            workspace_path=workspace,
            overlay_dir=overlay,
        )
        result = subprocess.run(
            spec.argv,
            capture_output=True,
            text=True,
            cwd=str(workspace),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
        )

        assert target.read_text() == "ok"
        assert "write-ok" in result.stdout
        assert "PermissionError" in result.stdout
