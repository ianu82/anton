from __future__ import annotations

from pathlib import Path

import pytest

import anton.windows_acl as windows_acl
from anton.windows_sandbox import WindowsAccessGrant


class TestWindowsAcl:
    def test_grant_spec_uses_recursive_dir_rights(self, tmp_path):
        directory = tmp_path / "workspace"
        directory.mkdir()

        command, is_dir = windows_acl._grant_spec(directory, "read", recursive=True)

        assert is_dir is True
        assert command == [
            "icacls",
            str(directory.resolve()),
            "/grant:r",
            "(OI)(CI)(RX)",
            "/t",
            "/c",
            "/q",
        ]

    def test_grant_access_to_sid_invokes_icacls(self, monkeypatch, tmp_path):
        monkeypatch.setattr(windows_acl.sys, "platform", "win32")
        directory = tmp_path / "workspace"
        directory.mkdir()

        calls: list[list[str]] = []

        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

        monkeypatch.setattr(
            windows_acl.subprocess,
            "run",
            lambda command, **kwargs: calls.append(command) or Result(),
        )

        windows_acl.grant_access_to_sid(
            "S-1-15-2-123",
            WindowsAccessGrant(path=str(directory), access="modify"),
        )

        assert calls == [
            [
                "icacls",
                str(directory.resolve()),
                "/grant:r",
                "*S-1-15-2-123:(OI)(CI)(M)",
                "/t",
                "/c",
                "/q",
            ]
        ]

    def test_grant_access_to_sid_raises_on_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr(windows_acl.sys, "platform", "win32")
        path = tmp_path / "file.txt"
        path.write_text("ok", encoding="utf-8")

        class Result:
            returncode = 1
            stderr = "Access is denied."
            stdout = ""

        monkeypatch.setattr(windows_acl.subprocess, "run", lambda command, **kwargs: Result())

        with pytest.raises(RuntimeError, match="Access is denied"):
            windows_acl.grant_access_to_sid(
                "S-1-15-2-123",
                WindowsAccessGrant(path=str(path), access="read", recursive=False),
            )
