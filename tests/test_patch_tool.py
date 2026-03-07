from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

from anton.chat import ChatSession
from anton.execution_policy import ExecutionMode
from anton.patch_tool import apply_workspace_edits
from anton.tools import PATCH_TOOL
from anton.workspace import Workspace


class TestPatchToolDefinition:
    def test_tool_definition_structure(self):
        assert PATCH_TOOL["name"] == "patch"
        props = PATCH_TOOL["input_schema"]["properties"]
        assert "edits" in props


class TestPatchToolApplication:
    def test_create_and_replace_file(self, tmp_path):
        workspace = Workspace(tmp_path)
        workspace.initialize()
        session = ChatSession(AsyncMock(), workspace=workspace)

        created = apply_workspace_edits(
            session,
            [
                {
                    "kind": "create",
                    "path": "hello.txt",
                    "new_text": "hello\nworld\n",
                }
            ],
        )
        assert "created hello.txt" in created
        assert (tmp_path / "hello.txt").read_text() == "hello\nworld\n"

        replaced = apply_workspace_edits(
            session,
            [
                {
                    "kind": "replace",
                    "path": "hello.txt",
                    "old_text": "world",
                    "new_text": "anton",
                }
            ],
        )
        assert "patched hello.txt" in replaced
        assert (tmp_path / "hello.txt").read_text() == "hello\nanton\n"

    def test_replace_requires_exact_single_match(self, tmp_path):
        workspace = Workspace(tmp_path)
        workspace.initialize()
        path = tmp_path / "dup.txt"
        path.write_text("repeat\nrepeat\n", encoding="utf-8")
        session = ChatSession(AsyncMock(), workspace=workspace)

        with pytest.raises(RuntimeError, match="expected exactly one match"):
            apply_workspace_edits(
                session,
                [
                    {
                        "kind": "replace",
                        "path": "dup.txt",
                        "old_text": "repeat",
                        "new_text": "once",
                    }
                ],
            )

    def test_path_escape_is_rejected(self, tmp_path):
        workspace = Workspace(tmp_path)
        workspace.initialize()
        outside = tmp_path.parent / "outside.txt"
        session = ChatSession(AsyncMock(), workspace=workspace)

        with pytest.raises(RuntimeError):
            apply_workspace_edits(
                session,
                [
                    {
                        "kind": "create",
                        "path": str(outside),
                        "new_text": "nope",
                    }
                ],
            )

    def test_read_only_mode_rejects_patch(self, tmp_path):
        workspace = Workspace(tmp_path)
        workspace.initialize()
        session = ChatSession(
            AsyncMock(),
            workspace=workspace,
            execution_mode=ExecutionMode.READ_ONLY,
        )

        with pytest.raises(RuntimeError, match="read_only"):
            apply_workspace_edits(
                session,
                [
                    {
                        "kind": "create",
                        "path": "blocked.txt",
                        "new_text": "blocked",
                    }
                ],
            )

    def test_symlinked_file_path_is_rejected(self, tmp_path):
        workspace = Workspace(tmp_path)
        workspace.initialize()
        target = tmp_path / "target.txt"
        target.write_text("hello\n", encoding="utf-8")
        link = tmp_path / "alias.txt"
        link.symlink_to(target)
        session = ChatSession(AsyncMock(), workspace=workspace)

        with pytest.raises(RuntimeError, match="symlink"):
            apply_workspace_edits(
                session,
                [
                    {
                        "kind": "replace",
                        "path": "alias.txt",
                        "old_text": "hello",
                        "new_text": "anton",
                    }
                ],
            )

    def test_symlinked_parent_path_is_rejected(self, tmp_path):
        workspace = Workspace(tmp_path)
        workspace.initialize()
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        target = real_dir / "hello.txt"
        target.write_text("hello\n", encoding="utf-8")
        alias_dir = tmp_path / "alias"
        alias_dir.symlink_to(real_dir, target_is_directory=True)
        session = ChatSession(AsyncMock(), workspace=workspace)

        with pytest.raises(RuntimeError, match="symlink"):
            apply_workspace_edits(
                session,
                [
                    {
                        "kind": "replace",
                        "path": "alias/hello.txt",
                        "old_text": "hello",
                        "new_text": "anton",
                    }
                ],
            )

    def test_hard_linked_file_path_is_rejected(self, tmp_path):
        workspace = Workspace(tmp_path)
        workspace.initialize()
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("outside\n", encoding="utf-8")
        linked = tmp_path / "linked.txt"
        os.link(outside, linked)
        session = ChatSession(AsyncMock(), workspace=workspace)

        with pytest.raises(RuntimeError, match="hard-linked"):
            apply_workspace_edits(
                session,
                [
                    {
                        "kind": "replace",
                        "path": "linked.txt",
                        "old_text": "outside",
                        "new_text": "inside",
                    }
                ],
            )
