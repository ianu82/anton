from __future__ import annotations

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
