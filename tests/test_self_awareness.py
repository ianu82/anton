from __future__ import annotations

from pathlib import Path

import pytest

from anton.context.self_awareness import ContextUpdate, SelfAwarenessContext


@pytest.fixture()
def ctx_dir(tmp_path):
    d = tmp_path / "context"
    d.mkdir()
    return d


@pytest.fixture()
def sa(ctx_dir):
    return SelfAwarenessContext(ctx_dir)


class TestBuild:
    def test_empty_dir_returns_empty_dict(self, sa):
        assert sa.build() == {}

    def test_reads_files(self, sa, ctx_dir):
        (ctx_dir / "project-overview.md").write_text("This is a Python project.")
        (ctx_dir / "conventions.md").write_text("Use black for formatting.")

        result = sa.build()
        assert len(result) == 2
        assert result["conventions.md"] == "Use black for formatting."
        assert result["project-overview.md"] == "This is a Python project."

    def test_skips_dotfiles(self, sa, ctx_dir):
        (ctx_dir / ".hidden").write_text("secret")
        (ctx_dir / "visible.md").write_text("content")

        result = sa.build()
        assert ".hidden" not in result
        assert "visible.md" in result

    def test_skips_subdirectories(self, sa, ctx_dir):
        sub = ctx_dir / "subdir"
        sub.mkdir()
        (sub / "nested.md").write_text("nested content")
        (ctx_dir / "top.md").write_text("top content")

        result = sa.build()
        assert len(result) == 1
        assert "top.md" in result

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        sa = SelfAwarenessContext(tmp_path / "nonexistent")
        assert sa.build() == {}


class TestBuildPromptSection:
    def test_empty_returns_empty_string(self, sa):
        assert sa.build_prompt_section() == ""

    def test_formats_markdown(self, sa, ctx_dir):
        (ctx_dir / "project-overview.md").write_text("A Python CLI tool.")

        section = sa.build_prompt_section()
        assert "## Self-Awareness Context" in section
        assert "### Project Overview" in section
        assert "A Python CLI tool." in section

    def test_multiple_files(self, sa, ctx_dir):
        (ctx_dir / "conventions.md").write_text("Use pytest.")
        (ctx_dir / "stack.md").write_text("Python 3.11")

        section = sa.build_prompt_section()
        assert "### Conventions" in section
        assert "### Stack" in section


class TestApplyUpdates:
    def test_create_file(self, sa, ctx_dir):
        updates = [ContextUpdate(file="new-file.md", content="Hello world")]
        actions = sa.apply_updates(updates)

        assert (ctx_dir / "new-file.md").read_text() == "Hello world"
        assert "Updated new-file.md" in actions[0]

    def test_overwrite_file(self, sa, ctx_dir):
        (ctx_dir / "existing.md").write_text("old content")
        updates = [ContextUpdate(file="existing.md", content="new content")]
        sa.apply_updates(updates)

        assert (ctx_dir / "existing.md").read_text() == "new content"

    def test_delete_file(self, sa, ctx_dir):
        (ctx_dir / "to-delete.md").write_text("bye")
        updates = [ContextUpdate(file="to-delete.md", content=None)]
        actions = sa.apply_updates(updates)

        assert not (ctx_dir / "to-delete.md").exists()
        assert "Deleted to-delete.md" in actions[0]

    def test_delete_nonexistent_is_noop(self, sa, ctx_dir):
        updates = [ContextUpdate(file="ghost.md", content=None)]
        actions = sa.apply_updates(updates)
        assert "did not exist" in actions[0]

    def test_creates_context_dir_if_missing(self, tmp_path):
        new_dir = tmp_path / "fresh" / "context"
        sa = SelfAwarenessContext(new_dir)
        updates = [ContextUpdate(file="hello.md", content="world")]
        sa.apply_updates(updates)

        assert (new_dir / "hello.md").read_text() == "world"

    def test_sanitizes_path_traversal(self, sa, ctx_dir):
        updates = [ContextUpdate(file="../escape.md", content="nope")]
        sa.apply_updates(updates)

        # Should NOT create a file in the parent directory
        assert not (ctx_dir.parent / "escape.md").exists()
        # Should create it safely inside the context dir
        assert (ctx_dir / "escape.md").read_text() == "nope"

    def test_sanitizes_absolute_path(self, sa, ctx_dir):
        updates = [ContextUpdate(file="/etc/passwd", content="nope")]
        sa.apply_updates(updates)
        assert not Path("/etc/passwd_test").exists()
        # The file should be created safely inside the context dir with sanitized name
        assert (ctx_dir / "passwd").read_text() == "nope"

    def test_multiple_updates(self, sa, ctx_dir):
        (ctx_dir / "old.md").write_text("old")
        updates = [
            ContextUpdate(file="new.md", content="new content"),
            ContextUpdate(file="old.md", content=None),
        ]
        actions = sa.apply_updates(updates)

        assert (ctx_dir / "new.md").exists()
        assert not (ctx_dir / "old.md").exists()
        assert len(actions) == 2


class TestSanitizeFilename:
    def test_simple_name(self):
        assert SelfAwarenessContext._sanitize_filename("hello.md") == "hello.md"

    def test_strips_directory(self):
        assert SelfAwarenessContext._sanitize_filename("foo/bar/baz.md") == "baz.md"

    def test_strips_leading_dots(self):
        assert SelfAwarenessContext._sanitize_filename(".hidden") == "hidden"

    def test_replaces_unsafe_chars(self):
        result = SelfAwarenessContext._sanitize_filename("hello world!.md")
        assert " " not in result
        assert "!" not in result
