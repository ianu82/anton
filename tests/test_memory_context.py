from __future__ import annotations

from pathlib import Path

import pytest

from anton.memory.context import MemoryContext
from anton.memory.learnings import LearningStore
from anton.memory.store import SessionStore


@pytest.fixture()
def session_store(tmp_path):
    return SessionStore(tmp_path)


@pytest.fixture()
def learning_store(tmp_path):
    return LearningStore(tmp_path)


@pytest.fixture()
def ctx(session_store, learning_store):
    return MemoryContext(session_store, learning_store)


class TestBuild:
    async def test_no_history_returns_empty_string(self, ctx):
        result = ctx.build("some task")
        assert result == ""

    async def test_with_summaries_includes_recent_activity(self, ctx, session_store):
        s1 = await session_store.start_session("task 1")
        await session_store.complete_session(s1, "Completed task 1 successfully")

        result = ctx.build("new task")
        assert "## Recent Activity" in result
        assert "Completed task 1 successfully" in result

    async def test_with_relevant_learnings_includes_them(self, ctx, learning_store):
        await learning_store.record("file_ops", "Always check existence", "File operations safety")

        result = ctx.build("read a file safely")
        assert "## Relevant Learnings" in result
        assert "Always check existence" in result

    async def test_with_both_includes_both_sections(self, ctx, session_store, learning_store):
        s1 = await session_store.start_session("task 1")
        await session_store.complete_session(s1, "Completed task 1")
        await learning_store.record("file_ops", "Check existence", "File operations")

        result = ctx.build("file operations task")
        assert "## Recent Activity" in result
        assert "## Relevant Learnings" in result

    async def test_no_matching_learnings_omits_section(self, ctx, learning_store):
        await learning_store.record("network", "HTTP patterns", "Network stuff")

        result = ctx.build("quantum computing")
        assert "## Relevant Learnings" not in result


class TestSkillMemories:
    def test_no_skill_dirs_no_section(self, session_store, learning_store):
        ctx = MemoryContext(session_store, learning_store)
        result = ctx.build("some task")
        assert "## Skill Notes" not in result

    def test_skill_memory_included_when_relevant(self, session_store, learning_store, tmp_path):
        # Create a skill with .memory/
        skill_dir = tmp_path / "skills" / "read_file"
        mem_dir = skill_dir / ".memory"
        mem_dir.mkdir(parents=True)
        (mem_dir / "notes.md").write_text("Always use UTF-8 encoding.")

        ctx = MemoryContext(
            session_store, learning_store,
            skill_dirs=[tmp_path / "skills"],
        )
        result = ctx.build("read a file from disk")
        assert "## Skill Notes" in result
        assert "read_file" in result
        assert "Always use UTF-8 encoding." in result

    def test_skill_memory_excluded_when_irrelevant(self, session_store, learning_store, tmp_path):
        # Create a skill that doesn't match the task
        skill_dir = tmp_path / "skills" / "deploy_server"
        mem_dir = skill_dir / ".memory"
        mem_dir.mkdir(parents=True)
        (mem_dir / "notes.md").write_text("Use port 8080.")

        ctx = MemoryContext(
            session_store, learning_store,
            skill_dirs=[tmp_path / "skills"],
        )
        result = ctx.build("read a file")
        assert "## Skill Notes" not in result

    def test_multiple_memory_files_combined(self, session_store, learning_store, tmp_path):
        skill_dir = tmp_path / "skills" / "write_file"
        mem_dir = skill_dir / ".memory"
        mem_dir.mkdir(parents=True)
        (mem_dir / "encoding.md").write_text("Always specify encoding.")
        (mem_dir / "safety.txt").write_text("Check directory exists first.")

        ctx = MemoryContext(
            session_store, learning_store,
            skill_dirs=[tmp_path / "skills"],
        )
        result = ctx.build("write a new file")
        assert "Always specify encoding." in result
        assert "Check directory exists first." in result

    def test_nonexistent_skill_dir_ignored(self, session_store, learning_store, tmp_path):
        ctx = MemoryContext(
            session_store, learning_store,
            skill_dirs=[tmp_path / "nonexistent"],
        )
        result = ctx.build("some task")
        assert "## Skill Notes" not in result

    def test_skill_without_memory_dir_ignored(self, session_store, learning_store, tmp_path):
        skill_dir = tmp_path / "skills" / "list_files"
        skill_dir.mkdir(parents=True)
        # No .memory/ subdirectory

        ctx = MemoryContext(
            session_store, learning_store,
            skill_dirs=[tmp_path / "skills"],
        )
        result = ctx.build("list all files")
        assert "## Skill Notes" not in result
