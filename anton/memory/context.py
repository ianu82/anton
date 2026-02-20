from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.memory.learnings import LearningStore
    from anton.memory.store import SessionStore


class MemoryContext:
    """Builds the memory context string injected into the planner's system prompt."""

    def __init__(
        self,
        session_store: SessionStore,
        learning_store: LearningStore,
        *,
        skill_dirs: list[Path] | None = None,
    ) -> None:
        self._sessions = session_store
        self._learnings = learning_store
        self._skill_dirs = skill_dirs or []

    def build(self, task: str) -> str:
        sections: list[str] = []

        # Recent session summaries
        summaries = self._sessions.get_recent_summaries(limit=3)
        if summaries:
            lines = []
            for i, s in enumerate(summaries, 1):
                # Truncate long summaries
                preview = s[:300] + "..." if len(s) > 300 else s
                lines.append(f"{i}. {preview}")
            sections.append("## Recent Activity\n" + "\n".join(lines))

        # Relevant learnings
        learnings = self._learnings.find_relevant(task, limit=3)
        if learnings:
            lines = []
            for item in learnings:
                lines.append(f"### {item['topic']}\n{item['content']}")
            sections.append("## Relevant Learnings\n" + "\n".join(lines))

        # Skill-local memories
        skill_notes = self._gather_skill_memories(task)
        if skill_notes:
            lines = []
            for skill_name, content in skill_notes:
                lines.append(f"### {skill_name}\n{content}")
            sections.append("## Skill Notes\n" + "\n".join(lines))

        return "\n\n".join(sections)

    def _gather_skill_memories(self, task: str) -> list[tuple[str, str]]:
        """Scan skill directories for .memory/ files relevant to the task.

        Uses word overlap between skill name and task for basic relevance filtering.
        """
        results: list[tuple[str, str]] = []
        task_words = set(task.lower().split())

        for skill_dir in self._skill_dirs:
            skill_dir = Path(skill_dir)
            if not skill_dir.is_dir():
                continue

            for child in skill_dir.iterdir():
                if not child.is_dir():
                    continue

                memory_dir = child / ".memory"
                if not memory_dir.is_dir():
                    continue

                skill_name = child.name
                # Basic relevance: check word overlap between skill name and task
                skill_words = set(skill_name.lower().replace("-", " ").replace("_", " ").split())
                if not skill_words & task_words:
                    continue

                # Read all .md and .txt files from .memory/
                notes: list[str] = []
                for mem_file in sorted(memory_dir.iterdir()):
                    if mem_file.suffix in (".md", ".txt") and mem_file.is_file():
                        try:
                            notes.append(mem_file.read_text(encoding="utf-8").strip())
                        except (OSError, UnicodeDecodeError):
                            continue

                if notes:
                    results.append((skill_name, "\n\n".join(notes)))

        return results
