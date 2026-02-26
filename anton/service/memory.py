from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anton.memory.learnings import LearningStore
from anton.memory.store import SessionStore


@dataclass(slots=True)
class MemoryContextBuilder:
    """Build prompt-safe retrieved memory context with provenance."""

    memory_dir: Path
    enabled: bool = True
    max_items: int = 4
    max_chars: int = 4000

    def build(self, task: str) -> tuple[str, list[dict[str, str]]]:
        if not self.enabled:
            return "", []

        session_store = SessionStore(self.memory_dir)
        learning_store = LearningStore(self.memory_dir)

        summaries = session_store.get_recent_summaries(limit=self.max_items)
        learnings = learning_store.find_relevant(task, limit=self.max_items)

        lines: list[str] = []
        provenance: list[dict[str, str]] = []

        if summaries:
            lines.append("Recent session summaries:")
            for idx, summary in enumerate(summaries, start=1):
                text = summary.strip().replace("\n", " ")
                if not text:
                    continue
                lines.append(f"- summary_{idx}: {text[:500]}")
                provenance.append({"kind": "session_summary", "id": f"summary_{idx}"})

        if learnings:
            lines.append("Relevant learnings:")
            for item in learnings:
                topic = str(item.get("topic", "general"))
                summary = str(item.get("summary", "")).strip()
                content = str(item.get("content", "")).strip().replace("\n", " ")
                merged = summary or content[:500]
                if not merged:
                    continue
                lines.append(f"- {topic}: {merged[:500]}")
                provenance.append({"kind": "learning", "id": topic})

        if not lines:
            return "", []

        context = "\n".join(lines)
        if len(context) > self.max_chars:
            context = context[: self.max_chars] + "\n... (memory context truncated)"

        wrapped = (
            "<retrieved_memory_context>\n"
            "Use this as optional background context. Prefer current task evidence when conflicts exist.\n"
            f"{context}\n"
            "</retrieved_memory_context>"
        )
        return wrapped, provenance
