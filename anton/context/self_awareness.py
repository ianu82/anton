from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ContextUpdate:
    """A single update to apply to the self-awareness context."""

    file: str
    content: str | None  # None = delete


class SelfAwarenessContext:
    """Reads and manages .anton/context/ files for LLM self-awareness."""

    def __init__(self, context_dir: Path) -> None:
        self._dir = context_dir

    def build(self) -> dict[str, str]:
        """Read all non-dotfiles from the context directory.

        Returns:
            Mapping of filename to file content.
        """
        result: dict[str, str] = {}
        if not self._dir.is_dir():
            return result

        for path in sorted(self._dir.iterdir()):
            if path.name.startswith(".") or path.is_dir():
                continue
            try:
                result[path.name] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

        return result

    def build_prompt_section(self) -> str:
        """Format context files as a markdown section for system prompt injection."""
        files = self.build()
        if not files:
            return ""

        lines = ["\n## Self-Awareness Context"]
        for filename, content in files.items():
            heading = filename.rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title()
            lines.append(f"\n### {heading}")
            lines.append(content.strip())

        return "\n".join(lines)

    def apply_updates(self, updates: list[ContextUpdate]) -> list[str]:
        """Write or delete context files. Returns list of actions taken.

        Filenames are sanitized to prevent path traversal.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        actions: list[str] = []

        for update in updates:
            safe_name = self._sanitize_filename(update.file)
            if not safe_name:
                actions.append(f"Skipped invalid filename: {update.file!r}")
                continue

            target = self._dir / safe_name

            if update.content is None:
                # Delete
                if target.exists():
                    target.unlink()
                    actions.append(f"Deleted {safe_name}")
                else:
                    actions.append(f"File {safe_name} did not exist (no-op)")
            else:
                # Write / overwrite
                target.write_text(update.content, encoding="utf-8")
                actions.append(f"Updated {safe_name}")

        return actions

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize a filename to prevent path traversal and invalid chars."""
        # Strip directory components
        name = Path(name).name
        # Remove leading dots
        name = name.lstrip(".")
        # Replace unsafe chars
        name = re.sub(r"[^\w.\-]", "-", name)
        # Collapse multiple dashes
        name = re.sub(r"-{2,}", "-", name)
        return name.strip("-")
