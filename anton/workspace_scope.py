from __future__ import annotations

import os
from pathlib import Path


SAFE_MODE_EXCLUDED_DIR_NAMES = frozenset({
    ".venv",
    "venv",
    "env",
    ".direnv",
    "node_modules",
})
SAFE_MODE_EXCLUDED_RELATIVE_DIRS = (
    (".anton", "scratchpad-venvs"),
)


def _resolve_root(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def safe_mode_excluded_dirs(root: str | Path) -> list[Path]:
    """Return existing workspace paths excluded from safe-mode scope."""
    resolved_root = _resolve_root(root)
    excluded: list[Path] = []
    seen: set[Path] = set()
    try:
        with os.scandir(resolved_root) as entries:
            for entry in entries:
                if entry.name not in SAFE_MODE_EXCLUDED_DIR_NAMES:
                    continue
                if entry.is_dir(follow_symlinks=False) or entry.is_symlink():
                    path = Path(entry.path)
                    if path not in seen:
                        excluded.append(path)
                        seen.add(path)
    except FileNotFoundError:
        return []
    for relative_parts in SAFE_MODE_EXCLUDED_RELATIVE_DIRS:
        candidate = resolved_root.joinpath(*relative_parts)
        try:
            if candidate.is_dir() or candidate.is_symlink():
                if candidate not in seen:
                    excluded.append(candidate)
                    seen.add(candidate)
        except OSError:
            continue
    return sorted(excluded, key=lambda item: item.name)


def is_safe_mode_excluded_dir(root: str | Path, candidate: str | Path) -> bool:
    resolved_root = _resolve_root(root)
    candidate_path = Path(candidate)
    try:
        relative = candidate_path.relative_to(resolved_root)
    except ValueError:
        return False
    return (
        (len(relative.parts) == 1 and relative.parts[0] in SAFE_MODE_EXCLUDED_DIR_NAMES)
        or tuple(relative.parts) in SAFE_MODE_EXCLUDED_RELATIVE_DIRS
    )
