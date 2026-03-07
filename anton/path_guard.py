from __future__ import annotations

import os
import stat
from pathlib import Path

from anton.workspace_scope import is_safe_mode_excluded_dir

_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_reparse_point(st: os.stat_result) -> bool:
    attributes = getattr(st, "st_file_attributes", 0)
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _iter_relative_components(root: Path, candidate: Path) -> list[str]:
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Path escapes the workspace: {candidate}") from exc
    return list(relative.parts)


def _entry_alias_type(path: Path) -> str | None:
    try:
        if path.is_symlink():
            return "symlink"
        st = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    if _is_reparse_point(st):
        return "reparse point"
    if path.is_file() and st.st_nlink > 1:
        return "hard-linked file"
    return None


def ensure_workspace_path_is_direct(root: Path, requested_path: str | Path) -> Path:
    resolved_root = _resolve_path(root)
    candidate = Path(requested_path)
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    lexical = Path(os.path.normpath(str(candidate)))
    _iter_relative_components(resolved_root, lexical)

    current = resolved_root
    for component in _iter_relative_components(resolved_root, lexical):
        current = current / component
        alias_type = _entry_alias_type(current)
        if alias_type is not None:
            raise RuntimeError(f"Refusing to use {alias_type} path inside workspace: {current}")

    resolved = lexical.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"Path escapes the workspace: {requested_path}") from exc
    return resolved


def assert_workspace_tree_is_direct(root: Path, *, mode_name: str) -> None:
    resolved_root = _resolve_path(root)
    stack = [resolved_root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                entry_path = Path(entry.path)
                if is_safe_mode_excluded_dir(resolved_root, entry_path):
                    continue
                if entry.is_symlink():
                    raise RuntimeError(
                        f"{mode_name} mode rejected this workspace because it contains a symlink: {entry.path}"
                    )
                st = entry.stat(follow_symlinks=False)
                if _is_reparse_point(st):
                    raise RuntimeError(
                        f"{mode_name} mode rejected this workspace because it contains a reparse point: {entry.path}"
                    )
                if stat.S_ISDIR(st.st_mode):
                    stack.append(entry_path)
                    continue
                if stat.S_ISREG(st.st_mode) and st.st_nlink > 1:
                    raise RuntimeError(
                        f"{mode_name} mode rejected this workspace because it contains a hard-linked file: {entry.path}"
                    )
