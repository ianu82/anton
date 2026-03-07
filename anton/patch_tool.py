from __future__ import annotations

import os
from pathlib import Path

from anton.execution_policy import ExecutionMode
from anton.path_guard import ensure_workspace_path_is_direct


def workspace_root(session) -> Path:
    if session._workspace is not None:
        return session._workspace.base.resolve()
    return Path.cwd().resolve()


def resolve_workspace_path(root: Path, raw_path: str) -> Path:
    try:
        return ensure_workspace_path_is_direct(root, raw_path)
    except RuntimeError as exc:
        raise RuntimeError(str(exc).replace("Path escapes", "Patch path escapes")) from exc


def _ensure_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Refusing to patch non-text file: {path}") from exc


def _ensure_mutation_allowed(session, *, target_path: Path) -> None:
    mode = session._scratchpads.execution_mode
    if mode is ExecutionMode.READ_ONLY:
        raise RuntimeError("Rejected: execution mode 'read_only' does not allow workspace edits.")
    root = workspace_root(session)
    try:
        target_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Patch path escapes the workspace: {target_path}") from exc


def apply_workspace_edits(session, edits: list[dict]) -> str:
    if not edits:
        return "No patch edits provided."

    root = workspace_root(session)
    results: list[str] = []

    for idx, raw_edit in enumerate(edits, start=1):
        if not isinstance(raw_edit, dict):
            raise RuntimeError(f"Edit {idx} is not an object.")

        kind = str(raw_edit.get("kind", "")).strip().lower()
        raw_path = str(raw_edit.get("path", "")).strip()
        if kind not in {"create", "replace", "delete"}:
            raise RuntimeError(f"Edit {idx} has unsupported kind '{kind}'.")
        if not raw_path:
            raise RuntimeError(f"Edit {idx} is missing a path.")

        path = resolve_workspace_path(root, raw_path)
        _ensure_mutation_allowed(session, target_path=path)

        if kind == "create":
            if path.exists():
                raise RuntimeError(f"Create edit {idx} failed because the file already exists: {path}")
            new_text = str(raw_edit.get("new_text", ""))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_text, encoding="utf-8")
            results.append(f"created {os.path.relpath(path, root)}")
            continue

        if not path.is_file():
            raise RuntimeError(f"Edit {idx} requires an existing file: {path}")

        current = _ensure_text_file(path)
        if kind == "delete":
            expected = raw_edit.get("old_text")
            if expected is not None and current != str(expected):
                raise RuntimeError(f"Delete edit {idx} failed verification for {path}")
            path.unlink()
            results.append(f"deleted {os.path.relpath(path, root)}")
            continue

        old_text = str(raw_edit.get("old_text", ""))
        if not old_text:
            raise RuntimeError(f"Replace edit {idx} requires old_text.")
        occurrences = current.count(old_text)
        if occurrences != 1:
            raise RuntimeError(
                f"Replace edit {idx} expected exactly one match in {path}, found {occurrences}."
            )
        new_text = str(raw_edit.get("new_text", ""))
        updated = current.replace(old_text, new_text, 1)
        path.write_text(updated, encoding="utf-8")
        results.append(f"patched {os.path.relpath(path, root)}")

    return "Patch applied: " + ", ".join(results)
