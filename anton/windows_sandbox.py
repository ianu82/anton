from __future__ import annotations

import base64
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from anton.execution_policy import ExecutionMode


def _resolve(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


@dataclass(frozen=True)
class WindowsAccessGrant:
    path: str
    access: str
    effect: str = "grant"
    recursive: bool = True


@dataclass(frozen=True)
class WindowsSandboxConfig:
    mode: str
    executable: str
    args: tuple[str, ...]
    workspace_path: str
    overlay_dir: str
    excluded_paths: tuple[str, ...] = ()
    runtime_path: str | None = None
    anton_root: str | None = None
    python_roots: tuple[str, ...] = ()
    extra_write_paths: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        *,
        mode: ExecutionMode | str,
        executable: str,
        args: list[str],
        workspace_path: Path,
        overlay_dir: Path,
        excluded_paths: list[Path] | None = None,
        runtime_path: Path | None = None,
        anton_root: Path | None = None,
        python_roots: list[Path] | None = None,
        extra_write_paths: list[Path] | None = None,
    ) -> "WindowsSandboxConfig":
        resolved_mode = ExecutionMode.coerce(mode)
        return cls(
            mode=resolved_mode.value,
            executable=_resolve(executable),
            args=tuple(args),
            workspace_path=_resolve(workspace_path),
            overlay_dir=_resolve(overlay_dir),
            excluded_paths=tuple(_resolve(path) for path in (excluded_paths or [])),
            runtime_path=_resolve(runtime_path) if runtime_path is not None else None,
            anton_root=_resolve(anton_root) if anton_root is not None else None,
            python_roots=tuple(_resolve(path) for path in (python_roots or [])),
            extra_write_paths=tuple(_resolve(path) for path in (extra_write_paths or [])),
        )

    def to_base64(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(payload).decode("ascii")

    @classmethod
    def from_base64(cls, value: str) -> "WindowsSandboxConfig":
        payload = base64.urlsafe_b64decode(value.encode("ascii"))
        raw = json.loads(payload.decode("utf-8"))
        return cls(
            mode=str(raw["mode"]),
            executable=str(raw["executable"]),
            args=tuple(raw.get("args", [])),
            workspace_path=str(raw["workspace_path"]),
            overlay_dir=str(raw["overlay_dir"]),
            excluded_paths=tuple(str(item) for item in raw.get("excluded_paths", [])),
            runtime_path=str(raw["runtime_path"]) if raw.get("runtime_path") else None,
            anton_root=str(raw["anton_root"]) if raw.get("anton_root") else None,
            python_roots=tuple(str(item) for item in raw.get("python_roots", [])),
            extra_write_paths=tuple(str(item) for item in raw.get("extra_write_paths", [])),
        )

    def access_grants(self) -> list[WindowsAccessGrant]:
        mode = ExecutionMode.coerce(self.mode)
        ordered: list[WindowsAccessGrant] = []
        seen: set[tuple[str, str]] = set()

        def add(path: str | None, access: str, *, effect: str = "grant") -> None:
            if not path:
                return
            key = (path, access, effect)
            if key in seen:
                return
            seen.add(key)
            ordered.append(WindowsAccessGrant(path=path, access=access, effect=effect))

        add(self.overlay_dir, "modify")
        for path in self.extra_write_paths:
            add(path, "modify")
        add(self.runtime_path, "read")
        add(self.anton_root, "read")
        for path in self.python_roots:
            add(path, "read")
        if mode is ExecutionMode.WORKSPACE_WRITE:
            add(self.workspace_path, "modify")
        elif mode is ExecutionMode.READ_ONLY:
            add(self.workspace_path, "read")
        for path in self.excluded_paths:
            add(path, "full", effect="deny")
        return ordered


def windows_launcher_command(config: WindowsSandboxConfig) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "anton.windows_launcher",
        config.to_base64(),
    )
