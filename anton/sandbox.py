from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from anton.execution_policy import ExecutionMode
from anton.path_guard import assert_workspace_tree_is_direct
from anton.windows_sandbox import WindowsSandboxConfig, windows_launcher_command


@dataclass(frozen=True)
class SandboxLaunchSpec:
    argv: tuple[str, ...]
    runner: str
    profile_text: str = ""


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _find_sandbox_exec() -> str | None:
    return shutil.which("sandbox-exec") or ("/usr/bin/sandbox-exec" if Path("/usr/bin/sandbox-exec").exists() else None)


def _find_bwrap() -> str | None:
    return shutil.which("bwrap")


def ensure_execution_mode_supported(mode: ExecutionMode | str) -> ExecutionMode:
    resolved = ExecutionMode.coerce(mode)
    if resolved is ExecutionMode.FULL_TRUST:
        return resolved

    if sys.platform == "win32":
        raise RuntimeError(
            "Execution modes 'workspace_write' and 'read_only' are not supported on Windows yet. "
            "Use full_trust on Windows."
        )
    if sys.platform == "darwin":
        if _find_sandbox_exec() is None:
            raise RuntimeError(
                "sandbox-exec is unavailable on this macOS host, so safe execution modes cannot start."
            )
        return resolved
    if sys.platform.startswith("linux"):
        if _find_bwrap() is None:
            raise RuntimeError(
                "bubblewrap (bwrap) is required for safe execution modes on Linux."
            )
        return resolved

    raise RuntimeError(
        f"Execution mode '{resolved.value}' is not supported on platform '{sys.platform}'."
    )


def _darwin_profile(
    mode: ExecutionMode,
    *,
    workspace_path: Path,
    overlay_dir: Path,
    extra_write_paths: list[Path],
) -> str:
    writable_paths = [
        _resolve_path(overlay_dir),
    ]
    writable_paths.extend(_resolve_path(path) for path in extra_write_paths)
    if mode is ExecutionMode.WORKSPACE_WRITE:
        writable_paths.append(_resolve_path(workspace_path))

    write_rules = "\n".join(
        f'    (subpath "{path}")'
        for path in writable_paths
    )
    return (
        "(version 1)\n"
        '(import "system.sb")\n'
        "(deny default)\n"
        "(allow process*)\n"
        "(deny network*)\n"
        "(allow file-read*)\n"
        "(allow file-write*\n"
        f"{write_rules}\n"
        ")\n"
    )


def _linux_bwrap_argv(
    mode: ExecutionMode,
    *,
    workspace_path: Path,
    overlay_dir: Path,
    extra_write_paths: list[Path],
) -> list[str]:
    bwrap = _find_bwrap()
    if bwrap is None:
        raise RuntimeError("bubblewrap (bwrap) is required for safe execution modes on Linux.")

    workspace = _resolve_path(workspace_path)
    overlay = _resolve_path(overlay_dir)

    argv = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--ro-bind",
        "/",
        "/",
        "--bind",
        str(overlay),
        str(overlay),
        "--unshare-net",
    ]
    for path in extra_write_paths:
        resolved = _resolve_path(path)
        argv.extend(["--bind", str(resolved), str(resolved)])
    if mode is ExecutionMode.WORKSPACE_WRITE:
        argv.extend(["--bind", str(workspace), str(workspace)])
    else:
        argv.extend(["--ro-bind", str(workspace), str(workspace)])
    return argv


def _windows_launch_spec(
    mode: ExecutionMode,
    *,
    executable: str,
    args: list[str],
    workspace_path: Path,
    overlay_dir: Path,
    runtime_path: Path | None = None,
    anton_root: Path | None = None,
    python_roots: list[Path] | None = None,
    extra_write_paths: list[Path] | None = None,
) -> SandboxLaunchSpec:
    config = WindowsSandboxConfig.build(
        mode=mode,
        executable=executable,
        args=args,
        workspace_path=workspace_path,
        overlay_dir=overlay_dir,
        runtime_path=runtime_path,
        anton_root=anton_root,
        python_roots=python_roots,
        extra_write_paths=extra_write_paths,
    )
    return SandboxLaunchSpec(
        argv=windows_launcher_command(config),
        runner=f"windows_{mode.value}",
        profile_text=config.to_base64(),
    )


def build_sandbox_launch(
    mode: ExecutionMode | str,
    *,
    executable: str,
    args: list[str],
    workspace_path: Path | None,
    overlay_dir: Path,
    runtime_path: Path | None = None,
    anton_root: Path | None = None,
    python_roots: list[Path] | None = None,
    extra_write_paths: list[Path] | None = None,
) -> SandboxLaunchSpec:
    resolved = ensure_execution_mode_supported(mode)
    if resolved is ExecutionMode.FULL_TRUST:
        return SandboxLaunchSpec(argv=(executable, *args), runner="none")
    if workspace_path is None:
        raise RuntimeError(f"Execution mode '{resolved.value}' requires a workspace path.")

    workspace = _resolve_path(workspace_path)
    overlay = _resolve_path(overlay_dir)
    extra_paths = [_resolve_path(path) for path in (extra_write_paths or [])]
    assert_workspace_tree_is_direct(workspace, mode_name=resolved.value)

    if sys.platform == "win32":
        return _windows_launch_spec(
            resolved,
            executable=executable,
            args=args,
            workspace_path=workspace,
            overlay_dir=overlay,
            runtime_path=runtime_path,
            anton_root=anton_root,
            python_roots=python_roots,
            extra_write_paths=extra_paths,
        )

    if sys.platform == "darwin":
        profile = _darwin_profile(
            resolved,
            workspace_path=workspace,
            overlay_dir=overlay,
            extra_write_paths=extra_paths,
        )
        sandbox_exec = _find_sandbox_exec()
        if sandbox_exec is None:
            raise RuntimeError("sandbox-exec is unavailable on this macOS host.")
        return SandboxLaunchSpec(
            argv=(sandbox_exec, "-p", profile, executable, *args),
            runner=f"unix_{resolved.value}",
            profile_text=profile,
        )

    argv = _linux_bwrap_argv(
        resolved,
        workspace_path=workspace,
        overlay_dir=overlay,
        extra_write_paths=extra_paths,
    )
    if runtime_path is not None:
        runtime = _resolve_path(runtime_path)
        argv.extend(["--ro-bind", str(runtime), str(runtime)])
    argv.extend([executable, *args])
    return SandboxLaunchSpec(argv=tuple(argv), runner=f"unix_{resolved.value}")
