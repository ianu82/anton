from __future__ import annotations

import os
import subprocess
import shutil
import ssl
import sys
from dataclasses import dataclass
from pathlib import Path

from anton.execution_policy import ExecutionMode
from anton.path_guard import assert_workspace_tree_is_direct
from anton.workspace_scope import safe_mode_excluded_dirs
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


def _resolve_python_roots(python_roots: list[Path] | None) -> list[Path]:
    roots = python_roots or [Path(sys.base_prefix)]
    return [_resolve_path(path) for path in roots]


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    ordered: list[Path] = []
    for path in sorted((_resolve_path(path) for path in paths), key=lambda item: (len(item.parts), str(item))):
        if any(path == existing or path.is_relative_to(existing) for existing in ordered):
            continue
        ordered.append(path)
    return ordered


def _certificate_paths() -> list[Path]:
    verify_paths = ssl.get_default_verify_paths()
    candidates = [
        getattr(verify_paths, "cafile", None),
        getattr(verify_paths, "capath", None),
        getattr(verify_paths, "openssl_cafile", None),
        getattr(verify_paths, "openssl_capath", None),
        os.environ.get("SSL_CERT_FILE"),
        os.environ.get("SSL_CERT_DIR"),
        os.environ.get("REQUESTS_CA_BUNDLE"),
        os.environ.get("CURL_CA_BUNDLE"),
    ]
    paths: list[Path] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            paths.append(path if path.is_dir() else path.parent)
    return _dedupe_paths(paths)


def _unix_read_roots(
    *,
    executable: str,
    workspace_path: Path,
    overlay_dir: Path,
    runtime_path: Path | None,
    anton_root: Path | None,
    python_roots: list[Path],
    extra_write_paths: list[Path],
) -> list[Path]:
    roots = [
        _resolve_path(workspace_path),
        _resolve_path(overlay_dir),
        *(_resolve_path(path) for path in extra_write_paths),
        *python_roots,
    ]
    if runtime_path is not None:
        roots.append(_resolve_path(runtime_path))
    if anton_root is not None:
        roots.append(_resolve_path(anton_root))

    raw_executable = Path(executable).expanduser()
    if raw_executable.is_absolute():
        roots.append(raw_executable if raw_executable.is_dir() else raw_executable.parent)

    executable_path = _resolve_path(executable)
    roots.append(executable_path if executable_path.is_dir() else executable_path.parent)
    roots.extend(_certificate_paths())
    return _dedupe_paths(roots)


def _linux_dependency_dirs(executable: str) -> list[Path]:
    ldd = shutil.which("ldd")
    if ldd is None:
        return []

    try:
        result = subprocess.run(
            [ldd, executable],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []

    roots: list[Path] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "=>" in stripped:
            candidate = stripped.split("=>", 1)[1].strip().split(" ", 1)[0]
        else:
            candidate = stripped.split(" ", 1)[0]
        if not candidate.startswith("/"):
            continue
        path = Path(candidate)
        if path.exists():
            roots.append(path.resolve().parent)
    return _dedupe_paths(roots)


def _linux_runtime_read_roots(
    *,
    executable: str,
    runtime_path: Path | None,
    anton_root: Path | None,
    python_roots: list[Path],
) -> list[Path]:
    roots = [*python_roots, *_certificate_paths(), *_linux_dependency_dirs(executable)]
    if runtime_path is not None:
        roots.append(_resolve_path(runtime_path))
    if anton_root is not None:
        roots.append(_resolve_path(anton_root))

    raw_executable = Path(executable).expanduser()
    if raw_executable.is_absolute():
        roots.append(raw_executable if raw_executable.is_dir() else raw_executable.parent)

    resolved_executable = _resolve_path(executable)
    roots.append(resolved_executable if resolved_executable.is_dir() else resolved_executable.parent)
    return _dedupe_paths(roots)


def _linux_parent_dirs(paths: list[Path]) -> list[Path]:
    parents: set[Path] = set()
    for path in paths:
        current = path if path.is_dir() else path.parent
        while current != current.parent:
            parents.add(current)
            current = current.parent
    return sorted(parents, key=lambda item: (len(item.parts), str(item)))


def _darwin_read_rules(path: Path) -> list[str]:
    if path.is_file():
        return [f'    (literal "{path}")']
    return [
        f'    (literal "{path}")',
        f'    (subpath "{path}")',
    ]


def ensure_execution_mode_supported(mode: ExecutionMode | str) -> ExecutionMode:
    resolved = ExecutionMode.coerce(mode)
    if resolved is ExecutionMode.FULL_TRUST:
        return resolved

    if sys.platform == "win32":
        return resolved
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
    executable: str,
    workspace_path: Path,
    overlay_dir: Path,
    excluded_paths: list[Path],
    runtime_path: Path | None,
    anton_root: Path | None,
    python_roots: list[Path],
    extra_write_paths: list[Path],
) -> str:
    writable_paths = [
        _resolve_path(overlay_dir),
    ]
    writable_paths.extend(_resolve_path(path) for path in extra_write_paths)
    if mode is ExecutionMode.WORKSPACE_WRITE:
        writable_paths.append(_resolve_path(workspace_path))

    read_roots = _unix_read_roots(
        executable=executable,
        workspace_path=workspace_path,
        overlay_dir=overlay_dir,
        runtime_path=runtime_path,
        anton_root=anton_root,
        python_roots=python_roots,
        extra_write_paths=extra_write_paths,
    )
    raw_executable = Path(executable).expanduser()
    if raw_executable.is_absolute():
        if raw_executable.exists() and all(str(raw_executable) != str(existing) for existing in read_roots):
            read_roots.insert(0, raw_executable)
        raw_ancestors = [raw_executable if raw_executable.is_dir() else raw_executable.parent, *raw_executable.parents[:4]]
        for ancestor in raw_ancestors:
            if ancestor.exists() and all(str(ancestor) != str(existing) for existing in read_roots):
                read_roots.insert(0, ancestor)
    resolved_executable = _resolve_path(executable)
    if resolved_executable.exists() and all(str(resolved_executable) != str(existing) for existing in read_roots):
        read_roots.insert(0, resolved_executable)
    resolved_ancestors = [
        resolved_executable if resolved_executable.is_dir() else resolved_executable.parent,
        *resolved_executable.parents[:4],
    ]
    for ancestor in resolved_ancestors:
        if ancestor.exists() and all(str(ancestor) != str(existing) for existing in read_roots):
            read_roots.insert(0, ancestor)
    read_rules = "\n".join(rule for path in read_roots for rule in _darwin_read_rules(path))
    deny_rules = "\n".join(rule for path in excluded_paths for rule in _darwin_read_rules(path))
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
        "(allow file-read-metadata)\n"
        + (
        "(allow file-read-data\n"
        f"{read_rules}\n"
        ")\n"
        "(allow file-write*\n"
        f"{write_rules}\n"
        ")\n"
        )
        + (
            "(deny file-read-data\n"
            f"{deny_rules}\n"
            ")\n"
            "(deny file-write*\n"
            f"{deny_rules}\n"
            ")\n"
            if deny_rules
            else ""
        )
    )


def _linux_bwrap_argv(
    mode: ExecutionMode,
    *,
    executable: str,
    workspace_path: Path,
    overlay_dir: Path,
    excluded_paths: list[Path],
    runtime_path: Path | None,
    anton_root: Path | None,
    python_roots: list[Path],
    extra_write_paths: list[Path],
) -> list[str]:
    bwrap = _find_bwrap()
    if bwrap is None:
        raise RuntimeError("bubblewrap (bwrap) is required for safe execution modes on Linux.")

    workspace = _resolve_path(workspace_path)
    overlay = _resolve_path(overlay_dir)
    writable_roots = [overlay, *(_resolve_path(path) for path in extra_write_paths)]
    if mode is ExecutionMode.WORKSPACE_WRITE:
        writable_roots.append(workspace)

    read_only_roots = _linux_runtime_read_roots(
        executable=executable,
        runtime_path=runtime_path,
        anton_root=anton_root,
        python_roots=python_roots,
    )
    if mode is ExecutionMode.READ_ONLY:
        read_only_roots.append(workspace)
    read_only_roots = _dedupe_paths(read_only_roots)
    mount_roots = _dedupe_paths([*read_only_roots, *writable_roots])

    argv = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--unshare-net",
        "--chdir",
        str(workspace),
    ]

    for parent in _linux_parent_dirs(mount_roots):
        argv.extend(["--dir", str(parent)])
    for path in read_only_roots:
        argv.extend(["--ro-bind", str(path), str(path)])
    for path in writable_roots:
        argv.extend(["--bind", str(path), str(path)])
    for path in excluded_paths:
        argv.extend(["--tmpfs", str(path)])
    return argv


def _windows_launch_spec(
    mode: ExecutionMode,
    *,
    executable: str,
    args: list[str],
    workspace_path: Path,
    overlay_dir: Path,
    excluded_paths: list[Path],
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
        excluded_paths=excluded_paths,
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
    resolved_python_roots = _resolve_python_roots(python_roots)
    assert_workspace_tree_is_direct(workspace, mode_name=resolved.value)
    excluded_paths = safe_mode_excluded_dirs(workspace)

    if sys.platform == "win32":
        return _windows_launch_spec(
            resolved,
            executable=executable,
            args=args,
            workspace_path=workspace,
            overlay_dir=overlay,
            excluded_paths=excluded_paths,
            runtime_path=runtime_path,
            anton_root=anton_root,
            python_roots=resolved_python_roots,
            extra_write_paths=extra_paths,
        )

    if sys.platform == "darwin":
        profile = _darwin_profile(
            resolved,
            executable=executable,
            workspace_path=workspace,
            overlay_dir=overlay,
            excluded_paths=excluded_paths,
            runtime_path=runtime_path,
            anton_root=anton_root,
            python_roots=resolved_python_roots,
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
        executable=executable,
        workspace_path=workspace,
        overlay_dir=overlay,
        excluded_paths=excluded_paths,
        runtime_path=runtime_path,
        anton_root=anton_root,
        python_roots=resolved_python_roots,
        extra_write_paths=extra_paths,
    )
    argv.extend([executable, *args])
    return SandboxLaunchSpec(argv=tuple(argv), runner=f"unix_{resolved.value}")
