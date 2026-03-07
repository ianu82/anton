from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anton import __version__


PACKAGE_EVENT_TYPES = {
    "missing_import",
    "explicit_install",
    "legacy_auto_install",
    "profile_hydrate",
    "install_failure",
}


@dataclass(frozen=True)
class PackageSummaryRow:
    name: str
    events: int
    failures: int
    workspaces: int


@dataclass(frozen=True)
class PackageTelemetrySummary:
    total_events: int
    event_counts: list[tuple[str, int]]
    package_rows: list[PackageSummaryRow]
    profile_counts: list[tuple[str, int]]


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    description: str
    packages: tuple[str, ...]
    extends: tuple[str, ...] = ()
    post_install: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class InstalledRuntime:
    profile: str
    anton_version: str
    lock_hash: str
    path: Path
    package_count: int
    hydrated: bool


_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+")


def telemetry_dir() -> Path:
    override = os.environ.get("ANTON_TELEMETRY_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".anton" / "telemetry"


def package_events_path() -> Path:
    return telemetry_dir() / "package_events.jsonl"


def runtime_home() -> Path:
    override = os.environ.get("ANTON_RUNTIME_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".anton" / "runtimes"


def runtime_profile_dir() -> Path:
    override = os.environ.get("ANTON_RUNTIME_PROFILE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / "runtime_profiles"


def runtime_version_dir(anton_version: str = __version__) -> Path:
    return runtime_home() / anton_version


def runtime_manifest_path(profile_name: str) -> Path:
    return runtime_profile_dir() / f"{profile_name}.json"


def _resolve_workspace_path(workspace_path: str | Path | None = None) -> str:
    candidate = workspace_path or os.environ.get("ANTON_WORKSPACE_PATH") or os.getcwd()
    return str(Path(candidate).expanduser().resolve())


def workspace_hash(workspace_path: str | Path | None = None) -> str:
    resolved = _resolve_workspace_path(workspace_path)
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]


def log_package_event(
    event: str,
    *,
    package: str | None = None,
    profile: str | None = None,
    scratchpad: str | None = None,
    source: str | None = None,
    status: str | None = None,
    error: str | None = None,
    workspace_path: str | Path | None = None,
) -> None:
    if event not in PACKAGE_EVENT_TYPES:
        return

    try:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            "package": package,
            "profile": profile,
            "scratchpad": scratchpad,
            "source": source,
            "status": status,
            "workspace_hash": workspace_hash(workspace_path),
        }
        if error:
            payload["error"] = error[:500]

        path = package_events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        return


def load_package_events(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or package_events_path()
    if not target.is_file():
        return []

    events: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def summarize_package_events(events: list[dict[str, Any]]) -> PackageTelemetrySummary:
    event_counts = Counter()
    profile_counts = Counter()
    package_rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"events": 0, "failures": 0, "workspaces": set()}
    )

    for event in events:
        raw_event_name = event.get("event")
        event_name = str(raw_event_name).strip() if raw_event_name is not None else ""
        if event_name:
            event_counts[event_name] += 1

        raw_package = event.get("package")
        package = str(raw_package).strip() if raw_package is not None else ""
        if package:
            row = package_rows[package]
            row["events"] += 1
            if event_name == "install_failure":
                row["failures"] += 1
            raw_workspace = event.get("workspace_hash")
            workspace = str(raw_workspace).strip() if raw_workspace is not None else ""
            if workspace:
                row["workspaces"].add(workspace)

        raw_profile = event.get("profile")
        profile = str(raw_profile).strip() if raw_profile is not None else ""
        if profile:
            profile_counts[profile] += 1

    sorted_packages = sorted(
        (
            PackageSummaryRow(
                name=name,
                events=data["events"],
                failures=data["failures"],
                workspaces=len(data["workspaces"]),
            )
            for name, data in package_rows.items()
        ),
        key=lambda row: (-row.events, -row.failures, row.name),
    )

    return PackageTelemetrySummary(
        total_events=len(events),
        event_counts=sorted(event_counts.items(), key=lambda item: (-item[1], item[0])),
        package_rows=sorted_packages,
        profile_counts=sorted(profile_counts.items(), key=lambda item: (-item[1], item[0])),
    )


def _package_name(spec: str) -> str:
    match = _PACKAGE_NAME_RE.match(spec.strip())
    return match.group(0).lower() if match else spec.strip().lower()


def _load_manifest_blob(profile_name: str) -> dict[str, Any]:
    path = runtime_manifest_path(profile_name)
    if not path.is_file():
        raise ValueError(f"Unknown runtime profile: {profile_name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid runtime profile manifest: {path}")
    return data


def load_runtime_profile(profile_name: str) -> RuntimeProfile:
    seen: set[str] = set()
    resolved_packages: dict[str, str] = {}
    resolved_post_install: list[tuple[str, ...]] = []
    description = ""

    def visit(name: str) -> tuple[str, ...]:
        nonlocal description
        if name in seen:
            raise ValueError(f"Cyclic runtime profile dependency: {name}")
        seen.add(name)
        blob = _load_manifest_blob(name)
        extends = tuple(str(item) for item in blob.get("extends", []))
        for parent in extends:
            visit(parent)

        raw_description = blob.get("description", "")
        if name == profile_name:
            description = str(raw_description)

        for spec in blob.get("packages", []):
            spec_str = str(spec).strip()
            if spec_str:
                resolved_packages[_package_name(spec_str)] = spec_str

        for command in blob.get("post_install", []):
            if isinstance(command, list) and command:
                resolved_post_install.append(tuple(str(part) for part in command))
        return extends

    extends = visit(profile_name)
    ordered_packages = tuple(
        spec for _, spec in sorted(resolved_packages.items(), key=lambda item: item[0])
    )
    return RuntimeProfile(
        name=profile_name,
        description=description,
        packages=ordered_packages,
        extends=extends,
        post_install=tuple(resolved_post_install),
    )


def list_runtime_profiles() -> list[RuntimeProfile]:
    profiles: list[RuntimeProfile] = []
    for path in sorted(runtime_profile_dir().glob("*.json")):
        profiles.append(load_runtime_profile(path.stem))
    return profiles


def runtime_lock_hash(profile: RuntimeProfile) -> str:
    payload = {
        "name": profile.name,
        "packages": list(profile.packages),
        "post_install": [list(command) for command in profile.post_install],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def runtime_slug(profile: RuntimeProfile) -> str:
    return f"{profile.name}-{runtime_lock_hash(profile)}"


def runtime_dir(profile: RuntimeProfile, anton_version: str = __version__) -> Path:
    return runtime_version_dir(anton_version) / runtime_slug(profile)


def runtime_metadata_path(runtime_path: Path) -> Path:
    return runtime_path / "runtime.json"


def runtime_python_path(runtime_path: Path) -> Path:
    if sys.platform == "win32":
        return runtime_path / "Scripts" / "python.exe"
    return runtime_path / "bin" / "python"


def runtime_site_packages_path(runtime_path: Path) -> Path:
    for candidate in runtime_path.rglob("site-packages"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate site-packages inside {runtime_path}")


def runtime_packages_for_profile(profile_name: str) -> list[str]:
    return [_package_name(spec) for spec in load_runtime_profile(profile_name).packages]


def default_runtime_profile() -> str:
    return "base"


def _find_uv() -> str | None:
    uv = shutil.which("uv")
    if uv:
        return uv
    if sys.platform == "win32":
        candidates = (
            os.path.expanduser("~/.local/bin/uv.exe"),
            os.path.expanduser("~/.cargo/bin/uv.exe"),
        )
    else:
        candidates = (
            os.path.expanduser("~/.local/bin/uv"),
            os.path.expanduser("~/.cargo/bin/uv"),
        )
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _run_command(cmd: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _write_runtime_metadata(profile: RuntimeProfile, target: Path) -> None:
    metadata = {
        "profile": profile.name,
        "anton_version": __version__,
        "lock_hash": runtime_lock_hash(profile),
        "packages": list(profile.packages),
        "post_install": [list(command) for command in profile.post_install],
        "created_at": datetime.now(UTC).isoformat(),
    }
    runtime_metadata_path(target).write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _hydrate_test_runtime(profile: RuntimeProfile, target: Path, site_packages_value: str) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    python_path = runtime_python_path(target)
    python_path.parent.mkdir(parents=True, exist_ok=True)
    if not python_path.exists():
        try:
            python_path.symlink_to(Path(sys.executable))
        except OSError:
            shutil.copy2(sys.executable, python_path)

    site_target = target / "lib" / "anton-test" / "site-packages"
    site_target.parent.mkdir(parents=True, exist_ok=True)
    if site_target.exists() or site_target.is_symlink():
        if site_target.is_symlink() or site_target.is_file():
            site_target.unlink()
        else:
            shutil.rmtree(site_target)
    site_target.mkdir(parents=True, exist_ok=True)
    for raw_path in site_packages_value.split(os.pathsep):
        source_dir = Path(raw_path).expanduser()
        if not source_dir.is_dir():
            continue
        for child in source_dir.iterdir():
            target_child = site_target / child.name
            if target_child.exists() or target_child.is_symlink():
                continue
            try:
                target_child.symlink_to(child, target_is_directory=child.is_dir())
            except OSError:
                if child.is_file():
                    shutil.copy2(child, target_child)

    _write_runtime_metadata(profile, target)
    return target


def ensure_runtime(profile_name: str, *, workspace_path: str | Path | None = None) -> Path:
    profile = load_runtime_profile(profile_name)
    target = runtime_dir(profile)
    metadata_path = runtime_metadata_path(target)
    if metadata_path.is_file():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    log_package_event(
        "profile_hydrate",
        profile=profile.name,
        status="started",
        workspace_path=workspace_path,
    )

    try:
        test_site_packages = os.environ.get("ANTON_RUNTIME_TEST_SITE_PACKAGES")
        if test_site_packages:
            return _hydrate_test_runtime(profile, target, test_site_packages)

        uv = _find_uv()
        if uv is None:
            raise RuntimeError("uv is required to hydrate Anton runtimes.")

        create_result = _run_command(
            [uv, "venv", str(target), "--python", sys.executable, "--seed", "--quiet"],
            timeout=180,
        )
        if create_result.returncode != 0:
            raise RuntimeError(create_result.stderr or create_result.stdout or "uv venv failed")

        python_path = runtime_python_path(target)
        install_result = _run_command(
            [uv, "pip", "install", "--python", str(python_path), *profile.packages],
            timeout=1800,
        )
        if install_result.returncode != 0:
            raise RuntimeError(install_result.stderr or install_result.stdout or "uv pip install failed")

        for command in profile.post_install:
            resolved = [str(python_path) if part == "{python}" else part for part in command]
            post_result = _run_command(resolved, timeout=1800)
            if post_result.returncode != 0:
                raise RuntimeError(post_result.stderr or post_result.stdout or "runtime post-install failed")

        _write_runtime_metadata(profile, target)
        log_package_event(
            "profile_hydrate",
            profile=profile.name,
            status="success",
            workspace_path=workspace_path,
        )
        return target
    except Exception as exc:
        log_package_event(
            "profile_hydrate",
            profile=profile.name,
            status="failed",
            error=str(exc),
            workspace_path=workspace_path,
        )
        shutil.rmtree(target, ignore_errors=True)
        raise


def list_installed_runtimes() -> list[InstalledRuntime]:
    home = runtime_home()
    if not home.is_dir():
        return []

    installed: list[InstalledRuntime] = []
    for version_dir in sorted(home.iterdir()):
        if not version_dir.is_dir():
            continue
        for runtime_path in sorted(version_dir.iterdir()):
            if not runtime_path.is_dir():
                continue
            metadata_path = runtime_metadata_path(runtime_path)
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            installed.append(
                InstalledRuntime(
                    profile=str(metadata.get("profile", "")),
                    anton_version=str(metadata.get("anton_version", version_dir.name)),
                    lock_hash=str(metadata.get("lock_hash", runtime_path.name.rsplit("-", 1)[-1])),
                    path=runtime_path,
                    package_count=len(metadata.get("packages", [])),
                    hydrated=True,
                )
            )
    return installed


def garbage_collect_runtimes(*, keep_versions: set[str] | None = None) -> list[Path]:
    keep = keep_versions or {__version__}
    home = runtime_home()
    if not home.is_dir():
        return []

    removed: list[Path] = []
    for version_dir in home.iterdir():
        if not version_dir.is_dir():
            continue
        if version_dir.name in keep:
            continue
        shutil.rmtree(version_dir, ignore_errors=True)
        removed.append(version_dir)
    return removed
