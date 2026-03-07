from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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


def telemetry_dir() -> Path:
    override = os.environ.get("ANTON_TELEMETRY_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".anton" / "telemetry"


def package_events_path() -> Path:
    return telemetry_dir() / "package_events.jsonl"


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
