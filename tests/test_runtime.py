from __future__ import annotations

import json

from anton.runtime import (
    load_package_events,
    log_package_event,
    package_events_path,
    summarize_package_events,
    telemetry_dir,
)


class TestPackageTelemetry:
    def test_log_and_load_events(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTON_TELEMETRY_DIR", str(tmp_path / "telemetry"))

        log_package_event(
            "explicit_install",
            package="cowsay",
            scratchpad="main",
            source="install_action",
            status="started",
            workspace_path=tmp_path,
        )
        log_package_event(
            "install_failure",
            package="bogus",
            scratchpad="main",
            source="module_not_found",
            status="exit_1",
            error="No matching distribution found",
            workspace_path=tmp_path,
        )

        path = package_events_path()
        assert path == telemetry_dir() / "package_events.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert rows[0]["event"] == "explicit_install"
        assert rows[1]["event"] == "install_failure"

        loaded = load_package_events(path)
        assert loaded == rows

    def test_summarize_events(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTON_TELEMETRY_DIR", str(tmp_path / "telemetry"))

        log_package_event("missing_import", package="bs4", workspace_path=tmp_path)
        log_package_event("legacy_auto_install", package="bs4", workspace_path=tmp_path)
        log_package_event("profile_hydrate", profile="base", workspace_path=tmp_path)
        log_package_event("install_failure", package="bs4", workspace_path=tmp_path)

        summary = summarize_package_events(load_package_events())
        assert summary.total_events == 4
        assert ("missing_import", 1) in summary.event_counts
        assert ("base", 1) in summary.profile_counts
        assert summary.package_rows[0].name == "bs4"
        assert summary.package_rows[0].events == 3
        assert summary.package_rows[0].failures == 1
