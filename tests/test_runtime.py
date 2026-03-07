from __future__ import annotations

from anton.runtime import log_package_event, summarize_package_events


class TestPackageTelemetry:
    def test_log_and_summarize_events(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTON_TELEMETRY_DIR", str(tmp_path / "telemetry"))

        log_package_event("explicit_install", package="cowsay", workspace_path=tmp_path)
        log_package_event("install_failure", package="bogus", workspace_path=tmp_path)
        log_package_event("missing_import", package="pandas", profile="base", workspace_path=tmp_path)

        from anton.runtime import load_package_events

        events = load_package_events()
        summary = summarize_package_events(events)

        assert summary.total_events == 3
        assert ("explicit_install", 1) in summary.event_counts
        assert ("install_failure", 1) in summary.event_counts
        assert ("missing_import", 1) in summary.event_counts
        assert ("base", 1) in summary.profile_counts
        assert any(row.name == "cowsay" for row in summary.package_rows)
        assert any(row.name == "bogus" and row.failures == 1 for row in summary.package_rows)
