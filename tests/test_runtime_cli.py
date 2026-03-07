from __future__ import annotations

from anton.cli import runtime_stats
from anton.runtime import log_package_event


class TestRuntimeStatsCommand:
    def test_runtime_stats_prints_summary(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("ANTON_TELEMETRY_DIR", str(tmp_path / "telemetry"))

        log_package_event("explicit_install", package="cowsay", workspace_path=tmp_path)
        log_package_event("install_failure", package="bogus", workspace_path=tmp_path)

        runtime_stats()
        out = capsys.readouterr().out

        assert "Package telemetry:" in out
        assert "Total events:" in out
        assert "explicit_install" in out
        assert "cowsay" in out
        assert "bogus" in out

    def test_runtime_stats_handles_empty_log(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("ANTON_TELEMETRY_DIR", str(tmp_path / "telemetry"))

        runtime_stats()
        out = capsys.readouterr().out

        assert "No package telemetry recorded yet" in out
