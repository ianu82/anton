from __future__ import annotations

from anton.cli import runtime_gc, runtime_install, runtime_list, runtime_stats
from anton.runtime import load_runtime_profile, log_package_event, runtime_dir


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


class TestRuntimeManagementCommands:
    def test_runtime_list_prints_profiles(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("ANTON_RUNTIME_HOME", str(tmp_path / "runtimes"))

        runtime_list()
        out = capsys.readouterr().out

        assert "Anton Runtime Profiles" in out
        assert "base" in out
        assert "browser" in out
        assert "ml" in out

    def test_runtime_install_prints_target(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("ANTON_RUNTIME_HOME", str(tmp_path / "runtimes"))
        target = runtime_dir(load_runtime_profile("base"))
        monkeypatch.setattr("anton.cli.ensure_runtime", None, raising=False)
        monkeypatch.setattr("anton.runtime.ensure_runtime", lambda profile: target)

        runtime_install("base")
        out = capsys.readouterr().out

        assert "Hydrated runtime profile base" in out
        normalized = out.replace("\n", "")
        assert "runtimes/0.2.7/base-" in normalized

    def test_runtime_gc_reports_removed_versions(self, monkeypatch, capsys):
        monkeypatch.setattr("anton.runtime.garbage_collect_runtimes", lambda: ["/tmp/r1", "/tmp/r2"])

        runtime_gc()
        out = capsys.readouterr().out

        assert "Removed 2 runtime version(s)." in out
        assert "/tmp/r1" in out
        assert "/tmp/r2" in out
