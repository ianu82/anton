from __future__ import annotations

import json

from anton.runtime import (
    RuntimeProfile,
    ensure_runtime,
    garbage_collect_runtimes,
    list_installed_runtimes,
    list_runtime_profiles,
    load_package_events,
    load_runtime_profile,
    log_package_event,
    package_events_path,
    runtime_dir,
    runtime_lock_hash,
    runtime_metadata_path,
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


class TestManagedRuntimes:
    def test_load_runtime_profile_resolves_extends(self):
        base = load_runtime_profile("base")
        ml = load_runtime_profile("ml")
        browser = load_runtime_profile("browser")

        assert isinstance(base, RuntimeProfile)
        assert "pandas==3.0.1" in base.packages
        assert "scikit-learn==1.8.0" in ml.packages
        assert "pandas==3.0.1" in ml.packages
        assert browser.post_install == (("{python}", "-m", "playwright", "install", "chromium"),)

    def test_runtime_lock_hash_changes_with_profile_content(self):
        base = load_runtime_profile("base")
        ml = load_runtime_profile("ml")
        assert runtime_lock_hash(base) != runtime_lock_hash(ml)

    def test_ensure_runtime_hydrates_once(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTON_RUNTIME_HOME", str(tmp_path / "runtimes"))

        commands = []

        def fake_run_command(cmd, *, timeout=600):
            commands.append(cmd)

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        monkeypatch.setattr("anton.runtime._run_command", fake_run_command)
        monkeypatch.setattr("anton.runtime._find_uv", lambda: "/tmp/fake-uv")

        target = ensure_runtime("base", workspace_path=tmp_path)
        metadata = runtime_metadata_path(target)
        assert metadata.is_file()
        assert len(commands) == 2

        second = ensure_runtime("base", workspace_path=tmp_path)
        assert second == target
        assert len(commands) == 2

    def test_list_installed_runtimes(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTON_RUNTIME_HOME", str(tmp_path / "runtimes"))
        profile = load_runtime_profile("base")
        target = runtime_dir(profile)
        target.mkdir(parents=True, exist_ok=True)
        runtime_metadata_path(target).write_text(
            '{"profile":"base","anton_version":"0.2.7","lock_hash":"abc123","packages":["pandas==3.0.1"]}',
            encoding="utf-8",
        )

        installed = list_installed_runtimes()
        assert len(installed) == 1
        assert installed[0].profile == "base"
        assert installed[0].package_count == 1

    def test_garbage_collect_keeps_current_version(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTON_RUNTIME_HOME", str(tmp_path / "runtimes"))
        current = tmp_path / "runtimes" / "0.2.7"
        old = tmp_path / "runtimes" / "0.2.6"
        (current / "base-keep").mkdir(parents=True)
        (old / "base-old").mkdir(parents=True)

        removed = garbage_collect_runtimes()
        assert removed == [old]
        assert current.exists()
        assert not old.exists()

    def test_list_runtime_profiles(self):
        names = [profile.name for profile in list_runtime_profiles()]
        assert names == ["base", "browser", "ml"]
