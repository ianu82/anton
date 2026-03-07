from __future__ import annotations

import os
import site
import sysconfig

from anton.runtime import (
    ensure_runtime,
    list_installed_runtimes,
    load_runtime_profile,
    log_package_event,
    runtime_metadata_path,
    runtime_site_packages_path,
    summarize_package_events,
)


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


class TestManagedRuntimeProfiles:
    def test_profile_inheritance(self):
        ml = load_runtime_profile("ml")

        assert ml.name == "ml"
        assert "base" in ml.extends
        assert any(spec.startswith("scikit-learn==") for spec in ml.packages)
        assert any(spec.startswith("pandas==") for spec in ml.packages)

    def test_ensure_runtime_hydrates_test_runtime(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTON_RUNTIME_HOME", str(tmp_path / "runtimes"))
        site_paths = [sysconfig.get_paths()["purelib"], site.getusersitepackages()]
        monkeypatch.setenv("ANTON_RUNTIME_TEST_SITE_PACKAGES", os.pathsep.join(site_paths))

        target = ensure_runtime("base", workspace_path=tmp_path)

        assert runtime_metadata_path(target).is_file()
        assert runtime_site_packages_path(target).is_dir()

        installed = list_installed_runtimes()
        assert len(installed) == 1
        assert installed[0].profile == "base"
