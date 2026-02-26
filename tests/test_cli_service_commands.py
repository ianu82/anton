from __future__ import annotations

from typer.testing import CliRunner

import anton.cli as cli


def _disable_boot_checks(monkeypatch):
    monkeypatch.setattr(cli, "_ensure_dependencies", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("anton.updater.check_and_update", lambda *_args, **_kwargs: None)


def test_skills_list_cli(monkeypatch):
    _disable_boot_checks(monkeypatch)

    def fake_service_request(**kwargs):
        assert kwargs["method"] == "GET"
        assert kwargs["path"] == "/skills"
        return {
            "skills": [
                {
                    "id": "skill-1",
                    "name": "cohort_churn",
                    "latest_version": 2,
                    "description": "Analyze churn",
                }
            ]
        }

    monkeypatch.setattr(cli, "_service_request", fake_service_request)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["skills", "list", "--service-url", "http://127.0.0.1:8000"])
    assert result.exit_code == 0
    assert "cohort_churn" in result.output
    assert "skill-1" in result.output


def test_skills_run_cli(monkeypatch):
    _disable_boot_checks(monkeypatch)

    captured: dict = {}

    def fake_service_request(**kwargs):
        captured.update(kwargs)
        return {
            "run_id": "run-1",
            "status": "completed",
            "skill_version": 1,
            "reply": "done",
        }

    monkeypatch.setattr(cli, "_service_request", fake_service_request)
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "skills",
            "run",
            "skill-1",
            "--session-id",
            "session-1",
            "--params",
            '{"metric":"churn"}',
        ],
    )
    assert result.exit_code == 0
    assert captured["method"] == "POST"
    assert captured["path"] == "/skills/skill-1/run"
    assert captured["payload"]["params"] == {"metric": "churn"}
    assert captured["payload"]["session_id"] == "session-1"
    assert "run-1" in result.output


def test_schedules_create_cli(monkeypatch):
    _disable_boot_checks(monkeypatch)

    captured: dict = {}

    def fake_service_request(**kwargs):
        captured.update(kwargs)
        return {"id": "schedule-1", "status": "active"}

    monkeypatch.setattr(cli, "_service_request", fake_service_request)
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "schedules",
            "create",
            "--session-id",
            "session-1",
            "--skill-id",
            "skill-1",
            "--params",
            '{"metric":"churn"}',
            "--interval-seconds",
            "60",
        ],
    )
    assert result.exit_code == 0
    assert captured["method"] == "POST"
    assert captured["path"] == "/scheduled-runs"
    assert captured["payload"]["session_id"] == "session-1"
    assert captured["payload"]["skill_id"] == "skill-1"
    assert captured["payload"]["interval_seconds"] == 60
    assert "schedule-1" in result.output


def test_cli_invalid_json_params(monkeypatch):
    _disable_boot_checks(monkeypatch)
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "skills",
            "run",
            "skill-1",
            "--session-id",
            "session-1",
            "--params",
            "{not-json",
        ],
    )
    assert result.exit_code == 1
    assert "Invalid JSON for params" in result.output
