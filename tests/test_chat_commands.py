from __future__ import annotations

from rich.console import Console

import pytest

from anton.chat import (
    _chat_parse_json_object,
    _handle_schedules_slash_command,
    _handle_skills_slash_command,
    _print_slash_help,
    _split_positional_and_options,
)


def test_print_slash_help_includes_skills_and_schedules():
    console = Console(record=True, width=160)
    _print_slash_help(console)
    rendered = console.export_text()
    assert "/skills" in rendered
    assert "/schedules" in rendered


def test_split_positional_and_options_parses_values_and_flags():
    positionals, options = _split_positional_and_options(
        ["alpha", "--limit", "10", "--wait", "beta"],
        value_options={"--limit"},
        flag_options={"--wait"},
    )
    assert positionals == ["alpha", "beta"]
    assert options == {"--limit": "10", "--wait": True}


def test_chat_parse_json_object_rejects_non_object():
    with pytest.raises(ValueError):
        _chat_parse_json_object("[]", "params")


@pytest.mark.asyncio
async def test_handle_skills_run_slash_command_builds_expected_payload(monkeypatch):
    captured: dict = {}

    def fake_service_request(**kwargs):
        captured.update(kwargs)
        return {
            "run_id": "run-1",
            "status": "completed",
            "skill_version": 2,
            "reply": "done",
        }

    monkeypatch.setattr("anton.chat._chat_service_request", fake_service_request)
    console = Console(record=True, width=200)
    handled = await _handle_skills_slash_command(
        console,
        '/skills run skill-1 --session-id session-1 --params \'{"metric":"churn"}\' --version 2 --no-wait',
    )
    assert handled is True
    assert captured["method"] == "POST"
    assert captured["path"] == "/skills/skill-1/run"
    assert captured["payload"]["session_id"] == "session-1"
    assert captured["payload"]["params"] == {"metric": "churn"}
    assert captured["payload"]["version"] == 2
    assert captured["payload"]["wait_for_completion"] is False


@pytest.mark.asyncio
async def test_handle_schedules_create_slash_command_builds_expected_payload(monkeypatch):
    captured: dict = {}

    def fake_service_request(**kwargs):
        captured.update(kwargs)
        return {"id": "schedule-1", "status": "active"}

    monkeypatch.setattr("anton.chat._chat_service_request", fake_service_request)
    console = Console(record=True, width=200)
    handled = await _handle_schedules_slash_command(
        console,
        '/schedules create --session-id session-1 --skill-id skill-1 --params \'{"metric":"churn"}\' '
        '--interval-seconds 60 --start-in-seconds 5 --paused',
    )
    assert handled is True
    assert captured["method"] == "POST"
    assert captured["path"] == "/scheduled-runs"
    assert captured["payload"]["session_id"] == "session-1"
    assert captured["payload"]["skill_id"] == "skill-1"
    assert captured["payload"]["params"] == {"metric": "churn"}
    assert captured["payload"]["interval_seconds"] == 60
    assert captured["payload"]["start_in_seconds"] == 5
    assert captured["payload"]["active"] is False


@pytest.mark.asyncio
async def test_skills_slash_command_shows_service_hint_on_connection_error(monkeypatch):
    def fake_service_request(**_kwargs):
        raise RuntimeError("Could not reach service.")

    monkeypatch.setattr("anton.chat._chat_service_request", fake_service_request)
    console = Console(record=True, width=200)
    handled = await _handle_skills_slash_command(console, "/skills list")
    assert handled is True
    rendered = console.export_text()
    assert "Start service with: anton serve" in rendered
