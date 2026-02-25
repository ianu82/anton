from __future__ import annotations

from pathlib import Path
import time
import uuid

from fastapi.testclient import TestClient

from anton.config.settings import AntonSettings
from anton.llm.provider import LLMResponse, StreamComplete, StreamTextDelta, Usage
from anton.service.app import create_app


class _TextOnlyLLM:
    coding_model = "dummy-coder"

    async def plan_stream(self, **kwargs):
        response = LLMResponse(
            content="scheduled result",
            tool_calls=[],
            usage=Usage(input_tokens=2, output_tokens=3),
            stop_reason="end_turn",
        )
        yield StreamTextDelta(text="scheduled result")
        yield StreamComplete(response=response)

    async def plan(self, **kwargs):
        return LLMResponse(
            content="scheduled result",
            tool_calls=[],
            usage=Usage(input_tokens=2, output_tokens=3),
            stop_reason="end_turn",
        )


def _build_settings(workspace: Path) -> AntonSettings:
    settings = AntonSettings()
    settings.resolve_workspace(str(workspace))
    return settings


def test_schedule_create_list_pause_resume(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: _TextOnlyLLM())
    app = create_app(_build_settings(tmp_path))
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={"workspace_path": str(tmp_path / "sched-session")}).json()["session_id"]
        skill_name = f"schedule_skill_{uuid.uuid4().hex[:8]}"
        skill = client.post(
            "/skills",
            json={
                "name": skill_name,
                "description": "Scheduled prompt",
                "prompt_template": "Summarize {metric} for {period}.",
            },
        ).json()

        created = client.post(
            "/scheduled-runs",
            json={
                "name": "daily-kpi",
                "session_id": session_id,
                "skill_id": skill["id"],
                "params": {"metric": "churn", "period": "Q1"},
                "interval_seconds": 3600,
                "start_in_seconds": 10,
                "active": True,
            },
        )
        assert created.status_code == 200
        schedule = created.json()
        assert schedule["status"] == "active"
        assert schedule["name"] == "daily-kpi"

        listed = client.get("/scheduled-runs", params={"status": "active"}).json()["scheduled_runs"]
        assert any(item["id"] == schedule["id"] for item in listed)

        paused = client.post(f"/scheduled-runs/{schedule['id']}/pause")
        assert paused.status_code == 200
        assert paused.json()["status"] == "paused"

        resumed = client.post(f"/scheduled-runs/{schedule['id']}/resume")
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "active"


def test_schedule_trigger_executes_run_and_updates_state(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: _TextOnlyLLM())
    app = create_app(_build_settings(tmp_path))
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={"workspace_path": str(tmp_path / "sched-trigger")}).json()["session_id"]
        skill = client.post(
            "/skills",
            json={
                "name": f"trigger_skill_{uuid.uuid4().hex[:8]}",
                "description": "Trigger prompt",
                "prompt_template": "Show {metric} by {dimension}.",
            },
        ).json()
        schedule = client.post(
            "/scheduled-runs",
            json={
                "session_id": session_id,
                "skill_id": skill["id"],
                "params": {"metric": "churn", "dimension": "cohort"},
                "interval_seconds": 120,
            },
        ).json()

        before_trigger = time.time()
        triggered = client.post(
            f"/scheduled-runs/{schedule['id']}/trigger",
            json={"wait_for_completion": True},
        )
        assert triggered.status_code == 200
        body = triggered.json()
        assert body["status"] == "completed"
        assert body["rendered_prompt"] == "Show churn by cohort."
        assert body["schedule"]["last_run_id"] == body["run_id"]
        assert body["schedule"]["next_run_at"] > before_trigger

        run = client.get(f"/runs/{body['run_id']}").json()
        assert run["prompt"] == "Show churn by cohort."


def test_schedule_create_requires_valid_skill_params(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: _TextOnlyLLM())
    app = create_app(_build_settings(tmp_path))
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={"workspace_path": str(tmp_path / "sched-bad")}).json()["session_id"]
        skill = client.post(
            "/skills",
            json={
                "name": f"bad_schedule_skill_{uuid.uuid4().hex[:8]}",
                "description": "Bad schedule prompt",
                "prompt_template": "Show {metric} by {dimension}.",
            },
        ).json()

        created = client.post(
            "/scheduled-runs",
            json={
                "session_id": session_id,
                "skill_id": skill["id"],
                "params": {"metric": "churn"},
                "interval_seconds": 120,
            },
        )
        assert created.status_code == 400
        assert "Missing required skill params" in created.json()["detail"]
