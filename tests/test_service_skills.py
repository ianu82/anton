from __future__ import annotations

from pathlib import Path
import uuid

from fastapi.testclient import TestClient

from anton.config.settings import AntonSettings
from anton.llm.provider import LLMResponse, StreamComplete, StreamTextDelta, Usage
from anton.service.app import create_app


class _TextOnlyLLM:
    coding_model = "dummy-coder"

    async def plan_stream(self, **kwargs):
        response = LLMResponse(
            content="skill result",
            tool_calls=[],
            usage=Usage(input_tokens=2, output_tokens=4),
            stop_reason="end_turn",
        )
        yield StreamTextDelta(text="skill result")
        yield StreamComplete(response=response)

    async def plan(self, **kwargs):
        return LLMResponse(
            content="skill result",
            tool_calls=[],
            usage=Usage(input_tokens=2, output_tokens=4),
            stop_reason="end_turn",
        )


def _build_settings(workspace: Path) -> AntonSettings:
    settings = AntonSettings()
    settings.resolve_workspace(str(workspace))
    return settings


def test_skill_registry_create_list_get(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: _TextOnlyLLM())

    app = create_app(_build_settings(tmp_path))
    with TestClient(app) as client:
        skill_name = f"cohort_churn_{uuid.uuid4().hex[:8]}"
        created = client.post(
            "/skills",
            json={
                "name": skill_name,
                "description": "Cohort churn analysis prompt",
                "prompt_template": "Show churn by {dimension} for {period}.",
                "metadata": {"team": "analytics"},
            },
        )
        assert created.status_code == 200
        skill = created.json()
        assert skill["name"] == skill_name
        assert skill["latest_version"] == 1
        assert skill["latest_required_params"] == ["dimension", "period"]

        listed = client.get("/skills").json()["skills"]
        assert any(item["id"] == skill["id"] for item in listed)

        fetched = client.get(f"/skills/{skill['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["latest_prompt_template"] == "Show churn by {dimension} for {period}."


def test_skill_versioning_and_run(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: _TextOnlyLLM())

    app = create_app(_build_settings(tmp_path))
    with TestClient(app) as client:
        skill_name = f"kpi_rollup_{uuid.uuid4().hex[:8]}"
        session_id = client.post("/sessions", json={"workspace_path": str(tmp_path / "skills-run")}).json()["session_id"]
        created = client.post(
            "/skills",
            json={
                "name": skill_name,
                "description": "KPI rollup skill",
                "prompt_template": "Summarize {metric} for {period}.",
            },
        )
        assert created.status_code == 200
        skill_id = created.json()["id"]

        updated = client.post(
            f"/skills/{skill_id}/versions",
            json={
                "prompt_template": "Summarize {metric} by {dimension} for {period}.",
            },
        )
        assert updated.status_code == 200
        assert updated.json()["latest_version"] == 2

        run = client.post(
            f"/skills/{skill_id}/run",
            json={
                "session_id": session_id,
                "version": 2,
                "params": {
                    "metric": "churn",
                    "dimension": "cohort",
                    "period": "Q1",
                },
            },
        )
        assert run.status_code == 200
        body = run.json()
        assert body["status"] == "completed"
        assert body["skill_version"] == 2
        assert body["rendered_prompt"] == "Summarize churn by cohort for Q1."

        stored_run = client.get(f"/runs/{body['run_id']}")
        assert stored_run.status_code == 200
        assert stored_run.json()["prompt"] == "Summarize churn by cohort for Q1."


def test_skill_run_missing_params_returns_400(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: _TextOnlyLLM())

    app = create_app(_build_settings(tmp_path))
    with TestClient(app) as client:
        skill_name = f"segment_summary_{uuid.uuid4().hex[:8]}"
        session_id = client.post("/sessions", json={"workspace_path": str(tmp_path / "skills-bad-run")}).json()["session_id"]
        created = client.post(
            "/skills",
            json={
                "name": skill_name,
                "description": "Segment summary",
                "prompt_template": "Summarize {segment} performance.",
            },
        )
        assert created.status_code == 200
        skill_id = created.json()["id"]

        run = client.post(
            f"/skills/{skill_id}/run",
            json={
                "session_id": session_id,
                "params": {},
            },
        )
        assert run.status_code == 400
        assert "Missing required skill params" in run.json()["detail"]


def test_skill_rejects_complex_placeholder(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: _TextOnlyLLM())

    app = create_app(_build_settings(tmp_path))
    with TestClient(app) as client:
        skill_name = f"invalid_skill_{uuid.uuid4().hex[:8]}"
        created = client.post(
            "/skills",
            json={
                "name": skill_name,
                "description": "Invalid placeholder",
                "prompt_template": "Summarize {payload.metric}.",
            },
        )
        assert created.status_code == 400
        assert "Unsupported placeholder" in created.json()["detail"]


def test_skill_run_rejects_zero_version(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: _TextOnlyLLM())

    app = create_app(_build_settings(tmp_path))
    with TestClient(app) as client:
        skill = client.post(
            "/skills",
            json={
                "name": f"version_zero_{uuid.uuid4().hex[:8]}",
                "description": "Version check",
                "prompt_template": "Summarize {metric}.",
            },
        ).json()
        session_id = client.post("/sessions", json={"workspace_path": str(tmp_path / "skills-version-zero")}).json()[
            "session_id"
        ]

        run = client.post(
            f"/skills/{skill['id']}/run",
            json={
                "session_id": session_id,
                "version": 0,
                "params": {"metric": "churn"},
            },
        )
        assert run.status_code == 400
        assert "Skill version must be >= 1." in run.json()["detail"]
