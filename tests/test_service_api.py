from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from anton.config.settings import AntonSettings
from anton.llm.provider import LLMResponse, StreamComplete, StreamTextDelta, ToolCall, Usage
from anton.service.app import create_app


class TextOnlyLLM:
    coding_model = "dummy-coder"

    async def plan_stream(self, **kwargs):
        response = LLMResponse(
            content="done",
            tool_calls=[],
            usage=Usage(input_tokens=3, output_tokens=5),
            stop_reason="end_turn",
        )
        yield StreamTextDelta(text="done")
        yield StreamComplete(response=response)

    async def plan(self, **kwargs):
        return LLMResponse(
            content="done",
            tool_calls=[],
            usage=Usage(input_tokens=3, output_tokens=5),
            stop_reason="end_turn",
        )


class ApprovalLLM:
    coding_model = "dummy-coder"

    def __init__(self) -> None:
        self._stream_calls = 0

    async def plan_stream(self, **kwargs):
        self._stream_calls += 1
        if self._stream_calls == 1:
            response = LLMResponse(
                content="attempting write",
                tool_calls=[
                    ToolCall(
                        id="tc-write-1",
                        name="connector",
                        input={
                            "action": "write",
                            "connector_id": "warehouse",
                            "query": "DELETE FROM users",
                        },
                    )
                ],
                usage=Usage(input_tokens=7, output_tokens=9),
                stop_reason="tool_use",
            )
            yield StreamTextDelta(text="attempting write")
            yield StreamComplete(response=response)
            return

        response = LLMResponse(
            content="waiting for approval",
            tool_calls=[],
            usage=Usage(input_tokens=2, output_tokens=3),
            stop_reason="end_turn",
        )
        yield StreamTextDelta(text="waiting for approval")
        yield StreamComplete(response=response)

    async def plan(self, **kwargs):
        return LLMResponse(
            content="waiting for approval",
            tool_calls=[],
            usage=Usage(input_tokens=2, output_tokens=3),
            stop_reason="end_turn",
        )


class SlowLLM:
    coding_model = "dummy-coder"

    async def plan_stream(self, **kwargs):
        await asyncio.sleep(0.2)
        response = LLMResponse(
            content="slow result",
            tool_calls=[],
            usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
        )
        yield StreamTextDelta(text="slow result")
        yield StreamComplete(response=response)

    async def plan(self, **kwargs):
        await asyncio.sleep(0.2)
        return LLMResponse(
            content="slow result",
            tool_calls=[],
            usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
        )


def _build_settings(workspace: Path) -> AntonSettings:
    settings = AntonSettings()
    settings.resolve_workspace(str(workspace))
    return settings


def test_service_session_turn_events_metrics(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: TextOnlyLLM())
    app = create_app(_build_settings(tmp_path))
    client = TestClient(app)

    create_resp = client.post("/sessions", json={"workspace_path": str(tmp_path / "ws1")})
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    turn_resp = client.post(f"/sessions/{session_id}/turn", json={"message": "hello"})
    assert turn_resp.status_code == 200
    body = turn_resp.json()
    assert body["status"] == "completed"
    assert body["reply"] == "done"

    run_id = body["run_id"]

    events_resp = client.get(f"/sessions/{session_id}/events")
    assert events_resp.status_code == 200
    assert len(events_resp.json()["events"]) > 0

    run_resp = client.get(f"/runs/{run_id}")
    assert run_resp.status_code == 200
    assert run_resp.json()["status"] == "completed"

    trace_resp = client.get(f"/runs/{run_id}/trace")
    assert trace_resp.status_code == 200
    trace = trace_resp.json()
    assert trace["run"]["id"] == run_id
    assert isinstance(trace["events"], list)

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert metrics["run_count"] >= 1
    assert metrics["success_rate"] >= 0.0


def test_service_approval_flow(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: ApprovalLLM())
    app = create_app(_build_settings(tmp_path))
    client = TestClient(app)

    create_resp = client.post("/sessions", json={"workspace_path": str(tmp_path / "ws2")})
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    turn_resp = client.post(f"/sessions/{session_id}/turn", json={"message": "delete stale users"})
    assert turn_resp.status_code == 200
    body = turn_resp.json()
    assert body["status"] == "approval_required"
    assert len(body["pending_approval_ids"]) == 1

    pending = client.get("/approvals").json()["approvals"]
    assert len(pending) >= 1
    approval_id = pending[0]["id"]

    decision_resp = client.post(
        f"/approvals/{approval_id}/decision",
        json={"approved": True, "note": "approved in test"},
    )
    assert decision_resp.status_code == 200
    assert decision_resp.json()["status"] == "approved"

    approved = client.get("/approvals", params={"status": "approved"}).json()["approvals"]
    assert any(item["id"] == approval_id for item in approved)


def test_session_event_isolation(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: TextOnlyLLM())
    app = create_app(_build_settings(tmp_path))
    client = TestClient(app)

    s1 = client.post("/sessions", json={"workspace_path": str(tmp_path / "iso1")}).json()["session_id"]
    s2 = client.post("/sessions", json={"workspace_path": str(tmp_path / "iso2")}).json()["session_id"]

    r1 = client.post(f"/sessions/{s1}/turn", json={"message": "first"}).json()
    r2 = client.post(f"/sessions/{s2}/turn", json={"message": "second"}).json()

    events_1 = client.get(f"/sessions/{s1}/events").json()["events"]
    events_2 = client.get(f"/sessions/{s2}/events").json()["events"]

    run_ids_1 = {event["run_id"] for event in events_1}
    run_ids_2 = {event["run_id"] for event in events_2}

    assert r1["run_id"] in run_ids_1
    assert r2["run_id"] in run_ids_2
    assert r2["run_id"] not in run_ids_1
    assert r1["run_id"] not in run_ids_2


def test_idempotency_key_reuses_run(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: TextOnlyLLM())
    app = create_app(_build_settings(tmp_path))
    client = TestClient(app)

    session_id = client.post("/sessions", json={"workspace_path": str(tmp_path / "idem")}).json()["session_id"]
    req = {
        "message": "hello idempotent",
        "idempotency_key": "key-123",
    }

    first = client.post(f"/sessions/{session_id}/turn", json=req).json()
    second = client.post(f"/sessions/{session_id}/turn", json=req).json()

    assert first["run_id"] == second["run_id"]
    assert second["reused"] is True
    assert second["status"] == "completed"


def test_queue_mode_and_cancel(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: SlowLLM())
    settings = _build_settings(tmp_path)
    settings.service_worker_mode = "queue"
    settings.service_queue_worker_count = 1

    app = create_app(settings)
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={"workspace_path": str(tmp_path / "queue-cancel")}).json()["session_id"]

        queued = client.post(
            f"/sessions/{session_id}/turn",
            json={
                "message": "slow work",
                "wait_for_completion": False,
                "idempotency_key": "queue-cancel-1",
            },
        ).json()
        run_id = queued["run_id"]
        assert queued["status"] == "queued"

        cancel = client.post(f"/runs/{run_id}/cancel").json()
        assert cancel["cancel_requested"] is True

        deadline = time.time() + 5
        status = "running"
        while time.time() < deadline:
            run = client.get(f"/runs/{run_id}").json()
            status = run["status"]
            if status == "cancelled":
                break
            time.sleep(0.1)

        assert status == "cancelled"


def test_list_session_runs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: TextOnlyLLM())
    app = create_app(_build_settings(tmp_path))
    client = TestClient(app)

    session_id = client.post("/sessions", json={"workspace_path": str(tmp_path / "runs-list")}).json()["session_id"]
    client.post(f"/sessions/{session_id}/turn", json={"message": "one"})
    client.post(f"/sessions/{session_id}/turn", json={"message": "two"})

    listed = client.get(f"/sessions/{session_id}/runs").json()["runs"]
    assert len(listed) >= 2
    assert listed[0]["status"] in {"completed", "running", "queued", "approval_required", "cancelled", "failed"}
