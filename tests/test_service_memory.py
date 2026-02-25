from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from anton.config.settings import AntonSettings
from anton.llm.provider import LLMResponse, StreamComplete, StreamTextDelta, Usage
from anton.memory.learnings import LearningStore
from anton.memory.store import SessionStore
from anton.service.app import create_app
from anton.service.memory import MemoryContextBuilder


class _TextOnlyLLM:
    coding_model = "dummy-coder"

    async def plan_stream(self, **kwargs):
        response = LLMResponse(
            content="ok",
            tool_calls=[],
            usage=Usage(input_tokens=2, output_tokens=3),
            stop_reason="end_turn",
        )
        yield StreamTextDelta(text="ok")
        yield StreamComplete(response=response)

    async def plan(self, **kwargs):
        return LLMResponse(
            content="ok",
            tool_calls=[],
            usage=Usage(input_tokens=2, output_tokens=3),
            stop_reason="end_turn",
        )


def test_memory_context_builder_collects_summaries_and_learnings(tmp_path: Path):
    memory_dir = tmp_path / ".anton"
    memory_dir.mkdir(parents=True, exist_ok=True)

    session_store = SessionStore(memory_dir)
    learning_store = LearningStore(memory_dir)

    async def seed() -> None:
        session_id = await session_store.start_session("Investigate churn")
        await session_store.complete_session(session_id, "We found churn concentrated in April cohort.")
        await learning_store.record(
            "cohort_analysis",
            "Always compare cohort retention curves before aggregate churn.",
            "Use retention curves before aggregate metrics",
        )

    asyncio.run(seed())

    builder = MemoryContextBuilder(memory_dir=memory_dir, enabled=True, max_items=3, max_chars=2000)
    context, provenance = builder.build("show churn by cohort")

    assert "retrieved_memory_context" in context
    assert "cohort" in context.lower()
    assert len(provenance) >= 1


def test_service_emits_memory_retrieval_event(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anton.service.runtime.LLMClient.from_settings", lambda _: _TextOnlyLLM())

    settings = AntonSettings()
    settings.resolve_workspace(str(tmp_path))

    app = create_app(settings)
    with TestClient(app) as client:
        workspace = tmp_path / "mem-ws"
        session_resp = client.post("/sessions", json={"workspace_path": str(workspace)})
        session_id = session_resp.json()["session_id"]

        learning_store = LearningStore(Path(settings.memory_dir))

        async def seed_learning() -> None:
            await learning_store.record(
                "churn_patterns",
                "April promo cohorts have higher churn unless onboarding email is sent.",
                "Promo cohorts need onboarding follow-up",
            )

        asyncio.run(seed_learning())

        turn = client.post(f"/sessions/{session_id}/turn", json={"message": "analyze churn cohorts"})
        assert turn.status_code == 200

        events = client.get(f"/sessions/{session_id}/events").json()["events"]
        memory_events = [event for event in events if event["event_type"] == "memory_retrieval"]
        assert len(memory_events) >= 1
        assert memory_events[-1]["payload"]["retrieved_count"] >= 1
