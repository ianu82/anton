from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from anton.config.settings import AntonSettings
from anton.service.runtime import RuntimeManager
from anton.service.store import ServiceStore


class SessionCreateRequest(BaseModel):
    session_id: str | None = Field(default=None, description="Optional session id")
    workspace_path: str | None = Field(default=None, description="Optional workspace path")
    metadata: dict[str, Any] | None = Field(default=None)


class TurnRequest(BaseModel):
    message: str


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    note: str | None = None


def create_app(settings: AntonSettings | None = None) -> FastAPI:
    resolved_settings = settings or AntonSettings()
    resolved_settings.resolve_workspace(str(Path.cwd()))

    service_root = Path(resolved_settings.workspace_path) / ".anton" / "service"
    store = ServiceStore(
        service_root / "service.db",
        audit_log_path=service_root / "audit.log.jsonl",
    )
    runtime = RuntimeManager(resolved_settings, store)

    app = FastAPI(title="Anton MVP Service", version="0.1.0")
    app.state.settings = resolved_settings
    app.state.store = store
    app.state.runtime = runtime

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/sessions")
    async def create_session(req: SessionCreateRequest) -> dict[str, Any]:
        try:
            return await runtime.create_session(
                session_id=req.session_id,
                workspace_path=req.workspace_path,
                metadata=req.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/turn")
    async def run_turn(session_id: str, req: TurnRequest) -> dict[str, Any]:
        try:
            return await runtime.run_turn(session_id=session_id, message=req.message)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/events")
    async def list_events(session_id: str, since_id: int = 0, limit: int = 500) -> dict[str, Any]:
        events = store.list_events(session_id=session_id, since_id=since_id, limit=limit)
        return {
            "session_id": session_id,
            "events": events,
            "next_since_id": (events[-1]["id"] if events else since_id),
        }

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id '{run_id}'.")
        return run

    @app.get("/runs/{run_id}/artifacts")
    async def list_artifacts(run_id: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "artifacts": store.list_artifacts(run_id=run_id),
        }

    @app.get("/runs/{run_id}/trace")
    async def get_run_trace(run_id: str) -> dict[str, Any]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id '{run_id}'.")
        return {
            "run": run,
            "events": store.list_run_events(run_id=run_id),
            "artifacts": store.list_artifacts(run_id=run_id),
        }

    @app.get("/approvals")
    async def list_approvals(session_id: str | None = None, status: str = "pending", limit: int = 200) -> dict[str, Any]:
        return {
            "approvals": store.list_approvals(session_id=session_id, status=status, limit=limit)
        }

    @app.post("/approvals/{approval_id}/decision")
    async def decide_approval(approval_id: str, req: ApprovalDecisionRequest) -> dict[str, Any]:
        result = store.set_approval_decision(approval_id=approval_id, approved=req.approved, note=req.note)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Unknown approval_id '{approval_id}'.")
        return result

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return store.metrics_summary()

    return app
