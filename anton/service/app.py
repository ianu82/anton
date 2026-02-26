from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import sqlite3
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from anton.config.settings import AntonSettings
from anton.service.runtime import RuntimeManager
from anton.service.skills import extract_template_fields, render_prompt_template
from anton.service.store import ServiceStore


class SessionCreateRequest(BaseModel):
    session_id: str | None = Field(default=None, description="Optional session id")
    workspace_path: str | None = Field(default=None, description="Optional workspace path")
    metadata: dict[str, Any] | None = Field(default=None)


class TurnRequest(BaseModel):
    message: str
    idempotency_key: str | None = None
    wait_for_completion: bool = True
    wait_timeout_seconds: float = 300.0


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    note: str | None = None


class SkillCreateRequest(BaseModel):
    name: str
    description: str = ""
    prompt_template: str
    metadata: dict[str, Any] | None = None


class SkillVersionRequest(BaseModel):
    prompt_template: str


class SkillRunRequest(BaseModel):
    session_id: str
    params: dict[str, str] | None = None
    version: int | None = None
    idempotency_key: str | None = None
    wait_for_completion: bool = True
    wait_timeout_seconds: float = 300.0


class ScheduleCreateRequest(BaseModel):
    name: str | None = None
    session_id: str
    skill_id: str
    skill_version: int | None = None
    params: dict[str, str] | None = None
    interval_seconds: int = 3600
    start_in_seconds: int = 0
    active: bool = True


class ScheduleTriggerRequest(BaseModel):
    idempotency_key: str | None = None
    wait_for_completion: bool = False
    wait_timeout_seconds: float = 300.0


def create_app(settings: AntonSettings | None = None) -> FastAPI:
    resolved_settings = settings or AntonSettings()
    resolved_settings.resolve_workspace(str(resolved_settings.workspace_path))

    service_root = Path(resolved_settings.workspace_path) / ".anton" / "service"
    store = ServiceStore(
        service_root / "service.db",
        audit_log_path=service_root / "audit.log.jsonl",
    )
    runtime = RuntimeManager(resolved_settings, store)
    scheduler_enabled = bool(resolved_settings.service_scheduler_enabled)
    scheduler_poll_seconds = max(0.1, float(resolved_settings.service_scheduler_poll_seconds))
    scheduler_batch_size = max(1, int(resolved_settings.service_scheduler_batch_size))
    scheduler_task: asyncio.Task | None = None

    async def _run_skill_once(
        *,
        skill_id: str,
        version: int | None,
        session_id: str,
        params: dict[str, str] | None,
        idempotency_key: str | None,
        wait_for_completion: bool,
        wait_timeout_seconds: float,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        template = store.get_skill_template(skill_id=skill_id, version=version)
        if template is None:
            raise HTTPException(status_code=404, detail=f"Unknown skill_id/version combination for '{skill_id}'.")
        try:
            rendered_prompt = render_prompt_template(
                template["prompt_template"],
                required_params=template["required_params"],
                params=params,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            result = await runtime.run_turn(
                session_id=session_id,
                message=rendered_prompt,
                idempotency_key=idempotency_key,
                wait_for_completion=wait_for_completion,
                wait_timeout_seconds=wait_timeout_seconds,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return template, rendered_prompt, result

    async def _scheduler_loop() -> None:
        while True:
            now = time.time()
            due = [
                schedule
                for schedule in store.list_schedules(status="active", limit=scheduler_batch_size)
                if float(schedule["next_run_at"]) <= now
            ]
            due.sort(key=lambda item: float(item["next_run_at"]))

            for schedule in due:
                schedule_id = schedule["id"]
                schedule_run_id = f"schedule-{schedule_id}"
                try:
                    template, _rendered_prompt, result = await _run_skill_once(
                        skill_id=schedule["skill_id"],
                        version=schedule["skill_version"],
                        session_id=schedule["session_id"],
                        params=schedule["params"],
                        idempotency_key=f"schedule:{schedule_id}:{int(now)}",
                        wait_for_completion=False,
                        wait_timeout_seconds=float(resolved_settings.service_default_wait_timeout_seconds),
                    )
                    store.record_schedule_trigger(schedule_id=schedule_id, run_id=result["run_id"])
                    store.append_event(
                        session_id=schedule["session_id"],
                        run_id=result["run_id"],
                        event_type="schedule_triggered",
                        payload={
                            "schedule_id": schedule_id,
                            "skill_id": template["skill_id"],
                            "skill_version": template["version"],
                            "mode": "automatic",
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except HTTPException as exc:
                    store.set_schedule_status(schedule_id=schedule_id, status="paused")
                    store.append_event(
                        session_id=schedule["session_id"],
                        run_id=schedule_run_id,
                        event_type="schedule_trigger_failed",
                        payload={"schedule_id": schedule_id, "detail": str(exc.detail)},
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    store.set_schedule_status(schedule_id=schedule_id, status="paused")
                    store.append_event(
                        session_id=schedule["session_id"],
                        run_id=schedule_run_id,
                        event_type="schedule_trigger_failed",
                        payload={"schedule_id": schedule_id, "detail": str(exc)},
                    )

            await asyncio.sleep(scheduler_poll_seconds)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal scheduler_task
        await runtime.start()
        if scheduler_enabled:
            scheduler_task = asyncio.create_task(_scheduler_loop(), name="anton-scheduler")
        try:
            yield
        finally:
            if scheduler_task is not None:
                scheduler_task.cancel()
                await asyncio.gather(scheduler_task, return_exceptions=True)
                scheduler_task = None
            await runtime.close()

    app = FastAPI(title="Anton MVP Service", version="0.1.0", lifespan=lifespan)
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
            return await runtime.run_turn(
                session_id=session_id,
                message=req.message,
                idempotency_key=req.idempotency_key,
                wait_for_completion=req.wait_for_completion,
                wait_timeout_seconds=req.wait_timeout_seconds,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/runs")
    async def list_runs(session_id: str, limit: int = 100) -> dict[str, Any]:
        if store.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown session_id '{session_id}'.")
        return {
            "session_id": session_id,
            "runs": store.list_runs(session_id=session_id, limit=limit),
        }

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

    @app.post("/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> dict[str, Any]:
        try:
            return await runtime.cancel_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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

    @app.post("/skills")
    async def create_skill(req: SkillCreateRequest) -> dict[str, Any]:
        name = req.name.strip()
        prompt_template = req.prompt_template.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Skill name cannot be empty.")
        if not prompt_template:
            raise HTTPException(status_code=400, detail="Skill prompt_template cannot be empty.")
        try:
            required_params = extract_template_fields(prompt_template)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            return store.create_skill(
                name=name,
                description=req.description.strip(),
                prompt_template=prompt_template,
                required_params=required_params,
                metadata=req.metadata,
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail=f"Skill name '{name}' already exists.") from exc

    @app.get("/skills")
    async def list_skills(limit: int = 200) -> dict[str, Any]:
        return {"skills": store.list_skills(limit=limit)}

    @app.get("/skills/{skill_id}")
    async def get_skill(skill_id: str) -> dict[str, Any]:
        skill = store.get_skill(skill_id)
        if skill is None:
            raise HTTPException(status_code=404, detail=f"Unknown skill_id '{skill_id}'.")
        return skill

    @app.post("/skills/{skill_id}/versions")
    async def create_skill_version(skill_id: str, req: SkillVersionRequest) -> dict[str, Any]:
        prompt_template = req.prompt_template.strip()
        if not prompt_template:
            raise HTTPException(status_code=400, detail="Skill prompt_template cannot be empty.")
        try:
            required_params = extract_template_fields(prompt_template)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        skill = store.add_skill_version(
            skill_id=skill_id,
            prompt_template=prompt_template,
            required_params=required_params,
        )
        if skill is None:
            raise HTTPException(status_code=404, detail=f"Unknown skill_id '{skill_id}'.")
        return skill

    @app.post("/skills/{skill_id}/run")
    async def run_skill(skill_id: str, req: SkillRunRequest) -> dict[str, Any]:
        template, rendered_prompt, result = await _run_skill_once(
            skill_id=skill_id,
            version=req.version,
            session_id=req.session_id,
            params=req.params,
            idempotency_key=req.idempotency_key,
            wait_for_completion=req.wait_for_completion,
            wait_timeout_seconds=req.wait_timeout_seconds,
        )
        return {
            "skill_id": skill_id,
            "skill_version": template["version"],
            "rendered_prompt": rendered_prompt,
            **result,
        }

    @app.post("/scheduled-runs")
    async def create_schedule(req: ScheduleCreateRequest) -> dict[str, Any]:
        if req.interval_seconds < 1:
            raise HTTPException(status_code=400, detail="interval_seconds must be >= 1.")
        if req.start_in_seconds < 0:
            raise HTTPException(status_code=400, detail="start_in_seconds must be >= 0.")
        if store.get_session(req.session_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown session_id '{req.session_id}'.")
        template = store.get_skill_template(skill_id=req.skill_id, version=req.skill_version)
        if template is None:
            raise HTTPException(status_code=404, detail=f"Unknown skill_id/version combination for '{req.skill_id}'.")
        try:
            render_prompt_template(
                template["prompt_template"],
                required_params=template["required_params"],
                params=req.params,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        schedule_name = req.name.strip() if req.name else f"{template['name']}-every-{req.interval_seconds}s"
        if not schedule_name:
            raise HTTPException(status_code=400, detail="Schedule name cannot be empty.")

        return store.create_schedule(
            name=schedule_name,
            session_id=req.session_id,
            skill_id=req.skill_id,
            skill_version=req.skill_version,
            params=req.params or {},
            interval_seconds=req.interval_seconds,
            start_at=time.time() + req.start_in_seconds,
            status=("active" if req.active else "paused"),
        )

    @app.get("/scheduled-runs")
    async def list_schedules(status: str | None = None, limit: int = 200) -> dict[str, Any]:
        return {"scheduled_runs": store.list_schedules(status=status, limit=limit)}

    @app.get("/scheduled-runs/{schedule_id}")
    async def get_schedule(schedule_id: str) -> dict[str, Any]:
        schedule = store.get_schedule(schedule_id)
        if schedule is None:
            raise HTTPException(status_code=404, detail=f"Unknown schedule_id '{schedule_id}'.")
        return schedule

    @app.post("/scheduled-runs/{schedule_id}/trigger")
    async def trigger_schedule(schedule_id: str, req: ScheduleTriggerRequest) -> dict[str, Any]:
        schedule = store.get_schedule(schedule_id)
        if schedule is None:
            raise HTTPException(status_code=404, detail=f"Unknown schedule_id '{schedule_id}'.")

        template, rendered_prompt, result = await _run_skill_once(
            skill_id=schedule["skill_id"],
            version=schedule["skill_version"],
            session_id=schedule["session_id"],
            params=schedule["params"],
            idempotency_key=req.idempotency_key,
            wait_for_completion=req.wait_for_completion,
            wait_timeout_seconds=req.wait_timeout_seconds,
        )
        updated = store.record_schedule_trigger(schedule_id=schedule_id, run_id=result["run_id"])
        store.append_event(
            session_id=schedule["session_id"],
            run_id=result["run_id"],
            event_type="schedule_triggered",
            payload={
                "schedule_id": schedule_id,
                "skill_id": template["skill_id"],
                "skill_version": template["version"],
                "mode": "manual",
            },
        )
        return {
            "schedule_id": schedule_id,
            "skill_id": template["skill_id"],
            "skill_version": template["version"],
            "rendered_prompt": rendered_prompt,
            "schedule": updated,
            **result,
        }

    @app.post("/scheduled-runs/{schedule_id}/pause")
    async def pause_schedule(schedule_id: str) -> dict[str, Any]:
        schedule = store.set_schedule_status(schedule_id=schedule_id, status="paused")
        if schedule is None:
            raise HTTPException(status_code=404, detail=f"Unknown schedule_id '{schedule_id}'.")
        return schedule

    @app.post("/scheduled-runs/{schedule_id}/resume")
    async def resume_schedule(schedule_id: str) -> dict[str, Any]:
        schedule = store.set_schedule_status(schedule_id=schedule_id, status="active")
        if schedule is None:
            raise HTTPException(status_code=404, detail=f"Unknown schedule_id '{schedule_id}'.")
        return schedule

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return store.metrics_summary()

    return app
