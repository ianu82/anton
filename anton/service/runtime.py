from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anton.chat import ChatSession
from anton.connectors import ConnectorHub
from anton.context.self_awareness import SelfAwarenessContext
from anton.governance import BudgetConfig, BudgetTracker, PolicyConfig, PolicyEngine, ToolGateResult
from anton.llm.client import LLMClient
from anton.llm.provider import StreamComplete, StreamTextDelta
from anton.service.memory import MemoryContextBuilder
from anton.service.store import ServiceStore
from anton.workspace import Workspace


@dataclass
class SessionRuntime:
    session_id: str
    workspace_path: Path
    workspace: Workspace
    session: ChatSession
    memory_context: MemoryContextBuilder
    lock: asyncio.Lock


@dataclass
class QueuedRunRequest:
    run_id: str
    session_id: str
    message: str
    run: dict[str, Any]


class RunGovernance:
    def __init__(
        self,
        *,
        session_id: str,
        run_id: str,
        store: ServiceStore,
        policy: PolicyEngine,
        budget: BudgetTracker,
    ) -> None:
        self._session_id = session_id
        self._run_id = run_id
        self._store = store
        self._policy = policy
        self._budget = budget
        self.pending_approvals: list[str] = []

    @staticmethod
    def _fingerprint(tool_name: str, tool_input: dict[str, Any]) -> str:
        raw = json.dumps({"name": tool_name, "input": tool_input}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def tool_gate(self, tool_name: str, tool_input: dict[str, Any]) -> ToolGateResult:
        self._budget.record_tool_call()
        budget_violation = self._budget.check()
        if budget_violation is not None:
            return ToolGateResult(allow=False, message=budget_violation)

        decision = self._policy.evaluate(tool_name, tool_input)
        if decision.allow:
            return ToolGateResult(allow=True)

        if decision.requires_approval:
            fingerprint = self._fingerprint(tool_name, tool_input)
            prior = self._store.find_approval_decision(session_id=self._session_id, fingerprint=fingerprint)
            if prior == "approved":
                return ToolGateResult(allow=True)
            if prior == "rejected":
                return ToolGateResult(allow=False, message="Request rejected by reviewer.")

            approval_id = self._store.create_approval(
                session_id=self._session_id,
                run_id=self._run_id,
                tool_name=tool_name,
                tool_input=tool_input,
                fingerprint=fingerprint,
                reason=decision.reason,
            )
            self.pending_approvals.append(approval_id)
            return ToolGateResult(
                allow=False,
                message=decision.reason,
                pending_approval_id=approval_id,
            )

        return ToolGateResult(allow=False, message=decision.reason)

    def usage_hook(self, usage) -> None:
        self._budget.record_usage(usage)
        self._store.append_usage(
            session_id=self._session_id,
            run_id=self._run_id,
            metric="tokens",
            amount=float(usage.input_tokens + usage.output_tokens),
            metadata={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
        )

    def audit_hook(self, event_name: str, payload: dict[str, Any]) -> None:
        self._store.append_event(
            session_id=self._session_id,
            run_id=self._run_id,
            event_type=event_name,
            payload=payload,
        )


class RuntimeManager:
    def __init__(self, settings, store: ServiceStore) -> None:
        self._settings = settings
        self._store = store
        self._runtimes: dict[str, SessionRuntime] = {}
        self._manager_lock = asyncio.Lock()

        self._worker_mode = getattr(settings, "service_worker_mode", "local").strip().lower() or "local"
        if self._worker_mode not in {"local", "queue"}:
            self._worker_mode = "local"

        self._run_queue: asyncio.Queue[QueuedRunRequest] | None = None
        self._queue_workers: list[asyncio.Task] = []
        configured_workers = int(getattr(settings, "service_queue_worker_count", 2))
        self._queue_worker_count = max(1, configured_workers)
        self._workers_started = False

        self._active_run_tasks: dict[str, asyncio.Task] = {}
        self._run_futures: dict[str, asyncio.Future] = {}

    async def start(self) -> None:
        if self._worker_mode == "queue":
            await self._ensure_workers_started()

    async def close(self) -> None:
        for task in list(self._active_run_tasks.values()):
            task.cancel()

        if self._queue_workers:
            for worker in self._queue_workers:
                worker.cancel()
            await asyncio.gather(*self._queue_workers, return_exceptions=True)
            self._queue_workers.clear()

        for future in list(self._run_futures.values()):
            if not future.done():
                future.cancel()
        self._run_futures.clear()

        for runtime in list(self._runtimes.values()):
            await runtime.session.close()

    async def _ensure_workers_started(self) -> None:
        if self._workers_started:
            return
        async with self._manager_lock:
            if self._workers_started:
                return
            self._run_queue = asyncio.Queue()
            for idx in range(self._queue_worker_count):
                task = asyncio.create_task(self._queue_worker_loop(idx), name=f"anton-queue-worker-{idx}")
                self._queue_workers.append(task)
            self._workers_started = True

    async def _queue_worker_loop(self, worker_index: int) -> None:
        assert self._run_queue is not None
        while True:
            request = await self._run_queue.get()
            try:
                self._store.set_run_status(run_id=request.run_id, status="running")
                result = await self._execute_run(
                    run_id=request.run_id,
                    session_id=request.session_id,
                    message=request.message,
                    run=request.run,
                )
                future = self._run_futures.pop(request.run_id, None)
                if future is not None and not future.done():
                    future.set_result(result)
            except Exception as exc:
                self._store.complete_run(
                    run_id=request.run_id,
                    status="failed",
                    response="",
                    error=f"Worker execution failed: {exc}",
                    input_tokens=0,
                    output_tokens=0,
                    pending_approval_ids=[],
                )
                future = self._run_futures.pop(request.run_id, None)
                if future is not None and not future.done():
                    future.set_exception(exc)
            finally:
                self._run_queue.task_done()
                self._store.append_event(
                    session_id=request.session_id,
                    run_id=request.run_id,
                    event_type="worker_finished",
                    payload={"worker_index": worker_index},
                )

    async def create_session(
        self,
        *,
        session_id: str | None,
        workspace_path: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        assigned_id = session_id or uuid.uuid4().hex[:12]
        workspace_root = Path(workspace_path).expanduser().resolve() if workspace_path else Path(self._settings.workspace_path)

        async with self._manager_lock:
            if assigned_id in self._runtimes:
                raise ValueError(f"Session '{assigned_id}' already exists in runtime manager.")
            if self._store.get_session(assigned_id) is not None:
                raise ValueError(f"Session '{assigned_id}' already exists.")
            try:
                self._store.create_session(
                    session_id=assigned_id,
                    workspace_path=str(workspace_root),
                    metadata=metadata,
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Session '{assigned_id}' already exists.") from exc
            runtime = self._build_runtime(assigned_id, workspace_root)
            self._runtimes[assigned_id] = runtime

        return {
            "session_id": assigned_id,
            "workspace_path": str(workspace_root),
            "metadata": metadata or {},
        }

    async def get_or_restore_session(self, session_id: str) -> SessionRuntime:
        runtime = self._runtimes.get(session_id)
        if runtime is not None:
            return runtime

        async with self._manager_lock:
            runtime = self._runtimes.get(session_id)
            if runtime is not None:
                return runtime

            session_row = self._store.get_session(session_id)
            if session_row is None:
                raise KeyError(f"Unknown session_id '{session_id}'.")

            workspace_root = Path(session_row["workspace_path"]).expanduser().resolve()
            runtime = self._build_runtime(session_id, workspace_root)
            self._runtimes[session_id] = runtime
            return runtime

    async def run_turn(
        self,
        *,
        session_id: str,
        message: str,
        idempotency_key: str | None = None,
        wait_for_completion: bool = True,
        wait_timeout_seconds: float = 300.0,
    ) -> dict[str, Any]:
        await self.get_or_restore_session(session_id)

        if idempotency_key:
            existing_id = self._store.run_id_for_idempotency_key(
                session_id=session_id,
                idempotency_key=idempotency_key,
            )
            if existing_id:
                existing = self._store.get_run(existing_id)
                if existing is not None:
                    return {
                        "run_id": existing_id,
                        "status": existing["status"],
                        "reply": existing.get("response", "") or "",
                        "error": existing.get("error"),
                        "pending_approval_ids": existing.get("pending_approval_ids", []),
                        "total_tokens": int(existing.get("input_tokens", 0)) + int(existing.get("output_tokens", 0)),
                        "tool_calls": 0,
                        "run": existing,
                        "reused": True,
                    }

        run_id = uuid.uuid4().hex[:12]
        if idempotency_key:
            reserved_id = self._store.reserve_idempotency_key(
                session_id=session_id,
                idempotency_key=idempotency_key,
                run_id=run_id,
            )
            if reserved_id != run_id:
                existing = self._store.get_run(reserved_id)
                if existing is None:
                    raise RuntimeError(
                        f"Idempotency key '{idempotency_key}' points to missing run '{reserved_id}'."
                    )
                return {
                    "run_id": reserved_id,
                    "status": existing["status"],
                    "reply": existing.get("response", "") or "",
                    "error": existing.get("error"),
                    "pending_approval_ids": existing.get("pending_approval_ids", []),
                    "total_tokens": int(existing.get("input_tokens", 0)) + int(existing.get("output_tokens", 0)),
                    "tool_calls": 0,
                    "run": existing,
                    "reused": True,
                }

        run = self._store.create_run(
            run_id=run_id,
            session_id=session_id,
            prompt=message,
            worker_mode=self._worker_mode,
        )

        self._store.append_event(
            session_id=session_id,
            run_id=run_id,
            event_type="run_created",
            payload={
                "worker_mode": self._worker_mode,
                "idempotency_key": idempotency_key or "",
            },
        )

        if self._worker_mode == "queue":
            await self._ensure_workers_started()
            assert self._run_queue is not None

            loop = asyncio.get_running_loop()
            result_future: asyncio.Future = loop.create_future()
            self._run_futures[run_id] = result_future
            self._store.set_run_status(run_id=run_id, status="queued")

            self._store.append_event(
                session_id=session_id,
                run_id=run_id,
                event_type="run_queued",
                payload={
                    "queue_depth": self._run_queue.qsize(),
                },
            )

            await self._run_queue.put(
                QueuedRunRequest(
                    run_id=run_id,
                    session_id=session_id,
                    message=message,
                    run=run,
                )
            )

            if not wait_for_completion:
                return {
                    "run_id": run_id,
                    "status": "queued",
                    "reply": "",
                    "error": None,
                    "pending_approval_ids": [],
                    "total_tokens": 0,
                    "tool_calls": 0,
                    "run": run,
                    "reused": False,
                }

            try:
                result = await asyncio.wait_for(result_future, timeout=max(1.0, wait_timeout_seconds))
            except asyncio.TimeoutError:
                return {
                    "run_id": run_id,
                    "status": "running",
                    "reply": "",
                    "error": None,
                    "pending_approval_ids": [],
                    "total_tokens": 0,
                    "tool_calls": 0,
                    "run": run,
                    "reused": False,
                }
            return result

        if wait_for_completion:
            result = await self._execute_run(
                run_id=run_id,
                session_id=session_id,
                message=message,
                run=run,
            )
            result["reused"] = False
            return result

        loop = asyncio.get_running_loop()
        task = loop.create_task(
            self._execute_run(
                run_id=run_id,
                session_id=session_id,
                message=message,
                run=run,
            ),
            name=f"anton-local-run-{run_id}",
        )
        self._run_futures[run_id] = task

        def _finalize_local_background_run(completed: asyncio.Task) -> None:
            self._run_futures.pop(run_id, None)
            try:
                completed.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                run_row = self._store.get_run(run_id)
                if run_row is not None and run_row.get("completed_at") is None:
                    self._store.complete_run(
                        run_id=run_id,
                        status="failed",
                        response="",
                        error=f"Background execution failed: {exc}",
                        input_tokens=0,
                        output_tokens=0,
                        pending_approval_ids=[],
                    )
                self._store.append_event(
                    session_id=session_id,
                    run_id=run_id,
                    event_type="background_run_failed",
                    payload={"error": str(exc)},
                )

        task.add_done_callback(_finalize_local_background_run)
        return {
            "run_id": run_id,
            "status": "running",
            "reply": "",
            "error": None,
            "pending_approval_ids": [],
            "total_tokens": 0,
            "tool_calls": 0,
            "run": run,
            "reused": False,
        }

    async def cancel_run(self, run_id: str) -> dict[str, Any]:
        run = self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id '{run_id}'.")

        if run.get("completed_at") is not None:
            return {
                "run_id": run_id,
                "status": run["status"],
                "cancel_requested": False,
                "message": "Run already finished.",
            }

        updated = self._store.request_run_cancel(run_id)
        if not updated:
            return {
                "run_id": run_id,
                "status": run["status"],
                "cancel_requested": False,
                "message": "Run is already finalized.",
            }

        task = self._active_run_tasks.get(run_id)
        if task is not None:
            task.cancel()

        self._store.append_event(
            session_id=run["session_id"],
            run_id=run_id,
            event_type="run_cancel_requested",
            payload={},
        )

        return {
            "run_id": run_id,
            "status": "cancellation_requested",
            "cancel_requested": True,
            "message": "Cancellation has been requested.",
        }

    async def _execute_run(
        self,
        *,
        run_id: str,
        session_id: str,
        message: str,
        run: dict[str, Any],
    ) -> dict[str, Any]:
        runtime = await self.get_or_restore_session(session_id)
        session_row = self._store.get_session(session_id)
        metadata = session_row.get("metadata", {}) if session_row else {}
        raw_auth = metadata.get("auth_context") if isinstance(metadata, dict) else None
        auth_context = ConnectorHub.auth_context_from_dict(raw_auth if isinstance(raw_auth, dict) else None)
        memory_text, memory_provenance = runtime.memory_context.build(message)
        message_for_turn = message
        if memory_text:
            message_for_turn = f"{message}\n\n{memory_text}"
        self._store.append_event(
            session_id=session_id,
            run_id=run_id,
            event_type="memory_retrieval",
            payload={
                "retrieved_count": len(memory_provenance),
                "provenance": memory_provenance,
            },
        )

        policy = PolicyEngine(
            PolicyConfig(
                max_estimated_seconds_without_approval=self._settings.max_estimated_seconds_without_approval,
                connector_max_query_limit=self._settings.connector_max_query_limit,
                connector_require_where_or_limit=self._settings.connector_require_where_or_limit,
            )
        )
        budget = BudgetTracker(
            BudgetConfig(
                max_tokens_per_run=self._settings.max_tokens_per_run,
                max_tool_calls_per_run=self._settings.max_tool_calls_per_run,
            )
        )
        governance = RunGovernance(
            session_id=session_id,
            run_id=run_id,
            store=self._store,
            policy=policy,
            budget=budget,
        )

        output_dir = runtime.workspace_path / ".anton" / "output"
        before_artifacts = self._snapshot_artifacts(output_dir)

        reply = ""
        error: str | None = None
        status = "running"

        if self._store.is_run_cancel_requested(run_id):
            status = "cancelled"
            self._store.complete_run(
                run_id=run_id,
                status=status,
                response="",
                error=None,
                input_tokens=0,
                output_tokens=0,
                pending_approval_ids=[],
            )
            return {
                "run_id": run_id,
                "status": status,
                "reply": "",
                "error": None,
                "pending_approval_ids": [],
                "total_tokens": 0,
                "tool_calls": 0,
                "run": run,
            }

        async with runtime.lock:
            run_task = asyncio.current_task()
            if run_task is not None:
                self._active_run_tasks[run_id] = run_task

            runtime.session.configure_connector_auth_context(auth_context)
            runtime.session.configure_run_hooks(
                tool_gate=governance.tool_gate,
                usage_hook=governance.usage_hook,
                audit_hook=governance.audit_hook,
            )

            try:
                async for event in runtime.session.turn_stream(message_for_turn):
                    if self._store.is_run_cancel_requested(run_id):
                        raise asyncio.CancelledError("Run cancelled by user request")

                    if isinstance(event, StreamTextDelta):
                        reply += event.text
                    elif isinstance(event, StreamComplete):
                        self._store.append_event(
                            session_id=session_id,
                            run_id=run_id,
                            event_type="stream_complete",
                            payload={
                                "stop_reason": event.response.stop_reason,
                                "input_tokens": event.response.usage.input_tokens,
                                "output_tokens": event.response.usage.output_tokens,
                            },
                        )
            except asyncio.CancelledError:
                status = "cancelled"
            except Exception as exc:  # pragma: no cover - defensive
                error = str(exc)
                status = "failed"
            finally:
                runtime.session.configure_run_hooks(
                    tool_gate=None,
                    usage_hook=None,
                    audit_hook=None,
                )
                runtime.session.configure_connector_auth_context(None)
                self._active_run_tasks.pop(run_id, None)

        after_artifacts = self._snapshot_artifacts(output_dir)
        new_files = sorted(after_artifacts - before_artifacts)
        for artifact in new_files:
            kind = artifact.suffix.lstrip(".") or "file"
            self._store.add_artifact(run_id=run_id, kind=kind, path=str(artifact))

        if status == "running":
            if error is not None:
                status = "failed"
            elif governance.pending_approvals:
                status = "approval_required"
            else:
                status = "completed"

        response = reply if status != "failed" else ""

        self._store.complete_run(
            run_id=run_id,
            status=status,
            response=response,
            error=error,
            input_tokens=budget.input_tokens,
            output_tokens=budget.output_tokens,
            pending_approval_ids=governance.pending_approvals,
        )

        return {
            "run_id": run_id,
            "status": status,
            "reply": response,
            "error": error,
            "pending_approval_ids": governance.pending_approvals,
            "total_tokens": budget.total_tokens,
            "tool_calls": budget.tool_calls,
            "run": run,
        }

    def _build_runtime(self, session_id: str, workspace_root: Path) -> SessionRuntime:
        workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = Workspace(workspace_root)
        workspace.initialize()
        workspace.apply_env_to_process()

        settings = self._settings.model_copy(deep=True)
        settings.resolve_workspace(str(workspace_root))

        llm = LLMClient.from_settings(settings)
        sa = SelfAwarenessContext(Path(settings.context_dir))
        connector_hub = ConnectorHub.from_settings(settings, workspace=workspace_root)

        runtime_context = (
            f"- Provider: {settings.planning_provider}\n"
            f"- Planning model: {settings.planning_model}\n"
            f"- Coding model: {settings.coding_model}\n"
            f"- Workspace: {workspace_root}\n"
            f"- Connector mode: {settings.connector_mode}\n"
        )

        api_key = settings.anthropic_api_key if settings.coding_provider == "anthropic" else settings.openai_api_key
        session = ChatSession(
            llm,
            self_awareness=sa,
            runtime_context=runtime_context,
            workspace=workspace,
            console=None,
            coding_provider=settings.coding_provider,
            coding_api_key=api_key or "",
            connector_hub=connector_hub,
        )

        memory_context = MemoryContextBuilder(
            memory_dir=Path(settings.memory_dir),
            enabled=bool(settings.memory_enabled),
            max_items=max(1, int(settings.memory_recall_max_items)),
            max_chars=max(1000, int(settings.memory_recall_max_chars)),
        )

        return SessionRuntime(
            session_id=session_id,
            workspace_path=workspace_root,
            workspace=workspace,
            session=session,
            memory_context=memory_context,
            lock=asyncio.Lock(),
        )

    @staticmethod
    def _snapshot_artifacts(output_dir: Path) -> set[Path]:
        if not output_dir.exists():
            return set()
        return {path.resolve() for path in output_dir.rglob("*") if path.is_file()}
