from __future__ import annotations

import asyncio
import hashlib
import json
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
from anton.service.store import ServiceStore
from anton.workspace import Workspace


@dataclass
class SessionRuntime:
    session_id: str
    workspace_path: Path
    workspace: Workspace
    session: ChatSession
    lock: asyncio.Lock


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
            self._store.create_session(
                session_id=assigned_id,
                workspace_path=str(workspace_root),
                metadata=metadata,
            )
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
    ) -> dict[str, Any]:
        runtime = await self.get_or_restore_session(session_id)
        run_id = uuid.uuid4().hex[:12]

        run = self._store.create_run(run_id=run_id, session_id=session_id, prompt=message)

        policy = PolicyEngine(
            PolicyConfig(
                max_estimated_seconds_without_approval=self._settings.max_estimated_seconds_without_approval,
                connector_max_query_limit=self._settings.connector_max_query_limit,
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

        async with runtime.lock:
            runtime.session.configure_run_hooks(
                tool_gate=governance.tool_gate,
                usage_hook=governance.usage_hook,
                audit_hook=governance.audit_hook,
            )

            try:
                async for event in runtime.session.turn_stream(message):
                    if isinstance(event, StreamTextDelta):
                        reply += event.text
                    elif isinstance(event, StreamComplete):
                        # usage hook already handles usage accounting
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
            except Exception as exc:
                error = str(exc)
            finally:
                runtime.session.configure_run_hooks(
                    tool_gate=None,
                    usage_hook=None,
                    audit_hook=None,
                )

        after_artifacts = self._snapshot_artifacts(output_dir)
        new_files = sorted(after_artifacts - before_artifacts)
        for artifact in new_files:
            kind = artifact.suffix.lstrip(".") or "file"
            self._store.add_artifact(run_id=run_id, kind=kind, path=str(artifact))

        if error is not None:
            status = "failed"
            response = ""
        elif governance.pending_approvals:
            status = "approval_required"
            response = reply
        else:
            status = "completed"
            response = reply

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

        return SessionRuntime(
            session_id=session_id,
            workspace_path=workspace_root,
            workspace=workspace,
            session=session,
            lock=asyncio.Lock(),
        )

    @staticmethod
    def _snapshot_artifacts(output_dir: Path) -> set[Path]:
        if not output_dir.exists():
            return set()
        return {path.resolve() for path in output_dir.rglob("*") if path.is_file()}
