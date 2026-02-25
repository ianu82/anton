from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from anton.service.runtime import RuntimeManager
from anton.service.store import ServiceStore


def load_tasks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Eval task file must be a JSON array.")
    out: list[dict[str, Any]] = []
    for idx, task in enumerate(payload, start=1):
        if not isinstance(task, dict):
            raise ValueError(f"Eval task #{idx} must be an object.")
        prompt = str(task.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"Eval task #{idx} missing 'prompt'.")
        out.append(task)
    return out


async def run_eval(
    *,
    settings,
    tasks_path: Path,
    output_path: Path,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    tasks = load_tasks(tasks_path)
    ws_root = (workspace_root or Path(settings.workspace_path)).resolve()
    service_root = ws_root / ".anton" / "service"

    store = ServiceStore(service_root / "eval.db")
    runtime = RuntimeManager(settings, store)

    results: list[dict[str, Any]] = []

    for idx, task in enumerate(tasks, start=1):
        name = str(task.get("name", f"task_{idx}"))
        expect_contains = task.get("expect_contains")

        workspace = ws_root / ".anton" / "eval" / name
        workspace.mkdir(parents=True, exist_ok=True)

        created = await runtime.create_session(
            session_id=None,
            workspace_path=str(workspace),
            metadata={"eval_task": name},
        )

        t0 = time.monotonic()
        turn = await runtime.run_turn(session_id=created["session_id"], message=str(task["prompt"]))
        latency = time.monotonic() - t0

        passed = turn["status"] == "completed"
        if isinstance(expect_contains, str) and expect_contains:
            passed = passed and expect_contains in (turn.get("reply") or "")

        results.append(
            {
                "name": name,
                "status": turn["status"],
                "passed": passed,
                "latency_seconds": latency,
                "run_id": turn["run_id"],
                "pending_approval_ids": turn.get("pending_approval_ids", []),
                "reply_preview": (turn.get("reply") or "")[:300],
            }
        )

    run_count = len(results)
    passed_count = sum(1 for result in results if result["passed"])
    latencies = sorted(result["latency_seconds"] for result in results)

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        idx = int((len(values) - 1) * p)
        return values[idx]

    summary = {
        "run_count": run_count,
        "passed_count": passed_count,
        "pass_rate": (passed_count / run_count) if run_count else 0.0,
        "latency_p50": pct(latencies, 0.5),
        "latency_p95": pct(latencies, 0.95),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    return summary


def run_eval_sync(
    *,
    settings,
    tasks_path: Path,
    output_path: Path,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        run_eval(
            settings=settings,
            tasks_path=tasks_path,
            output_path=output_path,
            workspace_root=workspace_root,
        )
    )
