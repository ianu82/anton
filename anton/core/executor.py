from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from anton.core.planner import Plan
from anton.events.bus import EventBus
from anton.events.types import Phase, StatusUpdate
from anton.skill.base import SkillResult
from anton.skill.registry import SkillRegistry

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    step_index: int
    skill_name: str
    result: SkillResult
    duration_seconds: float


@dataclass
class ExecutionResult:
    step_results: list[StepResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0


class Executor:
    def __init__(self, registry: SkillRegistry, bus: EventBus) -> None:
        self._registry = registry
        self._bus = bus

    async def execute_plan(self, plan: Plan, eta_seconds: float | None = None) -> ExecutionResult:
        results: list[StepResult] = []
        total_start = time.monotonic()
        total_steps = len(plan.steps)

        for i, step in enumerate(plan.steps):
            # Compute ETA: use initial estimate for first step, then observed pace
            step_eta: float | None = None
            if i == 0 and eta_seconds is not None:
                step_eta = eta_seconds
            elif i > 0:
                elapsed = time.monotonic() - total_start
                avg_per_step = elapsed / i
                step_eta = avg_per_step * (total_steps - i)

            await self._bus.publish(
                StatusUpdate(
                    phase=Phase.EXECUTING,
                    message=f"Step {i + 1}/{total_steps}: {step.description}",
                    eta_seconds=step_eta,
                )
            )

            skill = self._registry.get(step.skill_name)
            if skill is None:
                result = SkillResult(
                    output=None,
                    metadata={"error": f"Skill not found: {step.skill_name}"},
                )
                results.append(
                    StepResult(
                        step_index=i,
                        skill_name=step.skill_name,
                        result=result,
                        duration_seconds=0,
                    )
                )
                continue

            step_start = time.monotonic()
            try:
                result = await skill.execute(**step.parameters)
            except Exception as exc:
                result = SkillResult(
                    output=None,
                    metadata={"error": f"Skill crashed: {exc}"},
                )
                # Quarantine the broken skill so it doesn't keep failing
                self._quarantine_skill(skill.name, skill.source_path, exc)
            step_duration = time.monotonic() - step_start

            results.append(
                StepResult(
                    step_index=i,
                    skill_name=step.skill_name,
                    result=result,
                    duration_seconds=step_duration,
                )
            )

        total_duration = time.monotonic() - total_start
        return ExecutionResult(step_results=results, total_duration_seconds=total_duration)

    def _quarantine_skill(self, name: str, source_path: Path | None, exc: Exception) -> None:
        """Rename a broken skill to .broken and unregister it."""
        logger.warning("Quarantining broken skill '%s': %s", name, exc)
        self._registry.unregister(name)
        if source_path is not None and source_path.exists():
            broken_path = source_path.with_suffix(".py.broken")
            try:
                source_path.rename(broken_path)
                # Write the error details alongside
                error_path = source_path.with_suffix(".error")
                error_path.write_text(
                    f"Quarantined due to runtime error:\n{exc}\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
