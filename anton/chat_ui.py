from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console


@dataclass
class _ToolActivity:
    tool_id: str
    name: str
    json_parts: list[str] = field(default_factory=list)
    description: str = ""
    current_progress: str = ""
    step_count: int = 0


_TOOL_LABELS: dict[str, str] = {
    "execute_task": "Task",
    "scratchpad": "Scratchpad",
    "minds": "Minds",
    "update_context": "Context",
    "request_secret": "Secret",
}

_MAX_DESC = 60


def _tool_display_text(name: str, input_json: str) -> str:
    """Map tool name + raw JSON input to a human-readable description."""
    label = _TOOL_LABELS.get(name, name)
    try:
        data = json.loads(input_json)
    except (json.JSONDecodeError, TypeError):
        return label

    desc = ""
    if name == "execute_task":
        desc = data.get("task", "")
    elif name == "scratchpad":
        desc = data.get("action", "")
    elif name == "minds":
        action = data.get("action", "")
        question = data.get("question", "")
        desc = f"{action}: {question}" if question else action
    elif name == "update_context":
        updates = data.get("updates", [])
        desc = f"{len(updates)} file(s)"
    elif name == "request_secret":
        desc = data.get("variable_name", "")

    if desc:
        if len(desc) > _MAX_DESC:
            desc = desc[:_MAX_DESC - 1] + "\u2026"
        return f"{label}({desc})"
    return label

THINKING_MESSAGES = [
    "Consulting the sacred docs...",
    "Rebasing my neurons...",
    "Spinning up inference hamsters...",
    "Parsing the vibes...",
    "Asking the rubber duck...",
    "Aligning my attention heads...",
    "Searching the latent space...",
    "Unrolling the loops...",
    "Compiling thoughts...",
    "Warming up the transformer...",
    "Descending the gradient...",
    "Sampling from the posterior...",
    "Tokenizing reality...",
    "Running a forward pass...",
    "Traversing the context window...",
    "Optimizing the objective...",
    "Softmaxing the options...",
    "Backpropagating insights...",
    "Loading weights...",
    "Crunching embeddings...",
]

TOOL_MESSAGES = [
    "Rolling up sleeves...",
    "Firing up the agent...",
    "Handing off to the crew...",
    "Dispatching the task...",
    "Engaging autopilot...",
    "Letting the tools cook...",
]

PHASE_LABELS = {
    "memory_recall": "Memory",
    "planning": "Planning",
    "skill_discovery": "Skills",
    "skill_building": "Building",
    "executing": "Executing",
    "complete": "Complete",
    "failed": "Failed",
}


class StreamDisplay:
    """Manages a Rich Live display for streaming LLM responses."""

    def __init__(self, console: Console, toolbar: dict | None = None) -> None:
        self._console = console
        self._live: object | None = None
        self._buffer = ""
        self._started = False
        self._toolbar = toolbar
        self._activities: list[_ToolActivity] = []
        self._thinking_msg: str = ""

    def _set_status(self, text: str) -> None:
        if self._toolbar is not None:
            self._toolbar["status"] = text

    def start(self) -> None:
        msg = random.choice(THINKING_MESSAGES)  # noqa: S311
        self._thinking_msg = msg
        self._set_status(msg)
        spinner = Spinner("dots", text=Text(f" {msg}", style="anton.muted"))
        self._live = Live(
            spinner,
            console=self._console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()
        self._buffer = ""
        self._started = False
        self._activities = []

    def append_text(self, delta: str) -> None:
        if self._live is None:
            return
        self._buffer += delta
        self._started = True
        if self._activities:
            self._live.update(Group(self._build_activity_tree(), Markdown(self._buffer)))
        else:
            self._live.update(Markdown(self._buffer))

    def show_tool_result(self, content: str) -> None:
        """Display a tool result (e.g. scratchpad dump) directly to the user."""
        if self._live is None:
            return
        if self._buffer:
            self._buffer += "\n\n"
        self._buffer += content
        self._started = True
        if self._activities:
            self._live.update(Group(self._build_activity_tree(), Markdown(self._buffer)))
        else:
            self._live.update(Markdown(self._buffer))

    def show_tool_execution(self, task: str) -> None:
        """Backward-compatible wrapper — delegates to on_tool_use_start."""
        self.on_tool_use_start(f"_compat_{id(task)}", task)

    def on_tool_use_start(self, tool_id: str, name: str) -> None:
        """Track a new tool use and update the live display."""
        if self._live is None:
            return
        activity = _ToolActivity(tool_id=tool_id, name=name)
        self._activities.append(activity)
        self._refresh_live()

    def on_tool_use_delta(self, tool_id: str, json_delta: str) -> None:
        """Accumulate JSON input deltas for a tool use."""
        for act in self._activities:
            if act.tool_id == tool_id:
                act.json_parts.append(json_delta)
                return

    def on_tool_use_end(self, tool_id: str) -> None:
        """Finalize a tool use: parse accumulated JSON and set description."""
        for act in self._activities:
            if act.tool_id == tool_id:
                raw = "".join(act.json_parts)
                act.description = _tool_display_text(act.name, raw)
                self._refresh_live()
                return

    def update_progress(self, phase: str, message: str, eta: float | None = None) -> None:
        """Update the Live display with agent progress (phase + message + optional ETA)."""
        if self._live is None:
            return
        label = PHASE_LABELS.get(phase, phase)
        eta_str = f"  ~{int(eta)}s" if eta else ""
        status = f"{label}  {message}{eta_str}"
        self._set_status(status)

        # Associate progress with the last execute_task activity
        for act in reversed(self._activities):
            if act.name == "execute_task":
                act.current_progress = f"{label}  {message}{eta_str}"
                act.step_count += 1
                break

        self._refresh_live()

    def finish(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

        # Print finalized activity summary before the response
        if self._activities:
            self._console.print(self._build_activity_tree(final=True))

        # Print final rendered response
        if self._buffer:
            self._console.print(Text("anton> ", style="anton.cyan"), end="")
            self._console.print(Markdown(self._buffer))

        self._console.print()

    def abort(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    # --- Private helpers ---

    def _build_activity_tree(self, final: bool = False) -> Text:
        """Render the activity tree as styled Text."""
        lines = Text()
        for act in self._activities:
            label = act.description or _TOOL_LABELS.get(act.name, act.name)
            lines.append("  ")
            lines.append(label, style="bold")
            lines.append("\n")
            # Sub-progress for execute_task
            if act.name == "execute_task":
                if final and act.step_count > 0:
                    lines.append(f"  \u23bf Done ({act.step_count} step{'s' if act.step_count != 1 else ''})\n", style="anton.muted")
                elif not final and act.current_progress:
                    lines.append(f"  \u23bf {act.current_progress}\n", style="anton.muted")
        return lines

    def _refresh_live(self) -> None:
        """Recompose the live display with spinner + activity tree."""
        if self._live is None:
            return
        spinner = Spinner("dots", text=Text(f" {self._thinking_msg}", style="anton.muted"))
        tree = self._build_activity_tree()
        if self._started and self._buffer:
            self._live.update(Group(tree, Markdown(self._buffer)))
        else:
            self._live.update(Group(spinner, tree))
