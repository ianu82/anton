from __future__ import annotations

import asyncio
import inspect
import os
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import anthropic

from anton.clipboard import (
    cleanup_old_uploads,
    clipboard_unavailable_reason,
    grab_clipboard,
    is_clipboard_supported,
    parse_dropped_paths as _parse_dropped_paths,
    save_clipboard_image,
)
from anton.llm.prompts import CHAT_SYSTEM_PROMPT
from anton.llm.provider import (
    StreamComplete,
    StreamEvent,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
)
from anton.scratchpad import ScratchpadManager
from anton.tools import (
    CONNECTOR_TOOL,
    REQUEST_SECRET_TOOL,
    SCRATCHPAD_TOOL,
    UPDATE_CONTEXT_TOOL,
    dispatch_tool,
    format_cell_result,
    prepare_scratchpad_exec,
    prompt_secret,
)

if TYPE_CHECKING:
    from rich.console import Console

    from anton.connectors.hub import ConnectorHub
    from anton.config.settings import AntonSettings
    from anton.context.self_awareness import SelfAwarenessContext
    from anton.governance.hooks import AuditHook, ToolGate, UsageHook
    from anton.llm.client import LLMClient
    from anton.workspace import Workspace


_MAX_TOOL_ROUNDS = 25  # Hard limit on consecutive tool-call rounds per turn
_MAX_CONSECUTIVE_ERRORS = 5  # Stop if the same tool fails this many times in a row
_RESILIENCE_NUDGE_AT = 2  # Inject resilience nudge after this many consecutive errors
_RESILIENCE_NUDGE = (
    "\n\nSYSTEM: This tool has failed twice in a row. Before retrying the same approach or "
    "asking the user for help, try a creative workaround — different headers/user-agent, "
    "a public API, archive.org, an alternate library, or a completely different data source. "
    "Only involve the user if the problem truly requires something only they can provide."
)


class ChatSession:
    """Manages a multi-turn conversation with tool-call delegation."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        self_awareness: SelfAwarenessContext | None = None,
        runtime_context: str = "",
        workspace: Workspace | None = None,
        console: Console | None = None,
        coding_provider: str = "anthropic",
        coding_api_key: str = "",
        connector_hub: ConnectorHub | None = None,
        tool_gate: ToolGate | None = None,
        usage_hook: UsageHook | None = None,
        audit_hook: AuditHook | None = None,
    ) -> None:
        self._llm = llm_client
        self._self_awareness = self_awareness
        self._runtime_context = runtime_context
        self._workspace = workspace
        self._console = console
        self._connector_hub = connector_hub
        self._tool_gate = tool_gate
        self._usage_hook = usage_hook
        self._audit_hook = audit_hook
        self._history: list[dict] = []
        self._scratchpads = ScratchpadManager(
            coding_provider=coding_provider,
            coding_model=getattr(llm_client, "coding_model", ""),
            coding_api_key=coding_api_key,
            secret_handler=self._make_secret_handler(),
        )

    def configure_run_hooks(
        self,
        *,
        tool_gate: ToolGate | None = None,
        usage_hook: UsageHook | None = None,
        audit_hook: AuditHook | None = None,
    ) -> None:
        """Set optional governance hooks used during tool/LLM execution."""
        self._tool_gate = tool_gate
        self._usage_hook = usage_hook
        self._audit_hook = audit_hook

    def _make_secret_handler(self):
        """Create a closure that prompts the user for secrets via the console."""
        def handler(var_name: str, prompt_text: str) -> str | None:
            if self._workspace is None or self._console is None:
                return None
            # If already stored, return it without prompting
            if self._workspace.has_secret(var_name):
                return os.environ.get(var_name) or self._workspace.get_secret(var_name)
            value = prompt_secret(self._console, var_name, prompt_text)
            if not value:
                return None
            self._workspace.set_secret(var_name, value)
            return value
        return handler

    def _emit_audit(self, event_name: str, payload: dict) -> None:
        if self._audit_hook is not None:
            self._audit_hook(event_name, payload)

    def _emit_usage(self, usage) -> None:
        if self._usage_hook is not None:
            self._usage_hook(usage)

    async def _check_tool_gate(self, tool_name: str, tc_input: dict):
        from anton.governance.hooks import ToolGateResult

        if self._tool_gate is None:
            return ToolGateResult(allow=True)

        decision = self._tool_gate(tool_name, tc_input)
        if inspect.isawaitable(decision):
            decision = await decision
        return decision

    @property
    def history(self) -> list[dict]:
        return self._history

    def _build_system_prompt(self) -> str:
        prompt = CHAT_SYSTEM_PROMPT.format(runtime_context=self._runtime_context)
        if self._self_awareness is not None:
            sa_section = self._self_awareness.build_prompt_section()
            if sa_section:
                prompt += sa_section
        # Inject anton.md project context
        if self._workspace is not None:
            md_context = self._workspace.build_anton_md_context()
            if md_context:
                prompt += md_context
        return prompt

    # Packages the LLM is most likely to care about when writing scratchpad code.
    _NOTABLE_PACKAGES: set[str] = {
        "numpy", "pandas", "matplotlib", "seaborn", "scipy", "scikit-learn",
        "requests", "httpx", "aiohttp", "beautifulsoup4", "lxml",
        "pillow", "sympy", "networkx", "sqlalchemy", "pydantic",
        "rich", "tqdm", "click", "fastapi", "flask", "django",
        "openai", "anthropic", "tiktoken", "transformers", "torch",
        "polars", "pyarrow", "openpyxl", "xlsxwriter",
        "plotly", "bokeh", "altair",
        "pytest", "hypothesis",
        "yaml", "pyyaml", "toml", "tomli", "tomllib",
        "jinja2", "markdown", "pygments",
        "cryptography", "paramiko", "boto3",
    }

    def _build_tools(self) -> list[dict]:
        scratchpad_tool = dict(SCRATCHPAD_TOOL)
        pkg_list = self._scratchpads._available_packages
        if pkg_list:
            notable = sorted(
                p for p in pkg_list
                if p.lower() in self._NOTABLE_PACKAGES
            )
            if notable:
                pkg_line = ", ".join(notable)
                extra = f"\n\nInstalled packages ({len(pkg_list)} total, notable: {pkg_line})."
            else:
                extra = f"\n\nInstalled packages: {len(pkg_list)} total (standard library plus dependencies)."
            scratchpad_tool["description"] = SCRATCHPAD_TOOL["description"] + extra

        tools = [scratchpad_tool]
        if self._self_awareness is not None:
            tools.append(UPDATE_CONTEXT_TOOL)
        if self._workspace is not None:
            tools.append(REQUEST_SECRET_TOOL)
        if self._connector_hub is not None:
            tools.append(CONNECTOR_TOOL)
        return tools

    async def close(self) -> None:
        """Clean up scratchpads and other resources."""
        await self._scratchpads.close_all()

    async def turn(self, user_input: str | list[dict]) -> str:
        self._history.append({"role": "user", "content": user_input})

        system = self._build_system_prompt()
        tools = self._build_tools()

        response = await self._llm.plan(
            system=system,
            messages=self._history,
            tools=tools,
        )
        self._emit_usage(response.usage)
        self._emit_audit(
            "llm_response",
            {
                "mode": "non_stream",
                "stop_reason": response.stop_reason or "",
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "tool_calls": len(response.tool_calls),
            },
        )

        # Handle tool calls
        tool_round = 0
        error_streak: dict[str, int] = {}
        resilience_nudged: set[str] = set()

        while response.tool_calls:
            tool_round += 1
            if tool_round > _MAX_TOOL_ROUNDS:
                self._history.append({"role": "assistant", "content": response.content or ""})
                self._history.append({
                    "role": "user",
                    "content": (
                        f"SYSTEM: You have used {_MAX_TOOL_ROUNDS} tool-call rounds on this turn. "
                        "Stop retrying. Summarize what you accomplished and what failed, "
                        "then tell the user what they can do to unblock the issue."
                    ),
                })
                response = await self._llm.plan(
                    system=system,
                    messages=self._history,
                )
                break

            # Build assistant message with content blocks
            assistant_content: list[dict] = []
            if response.content:
                assistant_content.append({"type": "text", "text": response.content})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })
            self._history.append({"role": "assistant", "content": assistant_content})

            # Process each tool call via registry
            tool_results: list[dict] = []
            for tc in response.tool_calls:
                self._emit_audit(
                    "tool_call",
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    },
                )
                try:
                    gate = await self._check_tool_gate(tc.name, tc.input)
                    if not gate.allow:
                        if gate.pending_approval_id:
                            result_text = (
                                "Pending approval "
                                f"{gate.pending_approval_id}: {gate.message}"
                            )
                        else:
                            result_text = f"Rejected: {gate.message}"
                    else:
                        result_text = await dispatch_tool(self, tc.name, tc.input)
                except Exception as exc:
                    result_text = f"Tool '{tc.name}' failed: {exc}"

                result_text = _apply_error_tracking(
                    result_text, tc.name, error_streak, resilience_nudged,
                )
                self._emit_audit(
                    "tool_result",
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "content_preview": result_text[:500],
                    },
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_text,
                })

            self._history.append({"role": "user", "content": tool_results})

            # Get follow-up from LLM
            response = await self._llm.plan(
                system=system,
                messages=self._history,
                tools=tools,
            )
            self._emit_usage(response.usage)
            self._emit_audit(
                "llm_response",
                {
                    "mode": "non_stream",
                    "stop_reason": response.stop_reason or "",
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "tool_calls": len(response.tool_calls),
                },
            )

        # Text-only response
        reply = response.content or ""
        self._history.append({"role": "assistant", "content": reply})
        self._emit_audit("assistant_reply", {"mode": "non_stream", "content": reply[:2000]})
        return reply

    async def turn_stream(self, user_input: str | list[dict]) -> AsyncIterator[StreamEvent]:
        """Streaming version of turn(). Yields events as they arrive."""
        self._history.append({"role": "user", "content": user_input})

        async for event in self._stream_and_handle_tools():
            yield event

    async def _stream_and_handle_tools(self) -> AsyncIterator[StreamEvent]:
        """Stream one LLM call, handle tool loops, yield all events."""
        system = self._build_system_prompt()
        tools = self._build_tools()

        response: StreamComplete | None = None

        async for event in self._llm.plan_stream(
            system=system,
            messages=self._history,
            tools=tools,
        ):
            yield event
            if isinstance(event, StreamComplete):
                response = event
                self._emit_usage(event.response.usage)
                self._emit_audit(
                    "llm_response",
                    {
                        "mode": "stream",
                        "stop_reason": event.response.stop_reason or "",
                        "input_tokens": event.response.usage.input_tokens,
                        "output_tokens": event.response.usage.output_tokens,
                        "tool_calls": len(event.response.tool_calls),
                    },
                )

        if response is None:
            return

        llm_response = response.response

        # Tool-call loop with circuit breaker
        tool_round = 0
        error_streak: dict[str, int] = {}
        resilience_nudged: set[str] = set()

        while llm_response.tool_calls:
            tool_round += 1
            if tool_round > _MAX_TOOL_ROUNDS:
                self._history.append({"role": "assistant", "content": llm_response.content or ""})
                self._history.append({
                    "role": "user",
                    "content": (
                        f"SYSTEM: You have used {_MAX_TOOL_ROUNDS} tool-call rounds on this turn. "
                        "Stop retrying. Summarize what you accomplished and what failed, "
                        "then tell the user what they can do to unblock the issue."
                    ),
                })
                async for event in self._llm.plan_stream(
                    system=system,
                    messages=self._history,
                ):
                    yield event
                return

            # Build assistant message with content blocks
            assistant_content: list[dict] = []
            if llm_response.content:
                assistant_content.append({"type": "text", "text": llm_response.content})
            for tc in llm_response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })
            self._history.append({"role": "assistant", "content": assistant_content})

            # Process each tool call
            tool_results: list[dict] = []
            for tc in llm_response.tool_calls:
                self._emit_audit(
                    "tool_call",
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    },
                )
                try:
                    gate = await self._check_tool_gate(tc.name, tc.input)
                    if not gate.allow:
                        if gate.pending_approval_id:
                            result_text = (
                                "Pending approval "
                                f"{gate.pending_approval_id}: {gate.message}"
                            )
                        else:
                            result_text = f"Rejected: {gate.message}"
                    elif tc.name == "scratchpad" and tc.input.get("action") == "exec":
                        # Inline streaming exec — yields progress events
                        prep = await prepare_scratchpad_exec(self, tc.input)
                        if isinstance(prep, str):
                            result_text = prep
                        else:
                            pad, code, description, estimated_time, estimated_seconds = prep
                            from anton.scratchpad import Cell
                            cell = None
                            async for item in pad.execute_streaming(
                                code,
                                description=description,
                                estimated_time=estimated_time,
                                estimated_seconds=estimated_seconds,
                            ):
                                if isinstance(item, str):
                                    yield StreamTaskProgress(
                                        phase="scratchpad", message=item
                                    )
                                elif isinstance(item, Cell):
                                    cell = item
                            result_text = format_cell_result(cell) if cell else "No result produced."
                    else:
                        result_text = await dispatch_tool(self, tc.name, tc.input)
                        if tc.name == "scratchpad" and tc.input.get("action") == "dump":
                            yield StreamToolResult(content=result_text)
                            result_text = (
                                "The full notebook has been displayed to the user above. "
                                "Do not repeat it. Here is the content for your reference:\n\n"
                                + result_text
                            )
                except Exception as exc:
                    result_text = f"Tool '{tc.name}' failed: {exc}"

                result_text = _apply_error_tracking(
                    result_text, tc.name, error_streak, resilience_nudged,
                )
                self._emit_audit(
                    "tool_result",
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "content_preview": result_text[:500],
                    },
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_text,
                })

            self._history.append({"role": "user", "content": tool_results})

            # Stream follow-up
            response = None
            async for event in self._llm.plan_stream(
                system=system,
                messages=self._history,
                tools=tools,
            ):
                yield event
                if isinstance(event, StreamComplete):
                    response = event
                    self._emit_usage(event.response.usage)
                    self._emit_audit(
                        "llm_response",
                        {
                            "mode": "stream",
                            "stop_reason": event.response.stop_reason or "",
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                            "tool_calls": len(event.response.tool_calls),
                        },
                    )

            if response is None:
                return
            llm_response = response.response

        # Text-only final response — append to history
        reply = llm_response.content or ""
        self._history.append({"role": "assistant", "content": reply})
        self._emit_audit("assistant_reply", {"mode": "stream", "content": reply[:2000]})


def _apply_error_tracking(
    result_text: str,
    tool_name: str,
    error_streak: dict[str, int],
    resilience_nudged: set[str],
) -> str:
    """Track consecutive errors per tool and append nudge/circuit-breaker messages."""
    is_error = any(
        marker in result_text
        for marker in ("[error]", "Task failed:", "failed", "timed out", "Rejected:")
    )
    if is_error:
        error_streak[tool_name] = error_streak.get(tool_name, 0) + 1
    else:
        error_streak[tool_name] = 0
        resilience_nudged.discard(tool_name)

    streak = error_streak.get(tool_name, 0)
    if streak >= _RESILIENCE_NUDGE_AT and tool_name not in resilience_nudged:
        result_text += _RESILIENCE_NUDGE
        resilience_nudged.add(tool_name)

    if streak >= _MAX_CONSECUTIVE_ERRORS:
        result_text += (
            f"\n\nSYSTEM: The '{tool_name}' tool has failed {_MAX_CONSECUTIVE_ERRORS} times "
            "in a row. Stop retrying this approach. Either try a completely different "
            "strategy or tell the user what's going wrong so they can help."
        )

    return result_text


def _rebuild_session(
    *,
    settings: AntonSettings,
    state: dict,
    self_awareness,
    workspace,
    console: Console,
) -> ChatSession:
    """Rebuild LLMClient + ChatSession after settings change."""
    from anton.connectors import ConnectorHub
    from anton.llm.client import LLMClient

    state["llm_client"] = LLMClient.from_settings(settings)
    connector_hub = ConnectorHub.from_settings(
        settings,
        workspace=Path(settings.workspace_path),
    )
    runtime_context = (
        f"- Provider: {settings.planning_provider}\n"
        f"- Planning model: {settings.planning_model}\n"
        f"- Coding model: {settings.coding_model}\n"
        f"- Workspace: {settings.workspace_path}\n"
        f"- Connector mode: {settings.connector_mode}\n"
    )
    api_key = (
        settings.anthropic_api_key if settings.coding_provider == "anthropic"
        else settings.openai_api_key
    ) or ""
    return ChatSession(
        state["llm_client"],
        self_awareness=self_awareness,
        runtime_context=runtime_context,
        workspace=workspace,
        console=console,
        coding_provider=settings.coding_provider,
        coding_api_key=api_key,
        connector_hub=connector_hub,
    )


async def _handle_setup(
    console: Console,
    settings: AntonSettings,
    workspace: Workspace,
    state: dict,
    self_awareness,
    session: ChatSession,
) -> ChatSession:
    """Interactive setup wizard — reconfigure provider, model, and API key."""
    from rich.prompt import Prompt

    console.print()
    console.print("[anton.cyan]Current configuration:[/]")
    console.print(f"  Provider (planning): [bold]{settings.planning_provider}[/]")
    console.print(f"  Provider (coding):   [bold]{settings.coding_provider}[/]")
    console.print(f"  Planning model:      [bold]{settings.planning_model}[/]")
    console.print(f"  Coding model:        [bold]{settings.coding_model}[/]")
    console.print()

    # --- Provider ---
    providers = {"1": "anthropic", "2": "openai", "3": "openai-compatible"}
    current_num = {"anthropic": "1", "openai": "2", "openai-compatible": "3"}.get(settings.planning_provider, "1")
    console.print("[anton.cyan]Available providers:[/]")
    console.print(r"  [bold]1[/]  Anthropic (Claude)                    [dim]\[recommended][/]")
    console.print(r"  [bold]2[/]  OpenAI (GPT / o-series)               [dim]\[experimental][/]")
    console.print(r"  [bold]3[/]  OpenAI-compatible (custom endpoint)   [dim]\[experimental][/]")
    console.print()

    choice = Prompt.ask(
        "Select provider",
        choices=["1", "2", "3"],
        default=current_num,
        console=console,
    )
    provider = providers[choice]

    # --- Base URL (OpenAI-compatible only) ---
    if provider == "openai-compatible":
        current_base_url = settings.openai_base_url or ""
        console.print()
        base_url = Prompt.ask(
            f"API base URL [dim](e.g. http://localhost:11434/v1)[/]",
            default=current_base_url,
            console=console,
        )
        base_url = base_url.strip()
        if base_url:
            settings.openai_base_url = base_url
            workspace.set_secret("ANTON_OPENAI_BASE_URL", base_url)

    # --- API key ---
    key_attr = "anthropic_api_key" if provider == "anthropic" else "openai_api_key"
    current_key = getattr(settings, key_attr) or ""
    masked = current_key[:4] + "..." + current_key[-4:] if len(current_key) > 8 else "***"
    console.print()
    api_key = Prompt.ask(
        f"API key for {provider.title()} [dim](Enter to keep {masked})[/]",
        default="",
        console=console,
    )
    api_key = api_key.strip()

    # --- Models ---
    defaults = {
        "anthropic": ("claude-sonnet-4-6", "claude-haiku-4-5-20251001"),
        "openai": ("gpt-5-mini", "gpt-5-nano"),
    }
    default_planning, default_coding = defaults.get(provider, ("", ""))

    console.print()
    planning_model = Prompt.ask(
        "Planning model",
        default=settings.planning_model if provider == settings.planning_provider else default_planning,
        console=console,
    )
    coding_model = Prompt.ask(
        "Coding model",
        default=settings.coding_model if provider == settings.coding_provider else default_coding,
        console=console,
    )

    # --- Persist ---
    settings.planning_provider = provider
    settings.coding_provider = provider
    settings.planning_model = planning_model
    settings.coding_model = coding_model

    workspace.set_secret("ANTON_PLANNING_PROVIDER", provider)
    workspace.set_secret("ANTON_CODING_PROVIDER", provider)
    workspace.set_secret("ANTON_PLANNING_MODEL", planning_model)
    workspace.set_secret("ANTON_CODING_MODEL", coding_model)

    if api_key:
        setattr(settings, key_attr, api_key)
        key_name = f"ANTON_{provider.upper()}_API_KEY"
        workspace.set_secret(key_name, api_key)

    # Validate that we actually have an API key for the chosen provider
    final_key = getattr(settings, key_attr)
    if not final_key:
        console.print()
        console.print(f"[anton.error]No API key set for {provider}. Configuration not applied.[/]")
        console.print()
        return session

    console.print()
    console.print("[anton.success]Configuration updated.[/]")
    console.print()

    return _rebuild_session(
        settings=settings,
        state=state,
        self_awareness=self_awareness,
        workspace=workspace,
        console=console,
    )


def _format_file_message(text: str, paths: list[Path], console: Console) -> str:
    """Rewrite user input to include file contents for detected paths."""
    parts: list[str] = []

    # Determine what the user typed besides the paths
    remaining = text
    for p in paths:
        # Remove various representations of the path from the text
        for representation in (str(p), f"'{p}'", f'"{p}"', str(p).replace(" ", "\\ ")):
            remaining = remaining.replace(representation, "")
    remaining = remaining.strip()

    # Build the instruction
    if remaining:
        parts.append(remaining)
    else:
        if len(paths) == 1:
            parts.append(f"Analyze this file: {paths[0].name}")
        else:
            names = ", ".join(p.name for p in paths)
            parts.append(f"Analyze these files: {names}")

    # Attach each file
    for p in paths:
        suffix = p.suffix.lower()
        size = p.stat().st_size

        # Show what we're picking up
        console.print(f"  [anton.muted]attached: {p.name} ({_human_size(size)})[/]")

        # Skip very large files (>500KB) — just reference them
        if size > 512_000:
            parts.append(f"\n<file path=\"{p}\">\n(File too large to inline — {_human_size(size)}. "
                         f"Use the scratchpad to read it.)\n</file>")
            continue

        # Skip binary-looking files
        if suffix in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
                       ".pdf", ".zip", ".tar", ".gz", ".exe", ".dll", ".so",
                       ".pyc", ".pyo", ".whl", ".egg", ".db", ".sqlite"):
            parts.append(f"\n<file path=\"{p}\">\n(Binary file — {_human_size(size)}. "
                         f"Use the scratchpad to process it.)\n</file>")
            continue

        try:
            content = p.read_text(errors="replace")
        except Exception:
            parts.append(f"\n<file path=\"{p}\">\n(Could not read file.)\n</file>")
            continue

        parts.append(f"\n<file path=\"{p}\">\n{content}\n</file>")

    return "\n".join(parts)


def _format_clipboard_image_message(uploaded: object, user_text: str = "") -> list[dict]:
    """Build a multimodal LLM message for a clipboard image upload.

    Returns a list of content blocks (image + text) so the LLM can see
    the image directly. The file path is included so the LLM can pass
    it to the scratchpad if deeper processing is needed.
    """
    import base64

    text = user_text.strip() if user_text else "I've pasted an image from my clipboard. Analyze it."
    text += (
        f"\n\nThe image is also saved at: {uploaded.path}\n"
        f"({uploaded.width}x{uploaded.height}, {_human_size(uploaded.size_bytes)}). "
        f"If you need to process it programmatically, use that path in the scratchpad."
    )

    # Read and base64-encode the saved PNG
    image_data = Path(uploaded.path).read_bytes()
    b64 = base64.standard_b64encode(image_data).decode("ascii")

    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        },
        {
            "type": "text",
            "text": text,
        },
    ]


async def _ensure_clipboard(console: Console) -> bool:
    """Check clipboard support; offer to install Pillow if missing.

    Returns True if clipboard is ready to use, False otherwise.
    """
    reason = clipboard_unavailable_reason()
    if reason is None:
        return True
    if reason == "unsupported_platform":
        console.print("[anton.warning]Clipboard is not supported on this platform.[/]")
        return False
    # reason == "missing_pillow"
    console.print("[anton.muted]Clipboard image support requires Pillow.[/]")
    answer = console.input("[bold]Install Pillow now? (y/n):[/] ").strip().lower()
    if answer not in ("y", "yes"):
        console.print("[anton.muted]Skipped.[/]")
        return False
    console.print("[anton.muted]Installing Pillow...[/]")
    import subprocess
    proc = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable, "Pillow"],
            capture_output=True,
            timeout=120,
        ),
    )
    if proc.returncode == 0:
        console.print("[anton.success]Pillow installed. Clipboard is now available.[/]")
        return True
    else:
        # Fallback: try pip directly
        proc = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                [sys.executable, "-m", "pip", "install", "Pillow"],
                capture_output=True,
                timeout=120,
            ),
        )
        if proc.returncode == 0:
            console.print("[anton.success]Pillow installed. Clipboard is now available.[/]")
            return True
        console.print("[anton.error]Failed to install Pillow.[/]")
        return False


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.0f}{unit}" if unit == "B" else f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def _print_slash_help(console: Console) -> None:
    """Print available slash commands."""
    console.print()
    console.print("[anton.cyan]Available commands:[/]")
    console.print("  [bold]/setup[/]    — Configure provider, model, and API key")
    console.print("  [bold]/connect[/]  — Connect to a data source (guided credential collection)")
    console.print("  [bold]/paste[/]    — Attach clipboard image to your message")
    console.print("  [bold]/help[/]     — Show this help message")
    console.print("  [bold]exit[/]                  — Quit the chat")
    console.print()


def run_chat(console: Console, settings: AntonSettings) -> None:
    """Launch the interactive chat REPL."""
    asyncio.run(_chat_loop(console, settings))


async def _chat_loop(console: Console, settings: AntonSettings) -> None:
    from anton.connectors import ConnectorHub
    from anton.context.self_awareness import SelfAwarenessContext
    from anton.llm.client import LLMClient
    from anton.workspace import Workspace

    # Use a mutable container so closures always see the current client
    state: dict = {"llm_client": LLMClient.from_settings(settings)}

    # Self-awareness context
    self_awareness = SelfAwarenessContext(Path(settings.context_dir))

    # Workspace for anton.md and secret vault
    workspace = Workspace(settings.workspace_path)
    workspace.apply_env_to_process()

    # Clean up old clipboard uploads
    uploads_dir = Path(settings.workspace_path) / ".anton" / "uploads"
    cleanup_old_uploads(uploads_dir)

    # Build runtime context so the LLM knows what it's running on
    runtime_context = (
        f"- Provider: {settings.planning_provider}\n"
        f"- Planning model: {settings.planning_model}\n"
        f"- Coding model: {settings.coding_model}\n"
        f"- Workspace: {settings.workspace_path}\n"
        f"- Memory: {'enabled' if settings.memory_enabled else 'disabled'}\n"
        f"- Connector mode: {settings.connector_mode}"
    )

    coding_api_key = (
        settings.anthropic_api_key if settings.coding_provider == "anthropic"
        else settings.openai_api_key
    ) or ""
    connector_hub = ConnectorHub.from_settings(
        settings,
        workspace=Path(settings.workspace_path),
    )
    session = ChatSession(
        state["llm_client"],
        self_awareness=self_awareness,
        runtime_context=runtime_context,
        workspace=workspace,
        console=console,
        coding_provider=settings.coding_provider,
        coding_api_key=coding_api_key,
        connector_hub=connector_hub,
    )

    console.print("[anton.muted] Chat with Anton. Type '/help' for commands or 'exit' to quit.[/]")
    console.print(f"[anton.cyan_dim] {'━' * 40}[/]")
    console.print()

    from anton.chat_ui import StreamDisplay

    toolbar = {"stats": "", "status": ""}
    display = StreamDisplay(console, toolbar=toolbar)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI, HTML
    from prompt_toolkit.styles import Style as PTStyle

    def _bottom_toolbar():
        stats = toolbar["stats"]
        status = toolbar["status"]
        if not stats and not status:
            return ""
        width = os.get_terminal_size().columns
        gap = width - len(status) - len(stats)
        if gap < 1:
            gap = 1
        line = status + " " * gap + stats
        return HTML(f"\n<style fg='#555570'>{line}</style>")

    pt_style = PTStyle.from_dict({
        "bottom-toolbar": "noreverse nounderline bg:default",
    })

    prompt_session: PromptSession[str] = PromptSession(
        mouse_support=False,
        bottom_toolbar=_bottom_toolbar,
        style=pt_style,
    )

    try:
        while True:
            try:
                user_input = await prompt_session.prompt_async(ANSI("\033[1;38;2;0;255;159myou>\033[0m "))
            except EOFError:
                break

            stripped = user_input.strip()
            # message_content holds what we send to the LLM — may be str or
            # list[dict] (multimodal content blocks for images).
            message_content: str | list[dict] | None = None

            # Empty input → check clipboard for an image
            if not stripped:
                if is_clipboard_supported():
                    clip = grab_clipboard()
                    if clip.image:
                        uploaded = save_clipboard_image(clip.image.image, uploads_dir)
                        console.print(
                            f"  [anton.muted]attached: clipboard image "
                            f"({uploaded.width}x{uploaded.height}, "
                            f"{_human_size(uploaded.size_bytes)})[/]"
                        )
                        message_content = _format_clipboard_image_message(uploaded)
                    elif clip.file_paths:
                        stripped = _format_file_message("", clip.file_paths, console)
                if not stripped and message_content is None:
                    continue

            if message_content is None and stripped.lower() in ("exit", "quit", "bye"):
                break

            # Slash command dispatch
            if message_content is None and stripped.startswith("/"):
                parts = stripped.split(maxsplit=1)
                cmd = parts[0].lower()
                if cmd == "/setup":
                    session = await _handle_setup(
                        console, settings, workspace, state,
                        self_awareness, session,
                    )
                    continue
                elif cmd == "/help":
                    _print_slash_help(console)
                    continue
                elif cmd == "/paste":
                    if not await _ensure_clipboard(console):
                        continue
                    clip = grab_clipboard()
                    if clip.image:
                        uploaded = save_clipboard_image(clip.image.image, uploads_dir)
                        console.print(
                            f"  [anton.muted]attached: clipboard image "
                            f"({uploaded.width}x{uploaded.height}, "
                            f"{_human_size(uploaded.size_bytes)})[/]"
                        )
                        user_text = parts[1] if len(parts) > 1 else ""
                        message_content = _format_clipboard_image_message(uploaded, user_text)
                        # Fall through to turn_stream (don't continue)
                    else:
                        console.print("[anton.warning]No image found on clipboard.[/]")
                        continue
                elif cmd == "/connect":
                    user_text = parts[1] if len(parts) > 1 else ""
                    if user_text:
                        stripped = (
                            f"I want to connect to {user_text}. Ask me for the connection "
                            f"details I need to provide, collect them securely using "
                            f"request_secret, and then test the connection in the scratchpad."
                        )
                    else:
                        stripped = (
                            "I want to connect to a data source. Ask me what type "
                            "(PostgreSQL, MySQL, API, etc.), then collect the minimal "
                            "credentials needed using request_secret, and test the "
                            "connection in the scratchpad."
                        )
                    # Fall through to turn_stream
                else:
                    console.print(f"[anton.warning]Unknown command: {cmd}[/]")
                    continue

            # Detect dragged file paths and reformat the message
            if message_content is None:
                dropped = _parse_dropped_paths(stripped)
                if dropped:
                    stripped = _format_file_message(stripped, dropped, console)

            # Use multimodal content if set, otherwise the text string
            if message_content is None:
                message_content = stripped

            display.start()
            t0 = time.monotonic()
            ttft: float | None = None
            total_input = 0
            total_output = 0

            try:
                async for event in session.turn_stream(message_content):
                    if isinstance(event, StreamTextDelta):
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        display.append_text(event.text)
                    elif isinstance(event, StreamToolResult):
                        display.show_tool_result(event.content)
                    elif isinstance(event, StreamToolUseStart):
                        display.on_tool_use_start(event.id, event.name)
                    elif isinstance(event, StreamToolUseDelta):
                        display.on_tool_use_delta(event.id, event.json_delta)
                    elif isinstance(event, StreamToolUseEnd):
                        display.on_tool_use_end(event.id)
                    elif isinstance(event, StreamTaskProgress):
                        display.update_progress(
                            event.phase, event.message, event.eta_seconds
                        )
                    elif isinstance(event, StreamComplete):
                        total_input += event.response.usage.input_tokens
                        total_output += event.response.usage.output_tokens

                elapsed = time.monotonic() - t0
                parts = [f"{elapsed:.1f}s", f"{total_input} in / {total_output} out"]
                if ttft is not None:
                    parts.append(f"TTFT {int(ttft * 1000)}ms")
                toolbar["stats"] = "  ".join(parts)
                toolbar["status"] = ""
                display.finish()
            except anthropic.AuthenticationError:
                display.abort()
                console.print()
                console.print(
                    "[anton.error]Invalid API key. Let's set up a new one.[/]"
                )
                settings.anthropic_api_key = None
                from anton.cli import _ensure_api_key
                _ensure_api_key(settings)
                session = _rebuild_session(
                    settings=settings,
                    state=state,
                    self_awareness=self_awareness,
                    workspace=workspace,
                    console=console,
                )
            except KeyboardInterrupt:
                display.abort()
                console.print()
                break
            except Exception as exc:
                display.abort()
                console.print(f"[anton.error]Error: {exc}[/]")
                console.print()
    except KeyboardInterrupt:
        pass

    console.print()
    console.print("[anton.muted]See you.[/]")
    await session.close()
