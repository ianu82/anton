from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import anthropic

from anton.llm.prompts import CHAT_SYSTEM_PROMPT

if TYPE_CHECKING:
    from rich.console import Console

    from anton.config.settings import AntonSettings
    from anton.llm.client import LLMClient

EXECUTE_TASK_TOOL = {
    "name": "execute_task",
    "description": (
        "Execute a coding task autonomously through Anton's agent pipeline. "
        "Call this when you have enough context to act on the user's request."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "A clear, specific description of the task to execute.",
            },
        },
        "required": ["task"],
    },
}


class ChatSession:
    """Manages a multi-turn conversation with tool-call delegation."""

    def __init__(self, llm_client: LLMClient, run_task) -> None:
        self._llm = llm_client
        self._run_task = run_task
        self._history: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return self._history

    async def turn(self, user_input: str) -> str:
        self._history.append({"role": "user", "content": user_input})

        response = await self._llm.plan(
            system=CHAT_SYSTEM_PROMPT,
            messages=self._history,
            tools=[EXECUTE_TASK_TOOL],
        )

        # Handle tool calls (execute_task)
        while response.tool_calls:
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

            # Process each tool call
            tool_results: list[dict] = []
            for tc in response.tool_calls:
                if tc.name == "execute_task":
                    task_desc = tc.input.get("task", "")
                    try:
                        await self._run_task(task_desc)
                        result_text = f"Task completed: {task_desc}"
                    except Exception as exc:
                        result_text = f"Task failed: {exc}"
                else:
                    result_text = f"Unknown tool: {tc.name}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_text,
                })

            self._history.append({"role": "user", "content": tool_results})

            # Get follow-up from LLM
            response = await self._llm.plan(
                system=CHAT_SYSTEM_PROMPT,
                messages=self._history,
                tools=[EXECUTE_TASK_TOOL],
            )

        # Text-only response
        reply = response.content or ""
        self._history.append({"role": "assistant", "content": reply})
        return reply


def run_chat(console: Console, settings: AntonSettings) -> None:
    """Launch the interactive chat REPL."""
    asyncio.run(_chat_loop(console, settings))


async def _chat_loop(console: Console, settings: AntonSettings) -> None:
    from pathlib import Path

    from anton.channel.terminal import CLIChannel
    from anton.core.agent import Agent
    from anton.llm.client import LLMClient
    from anton.skill.registry import SkillRegistry

    # Use a mutable container so closures always see the current client
    state: dict = {"llm_client": LLMClient.from_settings(settings)}
    registry = SkillRegistry()

    builtin = Path(__file__).resolve().parent.parent / settings.skills_dir
    registry.discover(builtin)

    user_dir = Path(settings.user_skills_dir).expanduser()
    registry.discover(user_dir)

    memory = None
    learnings_store = None
    if settings.memory_enabled:
        from anton.memory.learnings import LearningStore
        from anton.memory.store import SessionStore

        memory_dir = Path(settings.memory_dir).expanduser()
        memory = SessionStore(memory_dir)
        learnings_store = LearningStore(memory_dir)

    channel = CLIChannel()

    async def _do_run_task(task: str) -> None:
        agent = Agent(
            channel=channel,
            llm_client=state["llm_client"],
            registry=registry,
            user_skills_dir=user_dir,
            memory=memory,
            learnings=learnings_store,
        )
        await agent.run(task)

    session = ChatSession(state["llm_client"], _do_run_task)

    console.print("[anton.muted]Chat with Anton. Type 'exit' to quit.[/]")
    console.print()

    try:
        while True:
            try:
                user_input = console.input("[bold]you>[/] ")
            except EOFError:
                break

            stripped = user_input.strip()
            if not stripped:
                continue
            if stripped.lower() in ("exit", "quit", "bye"):
                break

            try:
                reply = await session.turn(stripped)
                console.print(f"[anton.cyan]anton>[/] {reply}")
                console.print()
            except anthropic.AuthenticationError:
                console.print()
                console.print(
                    "[anton.error]Invalid API key. Let's set up a new one.[/]"
                )
                # Clear the bad key so _ensure_api_key prompts again
                settings.anthropic_api_key = None
                from anton.cli import _ensure_api_key
                _ensure_api_key(settings)
                # Rebuild the client and session with the new key
                state["llm_client"] = LLMClient.from_settings(settings)
                session = ChatSession(state["llm_client"], _do_run_task)
            except KeyboardInterrupt:
                console.print()
                break
            except Exception as exc:
                console.print(f"[anton.error]Error: {exc}[/]")
                console.print()
    except KeyboardInterrupt:
        pass

    console.print()
    console.print("[anton.muted]See you.[/]")
    await channel.close()
