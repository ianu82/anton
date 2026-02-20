from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from anton import __version__

app = typer.Typer(
    name="anton",
    help="Anton — autonomous coding copilot",
)


def _make_console() -> Console:
    from anton.channel.theme import build_rich_theme, detect_color_mode

    mode = detect_color_mode()
    return Console(theme=build_rich_theme(mode))


console = _make_console()


def _get_settings(ctx: typer.Context):
    """Retrieve the resolved AntonSettings from context."""
    return ctx.obj["settings"]


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    folder: str | None = typer.Option(
        None, "--folder", "-f", help="Workspace folder (defaults to cwd)"
    ),
) -> None:
    """Anton — autonomous coding copilot."""
    from anton.config.settings import AntonSettings

    settings = AntonSettings()
    settings.resolve_workspace(folder)

    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings

    if ctx.invoked_subcommand is None:
        from anton.channel.branding import render_banner
        from anton.chat import run_chat

        render_banner(console)
        _ensure_api_key(settings)
        run_chat(console, settings)


def _has_api_key(settings) -> bool:
    """Check if any API key is available."""
    if settings.anthropic_api_key:
        return True
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    return False


def _ensure_api_key(settings) -> None:
    """Prompt the user to configure a provider and API key if none is set."""
    if _has_api_key(settings):
        return

    console.print()
    console.print("[anton.warning]No API key configured.[/]")
    console.print()

    providers = {"1": "anthropic"}
    console.print("[anton.cyan]Available providers:[/]")
    console.print("  [bold]1[/]  Anthropic (Claude)")
    console.print()

    choice = Prompt.ask(
        "Select provider",
        choices=list(providers.keys()),
        default="1",
        console=console,
    )
    provider = providers[choice]

    console.print()
    api_key = Prompt.ask(
        f"Enter your {provider.title()} API key",
        console=console,
    )

    if not api_key.strip():
        console.print("[anton.error]No API key provided. Exiting.[/]")
        raise typer.Exit(1)

    api_key = api_key.strip()

    # Save to workspace .anton/.env for persistence
    env_dir = Path(settings.workspace_path) / ".anton"
    env_dir.mkdir(parents=True, exist_ok=True)
    env_file = env_dir / ".env"

    lines: list[str] = []
    key_name = f"ANTON_{provider.upper()}_API_KEY"
    replaced = False
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith(f"{key_name}="):
                lines.append(f"{key_name}={api_key}")
                replaced = True
            else:
                lines.append(line)
    if not replaced:
        lines.append(f"{key_name}={api_key}")

    env_file.write_text("\n".join(lines) + "\n")

    # Apply to current process so this run works
    os.environ[key_name] = api_key
    if provider == "anthropic":
        settings.anthropic_api_key = api_key

    console.print()
    console.print(f"[anton.success]Saved to {env_file}[/]")
    console.print()


@app.command("dashboard")
def dashboard() -> None:
    """Show the Anton status dashboard."""
    from anton.channel.branding import render_dashboard

    render_dashboard(console)


@app.command()
def run(
    ctx: typer.Context,
    task: str = typer.Argument(..., help="The task for Anton to complete"),
) -> None:
    """Give Anton a task and let it work autonomously."""
    from anton.channel.branding import render_banner

    settings = _get_settings(ctx)
    render_banner(console)
    asyncio.run(_run_task(task, settings))


async def _run_task(task: str, settings=None) -> None:
    from anton.channel.terminal import CLIChannel
    from anton.config.settings import AntonSettings
    from anton.context.self_awareness import SelfAwarenessContext
    from anton.core.agent import Agent
    from anton.llm.client import LLMClient
    from anton.skill.registry import SkillRegistry

    if settings is None:
        settings = AntonSettings()
        settings.resolve_workspace()

    _ensure_api_key(settings)
    channel = CLIChannel()

    try:
        llm_client = LLMClient.from_settings(settings)
        registry = SkillRegistry()

        # Discover built-in skills
        builtin = Path(__file__).resolve().parent.parent / settings.skills_dir
        registry.discover(builtin)

        # Discover user skills
        user_dir = Path(settings.user_skills_dir)
        registry.discover(user_dir)

        # Set up memory if enabled
        memory = None
        learnings_store = None
        if settings.memory_enabled:
            from anton.memory.learnings import LearningStore
            from anton.memory.store import SessionStore

            memory_dir = Path(settings.memory_dir)
            memory = SessionStore(memory_dir)
            learnings_store = LearningStore(memory_dir)

        # Self-awareness context
        self_awareness = SelfAwarenessContext(Path(settings.context_dir))

        agent = Agent(
            channel=channel,
            llm_client=llm_client,
            registry=registry,
            user_skills_dir=user_dir,
            memory=memory,
            learnings=learnings_store,
            self_awareness=self_awareness,
            skill_dirs=[builtin, user_dir],
        )
        await agent.run(task)
    finally:
        await channel.close()


@app.command("skills")
def list_skills(ctx: typer.Context) -> None:
    """List all discovered skills."""
    from anton.skill.registry import SkillRegistry

    settings = _get_settings(ctx)
    registry = SkillRegistry()

    builtin = Path(__file__).resolve().parent.parent / settings.skills_dir
    registry.discover(builtin)

    user_dir = Path(settings.user_skills_dir)
    registry.discover(user_dir)

    skills = registry.list_all()
    if not skills:
        console.print("[dim]No skills discovered.[/]")
        return

    table = Table(title="Discovered Skills")
    table.add_column("Name", style="anton.cyan")
    table.add_column("Description")
    table.add_column("Parameters")

    for s in skills:
        params = ", ".join(
            f"{k}: {v.get('type', '?')}"
            for k, v in s.parameters.get("properties", {}).items()
        )
        table.add_row(s.name, s.description, params or "—")

    console.print(table)


@app.command("sessions")
def list_sessions(ctx: typer.Context) -> None:
    """List recent sessions."""
    from anton.memory.store import SessionStore

    settings = _get_settings(ctx)
    memory_dir = Path(settings.memory_dir)
    store = SessionStore(memory_dir)

    sessions = store.list_sessions()
    if not sessions:
        console.print("[dim]No sessions found.[/]")
        return

    table = Table(title="Recent Sessions")
    table.add_column("ID", style="anton.cyan")
    table.add_column("Task")
    table.add_column("Status")
    table.add_column("Summary")

    for s in sessions:
        preview = s.get("summary_preview") or ""
        if len(preview) > 60:
            preview = preview[:60] + "..."
        table.add_row(s["id"], s.get("task", "")[:50], s.get("status", ""), preview)

    console.print(table)


@app.command("session")
def show_session(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="Session ID to display"),
) -> None:
    """Show session details and summary."""
    from anton.memory.store import SessionStore

    settings = _get_settings(ctx)
    memory_dir = Path(settings.memory_dir)
    store = SessionStore(memory_dir)

    session = store.get_session(session_id)
    if session is None:
        console.print(f"[red]Session {session_id} not found.[/]")
        raise typer.Exit(1)

    console.print(f"[bold]Session:[/] {session['id']}")
    console.print(f"[bold]Task:[/] {session.get('task', 'N/A')}")
    console.print(f"[bold]Status:[/] {session.get('status', 'N/A')}")

    summary = session.get("summary")
    if summary:
        console.print(f"\n[bold]Summary:[/]\n{summary}")


@app.command("learnings")
def list_learnings(ctx: typer.Context) -> None:
    """List all learnings with summaries."""
    from anton.memory.learnings import LearningStore

    settings = _get_settings(ctx)
    memory_dir = Path(settings.memory_dir)
    store = LearningStore(memory_dir)

    items = store.list_all()
    if not items:
        console.print("[dim]No learnings recorded yet.[/]")
        return

    table = Table(title="Learnings")
    table.add_column("Topic", style="anton.cyan")
    table.add_column("Summary")

    for item in items:
        table.add_row(item["topic"], item["summary"])

    console.print(table)


@app.command("channels")
def list_channels() -> None:
    """List available communication channels."""
    from anton.channel.registry import ChannelRegistry
    from anton.channel.terminal import CLIChannel, TerminalChannel
    from anton.channel.types import ChannelCapability, ChannelInfo, ChannelMeta

    registry = ChannelRegistry()
    registry.register(
        ChannelInfo(
            meta=ChannelMeta(
                id="cli",
                label="Terminal CLI",
                description="Rich interactive terminal",
                icon="\U0001f4bb",
                capabilities=[
                    ChannelCapability.TEXT_OUTPUT,
                    ChannelCapability.TEXT_INPUT,
                    ChannelCapability.INTERACTIVE,
                    ChannelCapability.RICH_FORMATTING,
                ],
                aliases=["terminal", "term"],
            ),
            factory=CLIChannel,
        )
    )

    channels = registry.list_all()
    if not channels:
        console.print("[dim]No channels registered.[/]")
        return

    table = Table(title="Channels")
    table.add_column("ID", style="anton.cyan")
    table.add_column("Label")
    table.add_column("Description")
    table.add_column("Capabilities")

    for ch in channels:
        caps = ", ".join(c.value for c in ch.meta.capabilities)
        table.add_row(
            f"{ch.meta.icon}  {ch.meta.id}" if ch.meta.icon else ch.meta.id,
            ch.meta.label,
            ch.meta.description,
            caps,
        )

    console.print(table)


@app.command("minion")
def minion_cmd(
    task: str = typer.Argument(..., help="Task for the minion to execute"),
    folder: str | None = typer.Option(
        None, "--folder", "-f", help="Workspace folder for the minion"
    ),
) -> None:
    """Spawn a minion to work on a task in a given folder."""
    console.print("[dim]Minion system is not yet implemented.[/]")
    console.print(f"[dim]  task: {task}[/]")
    if folder:
        console.print(f"[dim]  folder: {folder}[/]")


@app.command("minions")
def list_minions() -> None:
    """List tracked minions."""
    console.print("[dim]No minions tracked.[/]")


@app.command("version")
def version() -> None:
    """Show Anton version."""
    console.print(f"Anton v{__version__}")
