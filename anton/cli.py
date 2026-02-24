from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from anton import __version__

app = typer.Typer(
    name="anton",
    help="Anton — a self-evolving autonomous system",
)


def _make_console() -> Console:
    from anton.channel.theme import build_rich_theme, detect_color_mode

    mode = detect_color_mode()
    return Console(theme=build_rich_theme(mode))


console = _make_console()


def _get_settings(ctx: typer.Context):
    """Retrieve the resolved AntonSettings from context."""
    return ctx.obj["settings"]


def _ensure_workspace(settings) -> None:
    """Check workspace state and initialize if needed."""
    from anton.workspace import Workspace

    ws = Workspace(settings.workspace_path)

    # Apply existing .env variables to process
    ws.apply_env_to_process()

    if ws.is_initialized():
        return

    if ws.needs_confirmation():
        console.print()
        console.print(
            "[anton.warning]This folder already contains files that aren't part of Anton.[/]"
        )
        console.print(f"[dim]  Folder: {settings.workspace_path}[/]")
        console.print()
        if not Confirm.ask(
            "Initialize Anton workspace here?",
            default=True,
            console=console,
        ):
            raise typer.Exit(0)

    actions = ws.initialize()
    for action in actions:
        console.print(f"[anton.muted]  {action}[/]")
    if actions:
        console.print()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    folder: str | None = typer.Option(
        None, "--folder", "-f", help="Workspace folder (defaults to cwd)"
    ),
) -> None:
    """Anton — a self-evolving autonomous system."""
    from anton.config.settings import AntonSettings

    settings = AntonSettings()
    settings.resolve_workspace(folder)

    from anton.updater import check_and_update
    check_and_update(console, settings)

    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings

    if ctx.invoked_subcommand is None:
        from anton.channel.branding import render_banner
        from anton.chat import run_chat

        render_banner(console)
        _ensure_workspace(settings)
        _ensure_api_key(settings)
        run_chat(console, settings)


def _has_api_key(settings) -> bool:
    """Check if all configured providers have API keys."""
    providers = {settings.planning_provider, settings.coding_provider}
    for p in providers:
        if p == "anthropic" and not (settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")):
            return False
        if p in ("openai", "openai-compatible") and not (settings.openai_api_key or os.environ.get("OPENAI_API_KEY")):
            return False
    return True


def _ensure_api_key(settings) -> None:
    """Prompt the user to configure a provider and API key if none is set."""
    if _has_api_key(settings):
        return

    console.print()
    console.print("[anton.warning]No API key configured.[/]")
    console.print()

    providers = {"1": "anthropic", "2": "openai", "3": "openai-compatible"}
    console.print("[anton.cyan]Available providers:[/]")
    console.print(r"  [bold]1[/]  Anthropic (Claude)                    [dim]\[recommended][/]")
    console.print(r"  [bold]2[/]  OpenAI (GPT / o-series)               [dim]\[experimental][/]")
    console.print(r"  [bold]3[/]  OpenAI-compatible (custom endpoint)   [dim]\[experimental][/]")
    console.print()

    choice = Prompt.ask(
        "Select provider",
        choices=list(providers.keys()),
        default="1",
        console=console,
    )
    provider = providers[choice]

    console.print()

    # Use the workspace secret vault to store the key securely
    from anton.workspace import Workspace

    ws = Workspace(settings.workspace_path)

    # For OpenAI-compatible, ask for the base URL first
    if provider == "openai-compatible":
        base_url = Prompt.ask(
            "Enter the API base URL (e.g. http://localhost:11434/v1)",
            console=console,
        )
        if not base_url.strip():
            console.print("[anton.error]No base URL provided. Exiting.[/]")
            raise typer.Exit(1)
        base_url = base_url.strip()
        settings.openai_base_url = base_url
        ws.set_secret("ANTON_OPENAI_BASE_URL", base_url)
        console.print()

    api_key = Prompt.ask(
        f"Enter your API key",
        console=console,
    )

    if not api_key.strip():
        console.print("[anton.error]No API key provided. Exiting.[/]")
        raise typer.Exit(1)

    api_key = api_key.strip()
    key_name = "ANTON_OPENAI_API_KEY" if provider in ("openai", "openai-compatible") else "ANTON_ANTHROPIC_API_KEY"

    # Store via secret vault — never passes through LLM
    ws.set_secret(key_name, api_key)

    # Apply to current process and set provider config
    if provider == "anthropic":
        settings.anthropic_api_key = api_key
    elif provider == "openai":
        settings.openai_api_key = api_key
        settings.planning_provider = "openai"
        settings.coding_provider = "openai"
        settings.planning_model = "gpt-5-mini"
        settings.coding_model = "gpt-5-nano"
        ws.set_secret("ANTON_PLANNING_PROVIDER", "openai")
        ws.set_secret("ANTON_CODING_PROVIDER", "openai")
        ws.set_secret("ANTON_PLANNING_MODEL", "gpt-5-mini")
        ws.set_secret("ANTON_CODING_MODEL", "gpt-5-nano")
    elif provider == "openai-compatible":
        settings.openai_api_key = api_key
        settings.planning_provider = "openai-compatible"
        settings.coding_provider = "openai-compatible"

        console.print()
        planning_model = Prompt.ask("Planning model", console=console)
        coding_model = Prompt.ask("Coding model", console=console)

        settings.planning_model = planning_model
        settings.coding_model = coding_model
        ws.set_secret("ANTON_PLANNING_PROVIDER", "openai-compatible")
        ws.set_secret("ANTON_CODING_PROVIDER", "openai-compatible")
        ws.set_secret("ANTON_PLANNING_MODEL", planning_model)
        ws.set_secret("ANTON_CODING_MODEL", coding_model)

    console.print()
    console.print(f"[anton.success]Saved to {ws.env_path}[/]")
    console.print()


@app.command("setup")
def setup(ctx: typer.Context) -> None:
    """Configure provider, model, and API key."""
    settings = _get_settings(ctx)
    _ensure_workspace(settings)
    _ensure_api_key(settings)
    console.print("[anton.success]Setup complete.[/]")


@app.command("dashboard")
def dashboard() -> None:
    """Show the Anton status dashboard."""
    from anton.channel.branding import render_dashboard

    render_dashboard(console)


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


@app.command("version")
def version() -> None:
    """Show Anton version."""
    console.print(f"Anton v{__version__}")
