from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from anton import __version__


# ---------------------------------------------------------------------------
# Dependency checking — runs before anything that needs the heavy imports
# ---------------------------------------------------------------------------

# Core dependencies from pyproject.toml that anton needs at runtime
_REQUIRED_PACKAGES: dict[str, str] = {
    "anthropic": "anthropic>=0.42.0",
    "openai": "openai>=1.0",
    "pydantic": "pydantic>=2.0",
    "pydantic_settings": "pydantic-settings>=2.0",
    "prompt_toolkit": "prompt-toolkit>=3.0",
}
# typer and rich are already imported above — if they were missing we'd
# never reach this point, so no need to check them.


def _check_dependencies() -> list[str]:
    """Return list of missing package install specs."""
    import importlib

    missing: list[str] = []
    for module_name, install_spec in _REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(install_spec)
    return missing


def _find_uv() -> str | None:
    """Find the uv binary."""
    import shutil

    uv = shutil.which("uv")
    if uv:
        return uv

    if sys.platform == "win32":
        candidates = (
            os.path.expanduser("~/.local/bin/uv.exe"),
            os.path.expanduser("~/.cargo/bin/uv.exe"),
        )
    else:
        candidates = (
            os.path.expanduser("~/.local/bin/uv"),
            os.path.expanduser("~/.cargo/bin/uv"),
        )

    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _ensure_dependencies(console: Console) -> None:
    """Check for missing dependencies and offer to install them."""
    missing = _check_dependencies()
    if not missing:
        return

    console.print()
    console.print("[anton.warning]Missing dependencies detected:[/]")
    for pkg in missing:
        console.print(f"  [bold]- {pkg}[/]")
    console.print()

    # Check if install script is available locally (dev checkout)
    repo_root = Path(__file__).resolve().parent.parent
    if sys.platform == "win32":
        install_script = repo_root / "install.ps1"
    else:
        install_script = repo_root / "install.sh"
    uv = _find_uv()

    if uv:
        if Confirm.ask(
            f"Install missing packages with uv?",
            default=True,
            console=console,
        ):
            import subprocess

            console.print(f"[anton.muted]  Running: uv pip install {' '.join(missing)}[/]")
            result = subprocess.run(
                [uv, "pip", "install", "--python", sys.executable, *missing],
                capture_output=True,
            )
            if result.returncode == 0:
                console.print("[anton.success]  Dependencies installed.[/]")
                console.print("[anton.muted]  Please restart anton.[/]")
            else:
                console.print(f"[anton.error]  Install failed:[/]")
                console.print(result.stderr.decode() if result.stderr else result.stdout.decode())
                if install_script.is_file():
                    if sys.platform == "win32":
                        console.print(f"\n[anton.muted]  Or run the install script: powershell -File {install_script}[/]")
                    else:
                        console.print(f"\n[anton.muted]  Or run the install script: sh {install_script}[/]")
            raise typer.Exit(0)
    elif install_script.is_file():
        console.print(f"To install all dependencies, run:")
        if sys.platform == "win32":
            console.print(f"  [bold]powershell -File {install_script}[/]")
        else:
            console.print(f"  [bold]sh {install_script}[/]")
        console.print()
        raise typer.Exit(1)
    else:
        console.print("To install missing dependencies, run:")
        console.print(f"  [bold]pip install {' '.join(missing)}[/]")
        console.print()
        if sys.platform == "win32":
            console.print("[anton.muted]Or reinstall anton: irm https://raw.githubusercontent.com/mindsdb/anton/main/install.ps1 | iex[/]")
        else:
            console.print("[anton.muted]Or reinstall anton: curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh[/]")
        console.print()
        raise typer.Exit(1)

app = typer.Typer(
    name="anton",
    help="Anton — a self-evolving autonomous system",
)
skills_app = typer.Typer(help="Manage service skills (requires anton serve).")
schedules_app = typer.Typer(help="Manage scheduled runs (requires anton serve).")


def _make_console() -> Console:
    from anton.channel.theme import build_rich_theme, detect_color_mode

    mode = detect_color_mode()
    return Console(theme=build_rich_theme(mode))


console = _make_console()


def _service_base_url(service_url: str | None) -> str:
    url = service_url or os.environ.get("ANTON_SERVICE_BASE_URL", "http://127.0.0.1:8000")
    return url.rstrip("/")


def _parse_json_object(raw: str, label: str) -> dict[str, str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        console.print(f"[anton.error]Invalid JSON for {label}: {exc}[/]")
        raise typer.Exit(1) from exc
    if not isinstance(parsed, dict):
        console.print(f"[anton.error]{label} must be a JSON object.[/]")
        raise typer.Exit(1)
    normalized: dict[str, str] = {}
    for key, value in parsed.items():
        normalized[str(key)] = str(value)
    return normalized


def _service_request(
    *,
    method: str,
    base_url: str,
    path: str,
    payload: dict | None = None,
    query: dict[str, str | int | None] | None = None,
) -> dict:
    query_params = query or {}
    compact_query = {k: v for k, v in query_params.items() if v is not None}
    query_str = urllib.parse.urlencode(compact_query, doseq=True)
    url = f"{base_url}{path}"
    if query_str:
        url = f"{url}?{query_str}"

    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method.upper())

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.reason
        body = exc.read().decode("utf-8")
        if body:
            try:
                parsed = json.loads(body)
                detail = parsed.get("detail", detail)
            except json.JSONDecodeError:
                detail = body
        console.print(f"[anton.error]Service request failed ({exc.code}): {detail}[/]")
        raise typer.Exit(1) from exc
    except urllib.error.URLError as exc:
        console.print(f"[anton.error]Could not reach service at {base_url}: {exc.reason}[/]")
        raise typer.Exit(1) from exc


def _get_settings(ctx: typer.Context):
    """Retrieve the resolved AntonSettings from context."""
    return ctx.obj["settings"]


def _ensure_workspace(settings) -> None:
    """Check workspace state and initialize if needed.

    Boot logic:
    1. If $PWD/.anton exists → use it (local project), boot straight away
    2. If $HOME/.anton exists → use it (global project), boot straight away
    3. Neither exists → ask user: new local project or global project
    """
    from anton.workspace import Workspace

    local_path = settings.workspace_path
    global_path = Path.home()

    local_ws = Workspace(local_path)
    global_ws = Workspace(global_path)

    # 1. Local .anton exists → use it
    if local_ws.is_initialized():
        local_ws.apply_env_to_process()
        return

    # 2. Global ~/.anton exists and we're not already pointing at $HOME → use it
    if local_path != global_path and global_ws.is_initialized():
        settings.resolve_workspace(str(global_path))
        global_ws.apply_env_to_process()
        return

    # 3. Neither exists → ask user
    console.print()
    cwd_display = str(local_path)
    console.print("[anton.cyan]Where should Anton store its data?[/]")
    console.print(f"  [bold]1[/]  This folder  [dim]({cwd_display}/.anton)[/]")
    console.print(f"  [bold]2[/]  Global        [dim](~/.anton)[/]")
    console.print()
    choice = Prompt.ask(
        "Select",
        choices=["1", "2"],
        default="1",
        console=console,
    )

    if choice == "1":
        console.print(f"[anton.muted]  Creating project workspace in {cwd_display}/.anton[/]")
        ws = local_ws
    else:
        console.print(f"[anton.muted]  Creating global workspace in ~/.anton[/]")
        settings.resolve_workspace(str(global_path))
        ws = global_ws

    actions = ws.initialize()
    for action in actions:
        console.print(f"[anton.muted]  {action}[/]")
    ws.apply_env_to_process()
    console.print()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    folder: str | None = typer.Option(
        None, "--folder", "-f", help="Workspace folder (defaults to cwd)"
    ),
) -> None:
    """Anton — a self-evolving autonomous system."""
    _ensure_dependencies(console)

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


@app.command("serve")
def serve(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Run the Anton MVP service API."""
    settings = _get_settings(ctx)
    _ensure_workspace(settings)
    _ensure_api_key(settings)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - import guard
        console.print("[anton.error]Missing service dependencies.[/]")
        console.print("[anton.muted]Install with: pip install 'anton[service]'[/]")
        raise typer.Exit(1) from exc

    from anton.service import create_app

    app_instance = create_app(settings)
    uvicorn.run(app_instance, host=host, port=port, reload=reload)


@app.command("eval-mvp")
def eval_mvp(
    ctx: typer.Context,
    tasks: str = typer.Option(..., "--tasks", "-t", help="Path to JSON task suite"),
    output: str = typer.Option(".anton/eval/summary.json", "--output", "-o", help="Path to write summary JSON"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace root for evaluation runs"),
) -> None:
    """Run the MVP benchmark suite and write a summary report."""
    settings = _get_settings(ctx)
    _ensure_workspace(settings)
    _ensure_api_key(settings)

    from anton.service.eval import run_eval_sync

    tasks_path = Path(tasks).expanduser().resolve()
    if not tasks_path.exists():
        console.print(f"[anton.error]Task file not found: {tasks_path}[/]")
        raise typer.Exit(1)

    output_path = Path(output).expanduser().resolve()
    workspace_root = Path(workspace).expanduser().resolve() if workspace else None
    summary = run_eval_sync(
        settings=settings,
        tasks_path=tasks_path,
        output_path=output_path,
        workspace_root=workspace_root,
    )

    console.print("[anton.success]Evaluation complete.[/]")
    console.print(f"  Runs: {summary['run_count']}")
    console.print(f"  Passed: {summary['passed_count']}")
    console.print(f"  Pass rate: {summary['pass_rate']:.2%}")
    console.print(f"  p50 latency: {summary['latency_p50']:.2f}s")
    console.print(f"  p95 latency: {summary['latency_p95']:.2f}s")
    console.print(f"  Report: {output_path}")


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


@skills_app.command("list")
def skills_list(
    service_url: str | None = typer.Option(None, "--service-url", help="Anton service base URL"),
    limit: int = typer.Option(200, "--limit", help="Max skills to return"),
) -> None:
    """List skills from the Anton service."""
    base = _service_base_url(service_url)
    resp = _service_request(method="GET", base_url=base, path="/skills", query={"limit": limit})
    skills = resp.get("skills", [])
    if not skills:
        console.print("[dim]No skills found.[/]")
        return

    table = Table(title="Skills")
    table.add_column("ID", style="anton.cyan")
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Description")
    for skill in skills:
        table.add_row(
            str(skill.get("id", "")),
            str(skill.get("name", "")),
            str(skill.get("latest_version", "")),
            str(skill.get("description", "")),
        )
    console.print(table)


@skills_app.command("show")
def skills_show(
    skill_id: str = typer.Argument(..., help="Skill ID"),
    service_url: str | None = typer.Option(None, "--service-url", help="Anton service base URL"),
) -> None:
    """Show skill details."""
    base = _service_base_url(service_url)
    skill = _service_request(method="GET", base_url=base, path=f"/skills/{skill_id}")
    console.print_json(data=skill)


@skills_app.command("create")
def skills_create(
    name: str = typer.Argument(..., help="Skill name"),
    prompt_template: str = typer.Argument(..., help="Prompt template (e.g. 'Show {metric} for {period}.')"),
    description: str = typer.Option("", "--description", help="Skill description"),
    metadata: str = typer.Option("{}", "--metadata", help="Skill metadata as JSON object"),
    service_url: str | None = typer.Option(None, "--service-url", help="Anton service base URL"),
) -> None:
    """Create a new skill."""
    base = _service_base_url(service_url)
    metadata_obj = _parse_json_object(metadata, "metadata")
    created = _service_request(
        method="POST",
        base_url=base,
        path="/skills",
        payload={
            "name": name,
            "description": description,
            "prompt_template": prompt_template,
            "metadata": metadata_obj,
        },
    )
    console.print(f"[anton.success]Created skill {created.get('id')} ({created.get('name')}).[/]")
    console.print_json(data=created)


@skills_app.command("version")
def skills_version(
    skill_id: str = typer.Argument(..., help="Skill ID"),
    prompt_template: str = typer.Argument(..., help="New prompt template"),
    service_url: str | None = typer.Option(None, "--service-url", help="Anton service base URL"),
) -> None:
    """Add a new version to an existing skill."""
    base = _service_base_url(service_url)
    updated = _service_request(
        method="POST",
        base_url=base,
        path=f"/skills/{skill_id}/versions",
        payload={"prompt_template": prompt_template},
    )
    console.print(f"[anton.success]Updated skill {skill_id} to version {updated.get('latest_version')}.[/]")
    console.print_json(data=updated)


@skills_app.command("run")
def skills_run(
    skill_id: str = typer.Argument(..., help="Skill ID"),
    session_id: str = typer.Option(..., "--session-id", help="Target session ID"),
    params: str = typer.Option("{}", "--params", help="Skill params as JSON object"),
    version: int | None = typer.Option(None, "--version", help="Skill version (defaults to latest)"),
    idempotency_key: str | None = typer.Option(None, "--idempotency-key", help="Optional idempotency key"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for run completion"),
    wait_timeout_seconds: float = typer.Option(300.0, "--wait-timeout-seconds", help="Run wait timeout"),
    service_url: str | None = typer.Option(None, "--service-url", help="Anton service base URL"),
) -> None:
    """Run a skill for a session."""
    base = _service_base_url(service_url)
    params_obj = _parse_json_object(params, "params")
    result = _service_request(
        method="POST",
        base_url=base,
        path=f"/skills/{skill_id}/run",
        payload={
            "session_id": session_id,
            "params": params_obj,
            "version": version,
            "idempotency_key": idempotency_key,
            "wait_for_completion": wait,
            "wait_timeout_seconds": wait_timeout_seconds,
        },
    )
    console.print(
        f"[anton.success]Skill run {result.get('run_id')} status={result.get('status')} "
        f"version={result.get('skill_version')}[/]"
    )
    reply = str(result.get("reply", "") or "")
    if reply:
        console.print(f"\n[bold]Reply:[/]\n{reply}")
    console.print_json(data=result)


@schedules_app.command("list")
def schedules_list(
    service_url: str | None = typer.Option(None, "--service-url", help="Anton service base URL"),
    status: str | None = typer.Option(None, "--status", help="Optional status filter: active or paused"),
    limit: int = typer.Option(200, "--limit", help="Max schedules to return"),
) -> None:
    """List scheduled runs."""
    base = _service_base_url(service_url)
    resp = _service_request(
        method="GET",
        base_url=base,
        path="/scheduled-runs",
        query={"status": status, "limit": limit},
    )
    schedules = resp.get("scheduled_runs", [])
    if not schedules:
        console.print("[dim]No schedules found.[/]")
        return

    table = Table(title="Scheduled Runs")
    table.add_column("ID", style="anton.cyan")
    table.add_column("Name")
    table.add_column("Session")
    table.add_column("Skill")
    table.add_column("Status")
    table.add_column("Every(s)")
    table.add_column("Next Run")
    for item in schedules:
        table.add_row(
            str(item.get("id", "")),
            str(item.get("name", "")),
            str(item.get("session_id", "")),
            str(item.get("skill_id", "")),
            str(item.get("status", "")),
            str(item.get("interval_seconds", "")),
            str(item.get("next_run_at", "")),
        )
    console.print(table)


@schedules_app.command("show")
def schedules_show(
    schedule_id: str = typer.Argument(..., help="Schedule ID"),
    service_url: str | None = typer.Option(None, "--service-url", help="Anton service base URL"),
) -> None:
    """Show one scheduled run."""
    base = _service_base_url(service_url)
    schedule = _service_request(method="GET", base_url=base, path=f"/scheduled-runs/{schedule_id}")
    console.print_json(data=schedule)


@schedules_app.command("create")
def schedules_create(
    session_id: str = typer.Option(..., "--session-id", help="Session ID"),
    skill_id: str = typer.Option(..., "--skill-id", help="Skill ID"),
    params: str = typer.Option("{}", "--params", help="Schedule params as JSON object"),
    interval_seconds: int = typer.Option(3600, "--interval-seconds", help="Execution interval seconds"),
    name: str | None = typer.Option(None, "--name", help="Schedule name"),
    skill_version: int | None = typer.Option(None, "--skill-version", help="Skill version"),
    start_in_seconds: int = typer.Option(0, "--start-in-seconds", help="Delay first execution"),
    active: bool = typer.Option(True, "--active/--paused", help="Initial schedule status"),
    service_url: str | None = typer.Option(None, "--service-url", help="Anton service base URL"),
) -> None:
    """Create a scheduled run."""
    base = _service_base_url(service_url)
    params_obj = _parse_json_object(params, "params")
    created = _service_request(
        method="POST",
        base_url=base,
        path="/scheduled-runs",
        payload={
            "name": name,
            "session_id": session_id,
            "skill_id": skill_id,
            "skill_version": skill_version,
            "params": params_obj,
            "interval_seconds": interval_seconds,
            "start_in_seconds": start_in_seconds,
            "active": active,
        },
    )
    console.print(f"[anton.success]Created schedule {created.get('id')} ({created.get('status')}).[/]")
    console.print_json(data=created)


@schedules_app.command("trigger")
def schedules_trigger(
    schedule_id: str = typer.Argument(..., help="Schedule ID"),
    idempotency_key: str | None = typer.Option(None, "--idempotency-key", help="Optional idempotency key"),
    wait: bool = typer.Option(False, "--wait/--no-wait", help="Wait for run completion"),
    wait_timeout_seconds: float = typer.Option(300.0, "--wait-timeout-seconds", help="Run wait timeout"),
    service_url: str | None = typer.Option(None, "--service-url", help="Anton service base URL"),
) -> None:
    """Trigger a scheduled run now."""
    base = _service_base_url(service_url)
    result = _service_request(
        method="POST",
        base_url=base,
        path=f"/scheduled-runs/{schedule_id}/trigger",
        payload={
            "idempotency_key": idempotency_key,
            "wait_for_completion": wait,
            "wait_timeout_seconds": wait_timeout_seconds,
        },
    )
    console.print(
        f"[anton.success]Triggered schedule {schedule_id} run={result.get('run_id')} "
        f"status={result.get('status')}[/]"
    )
    console.print_json(data=result)


@schedules_app.command("pause")
def schedules_pause(
    schedule_id: str = typer.Argument(..., help="Schedule ID"),
    service_url: str | None = typer.Option(None, "--service-url", help="Anton service base URL"),
) -> None:
    """Pause a scheduled run."""
    base = _service_base_url(service_url)
    updated = _service_request(
        method="POST",
        base_url=base,
        path=f"/scheduled-runs/{schedule_id}/pause",
    )
    console.print(f"[anton.success]Paused schedule {schedule_id}.[/]")
    console.print_json(data=updated)


@schedules_app.command("resume")
def schedules_resume(
    schedule_id: str = typer.Argument(..., help="Schedule ID"),
    service_url: str | None = typer.Option(None, "--service-url", help="Anton service base URL"),
) -> None:
    """Resume a scheduled run."""
    base = _service_base_url(service_url)
    updated = _service_request(
        method="POST",
        base_url=base,
        path=f"/scheduled-runs/{schedule_id}/resume",
    )
    console.print(f"[anton.success]Resumed schedule {schedule_id}.[/]")
    console.print_json(data=updated)


@app.command("version")
def version() -> None:
    """Show Anton version."""
    console.print(f"Anton v{__version__}")


app.add_typer(skills_app, name="skills")
app.add_typer(schedules_app, name="schedules")
