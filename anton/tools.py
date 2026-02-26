"""Dynamic tool registry — decorator-based registration for chat tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anton.chat import ChatSession


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict
    handler: Callable  # async (session, tc_input) -> str
    stream_handler: Callable | None = None  # async generator version


_registry: dict[str, ToolDef] = {}


def tool(name: str, *, description: str, input_schema: dict):
    """Decorator to register a tool with its handler."""
    def decorator(fn):
        _registry[name] = ToolDef(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=fn,
        )
        return fn
    return decorator


def tool_stream(name: str):
    """Decorator to register a streaming handler for an existing tool."""
    def decorator(fn):
        if name in _registry:
            _registry[name].stream_handler = fn
        return fn
    return decorator


def get_tool(name: str) -> ToolDef | None:
    return _registry.get(name)


def all_tools() -> list[ToolDef]:
    return list(_registry.values())


def build_tool_schemas(available: list[str]) -> list[dict]:
    """Build API-ready tool schema dicts for the given tool names."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in _registry.values()
        if t.name in available
    ]


# ---------------------------------------------------------------------------
# Tool definitions + handlers (moved from chat.py)
# ---------------------------------------------------------------------------

UPDATE_CONTEXT_TOOL = {
    "name": "update_context",
    "description": (
        "Update self-awareness context files when you learn something important "
        "about the project or workspace. Use this to persist knowledge for future sessions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {
                            "type": "string",
                            "description": "Filename like 'project-overview.md'",
                        },
                        "content": {
                            "type": ["string", "null"],
                            "description": "New content, or null to delete the file",
                        },
                    },
                    "required": ["file", "content"],
                },
            },
        },
        "required": ["updates"],
    },
}

REQUEST_SECRET_TOOL = {
    "name": "request_secret",
    "description": (
        "Request a secret value (API key, token, password) from the user. "
        "The value is stored directly in .anton/.env and NEVER passed through the LLM. "
        "After calling this, you will be told the variable has been set — use it by name."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "variable_name": {
                "type": "string",
                "description": "Environment variable name to store (e.g. 'GITHUB_TOKEN', 'DATABASE_PASSWORD')",
            },
            "prompt_text": {
                "type": "string",
                "description": "What to ask the user (e.g. 'Please enter your GitHub personal access token')",
            },
        },
        "required": ["variable_name", "prompt_text"],
    },
}

CONNECTOR_TOOL = {
    "name": "connector",
    "description": (
        "Access configured data connectors through a governed adapter layer. "
        "Use this for schema discovery, query pushdown, and sampling data before "
        "moving results into scratchpad transformations.\n\n"
        "Actions:\n"
        "- list: List available connectors\n"
        "- schema: Describe a connector's tables/columns\n"
        "- query: Execute a read query with pushdown at the source\n"
        "- sample: Read a limited sample from a table\n"
        "- write: Execute write queries (requires policy approval by default)\n\n"
        "Prefer connector query/sample over pulling raw data into Python first. "
        "Use connector_id to target the source."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "schema", "query", "sample", "write"]},
            "connector_id": {"type": "string", "description": "Connector identifier"},
            "query": {"type": "string", "description": "SQL or source-native query"},
            "table": {"type": "string", "description": "Table name for sample action"},
            "limit": {"type": "integer", "description": "Row limit for query/sample actions"},
        },
        "required": ["action"],
    },
}


SCRATCHPAD_TOOL = {
    "name": "scratchpad",
    "description": (
        "Run Python code in a persistent scratchpad. Use this whenever you need to "
        "count characters, do math, parse data, transform text, or any task that "
        "benefits from precise computation rather than guessing. Variables, imports, "
        "and data persist across cells — like a notebook you drive programmatically.\n\n"
        "Actions:\n"
        "- exec: Run code in the scratchpad (creates it if needed)\n"
        "- view: See all cells and their outputs\n"
        "- reset: Restart the process, clearing all state (installed packages survive)\n"
        "- remove: Kill the scratchpad and delete its environment\n"
        "- dump: Show a clean notebook-style summary of cells (code + truncated output)\n"
        "- install: Install Python packages into the scratchpad's environment. "
        "Packages persist across resets.\n\n"
        "Use print() to produce output. Host Python packages are available by default. "
        "Include a 'packages' array on exec calls for any libraries your code needs — "
        "they'll be auto-installed before the cell runs (already-installed ones are skipped).\n"
        "get_llm() returns a pre-configured LLM client (sync) — call "
        "llm.complete(system=..., messages=[...]) for AI-powered computation.\n"
        "llm.generate_object(MyModel, system=..., messages=[...]) extracts structured "
        "data into Pydantic models. Supports single models and list[Model].\n"
        "agentic_loop(system=..., user_message=..., tools=[...], handle_tool=fn) "
        "runs a tool-call loop where the LLM reasons and calls your tools iteratively. "
        "handle_tool(name, inputs) -> str is a plain sync function.\n"
        "sample(var) inspects any variable with type-aware formatting — DataFrames get "
        "shape/dtypes/head, dicts get keys/values, lists get length/items. "
        "Defaults to 'preview' mode (compact); use sample(var, mode='full') for complete dump.\n"
        "All .anton/.env secrets are available as environment variables (os.environ).\n"
        "need_secret(variable_name, prompt_text) prompts the user for a secret (API key, "
        "password, token) directly from scratchpad code. The value is stored in .anton/.env "
        "and set in os.environ — never returned as a variable. Use this when your code "
        "discovers it needs a credential mid-execution.\n\n"
        "IMPORTANT: Cells have an inactivity timeout of 30 seconds — if a cell produces "
        "no output and no progress() calls for 30s, it is killed and all state is lost. "
        "For long-running code (API calls, data extraction, heavy computation), call "
        "progress(message) periodically to signal work is ongoing and reset the timer. "
        "The total timeout scales from your estimated_execution_time_seconds "
        "(roughly 2x the estimate). You MUST provide estimated_execution_time_seconds "
        "for every exec call. For very long operations, provide a realistic estimate "
        "and use progress() to keep the cell alive."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["exec", "view", "reset", "remove", "dump", "install"]},
            "name": {"type": "string", "description": "Scratchpad name"},
            "code": {
                "type": "string",
                "description": "Python code (exec only). Use print() for output.",
            },
            "packages": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Package names needed by this cell (exec or install). "
                "Listed after code so you know exactly what to include. "
                "Already-installed packages are skipped automatically.",
            },
            "one_line_description": {
                "type": "string",
                "description": "Brief description of what this cell does (e.g. 'Scrape listing prices'). Required for exec.",
            },
            "estimated_execution_time_seconds": {
                "type": "integer",
                "description": "Estimated execution time in seconds. Drives the total timeout (roughly 2x estimate). Use progress() for long cells.",
            },
        },
        "required": ["action", "name"],
    },
}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def handle_update_context(session: ChatSession, tc_input: dict) -> str:
    """Process an update_context tool call and return a result string."""
    if session._self_awareness is None:
        return "Context updates not available."

    from anton.context.self_awareness import ContextUpdate

    raw_updates = tc_input.get("updates", [])
    updates = [
        ContextUpdate(file=u["file"], content=u.get("content"))
        for u in raw_updates
        if isinstance(u, dict) and "file" in u
    ]

    if not updates:
        return "No valid updates provided."

    actions = session._self_awareness.apply_updates(updates)
    return "Context updated: " + "; ".join(actions)


def _password_input(prompt_label: str) -> str:
    """Read a password with hidden input. Separated for testability."""
    import getpass
    return getpass.getpass(prompt_label)


def prompt_secret(console, var_name: str, prompt_text: str) -> str:
    """Prompt the user for a secret with hidden input and context.

    Shows a contextual banner explaining what's needed and why.
    Input is hidden (standard password behavior).
    If the user enters nothing, asks to confirm empty value or retry.
    Returns the stripped value (may be empty if confirmed).
    """
    console.print()
    console.print(f"  [bold]Secret requested:[/] [anton.cyan]{var_name}[/]")
    console.print(f"  {prompt_text}")
    console.print(f"  [anton.muted]Stored in .anton/.env · never shown to the AI · input is hidden[/]")

    while True:
        value = _password_input(f"  {var_name}> ")
        value = value.strip()
        if value:
            return value
        # Empty input — confirm or retry
        answer = console.input(
            "  [anton.warning]No value entered.[/] Use empty password? [dim](y to confirm, Enter to retry):[/] "
        ).strip().lower()
        if answer in ("y", "yes"):
            return ""


def handle_request_secret(session: ChatSession, tc_input: dict) -> str:
    """Handle a request_secret tool call.

    Asks the user directly for the secret value, stores it in .env,
    and returns a confirmation — NEVER returns the actual secret value.
    """
    if session._workspace is None or session._console is None:
        return "Secret storage not available."

    var_name = tc_input.get("variable_name", "")
    prompt_text = tc_input.get("prompt_text", f"Enter value for {var_name}")

    if not var_name:
        return "No variable_name provided."

    # Check if already set
    if session._workspace.has_secret(var_name):
        return f"Variable {var_name} is already set in .anton/.env."

    # Ask user directly — this bypasses the LLM entirely
    value = prompt_secret(session._console, var_name, prompt_text)

    if not value:
        return f"No value provided for {var_name}. Variable not set."

    # Store securely — value never touches the LLM
    session._workspace.set_secret(var_name, value)
    return f"Variable {var_name} has been set in .anton/.env. You can now use it."


async def handle_connector(session: ChatSession, tc_input: dict) -> str:
    """Dispatch a connector tool call by action."""
    if session._connector_hub is None:
        return "Connector tool is not configured for this runtime."

    from anton.connectors import ConnectorError

    action = str(tc_input.get("action", "")).lower()
    connector_id = str(tc_input.get("connector_id", ""))
    limit = tc_input.get("limit", 1000)
    try:
        parsed_limit = int(limit)
    except (TypeError, ValueError):
        parsed_limit = 1000

    try:
        auth_context = session._connector_auth_context
        if action == "list":
            return await session._connector_hub.list_connectors(auth_context=auth_context)
        if action == "schema":
            if not connector_id:
                return "connector_id is required for schema action."
            return await session._connector_hub.schema(connector_id, auth_context=auth_context)
        if action == "query":
            if not connector_id:
                return "connector_id is required for query action."
            query = str(tc_input.get("query", "")).strip()
            if not query:
                return "query is required for query action."
            return await session._connector_hub.query(
                connector_id,
                query,
                limit=max(1, parsed_limit),
                auth_context=auth_context,
            )
        if action == "sample":
            if not connector_id:
                return "connector_id is required for sample action."
            table = str(tc_input.get("table", "")).strip()
            if not table:
                return "table is required for sample action."
            return await session._connector_hub.sample(
                connector_id,
                table,
                limit=max(1, parsed_limit),
                auth_context=auth_context,
            )
        if action == "write":
            if not connector_id:
                return "connector_id is required for write action."
            query = str(tc_input.get("query", "")).strip()
            if not query:
                return "query is required for write action."
            return await session._connector_hub.write(
                connector_id,
                query,
                auth_context=auth_context,
            )
        return f"Unknown connector action: {action}"
    except ConnectorError as exc:
        return f"Connector action failed: {exc}"


async def prepare_scratchpad_exec(session: ChatSession, tc_input: dict):
    """Validate and prepare a scratchpad exec call.

    Returns (pad, code, description, estimated_time, estimated_seconds) or
    a str error message if validation fails.
    """
    name = tc_input.get("name", "")
    code = tc_input.get("code", "")
    if not code or not code.strip():
        return "No code provided."

    pad = await session._scratchpads.get_or_create(name)

    # Auto-install packages before running the cell
    packages = tc_input.get("packages", [])
    if packages:
        install_result = await pad.install_packages(packages)
        if "Install failed" in install_result or "timed out" in install_result:
            return install_result

    description = tc_input.get("one_line_description", "")
    estimated_seconds = tc_input.get("estimated_execution_time_seconds", 0)
    if isinstance(estimated_seconds, str):
        try:
            estimated_seconds = int(estimated_seconds)
        except ValueError:
            estimated_seconds = 0

    estimated_time = f"{estimated_seconds}s" if estimated_seconds > 0 else ""
    return pad, code, description, estimated_time, estimated_seconds


def format_cell_result(cell) -> str:
    """Format a Cell into a tool result string.

    Every section is labeled so the LLM can tell what came from where:
    [output] — print() / stdout from the cell code
    [logs]   — library logging (httpx, urllib3, etc.) captured at INFO+
    [stderr] — warnings and stderr writes
    [error]  — Python traceback if the cell raised an exception
    """
    parts: list[str] = []
    if cell.stdout:
        stdout = cell.stdout
        if len(stdout) > 10_000:
            stdout = stdout[:10_000] + f"\n\n... (truncated, {len(stdout)} chars total)"
        parts.append(f"[output]\n{stdout}")
    if cell.logs if hasattr(cell, "logs") else False:
        logs = cell.logs.strip()
        if len(logs) > 3_000:
            logs = logs[:3_000] + "\n... (logs truncated)"
        parts.append(f"[logs]\n{logs}")
    if cell.stderr:
        parts.append(f"[stderr]\n{cell.stderr}")
    if cell.error:
        parts.append(f"[error]\n{cell.error}")
    if not parts:
        return "Code executed successfully (no output)."
    return "\n".join(parts)


async def handle_scratchpad(session: ChatSession, tc_input: dict) -> str:
    """Dispatch a scratchpad tool call by action."""
    action = tc_input.get("action", "")
    name = tc_input.get("name", "")

    if not name:
        return "Scratchpad name is required."

    if action == "exec":
        result = await prepare_scratchpad_exec(session, tc_input)
        if isinstance(result, str):
            return result
        pad, code, description, estimated_time, estimated_seconds = result

        cell = await pad.execute(
            code,
            description=description,
            estimated_time=estimated_time,
            estimated_seconds=estimated_seconds,
        )
        return format_cell_result(cell)

    elif action == "view":
        pad = session._scratchpads._pads.get(name)
        if pad is None:
            return f"No scratchpad named '{name}'."
        return pad.view()

    elif action == "reset":
        pad = session._scratchpads._pads.get(name)
        if pad is None:
            return f"No scratchpad named '{name}'."
        await pad.reset()
        return f"Scratchpad '{name}' reset. All state cleared."

    elif action == "remove":
        return await session._scratchpads.remove(name)

    elif action == "dump":
        pad = session._scratchpads._pads.get(name)
        if pad is None:
            return f"No scratchpad named '{name}'."
        return pad.render_notebook()

    elif action == "install":
        packages = tc_input.get("packages", [])
        if not packages:
            return "No packages specified."
        pad = await session._scratchpads.get_or_create(name)
        return await pad.install_packages(packages)

    else:
        return f"Unknown scratchpad action: {action}"


async def dispatch_tool(session: ChatSession, tool_name: str, tc_input: dict) -> str:
    """Dispatch a tool call by name. Returns result text."""
    if tool_name == "update_context":
        return handle_update_context(session, tc_input)
    elif tool_name == "request_secret":
        return handle_request_secret(session, tc_input)
    elif tool_name == "connector":
        return await handle_connector(session, tc_input)
    elif tool_name == "scratchpad":
        return await handle_scratchpad(session, tc_input)
    else:
        return f"Unknown tool: {tool_name}"
