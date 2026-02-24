"""Scratchpad — persistent Python subprocess for stateful, notebook-like execution."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import venv
from dataclasses import dataclass, field
from pathlib import Path

_CELL_TIMEOUT_DEFAULT = 120        # Default total timeout when no estimate given
_CELL_INACTIVITY_TIMEOUT = 30      # Max silence between output lines before killing
_INSTALL_TIMEOUT = 120
_MAX_OUTPUT = 10_000
_PROGRESS_MARKER = "__ANTON_PROGRESS__"


def _compute_timeouts(estimated_seconds: int) -> tuple[float, float]:
    """Compute (total_timeout, inactivity_timeout) from estimated execution time.

    - If estimate is 0: use defaults (120s total, 30s inactivity).
    - Otherwise: total = max(estimate * 2, estimate + 30) with no cap.
      Inactivity = min(max(estimate * 0.5, 30), 60).
    """
    if estimated_seconds <= 0:
        return float(_CELL_TIMEOUT_DEFAULT), float(_CELL_INACTIVITY_TIMEOUT)
    total = max(estimated_seconds * 2, estimated_seconds + 30)
    inactivity = min(max(estimated_seconds * 0.5, 30), 60)
    return float(total), float(inactivity)


_BOOT_SCRIPT = r'''
import io
import json
import os
import sys
import traceback

_CELL_DELIM = "__ANTON_CELL_END__"
_RESULT_START = "__ANTON_RESULT__"
_RESULT_END = "__ANTON_RESULT_END__"

# Persistent namespace across cells
namespace = {"__builtins__": __builtins__}

# --- Inject get_llm() for LLM access from scratchpad code ---
_scratchpad_model = os.environ.get("ANTON_SCRATCHPAD_MODEL", "")
if _scratchpad_model:
    try:
        import asyncio as _llm_asyncio

        _scratchpad_provider_name = os.environ.get("ANTON_SCRATCHPAD_PROVIDER", "anthropic")
        if _scratchpad_provider_name == "openai":
            from anton.llm.openai import OpenAIProvider as _ProviderClass
        else:
            from anton.llm.anthropic import AnthropicProvider as _ProviderClass

        _llm_provider = _ProviderClass()  # reads API key from env
        _llm_model = _scratchpad_model

        class _ScratchpadLLM:
            """Sync LLM wrapper for scratchpad use. Mirrors SkillLLM interface."""

            @property
            def model(self):
                return _llm_model

            def complete(self, *, system, messages, tools=None, tool_choice=None, max_tokens=4096):
                """Call the LLM synchronously. Returns an LLMResponse."""
                return _llm_asyncio.run(_llm_provider.complete(
                    model=_llm_model,
                    system=system,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=max_tokens,
                ))

            def generate_object(self, schema_class, *, system, messages, max_tokens=4096):
                """Generate a structured object matching a Pydantic model.

                Uses tool_choice to force the LLM to return structured data.
                Supports single models and list[Model].

                Args:
                    schema_class: A Pydantic BaseModel subclass, or list[Model].
                    system: System prompt.
                    messages: Conversation messages.
                    max_tokens: Max tokens for the LLM call.

                Returns:
                    An instance of schema_class (or a list of instances).
                """
                from pydantic import BaseModel as _BaseModel

                is_list = hasattr(schema_class, "__origin__") and schema_class.__origin__ is list
                if is_list:
                    inner_class = schema_class.__args__[0]

                    class _ArrayWrapper(_BaseModel):
                        items: list[inner_class]

                    schema = _ArrayWrapper.model_json_schema()
                    tool_name = f"{inner_class.__name__}_array"
                else:
                    schema = schema_class.model_json_schema()
                    tool_name = schema_class.__name__

                tool = {
                    "name": tool_name,
                    "description": f"Generate structured output matching the {tool_name} schema.",
                    "input_schema": schema,
                }

                response = self.complete(
                    system=system,
                    messages=messages,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool_name},
                    max_tokens=max_tokens,
                )

                if not response.tool_calls:
                    raise ValueError("LLM did not return structured output.")

                import json as _json
                raw = response.tool_calls[0].input

                if is_list:
                    wrapper = _ArrayWrapper.model_validate(raw)
                    return wrapper.items
                return schema_class.model_validate(raw)

        _scratchpad_llm_instance = _ScratchpadLLM()

        def get_llm():
            """Get a pre-configured LLM client. No API keys needed."""
            return _scratchpad_llm_instance

        def agentic_loop(*, system, user_message, tools, handle_tool, max_turns=10, max_tokens=4096):
            """Run a synchronous LLM tool-call loop.

            The LLM reasons, calls tools via handle_tool(name, inputs) -> str,
            and iterates until it produces a final text response.

            Args:
                system: System prompt for the LLM.
                user_message: Initial user message.
                tools: Tool definitions (Anthropic tool schema format).
                handle_tool: Callback (tool_name, tool_input) -> result_string.
                max_turns: Safety limit on LLM round-trips (default 10).
                max_tokens: Max tokens per LLM call.

            Returns:
                The final text response from the LLM.
            """
            llm = get_llm()
            messages = [{"role": "user", "content": user_message}]

            response = None
            for _ in range(max_turns):
                response = llm.complete(
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                )

                if not response.tool_calls:
                    return response.content

                # Build assistant message with text + tool_use blocks
                assistant_content = []
                if response.content:
                    assistant_content.append({"type": "text", "text": response.content})
                for tc in response.tool_calls:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
                messages.append({"role": "assistant", "content": assistant_content})

                # Execute each tool and collect results
                tool_results = []
                for tc in response.tool_calls:
                    try:
                        result = handle_tool(tc.name, tc.input)
                    except Exception as exc:
                        result = f"Error: {exc}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})

            # Hit max_turns
            return response.content if response else ""

        namespace["get_llm"] = get_llm
        namespace["agentic_loop"] = agentic_loop
    except Exception:
        pass  # LLM not available — not fatal (e.g. anthropic not installed)

# --- Inject Mind class for Minds/MindsDB access from scratchpad code ---
_minds_api_key = os.environ.get("MINDS_API_KEY", "")
if _minds_api_key:
    _minds_base_url = os.environ.get("MINDS_BASE_URL", "https://mdb.ai")

    class _MindResponse:
        """Streaming response from Mind.ask(). Iterate for text deltas."""

        def __init__(self, _client, _response, _mind):
            self._client = _client
            self._response = _response
            self._mind = _mind
            self.text = ""
            self.completed = False
            self.conversation_id = None
            self.message_id = None
            self._drained = False

        def __iter__(self):
            try:
                yield from self._iter_deltas()
            finally:
                self._close()

        def _iter_deltas(self):
            import json as _json
            for line in self._response.iter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    event = _json.loads(raw)
                except ValueError:
                    continue
                etype = event.get("type", "")
                if etype == "response.output_text.delta":
                    delta = event.get("delta", "")
                    if delta:
                        self.text += delta
                        yield delta
                elif etype == "response.completed":
                    resp_obj = event.get("response", {})
                    self.conversation_id = resp_obj.get("conversation_id")
                    self.message_id = resp_obj.get("id")
                    self.completed = True
                    self._mind._conversation_id = self.conversation_id
            self._drained = True

        def _close(self):
            try:
                self._response.close()
            except Exception:
                pass
            try:
                self._client.close()
            except Exception:
                pass
            self._drained = True

        def _auto_drain(self):
            if self._drained:
                return
            for _ in self:
                pass

        @staticmethod
        def _format_table(data):
            columns = data.get("column_names", [])
            rows = data.get("data", [])
            if not columns:
                return "No data returned."
            header = "| " + " | ".join(columns) + " |"
            sep = "| " + " | ".join("---" for _ in columns) + " |"
            lines = [header, sep]
            for row in rows:
                cells = [str(c) if c is not None else "" for c in row]
                lines.append("| " + " | ".join(cells) + " |")
            return "\n".join(lines)

        def get_data(self, *, limit=None, offset=0):
            """Fetch tabular results from this response.

            No args  → full CSV export (for pd.read_csv(io.StringIO(csv))).
            limit=N  → paginated markdown table (N rows, optional offset).

            Auto-drains the stream if you haven't iterated yet.
            If export fails, automatically falls back to result endpoint.
            """
            import httpx

            self._auto_drain()
            if not self.completed:
                raise RuntimeError(
                    "Stream did not complete — cannot fetch data. "
                    "The mind's text answer is available in response.text"
                )

            headers = {
                "Authorization": f"Bearer {self._mind._api_key}",
                "Content-Type": "application/json",
            }
            base = (
                f"/api/v1/conversations/{self.conversation_id}"
                f"/items/{self.message_id}"
            )

            with httpx.Client(
                base_url=self._mind._base_url, timeout=60, follow_redirects=True,
            ) as client:
                if limit is not None:
                    params = {"limit": limit}
                    if offset:
                        params["offset"] = offset
                    resp = client.get(f"{base}/result", headers=headers, params=params)
                    if resp.status_code >= 400:
                        raise RuntimeError(
                            f"get_data(limit={limit}) failed (HTTP {resp.status_code}): "
                            f"{resp.text[:500]}\n"
                            f"The mind's text answer is available in response.text"
                        )
                    return self._format_table(resp.json())

                # No limit → try export first, fall back to result
                resp = client.get(f"{base}/export", headers=headers)
                if resp.status_code < 400:
                    return resp.text

                # Export failed — fall back to result endpoint (markdown table)
                fallback = client.get(
                    f"{base}/result", headers=headers, params={"limit": 500},
                )
                if fallback.status_code < 400:
                    return self._format_table(fallback.json())

                # Both failed
                raise RuntimeError(
                    f"get_data() export failed (HTTP {resp.status_code}): "
                    f"{resp.text[:500]}\n"
                    f"get_data(limit=N) also failed (HTTP {fallback.status_code}): "
                    f"{fallback.text[:500]}\n"
                    f"The mind's text answer is available in response.text"
                )

    class Mind:
        """Streaming interface to query databases via MindsDB minds.

        Usage:
            mind = Mind('sales')
            response = mind.ask('top customers?')
            for chunk in response:
                print(chunk, end='')       # streaming text

            csv = response.get_data()            # full CSV export
            table = response.get_data(limit=100) # markdown table (100 rows)

        Follow-ups are automatic — subsequent ask() calls continue
        the same conversation.
        """

        def __init__(self, name):
            self.name = name
            self._api_key = _minds_api_key
            self._base_url = _minds_base_url
            self._conversation_id = None

        def ask(self, question):
            """Ask a question. Returns a MindResponse (iterate for streaming text)."""
            import httpx

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "input": question,
                "model": self.name,
                "stream": True,
            }
            if self._conversation_id:
                payload["conversation_id"] = self._conversation_id

            client = httpx.Client(
                base_url=self._base_url, timeout=120, follow_redirects=True,
            )
            try:
                request = client.build_request(
                    "POST", "/api/v1/responses", headers=headers, json=payload,
                )
                response = client.send(request, stream=True)
                response.raise_for_status()
            except Exception:
                client.close()
                raise

            return _MindResponse(_client=client, _response=response, _mind=self)

    namespace["Mind"] = Mind

# Read-execute loop
_real_stdout = sys.stdout
_real_stdin = sys.stdin

_PROGRESS_MARKER = "__ANTON_PROGRESS__"

def progress(message=""):
    """Signal that long-running work is still active. Resets the inactivity timer."""
    _real_stdout.write(_PROGRESS_MARKER + " " + str(message) + "\n")
    _real_stdout.flush()

namespace["progress"] = progress

# --- Logging capture ---
# Libraries like httpx, urllib3, etc. use Python logging. By default these
# messages are silently dropped (no handler configured). We set up a handler
# that writes to a per-cell StringIO so the LLM can see connection info,
# warnings, and errors from libraries.
import logging as _logging

class _CellLogHandler(_logging.Handler):
    """Logging handler that writes to whichever StringIO is current."""
    def __init__(self):
        super().__init__(level=_logging.INFO)
        self.buf = None
        self.setFormatter(_logging.Formatter("%(name)s: %(message)s"))

    def emit(self, record):
        if self.buf is not None:
            try:
                self.buf.write(self.format(record) + "\n")
            except Exception:
                pass

_cell_log_handler = _CellLogHandler()
_logging.root.addHandler(_cell_log_handler)
_logging.root.setLevel(_logging.INFO)

while True:
    lines = []
    try:
        for line in _real_stdin:
            stripped = line.rstrip("\n")
            if stripped == _CELL_DELIM:
                break
            lines.append(line)
        else:
            # EOF — parent closed stdin
            break
    except EOFError:
        break

    code = "".join(lines)
    if not code.strip():
        result = {"stdout": "", "stderr": "", "logs": "", "error": None}
        _real_stdout.write(_RESULT_START + "\n")
        _real_stdout.write(json.dumps(result) + "\n")
        _real_stdout.write(_RESULT_END + "\n")
        _real_stdout.flush()
        continue

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    log_buf = io.StringIO()
    error = None
    _cell_log_handler.buf = log_buf

    sys.stdout = out_buf
    sys.stderr = err_buf
    try:
        compiled = compile(code, "<scratchpad>", "exec")
        exec(compiled, namespace)
    except ModuleNotFoundError as _mnf:
        # Auto-install the missing module and retry the cell once
        _missing = _mnf.name
        if _missing:
            sys.stdout = _real_stdout
            sys.stderr = sys.__stderr__
            _cell_log_handler.buf = None
            _real_stdout.write(_PROGRESS_MARKER + " " + f"Installing {_missing}..." + "\n")
            _real_stdout.flush()
            import subprocess as _sp
            _uv_path = os.environ.get("ANTON_UV_PATH", "")
            if _uv_path:
                _pip = _sp.run(
                    [_uv_path, "pip", "install", "--python", sys.executable, _missing],
                    capture_output=True, timeout=120,
                )
            else:
                _pip = _sp.run(
                    [sys.executable, "-m", "pip", "install", _missing],
                    capture_output=True, timeout=120,
                )
            # Reset buffers and retry
            out_buf = io.StringIO()
            err_buf = io.StringIO()
            log_buf = io.StringIO()
            _cell_log_handler.buf = log_buf
            sys.stdout = out_buf
            sys.stderr = err_buf
            if _pip.returncode == 0:
                try:
                    exec(compiled, namespace)
                except Exception:
                    error = traceback.format_exc()
            else:
                error = (
                    f"ModuleNotFoundError: No module named '{_missing}'\n"
                    f"Auto-install failed:\n{_pip.stderr.decode()}"
                )
        else:
            error = traceback.format_exc()
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout = _real_stdout
        sys.stderr = sys.__stderr__
        _cell_log_handler.buf = None

    result = {
        "stdout": out_buf.getvalue(),
        "stderr": err_buf.getvalue(),
        "logs": log_buf.getvalue(),
        "error": error,
    }
    _real_stdout.write(_RESULT_START + "\n")
    _real_stdout.write(json.dumps(result) + "\n")
    _real_stdout.write(_RESULT_END + "\n")
    _real_stdout.flush()
'''

_CELL_DELIM = "__ANTON_CELL_END__"
_RESULT_START = "__ANTON_RESULT__"
_RESULT_END = "__ANTON_RESULT_END__"


@dataclass
class Cell:
    code: str
    stdout: str
    stderr: str
    error: str | None
    description: str = ""
    estimated_time: str = ""
    logs: str = ""


@dataclass
class Scratchpad:
    name: str
    cells: list[Cell] = field(default_factory=list)
    _proc: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _boot_path: str | None = field(default=None, repr=False)
    _coding_provider: str = field(default="anthropic", repr=False)
    _coding_model: str = field(default="", repr=False)
    _coding_api_key: str = field(default="", repr=False)
    _venv_dir: str | None = field(default=None, repr=False)
    _venv_python: str | None = field(default=None, repr=False)
    _installed_packages: set[str] = field(default_factory=set, repr=False)

    _MAX_VENV_RETRIES = 3

    def _ensure_venv(self) -> None:
        """Create a lightweight per-scratchpad venv (idempotent).

        Uses system_site_packages=True so the real system packages are visible.
        If we're running inside a parent venv, we also drop a .pth file so the
        parent venv's site-packages are visible in the child.

        If the venv is broken (stale symlinks, missing Python binary), it is
        deleted and recreated from scratch. Gives up after _MAX_VENV_RETRIES.
        """
        if self._venv_dir is not None and self._verify_venv_python():
            return

        last_error: Exception | None = None
        for attempt in range(1, self._MAX_VENV_RETRIES + 1):
            try:
                self._create_venv()
                if self._verify_venv_python():
                    self._setup_parent_site_packages()
                    return
                # Python binary exists but doesn't run — nuke and retry
                raise RuntimeError(f"venv Python binary at {self._venv_python} is not functional")
            except Exception as exc:
                last_error = exc
                # Clean up the broken venv before retrying
                self._nuke_venv()

        raise RuntimeError(
            f"Failed to create a working Python venv after {self._MAX_VENV_RETRIES} attempts. "
            f"Last error: {last_error}. "
            f"Try running: python3 -c 'print(\"ok\")' to verify your Python installation."
        )

    @staticmethod
    def _find_uv() -> str | None:
        """Return the path to the ``uv`` binary, or *None* if unavailable."""
        import subprocess as _sp
        # Fast path: already on PATH
        uv = shutil.which("uv")
        if uv:
            return uv
        # Common install locations
        for candidate in (
            os.path.expanduser("~/.local/bin/uv"),
            os.path.expanduser("~/.cargo/bin/uv"),
        ):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return None

    def _create_venv(self) -> None:
        """Allocate a venv directory and create the virtual environment.

        Prefers ``uv venv`` when available — it is faster, more reliable on
        macOS (doesn't break when Homebrew upgrades Python), and doesn't depend
        on the ``venv`` stdlib module being functional.  Falls back to
        ``venv.create()`` when ``uv`` isn't found.
        """
        import subprocess as _sp

        if sys.platform == "win32":
            self._venv_dir = str(Path("~/.anton/scratchpad-venv").expanduser())
            os.makedirs(self._venv_dir, exist_ok=True)
        else:
            self._venv_dir = tempfile.mkdtemp(prefix="anton_venv_")

        uv = self._find_uv()
        if uv:
            _sp.run(
                [uv, "venv", self._venv_dir,
                 "--python", sys.executable,
                 "--system-site-packages", "--seed", "--quiet"],
                check=True,
                capture_output=True,
                timeout=30,
            )
        else:
            venv.create(self._venv_dir, system_site_packages=True, with_pip=False, clear=True)

        bin_dir = os.path.join(self._venv_dir, "bin")
        if sys.platform == "win32":
            bin_dir = os.path.join(self._venv_dir, "Scripts")
        self._venv_python = os.path.join(bin_dir, "python")

    def _verify_venv_python(self) -> bool:
        """Check that the venv Python binary exists and can execute."""
        if self._venv_python is None:
            return False
        if not os.path.exists(self._venv_python):
            return False
        # Quick smoke test — run python with a trivial command
        try:
            import subprocess
            result = subprocess.run(
                [self._venv_python, "-c", "print('ok')"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0 and "ok" in result.stdout.decode()
        except Exception:
            return False

    def _nuke_venv(self) -> None:
        """Delete the venv directory entirely so it can be recreated."""
        if self._venv_dir is not None:
            try:
                shutil.rmtree(self._venv_dir)
            except OSError:
                pass
        self._venv_dir = None
        self._venv_python = None
        self._installed_packages.clear()

    def _setup_parent_site_packages(self) -> None:
        """Make parent venv's packages visible in the child venv."""
        if sys.prefix != sys.base_prefix:
            import site as _site
            parent_site = _site.getsitepackages()
            child_site = None
            for dirpath, dirnames, _ in os.walk(self._venv_dir):
                if "site-packages" in dirnames:
                    child_site = os.path.join(dirpath, "site-packages")
                    break
            if child_site and parent_site:
                pth_path = os.path.join(child_site, "_parent_venv.pth")
                with open(pth_path, "w") as f:
                    for sp in parent_site:
                        f.write(sp + "\n")

    async def start(self) -> None:
        """Write the boot script to a temp file and launch the subprocess."""
        self._ensure_venv()

        fd, path = tempfile.mkstemp(suffix=".py", prefix="anton_scratchpad_")
        os.write(fd, _BOOT_SCRIPT.encode())
        os.close(fd)
        self._boot_path = path

        env = os.environ.copy()
        if self._coding_model:
            env["ANTON_SCRATCHPAD_MODEL"] = self._coding_model
        if self._coding_provider:
            env["ANTON_SCRATCHPAD_PROVIDER"] = self._coding_provider
        # Ensure the SDKs can find API keys under their expected names.
        # Anton stores them as ANTON_*_API_KEY; the SDKs expect *_API_KEY.
        if "ANTHROPIC_API_KEY" not in env and "ANTON_ANTHROPIC_API_KEY" in env:
            env["ANTHROPIC_API_KEY"] = env["ANTON_ANTHROPIC_API_KEY"]
        if "OPENAI_API_KEY" not in env and "ANTON_OPENAI_API_KEY" in env:
            env["OPENAI_API_KEY"] = env["ANTON_OPENAI_API_KEY"]
        # If settings provided an explicit API key (e.g. from ~/.anton/.env or
        # Pydantic settings), inject it so the subprocess SDK can authenticate.
        if self._coding_api_key:
            sdk_key = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
            }.get(self._coding_provider, "")
            if sdk_key and sdk_key not in env:
                env[sdk_key] = self._coding_api_key
        # Pass uv path so the boot script can use it for auto-installing
        # missing modules (same installer that created the venv).
        uv = self._find_uv()
        if uv:
            env["ANTON_UV_PATH"] = uv

        # Ensure the anton package is importable in the subprocess (needed for
        # get_llm and skill loading). The boot script runs from a temp file, so
        # the project root isn't on sys.path by default.
        _anton_root = str(Path(__file__).resolve().parent.parent)
        python_path = env.get("PYTHONPATH", "")
        if _anton_root not in python_path:
            env["PYTHONPATH"] = _anton_root + (os.pathsep + python_path if python_path else "")

        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._venv_python, path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            # Python binary is missing or broken — nuke venv and raise
            self._nuke_venv()
            raise RuntimeError(
                f"Failed to start scratchpad: {exc}. "
                f"The Python venv has been deleted and will be recreated on next attempt."
            ) from exc

    async def execute(
        self,
        code: str,
        *,
        description: str = "",
        estimated_time: str = "",
        estimated_seconds: int = 0,
    ) -> Cell:
        """Send code to the subprocess, read the JSON result, return a Cell.

        Backward-compatible wrapper around execute_streaming() that drains
        all events and returns just the final Cell.
        """
        async for item in self.execute_streaming(
            code,
            description=description,
            estimated_time=estimated_time,
            estimated_seconds=estimated_seconds,
        ):
            if isinstance(item, Cell):
                return item
        # Should not reach here, but just in case
        return Cell(code=code, stdout="", stderr="", error="No result produced.")

    async def execute_streaming(
        self,
        code: str,
        *,
        description: str = "",
        estimated_time: str = "",
        estimated_seconds: int = 0,
    ):
        """Async generator that sends code and yields progress strings and a final Cell.

        Yields:
            str — progress messages from progress() calls in the cell code
            Cell — the final execution result (always the last item)
        """
        if self._proc is None or self._proc.returncode is not None:
            yield Cell(
                code=code,
                stdout="",
                stderr="",
                error="Scratchpad process is not running. Use reset to restart.",
                description=description,
                estimated_time=estimated_time,
            )
            return

        payload = code + "\n" + _CELL_DELIM + "\n"
        self._proc.stdin.write(payload.encode())  # type: ignore[union-attr]
        await self._proc.stdin.drain()  # type: ignore[union-attr]

        total_timeout, inactivity_timeout = _compute_timeouts(estimated_seconds)

        try:
            result_data: dict | None = None
            async for item in self._read_result(
                total_timeout=total_timeout,
                inactivity_timeout=inactivity_timeout,
            ):
                if isinstance(item, str):
                    yield item  # progress message
                else:
                    result_data = item
        except asyncio.TimeoutError as exc:
            self._proc.kill()
            await self._proc.wait()
            cell = Cell(
                code=code,
                stdout="",
                stderr="",
                error=f"{exc}. Process killed — state lost. Use reset to restart.",
                description=description,
                estimated_time=estimated_time,
            )
            self.cells.append(cell)
            yield cell
            return

        if result_data is None:
            result_data = {"stdout": "", "stderr": "", "error": "Process exited unexpectedly."}

        cell = Cell(
            code=code,
            stdout=result_data.get("stdout", ""),
            stderr=result_data.get("stderr", ""),
            error=result_data.get("error"),
            description=description,
            estimated_time=estimated_time,
            logs=result_data.get("logs", ""),
        )
        self.cells.append(cell)
        yield cell

    async def _read_result(
        self,
        *,
        total_timeout: float = _CELL_TIMEOUT_DEFAULT,
        inactivity_timeout: float = _CELL_INACTIVITY_TIMEOUT,
    ):
        """Async generator that reads lines from stdout until result delimiters.

        Yields:
            str — progress messages (lines starting with _PROGRESS_MARKER)
            dict — the final JSON result (always the last item)

        Raises asyncio.TimeoutError with a descriptive message.
        """
        import time as _time

        lines: list[str] = []
        in_result = False
        start = _time.monotonic()

        while True:
            elapsed = _time.monotonic() - start
            remaining_total = total_timeout - elapsed
            if remaining_total <= 0:
                raise asyncio.TimeoutError(
                    f"Cell timed out after {total_timeout:.0f}s total"
                )

            line_timeout = min(inactivity_timeout, remaining_total)
            try:
                raw = await asyncio.wait_for(
                    self._proc.stdout.readline(),  # type: ignore[union-attr]
                    timeout=line_timeout,
                )
            except asyncio.TimeoutError:
                # Determine which timeout was hit
                elapsed_now = _time.monotonic() - start
                if elapsed_now >= total_timeout - 0.5:
                    raise asyncio.TimeoutError(
                        f"Cell timed out after {total_timeout:.0f}s total"
                    ) from None
                raise asyncio.TimeoutError(
                    f"Cell killed after {inactivity_timeout:.0f}s of inactivity "
                    f"(no output or progress() calls)"
                ) from None

            if not raw:
                yield {"stdout": "", "stderr": "", "error": "Process exited unexpectedly."}
                return

            line = raw.decode().rstrip("\n")

            # Progress marker — yield to caller, don't store
            if line.startswith(_PROGRESS_MARKER):
                message = line[len(_PROGRESS_MARKER):].strip()
                yield message
                continue

            if line == _RESULT_START:
                in_result = True
                continue
            if line == _RESULT_END:
                break
            if in_result:
                lines.append(line)

        yield json.loads("\n".join(lines))

    def view(self) -> str:
        """Format all cells with their outputs."""
        if not self.cells:
            return f"Scratchpad '{self.name}' is empty."

        parts: list[str] = []
        for i, cell in enumerate(self.cells):
            header = f"--- Cell {i + 1}"
            if cell.description:
                header += f": {cell.description}"
            header += " ---"
            parts.append(header)
            parts.append(cell.code)
            if cell.stdout:
                parts.append(f"[output]\n{cell.stdout}")
            if cell.logs:
                parts.append(f"[logs]\n{cell.logs}")
            if cell.stderr:
                parts.append(f"[stderr]\n{cell.stderr}")
            if cell.error:
                parts.append(f"[error]\n{cell.error}")
            if not cell.stdout and not cell.logs and not cell.stderr and not cell.error:
                parts.append("(no output)")
        return "\n".join(parts)

    @staticmethod
    def _truncate_output(text: str, max_lines: int = 20, max_chars: int = 2000) -> str:
        """Truncate output to *max_lines* / *max_chars*, whichever is shorter."""
        lines = text.split("\n")
        # Apply line limit
        if len(lines) > max_lines:
            kept = "\n".join(lines[:max_lines])
            remaining = len(lines) - max_lines
            return kept + f"\n... ({remaining} more lines)"
        # Apply char limit (don't cut mid-line)
        if len(text) > max_chars:
            total = 0
            kept_lines: list[str] = []
            for line in lines:
                if total + len(line) + 1 > max_chars and kept_lines:
                    break
                kept_lines.append(line)
                total += len(line) + 1
            return "\n".join(kept_lines) + "\n... (truncated)"
        return text

    def render_notebook(self) -> str:
        """Return a clean markdown notebook-style summary of all cells."""
        # Filter out empty/whitespace-only cells
        numbered: list[tuple[int, Cell]] = []
        idx = 0
        for cell in self.cells:
            idx += 1
            if not cell.code.strip():
                continue
            numbered.append((idx, cell))

        if not numbered:
            return f"Scratchpad '{self.name}' has no cells."

        parts: list[str] = [f"## Scratchpad: {self.name} ({len(numbered)} cells)"]

        for i, (num, cell) in enumerate(numbered):
            header = f"\n### Cell {num}"
            if cell.description:
                header += f" \u2014 {cell.description}"
            parts.append(header)
            parts.append(f"```python\n{cell.code}\n```\n")

            if cell.error:
                # Show only the last traceback line
                last_line = cell.error.strip().split("\n")[-1]
                parts.append(f"**Error:** `{last_line}`")
                # If there was partial output before the error, show it
                if cell.stdout:
                    truncated = self._truncate_output(cell.stdout.rstrip("\n"))
                    parts.append(f"**Partial output:**\n```\n{truncated}\n```\n")
            elif cell.stdout:
                truncated = self._truncate_output(cell.stdout.rstrip("\n"))
                parts.append(f"**Output:**\n```\n{truncated}\n```\n")

            if cell.logs:
                truncated_logs = self._truncate_output(cell.logs.rstrip("\n"), max_lines=10, max_chars=1000)
                parts.append(f"**Logs:**\n```\n{truncated_logs}\n```\n")

            if i < len(numbered) - 1:
                parts.append("---")

        return "\n".join(parts)

    async def _stop_process(self) -> None:
        """Kill the subprocess and delete the boot script, but keep the venv."""
        if self._proc is not None and self._proc.returncode is None:
            try:
                self._proc.kill()
                await self._proc.wait()
            except ProcessLookupError:
                pass
        self._proc = None
        if self._boot_path is not None:
            try:
                os.unlink(self._boot_path)
            except OSError:
                pass
            self._boot_path = None

    async def reset(self) -> None:
        """Kill the process, clear cells, restart.

        If the venv is healthy, it's reused (installed packages survive).
        If the venv is broken, it's deleted and recreated from scratch.
        """
        await self._stop_process()
        self.cells.clear()
        # If the venv Python is broken, nuke it so _ensure_venv recreates it
        if not self._verify_venv_python():
            self._nuke_venv()
        await self.start()

    async def close(self) -> None:
        """Kill the process and clean up the boot script temp file and venv."""
        await self._stop_process()
        if self._venv_dir is not None:
            # On Windows, keep the fixed venv so firewall rules persist
            if sys.platform != "win32":
                try:
                    shutil.rmtree(self._venv_dir)
                except OSError:
                    pass
            self._venv_dir = None
            self._venv_python = None

    async def install_packages(self, packages: list[str]) -> str:
        """Install packages into the scratchpad's venv via pip (or uv pip)."""
        if not packages:
            return "No packages specified."
        # Skip packages we've already installed in this scratchpad
        needed = [p for p in packages if p.lower() not in self._installed_packages]
        if not needed:
            return "All packages already installed."
        self._ensure_venv()

        uv = self._find_uv()
        if uv:
            cmd = [uv, "pip", "install", "--python", self._venv_python, *needed]
        else:
            cmd = [self._venv_python, "-m", "pip", "install", "--no-input", *needed]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_INSTALL_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Install timed out after {_INSTALL_TIMEOUT}s."
        output = stdout.decode()
        if proc.returncode != 0:
            return f"Install failed (exit {proc.returncode}):\n{output}"
        # Track successfully installed packages
        for p in needed:
            self._installed_packages.add(p.lower())
        return output


class ScratchpadManager:
    """Manages named scratchpad instances."""

    def __init__(
        self,
        coding_provider: str = "anthropic",
        coding_model: str = "",
        coding_api_key: str = "",
    ) -> None:
        self._pads: dict[str, Scratchpad] = {}
        self._coding_provider: str = coding_provider
        self._coding_model: str = coding_model
        self._coding_api_key: str = coding_api_key
        self._available_packages: list[str] = self.probe_packages()

    @staticmethod
    def probe_packages() -> list[str]:
        """Return sorted list of installed package distribution names."""
        from importlib.metadata import distributions

        return sorted({d.metadata["Name"] for d in distributions()})

    async def get_or_create(self, name: str) -> Scratchpad:
        """Return existing pad or create + start a new one."""
        if name not in self._pads:
            pad = Scratchpad(
                name=name,
                _coding_provider=self._coding_provider,
                _coding_model=self._coding_model,
                _coding_api_key=self._coding_api_key,
            )
            await pad.start()
            self._pads[name] = pad
        return self._pads[name]

    async def remove(self, name: str) -> str:
        """Kill and delete a scratchpad."""
        pad = self._pads.pop(name, None)
        if pad is None:
            return f"No scratchpad named '{name}'."
        await pad.close()
        return f"Scratchpad '{name}' removed."

    def list_pads(self) -> list[str]:
        return list(self._pads.keys())

    async def close_all(self) -> None:
        """Cleanup all scratchpads on session end."""
        for pad in self._pads.values():
            await pad.close()
        self._pads.clear()
