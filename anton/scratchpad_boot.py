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
