"""Scratchpad — persistent Python subprocess for stateful, notebook-like execution."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_CELL_TIMEOUT = 30
_MAX_OUTPUT = 10_000

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

# Inject run_skill() if skill dirs are available
_skill_dirs_raw = os.environ.get("ANTON_SKILL_DIRS", "")
if _skill_dirs_raw:
    try:
        import importlib.util
        _skill_dirs = [d for d in _skill_dirs_raw.split(os.pathsep) if d]
        _registry = {}

        for skills_dir in _skill_dirs:
            from pathlib import Path as _Path
            skills_path = _Path(skills_dir)
            if not skills_path.is_dir():
                continue
            for skill_file in sorted(skills_path.glob("*/skill.py")):
                module_name = f"anton_skill_{skill_file.parent.name}"
                spec = importlib.util.spec_from_file_location(module_name, skill_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    continue
                for attr_name in dir(module):
                    obj = getattr(module, attr_name)
                    info = getattr(obj, "_skill_info", None)
                    if info is not None and hasattr(info, "name"):
                        _registry[info.name] = info

        def run_skill(name, **kwargs):
            """Run an Anton skill by name. Returns the skill's output."""
            import asyncio as _asyncio
            _skill = _registry.get(name)
            if _skill is None:
                raise ValueError(f"Unknown skill: {name}. Available: {list(_registry.keys())}")
            result = _asyncio.run(_skill.execute(**kwargs))
            return result.output

        namespace["run_skill"] = run_skill
    except Exception:
        pass  # Skills not available — not fatal

# Read-execute loop
_real_stdout = sys.stdout
_real_stdin = sys.stdin

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
        result = {"stdout": "", "stderr": "", "error": None}
        _real_stdout.write(_RESULT_START + "\n")
        _real_stdout.write(json.dumps(result) + "\n")
        _real_stdout.write(_RESULT_END + "\n")
        _real_stdout.flush()
        continue

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    error = None

    sys.stdout = out_buf
    sys.stderr = err_buf
    try:
        compiled = compile(code, "<scratchpad>", "exec")
        exec(compiled, namespace)
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout = _real_stdout
        sys.stderr = sys.__stderr__

    result = {
        "stdout": out_buf.getvalue(),
        "stderr": err_buf.getvalue(),
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


@dataclass
class Scratchpad:
    name: str
    cells: list[Cell] = field(default_factory=list)
    _proc: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _boot_path: str | None = field(default=None, repr=False)
    _skill_dirs: list[Path] = field(default_factory=list, repr=False)

    async def start(self) -> None:
        """Write the boot script to a temp file and launch the subprocess."""
        fd, path = tempfile.mkstemp(suffix=".py", prefix="anton_scratchpad_")
        os.write(fd, _BOOT_SCRIPT.encode())
        os.close(fd)
        self._boot_path = path

        env = os.environ.copy()
        if self._skill_dirs:
            env["ANTON_SKILL_DIRS"] = os.pathsep.join(str(d) for d in self._skill_dirs)

        self._proc = await asyncio.create_subprocess_exec(
            sys.executable, path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    async def execute(self, code: str) -> Cell:
        """Send code to the subprocess, read the JSON result, return a Cell."""
        if self._proc is None or self._proc.returncode is not None:
            # Process died — auto-note
            cell = Cell(
                code=code,
                stdout="",
                stderr="",
                error="Scratchpad process is not running. Use reset to restart.",
            )
            self.cells.append(cell)
            return cell

        payload = code + "\n" + _CELL_DELIM + "\n"
        self._proc.stdin.write(payload.encode())  # type: ignore[union-attr]
        await self._proc.stdin.drain()  # type: ignore[union-attr]

        try:
            result_data = await asyncio.wait_for(
                self._read_result(), timeout=_CELL_TIMEOUT
            )
        except asyncio.TimeoutError:
            self._proc.kill()
            await self._proc.wait()
            cell = Cell(
                code=code,
                stdout="",
                stderr="",
                error=f"Cell timed out after {_CELL_TIMEOUT}s. Process killed — state lost. Use reset to restart.",
            )
            self.cells.append(cell)
            return cell

        cell = Cell(
            code=code,
            stdout=result_data.get("stdout", ""),
            stderr=result_data.get("stderr", ""),
            error=result_data.get("error"),
        )
        self.cells.append(cell)
        return cell

    async def _read_result(self) -> dict:
        """Read lines from stdout until we get the result delimiters."""
        lines: list[str] = []
        in_result = False
        while True:
            raw = await self._proc.stdout.readline()  # type: ignore[union-attr]
            if not raw:
                # Process exited
                return {"stdout": "", "stderr": "", "error": "Process exited unexpectedly."}
            line = raw.decode().rstrip("\n")
            if line == _RESULT_START:
                in_result = True
                continue
            if line == _RESULT_END:
                break
            if in_result:
                lines.append(line)
        return json.loads("\n".join(lines))

    def view(self) -> str:
        """Format all cells with their outputs."""
        if not self.cells:
            return f"Scratchpad '{self.name}' is empty."

        parts: list[str] = []
        for i, cell in enumerate(self.cells):
            parts.append(f"--- Cell {i + 1} ---")
            parts.append(cell.code)
            if cell.stdout:
                parts.append(f"[stdout]\n{cell.stdout}")
            if cell.stderr:
                parts.append(f"[stderr]\n{cell.stderr}")
            if cell.error:
                parts.append(f"[error]\n{cell.error}")
            if not cell.stdout and not cell.stderr and not cell.error:
                parts.append("(no output)")
        return "\n".join(parts)

    async def reset(self) -> None:
        """Kill the process, clear cells, restart."""
        await self.close()
        self.cells.clear()
        await self.start()

    async def close(self) -> None:
        """Kill the process and clean up the boot script temp file."""
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


class ScratchpadManager:
    """Manages named scratchpad instances."""

    def __init__(self, skill_dirs: list[Path] | None = None) -> None:
        self._pads: dict[str, Scratchpad] = {}
        self._skill_dirs: list[Path] = skill_dirs or []

    async def get_or_create(self, name: str) -> Scratchpad:
        """Return existing pad or create + start a new one."""
        if name not in self._pads:
            pad = Scratchpad(name=name, _skill_dirs=self._skill_dirs)
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
