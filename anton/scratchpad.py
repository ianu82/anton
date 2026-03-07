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

from anton.runtime import (
    default_runtime_profile,
    ensure_runtime,
    load_runtime_profile,
    log_package_event,
    runtime_lock_hash,
    runtime_packages_for_profile,
    runtime_site_packages_path,
)

_CELL_TIMEOUT_DEFAULT = 120        # Default total timeout when no estimate given
_CELL_INACTIVITY_TIMEOUT = 30      # Max silence between output lines before killing
_CELL_INACTIVITY_AFTER_PROGRESS = 60  # Grace window after a progress() call
_INSTALL_TIMEOUT = 120
_MAX_OUTPUT = 10_000
_PROGRESS_MARKER = "__ANTON_PROGRESS__"
_KEEP_RECENT = 5  # Number of recent cells to keep during compaction


def _compute_timeouts(estimated_seconds: int) -> tuple[float, float]:
    """Compute (total_timeout, inactivity_timeout) from estimated execution time.

    - If estimate is 0: use defaults (120s total, 30s inactivity).
    - Otherwise: total = max(estimate * 2, estimate + 30) with no cap.
      Inactivity = max(estimate * 0.5, 30) — no hard cap, scales with estimate.
    """
    if estimated_seconds <= 0:
        return float(_CELL_TIMEOUT_DEFAULT), float(_CELL_INACTIVITY_TIMEOUT)
    total = max(estimated_seconds * 2, estimated_seconds + 30)
    inactivity = max(estimated_seconds * 0.5, 30)
    return float(total), float(inactivity)


_BOOT_SCRIPT_PATH = Path(__file__).parent / "scratchpad_boot.py"

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
    package_missing: dict[str, str] | None = None


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
    _workspace_path: Path | None = field(default=None, repr=False)
    _profile: str = field(default_factory=default_runtime_profile, repr=False)
    _runtime_path: Path | None = field(default=None, repr=False)
    _runtime_lock_hash: str = field(default="", repr=False)
    _venvs_base: Path = field(
        default_factory=lambda: Path("~/.anton/scratchpad-venvs").expanduser(),
        repr=False,
    )

    _MAX_VENV_RETRIES = 3

    def _overlay_dir(self) -> Path:
        override = os.environ.get("ANTON_SCRATCHPAD_BASE")
        if override:
            return Path(override).expanduser() / self.name
        return self._venvs_base / self.name

    def _overlay_metadata_path(self) -> Path:
        return self._overlay_dir() / ".anton-overlay.json"

    def _set_venv_paths(self) -> None:
        self._venv_dir = str(self._overlay_dir())
        if sys.platform == "win32":
            self._venv_python = os.path.join(self._venv_dir, "Scripts", "python.exe")
        else:
            self._venv_python = os.path.join(self._venv_dir, "bin", "python")

    def _ensure_runtime_profile(self) -> None:
        self._runtime_path = ensure_runtime(self._profile, workspace_path=self._workspace_path)
        self._runtime_lock_hash = runtime_lock_hash(load_runtime_profile(self._profile))

    def _load_overlay_metadata(self) -> dict | None:
        path = self._overlay_metadata_path()
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _write_overlay_metadata(self) -> None:
        if self._venv_dir is None:
            return
        payload = {
            "profile": self._profile,
            "runtime_lock_hash": self._runtime_lock_hash,
            "installed_packages": sorted(self._installed_packages),
        }
        self._overlay_metadata_path().write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _overlay_matches_runtime(self, metadata: dict | None) -> bool:
        if metadata is None:
            return False
        return (
            str(metadata.get("profile", "")) == self._profile
            and str(metadata.get("runtime_lock_hash", "")) == self._runtime_lock_hash
        )

    def _attach_runtime_site_packages(self) -> None:
        if self._venv_dir is None or self._runtime_path is None:
            return

        child_site = None
        for dirpath, dirnames, _ in os.walk(self._venv_dir):
            if "site-packages" in dirnames:
                child_site = Path(dirpath) / "site-packages"
                break
        if child_site is None:
            raise RuntimeError(f"Could not locate overlay site-packages in {self._venv_dir}")

        runtime_site = runtime_site_packages_path(self._runtime_path)
        (child_site / "_anton_runtime.pth").write_text(f"{runtime_site}\n", encoding="utf-8")

    def _ensure_venv(self) -> None:
        """Create or reuse a persistent overlay venv attached to an Anton runtime."""
        self._ensure_runtime_profile()
        self._set_venv_paths()

        metadata = self._load_overlay_metadata()
        if self._verify_venv_python() and self._overlay_matches_runtime(metadata):
            installed = metadata.get("installed_packages", []) if metadata else []
            self._installed_packages = {str(item).lower() for item in installed}
            self._attach_runtime_site_packages()
            return

        last_error: Exception | None = None
        for _attempt in range(1, self._MAX_VENV_RETRIES + 1):
            try:
                self._nuke_venv()
                self._create_venv()
                if self._verify_venv_python():
                    self._attach_runtime_site_packages()
                    self._write_overlay_metadata()
                    return
                raise RuntimeError(f"venv Python binary at {self._venv_python} is not functional")
            except Exception as exc:
                last_error = exc
                self._nuke_venv()

        raise RuntimeError(
            f"Failed to create a working scratchpad overlay after {self._MAX_VENV_RETRIES} attempts. "
            f"Last error: {last_error}. "
            f"Try running: python3 -c 'print(\"ok\")' to verify your Python installation."
        )

    @staticmethod
    def _find_uv() -> str | None:
        """Return the path to the ``uv`` binary, or *None* if unavailable."""
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

    def _create_venv(self) -> None:
        """Allocate the persistent overlay venv for this scratchpad."""
        import subprocess as _sp

        self._set_venv_paths()
        Path(self._venv_dir).parent.mkdir(parents=True, exist_ok=True)

        uv = self._find_uv()
        if uv:
            _sp.run(
                [uv, "venv", self._venv_dir, "--python", sys.executable, "--seed", "--quiet"],
                check=True,
                capture_output=True,
                timeout=30,
            )
        else:
            venv.create(self._venv_dir, system_site_packages=False, with_pip=False, clear=True)

        self._set_venv_paths()

    def _verify_venv_python(self) -> bool:
        """Check that the venv Python binary exists and can execute."""
        if self._venv_python is None:
            return False
        if not os.path.exists(self._venv_python):
            return False
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
        """Delete the overlay directory entirely so it can be recreated."""
        overlay_dir = self._overlay_dir()
        if overlay_dir.exists():
            try:
                shutil.rmtree(overlay_dir)
            except OSError:
                pass
        self._venv_dir = None
        self._venv_python = None
        self._installed_packages.clear()

    async def start(self) -> None:
        """Write the boot script to a temp file and launch the subprocess."""
        self._ensure_venv()

        boot_code = _BOOT_SCRIPT_PATH.read_text()
        fd, path = tempfile.mkstemp(suffix=".py", prefix="anton_scratchpad_")
        os.write(fd, boot_code.encode())
        os.close(fd)
        self._boot_path = path

        env = os.environ.copy()
        if self._coding_model:
            env["ANTON_SCRATCHPAD_MODEL"] = self._coding_model
        if self._coding_provider:
            env["ANTON_SCRATCHPAD_PROVIDER"] = self._coding_provider
        env["ANTON_RUNTIME_PROFILE"] = self._profile
        env["ANTON_SCRATCHPAD_NAME"] = self.name
        if self._workspace_path is not None:
            env["ANTON_WORKSPACE_PATH"] = str(self._workspace_path)
        if "ANTHROPIC_API_KEY" not in env and "ANTON_ANTHROPIC_API_KEY" in env:
            env["ANTHROPIC_API_KEY"] = env["ANTON_ANTHROPIC_API_KEY"]
        if "OPENAI_API_KEY" not in env and "ANTON_OPENAI_API_KEY" in env:
            env["OPENAI_API_KEY"] = env["ANTON_OPENAI_API_KEY"]
        if "OPENAI_BASE_URL" not in env and "ANTON_OPENAI_BASE_URL" in env:
            env["OPENAI_BASE_URL"] = env["ANTON_OPENAI_BASE_URL"]
        if self._coding_api_key:
            sdk_key = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "openai-compatible": "OPENAI_API_KEY",
            }.get(self._coding_provider, "")
            if sdk_key and sdk_key not in env:
                env[sdk_key] = self._coding_api_key
        uv = self._find_uv()
        if uv:
            env["ANTON_UV_PATH"] = uv

        anton_root = str(Path(__file__).resolve().parent.parent)
        python_path = env.get("PYTHONPATH", "")
        if anton_root not in python_path:
            env["PYTHONPATH"] = anton_root + (os.pathsep + python_path if python_path else "")

        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._venv_python,
                path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=(sys.platform != "win32"),
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
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
        """Send code to the subprocess, read the JSON result, return a Cell."""
        async for item in self.execute_streaming(
            code,
            description=description,
            estimated_time=estimated_time,
            estimated_seconds=estimated_seconds,
        ):
            if isinstance(item, Cell):
                return item
        return Cell(code=code, stdout="", stderr="", error="No result produced.")

    async def execute_streaming(
        self,
        code: str,
        *,
        description: str = "",
        estimated_time: str = "",
        estimated_seconds: int = 0,
        cancel_event: asyncio.Event | None = None,
    ):
        """Async generator that sends code and yields progress strings and a final Cell."""
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
                cancel_event=cancel_event,
            ):
                if isinstance(item, str):
                    yield item
                else:
                    result_data = item
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            self._kill_tree()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            error_msg = (
                f"{exc}. Process killed — state lost. Use reset to restart.\n\n"
                "If a database query was running, it may still be executing server-side.\n"
                "To check and cancel: run SHOW PROCESSLIST (MySQL) or\n"
                "SELECT * FROM information_schema.processlist WHERE status='running' and cancel with KILL <id>.\n"
                "For Snowflake: use SHOW RUNNING QUERIES and SELECT SYSTEM$CANCEL_ALL_QUERIES(<session_id>)."
            )
            cell = Cell(
                code=code,
                stdout="",
                stderr="",
                error=error_msg,
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
            package_missing=result_data.get("missing_import"),
        )
        self.cells.append(cell)
        yield cell

    async def _read_result(
        self,
        *,
        total_timeout: float = _CELL_TIMEOUT_DEFAULT,
        inactivity_timeout: float = _CELL_INACTIVITY_TIMEOUT,
        cancel_event: asyncio.Event | None = None,
    ):
        """Async generator that reads lines from stdout until result delimiters."""
        import time as _time

        lines: list[str] = []
        in_result = False
        start = _time.monotonic()
        current_inactivity = inactivity_timeout

        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError("Cancelled by user")
            elapsed = _time.monotonic() - start
            remaining_total = total_timeout - elapsed
            if remaining_total <= 0:
                raise asyncio.TimeoutError(f"Cell timed out after {total_timeout:.0f}s total")

            line_timeout = min(current_inactivity, remaining_total)
            try:
                raw = await asyncio.wait_for(
                    self._proc.stdout.readline(),  # type: ignore[union-attr]
                    timeout=line_timeout,
                )
            except asyncio.TimeoutError:
                elapsed_now = _time.monotonic() - start
                if elapsed_now >= total_timeout - 0.5:
                    raise asyncio.TimeoutError(
                        f"Cell timed out after {total_timeout:.0f}s total"
                    ) from None
                raise asyncio.TimeoutError(
                    f"Cell killed after {current_inactivity:.0f}s of inactivity "
                    f"(no output or progress() calls)"
                ) from None

            if not raw:
                yield {"stdout": "", "stderr": "", "error": "Process exited unexpectedly."}
                return

            line = raw.decode().rstrip("\r\n")
            if line.startswith(_PROGRESS_MARKER):
                current_inactivity = max(current_inactivity, _CELL_INACTIVITY_AFTER_PROGRESS)
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
            if cell.package_missing:
                missing = cell.package_missing
                parts.append(
                    "[package_missing]\n"
                    f"package={missing.get('package', '')}\n"
                    f"import={missing.get('import_name', '')}\n"
                    f"suggested_profile={missing.get('suggested_profile', '')}\n"
                    f"next_action={missing.get('next_action', '')}"
                )
            if not cell.stdout and not cell.logs and not cell.stderr and not cell.error:
                parts.append("(no output)")
        return "\n".join(parts)

    @staticmethod
    def _truncate_output(text: str, max_lines: int = 20, max_chars: int = 2000) -> str:
        """Truncate output to *max_lines* / *max_chars*, whichever is shorter."""
        lines = text.split("\n")
        if len(lines) > max_lines:
            kept = "\n".join(lines[:max_lines])
            remaining = len(lines) - max_lines
            return kept + f"\n... ({remaining} more lines)"
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
                last_line = cell.error.strip().split("\n")[-1]
                parts.append(f"**Error:** `{last_line}`")
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

    def _compact_cells(self) -> bool:
        """Collapse old cells into a single summary cell to reduce context size."""
        if len(self.cells) <= _KEEP_RECENT + 1:
            return False

        to_compact = self.cells[:-_KEEP_RECENT]
        recent = self.cells[-_KEEP_RECENT:]

        summary_lines: list[str] = []
        for i, cell in enumerate(to_compact, 1):
            status = "error" if cell.error else "ok"
            desc = cell.description or f"Cell {i}"
            first_line = ""
            output = cell.stdout or cell.error or ""
            if output:
                first_line = output.strip().split("\n")[0][:120]
            summary_lines.append(f"  [{status}] {desc}: {first_line}")

        summary_text = f"# Compacted {len(to_compact)} earlier cells:\n" + "\n".join(summary_lines)
        summary_cell = Cell(
            code="# (compacted — see summary above)",
            stdout=summary_text,
            stderr="",
            error=None,
            description=f"Summary of cells 1–{len(to_compact)}",
        )
        self.cells = [summary_cell] + recent
        return True

    async def cancel_running(self) -> None:
        """Kill the current execution and restart the subprocess."""
        if self._proc is None or self._proc.returncode is not None:
            return
        self._kill_tree()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        self.cells.append(
            Cell(
                code="# (cancelled by user)",
                stdout="",
                stderr="",
                error="Cancelled by user.",
                description="Cancelled",
            )
        )
        self._proc = None
        await self.start()

    async def _stop_process(self) -> None:
        """Kill the subprocess and delete the boot script, but keep the overlay."""
        if self._proc is not None and self._proc.returncode is None:
            try:
                self._kill_tree()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                pass
        if self._proc is not None:
            for attr in ("stdin", "stdout", "stderr"):
                pipe = getattr(self._proc, attr, None)
                if pipe and not pipe.is_closing() if hasattr(pipe, "is_closing") else False:
                    pipe.close()
        self._proc = None
        if self._boot_path is not None:
            try:
                os.unlink(self._boot_path)
            except OSError:
                pass
            self._boot_path = None

    def _kill_tree(self) -> None:
        """Kill the subprocess and all its children via process group."""
        if self._proc is None or self._proc.returncode is not None:
            return
        pid = self._proc.pid
        if sys.platform != "win32":
            import signal

            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass
        else:
            self._proc.kill()

    async def reset(self) -> None:
        """Kill the process, clear cells, restart."""
        await self._stop_process()
        self.cells.clear()
        if not self._verify_venv_python():
            self._nuke_venv()
        await self.start()

    async def close(self) -> None:
        """Kill the process and clean up the boot script temp file, but keep the overlay."""
        await self._stop_process()
        self._venv_dir = None
        self._venv_python = None

    async def destroy(self) -> None:
        """Kill the process and remove the persistent overlay."""
        await self._stop_process()
        self._nuke_venv()

    async def install_packages(self, packages: list[str], *, source: str = "scratchpad.install") -> str:
        """Install packages into the scratchpad overlay via pip (or uv pip)."""
        if not packages:
            return "No packages specified."
        needed = [p for p in packages if p.lower() not in self._installed_packages]
        if not needed:
            return "All packages already installed."
        self._ensure_venv()

        uv = self._find_uv()
        if uv:
            cmd = [uv, "pip", "install", "--python", self._venv_python, *needed]
        else:
            cmd = [self._venv_python, "-m", "pip", "install", "--no-input", *needed]

        for package in needed:
            log_package_event(
                "explicit_install",
                package=package,
                scratchpad=self.name,
                source=source,
                status="started",
                workspace_path=self._workspace_path,
            )

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
            for package in needed:
                log_package_event(
                    "install_failure",
                    package=package,
                    scratchpad=self.name,
                    source=source,
                    status="timed_out",
                    error=f"Install timed out after {_INSTALL_TIMEOUT}s.",
                    workspace_path=self._workspace_path,
                )
            return f"Install timed out after {_INSTALL_TIMEOUT}s."
        output = stdout.decode()
        if proc.returncode != 0:
            for package in needed:
                log_package_event(
                    "install_failure",
                    package=package,
                    scratchpad=self.name,
                    source=source,
                    status=f"exit_{proc.returncode}",
                    error=output,
                    workspace_path=self._workspace_path,
                )
            return f"Install failed (exit {proc.returncode}):\n{output}"
        for p in needed:
            self._installed_packages.add(p.lower())
        self._write_overlay_metadata()
        return output

    async def ensure_profile(self, profile: str) -> None:
        desired = profile or default_runtime_profile()
        if desired == self._profile:
            return
        await self._stop_process()
        self.cells.clear()
        self._profile = desired
        self._nuke_venv()


class ScratchpadManager:
    """Manages named scratchpad instances."""

    def __init__(
        self,
        coding_provider: str = "anthropic",
        coding_model: str = "",
        coding_api_key: str = "",
        workspace_path: Path | None = None,
    ) -> None:
        self._pads: dict[str, Scratchpad] = {}
        self._coding_provider: str = coding_provider
        self._coding_model: str = coding_model
        self._coding_api_key: str = coding_api_key
        self._workspace_path = workspace_path
        if workspace_path is not None:
            self._venvs_base = workspace_path / ".anton" / "scratchpad-venvs"
        else:
            self._venvs_base = Path("~/.anton/scratchpad-venvs").expanduser()
        self._available_packages: list[str] = self.probe_packages()

    @staticmethod
    def probe_packages() -> list[str]:
        """Return sorted package names from Anton's default managed runtime profile."""
        return sorted(runtime_packages_for_profile(default_runtime_profile()))

    def available_packages(self) -> list[str]:
        packages = set(self._available_packages)
        for pad in self._pads.values():
            packages.update(pad._installed_packages)
        return sorted(packages)

    async def get_or_create(self, name: str, *, profile: str | None = None) -> Scratchpad:
        """Return existing pad or create + start a new one."""
        desired_profile = profile or default_runtime_profile()
        if name not in self._pads:
            pad = Scratchpad(
                name=name,
                _coding_provider=self._coding_provider,
                _coding_model=self._coding_model,
                _coding_api_key=self._coding_api_key,
                _workspace_path=self._workspace_path,
                _profile=desired_profile,
                _venvs_base=self._venvs_base,
            )
            await pad.start()
            self._pads[name] = pad
        else:
            await self._pads[name].ensure_profile(desired_profile)
            if self._pads[name]._proc is None or self._pads[name]._proc.returncode is not None:
                await self._pads[name].start()
        return self._pads[name]

    async def remove(self, name: str) -> str:
        """Kill and delete a scratchpad."""
        pad = self._pads.pop(name, None)
        if pad is None:
            return f"No scratchpad named '{name}'."
        await pad.destroy()
        return f"Scratchpad '{name}' removed."

    def list_pads(self) -> list[str]:
        return list(self._pads.keys())

    async def cancel_all_running(self) -> None:
        """Cancel running executions in all scratchpads and restart them."""
        for pad in self._pads.values():
            await pad.cancel_running()

    async def close_all(self) -> None:
        """Cleanup all scratchpads on session end."""
        for pad in self._pads.values():
            await pad.close()
        self._pads.clear()
