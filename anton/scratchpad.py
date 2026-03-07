"""Scratchpad — persistent Python subprocess for stateful, notebook-like execution."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import venv
from collections.abc import Callable
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
_INSTALL_TIMEOUT = 120
_MAX_OUTPUT = 10_000
_PROGRESS_MARKER = "__ANTON_PROGRESS__"
_NEED_SECRET_MARKER = "__ANTON_NEED_SECRET__"
_SET_ENV_MARKER = "__ANTON_SET_ENV__"
_SECRET_DONE_MARKER = "__ANTON_SECRET_DONE__"


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
    _secret_handler: Callable[[str, str], str | None] | None = field(default=None, repr=False)
    _workspace_path: Path | None = field(default=None, repr=False)
    _profile: str = field(default_factory=default_runtime_profile, repr=False)
    _runtime_path: Path | None = field(default=None, repr=False)
    _runtime_lock_hash: str = field(default="", repr=False)

    _MAX_VENV_RETRIES = 3

    def _overlay_dir(self) -> Path:
        override = os.environ.get("ANTON_SCRATCHPAD_BASE")
        if override:
            return Path(override).expanduser() / self.name
        if self._workspace_path is not None:
            return self._workspace_path / ".anton" / "scratchpad-venvs" / self.name
        return Path("~/.anton/scratchpad-venvs").expanduser() / self.name

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
        """Allocate the persistent overlay venv for this scratchpad."""
        import subprocess as _sp

        self._set_venv_paths()
        Path(self._venv_dir).parent.mkdir(parents=True, exist_ok=True)

        uv = self._find_uv()
        if uv:
            _sp.run(
                [uv, "venv", self._venv_dir,
                 "--python", sys.executable,
                 "--seed", "--quiet"],
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
        if self._workspace_path is not None:
            env["ANTON_WORKSPACE_PATH"] = str(self._workspace_path)
        env["ANTON_RUNTIME_PROFILE"] = self._profile

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
            package_missing=result_data.get("missing_import"),
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

            # Secret request — handle inline, send response to subprocess stdin
            if line.startswith(_NEED_SECRET_MARKER):
                payload = json.loads(line[len(_NEED_SECRET_MARKER):].strip())
                var_name = payload["variable_name"]
                prompt_text = payload.get("prompt_text", f"Enter value for {var_name}")

                yield f"Waiting for secret '{var_name}' — enter below"

                value: str | None = None
                if self._secret_handler:
                    value = self._secret_handler(var_name, prompt_text)

                if value is not None:
                    env_msg = json.dumps({"variable_name": var_name, "value": value})
                    self._proc.stdin.write(f"{_SET_ENV_MARKER} {env_msg}\n".encode())
                    self._proc.stdin.write(f"{_SECRET_DONE_MARKER}\n".encode())
                    await self._proc.stdin.drain()
                    yield f"Secret '{var_name}' received"
                else:
                    self._proc.stdin.write(f"{_SECRET_DONE_MARKER} error\n".encode())
                    await self._proc.stdin.drain()
                    yield f"Secret '{var_name}' skipped"
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
            if cell.package_missing:
                missing = cell.package_missing
                parts.append(
                    "[package_missing]\n"
                    f"package={missing.get('package', '')}\n"
                    f"import={missing.get('import_name', '')}\n"
                    f"suggested_profile={missing.get('suggested_profile', '')}\n"
                    f"next_action={missing.get('next_action', '')}"
                )
            if cell.error:
                parts.append(f"[error]\n{cell.error}")
            if not cell.stdout and not cell.logs and not cell.stderr and not cell.error and not cell.package_missing:
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
            if cell.package_missing:
                missing = cell.package_missing
                parts.append(
                    "**Package missing:** "
                    f"`{missing.get('package', '')}` "
                    f"(import `{missing.get('import_name', '')}`; "
                    f"suggested profile `{missing.get('suggested_profile', '')}`)"
                )

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
        """Kill the process and clean up the boot script temp file, but keep the overlay."""
        await self._stop_process()
        self._venv_dir = None
        self._venv_python = None

    async def destroy(self) -> None:
        """Kill the process and remove the persistent overlay for this scratchpad."""
        await self._stop_process()
        self._nuke_venv()

    async def install_packages(self, packages: list[str], *, source: str = "scratchpad.install") -> str:
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
        # Track successfully installed packages
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
        self._venv_dir = None
        self._venv_python = None
        self._installed_packages.clear()


class ScratchpadManager:
    """Manages named scratchpad instances."""

    def __init__(
        self,
        coding_provider: str = "anthropic",
        coding_model: str = "",
        coding_api_key: str = "",
        secret_handler: Callable[[str, str], str | None] | None = None,
        workspace_path: Path | None = None,
    ) -> None:
        self._pads: dict[str, Scratchpad] = {}
        self._coding_provider: str = coding_provider
        self._coding_model: str = coding_model
        self._coding_api_key: str = coding_api_key
        self._secret_handler = secret_handler
        self._workspace_path = workspace_path
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
                _secret_handler=self._secret_handler,
                _workspace_path=self._workspace_path,
                _profile=desired_profile,
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

    async def close_all(self) -> None:
        """Cleanup all scratchpads on session end."""
        for pad in self._pads.values():
            await pad.close()
        self._pads.clear()
