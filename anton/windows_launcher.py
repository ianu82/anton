from __future__ import annotations

import os
import subprocess
import sys

from anton.windows_job import (
    assign_process_to_job,
    close_handle,
    create_kill_on_close_job,
    resume_thread,
)
from anton.windows_sandbox import WindowsSandboxConfig

CREATE_SUSPENDED = getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)


def launch_config(config: WindowsSandboxConfig) -> int:
    if sys.platform != "win32":
        raise RuntimeError("anton.windows_launcher can only run on Windows hosts.")

    job_handle = create_kill_on_close_job()
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            [config.executable, *config.args],
            cwd=config.workspace_path,
            env=os.environ.copy(),
            close_fds=False,
            creationflags=CREATE_SUSPENDED,
        )
        assign_process_to_job(job_handle, int(proc._handle))  # pyright: ignore[reportAttributeAccessIssue]
        resume_thread(int(proc._thread))  # pyright: ignore[reportAttributeAccessIssue]
        return int(proc.wait())
    except Exception:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        raise
    finally:
        close_handle(job_handle)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if sys.platform != "win32":
        raise RuntimeError("anton.windows_launcher can only run on Windows hosts.")
    if len(args) != 1:
        raise RuntimeError("anton.windows_launcher expects exactly one sandbox config payload.")
    config = WindowsSandboxConfig.from_base64(args[0])
    return launch_config(config)


if __name__ == "__main__":
    raise SystemExit(main())
