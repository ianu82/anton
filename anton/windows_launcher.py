from __future__ import annotations

import os
import sys

from anton.windows_acl import apply_access_grants
from anton.windows_identity import ensure_appcontainer_profile, free_sid, launch_appcontainer_process
from anton.windows_job import (
    assign_process_to_job,
    close_handle,
    create_kill_on_close_job,
    resume_thread,
    wait_for_process_exit,
)
from anton.windows_sandbox import WindowsSandboxConfig


def launch_config(config: WindowsSandboxConfig) -> int:
    if sys.platform != "win32":
        raise RuntimeError("anton.windows_launcher can only run on Windows hosts.")

    profile = None
    job_handle = create_kill_on_close_job()
    process_handle: int | None = None
    thread_handle: int | None = None
    try:
        profile = ensure_appcontainer_profile(config.mode)
        apply_access_grants(profile.sid_string, config.access_grants())
        handles = launch_appcontainer_process(
            profile=profile,
            executable=config.executable,
            args=config.args,
            cwd=config.workspace_path,
            env=os.environ.copy(),
        )
        process_handle = handles.process_handle
        thread_handle = handles.thread_handle
        assign_process_to_job(job_handle, process_handle)
        resume_thread(thread_handle)
        return wait_for_process_exit(process_handle)
    finally:
        if thread_handle is not None:
            close_handle(thread_handle)
        if process_handle is not None:
            close_handle(process_handle)
        close_handle(job_handle)
        if profile is not None:
            free_sid(profile.sid_ptr)


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
