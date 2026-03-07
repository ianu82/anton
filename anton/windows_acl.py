from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from anton.windows_sandbox import WindowsAccessGrant


def _grant_spec(path: Path, access: str, recursive: bool) -> tuple[list[str], bool]:
    resolved = path.resolve()
    is_dir = resolved.is_dir()
    if access == "modify":
        rights = "(OI)(CI)(M)" if is_dir else "(M)"
    elif access == "read":
        rights = "(OI)(CI)(RX)" if is_dir else "(RX)"
    else:
        raise ValueError(f"Unsupported Windows access grant: {access}")

    args = ["icacls", str(resolved), "/grant", rights]
    if recursive and is_dir:
        args.extend(["/t", "/c", "/q"])
    return args, is_dir


def grant_access_to_sid(sid_string: str, grant: WindowsAccessGrant) -> None:
    if sys.platform != "win32":
        raise RuntimeError("Windows ACL helpers are only available on Windows hosts.")

    grant_path = Path(grant.path)
    if not grant_path.exists():
        return

    command, _ = _grant_spec(grant_path, grant.access, grant.recursive)
    command[3] = f"*{sid_string}:{command[3]}"
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "icacls failed"
        raise RuntimeError(f"Failed to grant Windows sandbox access for {grant.path}: {stderr}")


def apply_access_grants(sid_string: str, grants: list[WindowsAccessGrant]) -> None:
    for grant in grants:
        grant_access_to_sid(sid_string, grant)
