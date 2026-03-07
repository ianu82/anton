from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from anton.windows_sandbox import WindowsAccessGrant


def _grant_spec(path: Path, access: str, recursive: bool, *, effect: str = "grant") -> tuple[list[str], bool]:
    resolved = path.resolve()
    is_dir = resolved.is_dir()
    if access == "modify":
        rights = "(OI)(CI)(M)" if is_dir else "(M)"
    elif access == "read":
        rights = "(OI)(CI)(RX)" if is_dir else "(RX)"
    elif access == "full":
        rights = "(OI)(CI)(F)" if is_dir else "(F)"
    else:
        raise ValueError(f"Unsupported Windows access grant: {access}")

    if effect == "grant":
        acl_flag = "/grant:r"
    elif effect == "deny":
        acl_flag = "/deny"
    else:
        raise ValueError(f"Unsupported Windows ACL effect: {effect}")

    args = ["icacls", str(resolved), acl_flag, rights]
    if recursive and is_dir:
        args.extend(["/t", "/c", "/q"])
    return args, is_dir


def grant_access_to_sid(sid_string: str, grant: WindowsAccessGrant) -> None:
    if sys.platform != "win32":
        raise RuntimeError("Windows ACL helpers are only available on Windows hosts.")

    grant_path = Path(grant.path)
    if not grant_path.exists():
        return

    command, _ = _grant_spec(grant_path, grant.access, grant.recursive, effect=grant.effect)
    command[3] = f"*{sid_string}:{command[3]}"
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "icacls failed"
        raise RuntimeError(
            f"Failed to {grant.effect} Windows sandbox access for {grant.path}: {stderr}"
        )


def apply_access_grants(sid_string: str, grants: list[WindowsAccessGrant]) -> None:
    for grant in grants:
        grant_access_to_sid(sid_string, grant)
