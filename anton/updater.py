"""Auto-update check for Anton."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def check_and_update(console, settings) -> None:
    """Check for a newer version of Anton and self-update if available.

    Skips silently on any error — the update check must never crash Anton.
    """
    try:
        _check_and_update(console, settings)
    except Exception:
        return


def _check_and_update(console, settings) -> None:
    if settings.disable_autoupdates:
        return

    if shutil.which("uv") is None:
        return

    # Check cache — skip if checked less than 1 hour ago
    cache_file = Path("~/.anton/.last_update_check").expanduser()
    try:
        if cache_file.is_file():
            last_check = float(cache_file.read_text().strip())
            if time.time() - last_check < 3600:
                return
    except (ValueError, OSError):
        pass

    # Fetch remote __init__.py to get __version__
    import urllib.request

    url = "https://raw.githubusercontent.com/mindsdb/anton/main/anton/__init__.py"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2) as resp:
            content = resp.read().decode("utf-8")
    except Exception:
        return

    # Parse remote version
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        return
    remote_version_str = match.group(1)

    # Compare versions
    from packaging.version import InvalidVersion, Version

    import anton

    try:
        local_ver = Version(anton.__version__)
        remote_ver = Version(remote_version_str)
    except InvalidVersion:
        return

    # Write cache timestamp regardless of whether update is needed
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(str(time.time()))
    except OSError:
        pass

    if remote_ver <= local_ver:
        return

    # Newer version available — upgrade
    console.print(f"  Updating anton {local_ver} \u2192 {remote_ver}...")

    try:
        result = subprocess.run(
            ["uv", "tool", "upgrade", "anton"],
            capture_output=True,
            timeout=60,
        )
    except Exception:
        console.print("  [dim]Update failed, continuing...[/]")
        return

    if result.returncode != 0:
        console.print("  [dim]Update failed, continuing...[/]")
        return

    console.print("  \u2713 Updated!")

    # Re-exec so the user gets the new version immediately
    if sys.platform == "win32":
        subprocess.Popen([sys.executable] + sys.argv)
        sys.exit(0)
    else:
        os.execvp(sys.argv[0], sys.argv)
