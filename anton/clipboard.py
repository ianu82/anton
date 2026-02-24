"""Clipboard image/file paste support.

Grabs images or file paths from the system clipboard (macOS/Windows),
saves uploads to .anton/uploads/, and provides path-parsing utilities
shared with the drag-and-drop logic in chat.py.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ClipboardImage:
    """An image grabbed from the clipboard."""

    image: Any  # PIL.Image.Image (typed as Any to avoid hard Pillow dep)
    width: int
    height: int
    mode: str


@dataclass
class ClipboardResult:
    """Result of inspecting the system clipboard."""

    image: ClipboardImage | None = None
    file_paths: list[Path] = field(default_factory=list)
    text: str = ""


@dataclass
class UploadedFile:
    """Metadata for a saved clipboard image."""

    path: Path
    original_type: str
    width: int
    height: int
    size_bytes: int
    format: str


# ---------------------------------------------------------------------------
# Clipboard inspection
# ---------------------------------------------------------------------------


def is_clipboard_supported() -> bool:
    """Return True if we can attempt clipboard image grabs on this platform."""
    if platform.system() not in ("Darwin", "Windows"):
        return False
    try:
        from PIL import ImageGrab  # noqa: F401

        return True
    except ImportError:
        return False


def clipboard_unavailable_reason() -> str | None:
    """Return a reason string if clipboard is unavailable, or None if OK.

    Distinguishes between unsupported platform and missing Pillow.
    """
    if platform.system() not in ("Darwin", "Windows"):
        return "unsupported_platform"
    try:
        from PIL import ImageGrab  # noqa: F401
        return None
    except ImportError:
        return "missing_pillow"


def grab_clipboard() -> ClipboardResult:
    """Inspect the system clipboard; try image first, then text/file paths."""
    result = ClipboardResult()

    # Try image via Pillow
    img = _grab_image()
    if img is not None:
        result.image = ClipboardImage(
            image=img,
            width=img.size[0],
            height=img.size[1],
            mode=img.mode,
        )
        return result

    # Fall back to text (may contain file paths)
    text = _grab_text()
    if text:
        # Check if the text looks like file paths
        paths = parse_dropped_paths(text)
        if paths:
            result.file_paths = paths
        else:
            result.text = text

    return result


def _grab_image() -> Any | None:
    """Attempt to grab an image from the clipboard via Pillow.

    Returns a PIL Image or None.  On macOS, ``grabclipboard()`` may return
    a *list* of file paths when the user copied files in Finder — we skip
    those (they'll be handled by the text path).
    """
    try:
        from PIL import ImageGrab
    except ImportError:
        return None

    try:
        clip = ImageGrab.grabclipboard()
    except Exception:
        return None

    if clip is None:
        return None

    # macOS quirk: Finder-copied files come back as a list of paths
    if isinstance(clip, list):
        return None

    # Ensure it's an actual PIL Image
    try:
        from PIL import Image

        if isinstance(clip, Image.Image):
            return clip
    except Exception:
        pass

    return None


def _grab_text() -> str:
    """Grab text from the clipboard using platform-native CLI tools."""
    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.run(
                ["pbpaste"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        elif system == "Windows":
            return subprocess.run(
                ["powershell", "-Command", "Get-Clipboard"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# File saving
# ---------------------------------------------------------------------------


def save_clipboard_image(image: Any, uploads_dir: Path) -> UploadedFile:
    """Save a PIL Image to *uploads_dir* as PNG and write a .meta.json sidecar.

    Parameters
    ----------
    image:
        A ``PIL.Image.Image`` (or the ``ClipboardImage.image`` attribute).
    uploads_dir:
        Directory to save into (created if missing).

    Returns
    -------
    UploadedFile with the saved path and metadata.
    """
    uploads_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    # Hash a few pixels for uniqueness
    raw = image.tobytes()[:4096]
    h = hashlib.sha256(raw).hexdigest()[:8]
    filename = f"clipboard_{ts}_{h}.png"
    filepath = uploads_dir / filename

    image.save(filepath, format="PNG")
    size_bytes = filepath.stat().st_size

    # Sidecar metadata
    meta = {
        "source": "clipboard",
        "timestamp": ts,
        "width": image.size[0],
        "height": image.size[1],
        "mode": image.mode,
        "format": "PNG",
        "size_bytes": size_bytes,
    }
    meta_path = filepath.with_suffix(".png.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    return UploadedFile(
        path=filepath,
        original_type="clipboard",
        width=image.size[0],
        height=image.size[1],
        size_bytes=size_bytes,
        format="PNG",
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def cleanup_old_uploads(uploads_dir: Path, max_age_days: int = 7) -> int:
    """Delete uploads older than *max_age_days*.  Returns count of files removed."""
    if not uploads_dir.is_dir():
        return 0

    cutoff = time.time() - (max_age_days * 86400)
    removed = 0

    for f in list(uploads_dir.iterdir()):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            continue

    return removed


# ---------------------------------------------------------------------------
# Path parsing (shared with drag-and-drop in chat.py)
# ---------------------------------------------------------------------------


def parse_dropped_paths(text: str) -> list[Path]:
    r"""Detect file paths from terminal drag-and-drop or clipboard text.

    When users drag files into the terminal, the shell pastes the path as:
      - /path/to/file           (macOS/Linux, no spaces)
      - /path/to/file\ name    (macOS, escaped spaces)
      - '/path/to/file name'   (macOS, quoted)
      - "C:\Users\foo\file"    (Windows, quoted)
      - C:\Users\foo\file      (Windows, no spaces)
    Multiple files may be separated by spaces or newlines.
    """
    paths: list[Path] = []

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            tokens = shlex.split(line)
        except ValueError:
            tokens = [line]

        for token in tokens:
            if len(token) < 2:
                continue
            candidate = Path(token)
            try:
                if candidate.is_absolute() and candidate.exists():
                    paths.append(candidate)
            except OSError:
                continue

    return paths
