from __future__ import annotations

import sys

from anton.windows_sandbox import WindowsSandboxConfig


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if sys.platform != "win32":
        raise RuntimeError("anton.windows_launcher can only run on Windows hosts.")
    if len(args) != 1:
        raise RuntimeError("anton.windows_launcher expects exactly one sandbox config payload.")
    WindowsSandboxConfig.from_base64(args[0])
    raise RuntimeError("Windows safe-mode launcher is not fully implemented yet.")


if __name__ == "__main__":
    raise SystemExit(main())
