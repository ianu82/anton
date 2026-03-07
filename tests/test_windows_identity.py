from __future__ import annotations

import ctypes

from anton.execution_policy import ExecutionMode
from anton.windows_identity import _build_environment_block, appcontainer_name_for_mode


class TestWindowsIdentity:
    def test_appcontainer_name_for_mode(self):
        assert appcontainer_name_for_mode(ExecutionMode.READ_ONLY) == "AntonScratchpadReadOnly"
        assert appcontainer_name_for_mode(ExecutionMode.WORKSPACE_WRITE) == "AntonScratchpadWorkspaceWrite"

    def test_environment_block_uses_double_nul_terminated_pairs(self):
        block = _build_environment_block({"B": "2", "A": "1"})
        assert "".join(block).startswith("A=1\0B=2\0\0")
        raw = ctypes.string_at(ctypes.addressof(block), ctypes.sizeof(block))
        assert raw.decode("utf-16-le").endswith("\0\0")
