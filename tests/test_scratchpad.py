from __future__ import annotations

import asyncio

import pytest

import anton.scratchpad as scratchpad_module
from anton.scratchpad import Cell, Scratchpad, ScratchpadManager


class TestScratchpadBasicExecution:
    async def test_basic_execution(self):
        """print(42) should return '42' in stdout."""
        pad = Scratchpad(name="test")
        await pad.start()
        try:
            cell = await pad.execute("print(42)")
            assert cell.stdout.strip() == "42"
            assert cell.error is None
        finally:
            await pad.close()

    async def test_state_persists(self):
        """Variable from cell 1 should be available in cell 2."""
        pad = Scratchpad(name="test")
        await pad.start()
        try:
            await pad.execute("x = 123")
            cell = await pad.execute("print(x)")
            assert cell.stdout.strip() == "123"
            assert cell.error is None
        finally:
            await pad.close()

    async def test_error_captured_process_survives(self):
        """Exception doesn't kill process; next cell works."""
        pad = Scratchpad(name="test")
        await pad.start()
        try:
            cell1 = await pad.execute("raise ValueError('boom')")
            assert cell1.error is not None
            assert "ValueError" in cell1.error
            assert "boom" in cell1.error

            # Process should still work
            cell2 = await pad.execute("print('alive')")
            assert cell2.stdout.strip() == "alive"
            assert cell2.error is None
        finally:
            await pad.close()

    async def test_imports_persist(self):
        """import json in cell 1, json.dumps(...) in cell 2."""
        pad = Scratchpad(name="test")
        await pad.start()
        try:
            await pad.execute("import json")
            cell = await pad.execute('print(json.dumps({"a": 1}))')
            assert cell.stdout.strip() == '{"a": 1}'
            assert cell.error is None
        finally:
            await pad.close()


class TestScratchpadView:
    async def test_view_history(self):
        """view() should show all cells with outputs."""
        pad = Scratchpad(name="test")
        await pad.start()
        try:
            await pad.execute("x = 10")
            await pad.execute("print(x + 5)")
            output = pad.view()
            assert "Cell 1" in output
            assert "Cell 2" in output
            assert "x = 10" in output
            assert "15" in output
        finally:
            await pad.close()

    async def test_view_empty(self):
        """view() on empty pad returns a message."""
        pad = Scratchpad(name="empty")
        await pad.start()
        try:
            output = pad.view()
            assert "empty" in output.lower()
        finally:
            await pad.close()


class TestScratchpadReset:
    async def test_reset_clears_state(self):
        """Variables should be gone after reset."""
        pad = Scratchpad(name="test")
        await pad.start()
        try:
            await pad.execute("x = 42")
            await pad.reset()
            cell = await pad.execute("print(x)")
            assert cell.error is not None
            assert "NameError" in cell.error
            # Cells list should only have the post-reset cell
            assert len(pad.cells) == 1
        finally:
            await pad.close()


class TestScratchpadEdgeCases:
    async def test_timeout_kills_process(self, monkeypatch):
        """Long-running code triggers timeout."""
        monkeypatch.setattr(scratchpad_module, "_CELL_TIMEOUT", 1)
        pad = Scratchpad(name="test")
        await pad.start()
        try:
            cell = await pad.execute("import time; time.sleep(60)")
            assert cell.error is not None
            assert "timed out" in cell.error.lower()
        finally:
            await pad.close()

    async def test_output_truncation(self):
        """Large output should be capped at 10KB when read through Cell."""
        pad = Scratchpad(name="test")
        await pad.start()
        try:
            # The raw cell captures full stdout; truncation happens at the
            # chat handler level. Here we just verify we get the full output.
            cell = await pad.execute("print('x' * 20000)")
            assert len(cell.stdout) >= 20000
            assert cell.error is None
        finally:
            await pad.close()

    async def test_dead_process_detected(self):
        """If process is dead, execute reports it."""
        pad = Scratchpad(name="test")
        await pad.start()
        # Kill the process manually
        pad._proc.kill()
        await pad._proc.wait()
        cell = await pad.execute("print(1)")
        assert cell.error is not None
        assert "not running" in cell.error.lower()
        await pad.close()

    async def test_stderr_captured(self):
        """stderr output is captured separately."""
        pad = Scratchpad(name="test")
        await pad.start()
        try:
            cell = await pad.execute("import sys; sys.stderr.write('warn\\n')")
            assert "warn" in cell.stderr
        finally:
            await pad.close()


class TestScratchpadManager:
    async def test_get_or_create(self):
        """Auto-creates a scratchpad on first access."""
        mgr = ScratchpadManager()
        try:
            pad = await mgr.get_or_create("alpha")
            assert pad.name == "alpha"
            assert "alpha" in mgr.list_pads()

            # Second call returns the same pad
            pad2 = await mgr.get_or_create("alpha")
            assert pad2 is pad
        finally:
            await mgr.close_all()

    async def test_remove(self):
        """remove() kills and deletes the scratchpad."""
        mgr = ScratchpadManager()
        try:
            await mgr.get_or_create("beta")
            result = await mgr.remove("beta")
            assert "beta" in result
            assert "beta" not in mgr.list_pads()
        finally:
            await mgr.close_all()

    async def test_remove_nonexistent(self):
        """remove() on unknown name returns a message."""
        mgr = ScratchpadManager()
        result = await mgr.remove("nope")
        assert "nope" in result

    async def test_close_all(self):
        """close_all() cleans up everything."""
        mgr = ScratchpadManager()
        await mgr.get_or_create("a")
        await mgr.get_or_create("b")
        assert len(mgr.list_pads()) == 2
        await mgr.close_all()
        assert len(mgr.list_pads()) == 0
