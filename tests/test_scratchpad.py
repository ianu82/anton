from __future__ import annotations

import asyncio
import os

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


class TestScratchpadRenderNotebook:
    async def test_render_notebook_basic(self):
        """Produces markdown with code blocks and output."""
        pad = Scratchpad(name="main")
        await pad.start()
        try:
            await pad.execute("x = 1")
            await pad.execute("print(x + 1)")
            md = pad.render_notebook()
            assert "## Scratchpad: main (2 cells)" in md
            assert "### Cell 1" in md
            assert "```python" in md
            assert "x = 1" in md
            assert "**Output:**" in md
            assert "2" in md
        finally:
            await pad.close()

    async def test_render_notebook_empty(self):
        """Empty pad returns a message."""
        pad = Scratchpad(name="empty")
        await pad.start()
        try:
            md = pad.render_notebook()
            assert "no cells" in md.lower()
        finally:
            await pad.close()

    async def test_render_notebook_skips_empty_cells(self):
        """Whitespace-only cells are filtered out."""
        pad = Scratchpad(name="gaps")
        await pad.start()
        try:
            await pad.execute("print('a')")
            await pad.execute("   \n  ")
            await pad.execute("print('b')")
            md = pad.render_notebook()
            assert "(2 cells)" in md
            assert "Cell 2" not in md  # whitespace cell skipped
            assert "Cell 1" in md
            assert "Cell 3" in md
        finally:
            await pad.close()

    async def test_render_notebook_truncates_long_output(self):
        """Long stdout shows 'more lines' indicator."""
        pad = Scratchpad(name="long")
        await pad.start()
        try:
            await pad.execute("for i in range(50): print(i)")
            md = pad.render_notebook()
            assert "more lines" in md
        finally:
            await pad.close()

    async def test_render_notebook_error_summary(self):
        """Only last traceback line shown, not full trace."""
        pad = Scratchpad(name="err")
        await pad.start()
        try:
            await pad.execute("raise ValueError('boom')")
            md = pad.render_notebook()
            assert "**Error:**" in md
            assert "ValueError: boom" in md
            # Full traceback details should NOT be present
            assert "Traceback" not in md
        finally:
            await pad.close()

    async def test_render_notebook_hides_stderr_without_error(self):
        """Warnings (stderr only, no error) are filtered out of output sections."""
        pad = Scratchpad(name="warn")
        await pad.start()
        try:
            await pad.execute("import sys; sys.stderr.write('some warning\\n')")
            md = pad.render_notebook()
            # stderr content should NOT appear as output
            assert "**Output:**" not in md
            assert "**Error:**" not in md
        finally:
            await pad.close()

    async def test_truncate_output_lines(self):
        """Respects line limit."""
        text = "\n".join(f"line {i}" for i in range(50))
        result = Scratchpad._truncate_output(text, max_lines=10)
        assert "line 0" in result
        assert "line 9" in result
        assert "line 10" not in result
        assert "(40 more lines)" in result

    async def test_truncate_output_chars(self):
        """Respects char limit."""
        text = "\n".join("x" * 80 for _ in range(5))
        result = Scratchpad._truncate_output(text, max_lines=100, max_chars=200)
        assert "(truncated)" in result
        assert len(result) < len(text)


class TestScratchpadEnvironment:
    async def test_env_vars_accessible(self, monkeypatch):
        """Secrets from .anton/.env (in os.environ) are accessible in scratchpad."""
        monkeypatch.setenv("MY_TEST_SECRET", "s3cret_value")
        pad = Scratchpad(name="env-test")
        await pad.start()
        try:
            cell = await pad.execute(
                "import os; print(os.environ.get('MY_TEST_SECRET', 'NOT_FOUND'))"
            )
            assert cell.stdout.strip() == "s3cret_value"
        finally:
            await pad.close()

    async def test_get_llm_available_when_model_set(self):
        """get_llm() should be injected when ANTON_SCRATCHPAD_MODEL is set."""
        pad = Scratchpad(name="llm-test", _coding_model="claude-test-model")
        await pad.start()
        try:
            cell = await pad.execute("llm = get_llm(); print(llm.model)")
            assert cell.stdout.strip() == "claude-test-model"
            assert cell.error is None
        finally:
            await pad.close()

    async def test_get_llm_not_available_without_model(self):
        """get_llm() should not be in namespace when no model is configured."""
        pad = Scratchpad(name="no-llm")
        await pad.start()
        try:
            cell = await pad.execute("get_llm()")
            assert cell.error is not None
            assert "NameError" in cell.error
        finally:
            await pad.close()

    async def test_agentic_loop_available_when_model_set(self):
        """agentic_loop() should be injected alongside get_llm()."""
        pad = Scratchpad(name="agentic-test", _coding_model="claude-test-model")
        await pad.start()
        try:
            cell = await pad.execute("print(callable(agentic_loop))")
            assert cell.stdout.strip() == "True"
            assert cell.error is None
        finally:
            await pad.close()

    async def test_agentic_loop_not_available_without_model(self):
        """agentic_loop() should not be in namespace when no model is configured."""
        pad = Scratchpad(name="no-agentic")
        await pad.start()
        try:
            cell = await pad.execute("agentic_loop()")
            assert cell.error is not None
            assert "NameError" in cell.error
        finally:
            await pad.close()

    async def test_generate_object_available_when_model_set(self):
        """generate_object() should be available on the LLM wrapper."""
        pad = Scratchpad(name="genobj-test", _coding_model="claude-test-model")
        await pad.start()
        try:
            cell = await pad.execute(
                "llm = get_llm(); print(hasattr(llm, 'generate_object') and callable(llm.generate_object))"
            )
            assert cell.stdout.strip() == "True"
            assert cell.error is None
        finally:
            await pad.close()

    async def test_api_key_bridged(self, monkeypatch):
        """ANTON_ANTHROPIC_API_KEY should be bridged to ANTHROPIC_API_KEY."""
        monkeypatch.setenv("ANTON_ANTHROPIC_API_KEY", "sk-ant-test-123")
        # Remove ANTHROPIC_API_KEY if set, to test the bridge
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        pad = Scratchpad(name="key-test", _coding_model="test-model")
        await pad.start()
        try:
            cell = await pad.execute(
                "import os; print(os.environ.get('ANTHROPIC_API_KEY', 'MISSING'))"
            )
            assert cell.stdout.strip() == "sk-ant-test-123"
        finally:
            await pad.close()


class TestScratchpadVenv:
    async def test_venv_created_on_start(self):
        """Venv directory should be created when the scratchpad starts."""
        pad = Scratchpad(name="venv-test")
        await pad.start()
        try:
            assert pad._venv_dir is not None
            assert os.path.isdir(pad._venv_dir)
            assert pad._venv_python is not None
            assert os.path.isfile(pad._venv_python)
        finally:
            await pad.close()

    async def test_venv_cleaned_on_close(self):
        """Venv directory should be removed when the scratchpad is closed."""
        pad = Scratchpad(name="venv-close")
        await pad.start()
        venv_dir = pad._venv_dir
        assert os.path.isdir(venv_dir)
        await pad.close()
        assert not os.path.exists(venv_dir)
        assert pad._venv_dir is None
        assert pad._venv_python is None

    async def test_venv_persists_across_reset(self):
        """Venv should survive a reset (only the process restarts)."""
        pad = Scratchpad(name="venv-reset")
        await pad.start()
        venv_dir = pad._venv_dir
        try:
            await pad.reset()
            assert pad._venv_dir == venv_dir
            assert os.path.isdir(venv_dir)
        finally:
            await pad.close()

    async def test_subprocess_uses_venv_python(self):
        """The subprocess should run with the venv's Python executable."""
        pad = Scratchpad(name="venv-exec")
        await pad.start()
        try:
            cell = await pad.execute("import sys; print(sys.executable)")
            assert cell.error is None
            assert pad._venv_dir in cell.stdout.strip()
        finally:
            await pad.close()

    async def test_system_packages_available(self):
        """System site-packages should be accessible (e.g. pydantic from parent env)."""
        pad = Scratchpad(name="venv-syspkg")
        await pad.start()
        try:
            cell = await pad.execute("import pydantic; print(pydantic.__name__)")
            assert cell.error is None
            assert cell.stdout.strip() == "pydantic"
        finally:
            await pad.close()


class TestScratchpadInstall:
    async def test_install_packages_success(self):
        """install_packages should install a package into the venv."""
        pad = Scratchpad(name="install-test")
        await pad.start()
        try:
            result = await pad.install_packages(["cowsay"])
            assert "cowsay" in result.lower() or "already satisfied" in result.lower()
            # Verify the package is importable
            cell = await pad.execute("import cowsay; print('ok')")
            assert cell.error is None
            assert cell.stdout.strip() == "ok"
        finally:
            await pad.close()

    async def test_install_empty_list(self):
        """install_packages with empty list returns a message."""
        pad = Scratchpad(name="install-empty")
        await pad.start()
        try:
            result = await pad.install_packages([])
            assert "no packages" in result.lower()
        finally:
            await pad.close()

    async def test_install_invalid_package(self):
        """install_packages with a bogus name should report failure."""
        pad = Scratchpad(name="install-bad")
        await pad.start()
        try:
            result = await pad.install_packages(["this-package-does-not-exist-xyz123"])
            assert "failed" in result.lower() or "error" in result.lower()
        finally:
            await pad.close()

    async def test_install_survives_reset(self):
        """Packages installed before a reset should still be available after."""
        pad = Scratchpad(name="install-reset")
        await pad.start()
        try:
            await pad.install_packages(["cowsay"])
            await pad.reset()
            cell = await pad.execute("import cowsay; print('ok')")
            assert cell.error is None
            assert cell.stdout.strip() == "ok"
        finally:
            await pad.close()
