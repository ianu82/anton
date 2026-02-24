"""Tests for the need_secret() IPC flow between scratchpad subprocess and parent."""

from __future__ import annotations

import os

import pytest

from anton.scratchpad import Cell, Scratchpad


class TestNeedSecretIPC:
    """Test the full need_secret() round-trip between subprocess and parent."""

    async def test_need_secret_happy_path(self):
        """need_secret() prompts parent, parent responds, env var is set in subprocess."""
        provided_args: list[tuple[str, str]] = []

        def handler(var_name: str, prompt_text: str) -> str | None:
            provided_args.append((var_name, prompt_text))
            return "my_secret_value"

        pad = Scratchpad(name="secret-happy", _secret_handler=handler)
        await pad.start()
        try:
            code = (
                "import os\n"
                "need_secret('TEST_SECRET', 'Enter your test secret')\n"
                "print(os.environ.get('TEST_SECRET', 'NOT_SET'))\n"
            )
            cell = await pad.execute(code)
            assert cell.error is None
            assert cell.stdout.strip() == "my_secret_value"
            assert len(provided_args) == 1
            assert provided_args[0] == ("TEST_SECRET", "Enter your test secret")
        finally:
            await pad.close()

    async def test_need_secret_already_in_env(self, monkeypatch):
        """If env var is already set, need_secret() returns immediately without prompting."""
        monkeypatch.setenv("ALREADY_SET_VAR", "existing_value")

        handler_called = False

        def handler(var_name: str, prompt_text: str) -> str | None:
            nonlocal handler_called
            handler_called = True
            return "should_not_be_used"

        pad = Scratchpad(name="secret-existing", _secret_handler=handler)
        await pad.start()
        try:
            code = (
                "import os\n"
                "need_secret('ALREADY_SET_VAR', 'This should not prompt')\n"
                "print(os.environ.get('ALREADY_SET_VAR', 'NOT_SET'))\n"
            )
            cell = await pad.execute(code)
            assert cell.error is None
            assert cell.stdout.strip() == "existing_value"
            # Handler should NOT have been called — the subprocess checks os.environ first
            assert not handler_called
        finally:
            await pad.close()

    async def test_need_secret_empty_value_raises(self):
        """If handler returns None (empty value), need_secret() raises RuntimeError."""
        def handler(var_name: str, prompt_text: str) -> str | None:
            return None  # Simulates user entering nothing

        pad = Scratchpad(name="secret-empty", _secret_handler=handler)
        await pad.start()
        try:
            code = "need_secret('EMPTY_VAR', 'Enter something')\n"
            cell = await pad.execute(code)
            assert cell.error is not None
            assert "RuntimeError" in cell.error
            assert "Failed to get secret" in cell.error
        finally:
            await pad.close()

    async def test_need_secret_no_handler(self):
        """If no secret handler is set, need_secret() raises RuntimeError."""
        pad = Scratchpad(name="secret-no-handler")  # No _secret_handler
        await pad.start()
        try:
            code = "need_secret('SOME_VAR', 'Enter something')\n"
            cell = await pad.execute(code)
            assert cell.error is not None
            assert "RuntimeError" in cell.error
        finally:
            await pad.close()

    async def test_need_secret_sets_env_for_subsequent_code(self):
        """After need_secret(), the env var is available in the same cell and later cells."""
        def handler(var_name: str, prompt_text: str) -> str | None:
            return "db_pass_123"

        pad = Scratchpad(name="secret-persist", _secret_handler=handler)
        await pad.start()
        try:
            # First cell: request the secret and use it
            code1 = (
                "import os\n"
                "need_secret('DB_PASSWORD', 'Enter DB password')\n"
                "print('cell1:', os.environ['DB_PASSWORD'])\n"
            )
            cell1 = await pad.execute(code1)
            assert cell1.error is None
            assert "cell1: db_pass_123" in cell1.stdout

            # Second cell: secret should still be in env (same subprocess)
            code2 = (
                "import os\n"
                "print('cell2:', os.environ.get('DB_PASSWORD', 'GONE'))\n"
            )
            cell2 = await pad.execute(code2)
            assert cell2.error is None
            assert "cell2: db_pass_123" in cell2.stdout
        finally:
            await pad.close()

    async def test_need_secret_no_double_prompt(self):
        """Calling need_secret() twice for the same var only prompts once."""
        call_count = 0

        def handler(var_name: str, prompt_text: str) -> str | None:
            nonlocal call_count
            call_count += 1
            return "the_value"

        pad = Scratchpad(name="secret-idempotent", _secret_handler=handler)
        await pad.start()
        try:
            code = (
                "import os\n"
                "need_secret('IDEM_VAR', 'Enter value')\n"
                "need_secret('IDEM_VAR', 'Enter value again')\n"
                "print(os.environ['IDEM_VAR'])\n"
            )
            cell = await pad.execute(code)
            assert cell.error is None
            assert cell.stdout.strip() == "the_value"
            # Handler should only be called once — second call sees os.environ
            assert call_count == 1
        finally:
            await pad.close()

    async def test_need_secret_default_prompt(self):
        """If no prompt_text given, a default is used."""
        provided_prompts: list[str] = []

        def handler(var_name: str, prompt_text: str) -> str | None:
            provided_prompts.append(prompt_text)
            return "val"

        pad = Scratchpad(name="secret-default-prompt", _secret_handler=handler)
        await pad.start()
        try:
            code = "need_secret('MY_KEY')\n"
            cell = await pad.execute(code)
            assert cell.error is None
            assert len(provided_prompts) == 1
            assert "MY_KEY" in provided_prompts[0]
        finally:
            await pad.close()

    async def test_need_secret_streaming_yields_progress(self):
        """execute_streaming() should yield a progress-like message for secret handling."""
        def handler(var_name: str, prompt_text: str) -> str | None:
            return "secret_val"

        pad = Scratchpad(name="secret-stream", _secret_handler=handler)
        await pad.start()
        try:
            code = (
                "need_secret('STREAM_VAR', 'Enter it')\n"
                "print('done')\n"
            )
            items = []
            async for item in pad.execute_streaming(code):
                items.append(item)

            progress_items = [i for i in items if isinstance(i, str)]
            cell_items = [i for i in items if isinstance(i, Cell)]

            # Should have a progress message about the secret
            assert any("STREAM_VAR" in p for p in progress_items)
            assert len(cell_items) == 1
            assert cell_items[0].stdout.strip() == "done"
        finally:
            await pad.close()

    async def test_need_secret_multiple_different_vars(self):
        """need_secret() can be called for multiple different variables in one cell."""
        secrets = {"VAR_A": "val_a", "VAR_B": "val_b"}

        def handler(var_name: str, prompt_text: str) -> str | None:
            return secrets.get(var_name)

        pad = Scratchpad(name="secret-multi", _secret_handler=handler)
        await pad.start()
        try:
            code = (
                "import os\n"
                "need_secret('VAR_A', 'Enter A')\n"
                "need_secret('VAR_B', 'Enter B')\n"
                "print(os.environ['VAR_A'], os.environ['VAR_B'])\n"
            )
            cell = await pad.execute(code)
            assert cell.error is None
            assert cell.stdout.strip() == "val_a val_b"
        finally:
            await pad.close()
