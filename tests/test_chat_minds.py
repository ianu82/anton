"""Chat integration tests for minds connect/disconnect and knowledge injection."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.chat import ChatSession, _handle_minds_command
from anton.llm.provider import LLMResponse, Usage


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


class TestMindsKnowledgeInjection:
    async def test_connected_mind_appears_in_system_prompt(self, tmp_path):
        """When .anton/minds/X.md exists, its content is injected into the system prompt."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        minds_dir = tmp_path / ".anton" / "minds"
        minds_dir.mkdir(parents=True)
        (minds_dir / "sales.md").write_text("This mind has sales data with customers and orders.")

        session = ChatSession(
            mock_llm,
            workspace=workspace,
            minds_api_key="test-key",
        )
        try:
            await session.turn("hello")
            call_kwargs = mock_llm.plan.call_args
            system = call_kwargs.kwargs.get("system", "")
            assert "## Connected Minds" in system
            assert "### sales" in system
            assert "sales data with customers and orders" in system
        finally:
            await session.close()

    async def test_no_minds_dir_means_no_injection(self):
        """When there's no .anton/minds directory, no mind sections are injected."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))

        session = ChatSession(mock_llm, minds_api_key="test-key")
        try:
            await session.turn("hello")
            call_kwargs = mock_llm.plan.call_args
            system = call_kwargs.kwargs.get("system", "")
            # The "## Connected Minds" section header should NOT appear
            assert "## Connected Minds" not in system
        finally:
            await session.close()

    async def test_empty_minds_dir_means_no_injection(self, tmp_path):
        """When .anton/minds exists but is empty, nothing is injected."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        (tmp_path / ".anton" / "minds").mkdir(parents=True)

        session = ChatSession(
            mock_llm,
            workspace=workspace,
            minds_api_key="test-key",
        )
        try:
            await session.turn("hello")
            call_kwargs = mock_llm.plan.call_args
            system = call_kwargs.kwargs.get("system", "")
            assert "## Connected Minds" not in system
        finally:
            await session.close()


class TestMindsConnect:
    async def test_connect_writes_llm_summary(self, tmp_path):
        """connect fetches mind info, catalogs datasources, gets LLM summary, writes file."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("This mind provides sales analytics."))
        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        session = ChatSession(
            mock_llm,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()

        with patch.object(
            session._minds, "get_mind", new_callable=AsyncMock,
            return_value={"name": "sales", "datasources": ["sales_db"]},
        ), patch.object(
            session._minds, "catalog", new_callable=AsyncMock,
            return_value="## customers\n  - id (integer)\n  - name (varchar)",
        ):
            await session._handle_minds_connect("sales", console)

        md_file = tmp_path / ".anton" / "minds" / "sales.md"
        assert md_file.exists()
        content = md_file.read_text()
        assert "sales analytics" in content

        await session.close()

    async def test_connect_fallback_on_llm_failure(self, tmp_path):
        """When LLM fails, raw catalog is stored as fallback."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(side_effect=Exception("LLM unavailable"))
        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        session = ChatSession(
            mock_llm,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()

        with patch.object(
            session._minds, "get_mind", new_callable=AsyncMock,
            return_value={"name": "sales", "datasources": ["db1"]},
        ), patch.object(
            session._minds, "catalog", new_callable=AsyncMock,
            return_value="## orders\n  - id (integer)",
        ):
            await session._handle_minds_connect("sales", console)

        md_file = tmp_path / ".anton" / "minds" / "sales.md"
        assert md_file.exists()
        content = md_file.read_text()
        assert "orders" in content

        await session.close()

    async def test_connect_uses_table_list_from_get_mind_when_catalog_fails(self, tmp_path):
        """When catalog 404s but get_mind has tables, use those."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("This mind has order and car data."))
        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        session = ChatSession(
            mock_llm,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()

        import httpx
        catalog_error = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock(status_code=404),
        )

        # get_mind returns datasource dict with tables (like the real API)
        with patch.object(
            session._minds, "get_mind", new_callable=AsyncMock,
            return_value={
                "name": "test",
                "datasources": [
                    {"name": "demo_db", "tables": ["orders", "used_car_price", "customers"]},
                ],
            },
        ), patch.object(
            session._minds, "catalog", new_callable=AsyncMock,
            side_effect=catalog_error,
        ):
            await session._handle_minds_connect("test", console)

        # Verify file was written — LLM got the table list
        md_file = tmp_path / ".anton" / "minds" / "test.md"
        assert md_file.exists()
        content = md_file.read_text()
        assert "order" in content.lower()

        # The LLM prompt should have received the table names
        plan_call = mock_llm.plan.call_args
        prompt_msg = plan_call.kwargs["messages"][0]["content"]
        assert "orders" in prompt_msg
        assert "used_car_price" in prompt_msg

        await session.close()

    async def test_connect_falls_back_to_ask_when_no_tables_in_metadata(self, tmp_path):
        """When catalog 404s AND get_mind has no tables, ask the mind directly."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("This mind has order tracking data."))
        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        session = ChatSession(
            mock_llm,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()

        import httpx
        catalog_error = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock(status_code=404),
        )

        # Datasource is a plain string (no tables metadata)
        with patch.object(
            session._minds, "get_mind", new_callable=AsyncMock,
            return_value={"name": "test", "datasources": ["demo_db"]},
        ), patch.object(
            session._minds, "catalog", new_callable=AsyncMock,
            side_effect=catalog_error,
        ), patch.object(
            session._minds, "ask", new_callable=AsyncMock,
            return_value="I have access to: orders (id, customer_id, total)",
        ) as mock_ask:
            await session._handle_minds_connect("test", console)

        # Verify it fell back to asking the mind
        mock_ask.assert_awaited_once()

        md_file = tmp_path / ".anton" / "minds" / "test.md"
        assert md_file.exists()

        await session.close()

    async def test_connect_rejects_invalid_name(self, tmp_path):
        """Invalid mind names are rejected."""
        mock_llm = AsyncMock()
        workspace = MagicMock()
        workspace.base = tmp_path

        session = ChatSession(
            mock_llm,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()

        await session._handle_minds_connect("bad/name", console)
        console.print.assert_called()
        assert not (tmp_path / ".anton" / "minds").exists()

        await session.close()

    async def test_connect_without_minds_configured(self, tmp_path):
        """Connect fails gracefully when minds not configured."""
        mock_llm = AsyncMock()
        workspace = MagicMock()
        workspace.base = tmp_path

        session = ChatSession(mock_llm, workspace=workspace)
        console = MagicMock()

        await session._handle_minds_connect("sales", console)
        call_args = [str(c) for c in console.print.call_args_list]
        assert any("not configured" in s.lower() or "setup" in s.lower() for s in call_args)

        await session.close()


class TestMindsDisconnect:
    def test_disconnect_removes_file(self, tmp_path):
        """Disconnect removes the mind's knowledge file."""
        mock_llm = AsyncMock()
        workspace = MagicMock()
        workspace.base = tmp_path

        minds_dir = tmp_path / ".anton" / "minds"
        minds_dir.mkdir(parents=True)
        md_file = minds_dir / "sales.md"
        md_file.write_text("Sales mind knowledge.")

        session = ChatSession(
            mock_llm,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()

        session._handle_minds_disconnect("sales", console)
        assert not md_file.exists()

    def test_disconnect_warns_when_not_connected(self, tmp_path):
        """Disconnect warns when the mind isn't connected."""
        mock_llm = AsyncMock()
        workspace = MagicMock()
        workspace.base = tmp_path

        (tmp_path / ".anton" / "minds").mkdir(parents=True)

        session = ChatSession(
            mock_llm,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()

        session._handle_minds_disconnect("nonexistent", console)
        call_args = [str(c) for c in console.print.call_args_list]
        assert any("not connected" in s.lower() for s in call_args)


class TestListMinds:
    async def test_list_minds_gets_api_endpoint(self):
        """list_minds() GETs /api/v1/minds and returns parsed JSON."""
        from anton.minds import MindsClient

        client = MindsClient(api_key="test-key", base_url="https://mdb.ai")
        fake_response = [
            {"name": "sales", "datasources": ["sales_db"]},
            {"name": "hr", "datasources": ["hr_db"]},
        ]

        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_resp)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        import httpx
        with patch.object(httpx, "AsyncClient", return_value=mock_http_client):
            result = await client.list_minds()

        assert result == fake_response
        mock_http_client.get.assert_awaited_once()
        call_args = mock_http_client.get.call_args
        assert call_args[0][0] == "/api/v1/minds"

    async def test_list_minds_returns_empty_list(self):
        """list_minds() returns empty list when no minds exist."""
        from anton.minds import MindsClient

        client = MindsClient(api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_resp)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        import httpx
        with patch.object(httpx, "AsyncClient", return_value=mock_http_client):
            result = await client.list_minds()

        assert result == []


class TestUnifiedMindsCommand:
    async def test_always_shows_status_first(self, tmp_path):
        """The unified command always calls _handle_minds_status first."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        session = ChatSession(
            mock_llm,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()
        settings = MagicMock()
        state = {"llm_client": mock_llm}

        # Prompt responses: "" for API key (keep), "https://mdb.ai" for base URL (keep)
        prompt_responses = iter(["", "https://mdb.ai"])

        with patch.dict(os.environ, {"MINDS_API_KEY": "test-key", "MINDS_BASE_URL": "https://mdb.ai"}), \
             patch.object(session, "_handle_minds_status") as mock_status, \
             patch.object(session._minds, "list_minds", new_callable=AsyncMock, return_value=[]), \
             patch("rich.prompt.Prompt") as mock_prompt_cls:
            mock_prompt_cls.ask.side_effect = lambda *a, **kw: next(prompt_responses)
            await _handle_minds_command(
                console, settings, workspace, state, None, session,
            )

        mock_status.assert_called_once_with(console)
        await session.close()

    async def test_prompts_for_api_key_when_missing(self, tmp_path):
        """When no API key is set, prompts the user for one."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        session = ChatSession(
            mock_llm,
            workspace=workspace,
        )
        console = MagicMock()
        settings = MagicMock()
        state = {"llm_client": mock_llm}

        # No MINDS_API_KEY, user provides nothing -> aborts
        with patch.dict(os.environ, {}, clear=False), \
             patch.dict(os.environ, {"MINDS_API_KEY": ""}, clear=False), \
             patch("rich.prompt.Prompt") as mock_prompt_cls:
            # Remove MINDS_API_KEY from environ
            os.environ.pop("MINDS_API_KEY", None)
            mock_prompt_cls.ask.return_value = ""
            result = await _handle_minds_command(
                console, settings, workspace, state, None, session,
            )

        # Should have warned about no API key
        printed = [str(c) for c in console.print.call_args_list]
        assert any("no api key" in s.lower() or "aborting" in s.lower() for s in printed)
        await session.close()

    async def test_toggle_connect_disconnect(self, tmp_path):
        """User can type a mind name to toggle connection."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Summary."))
        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        # Pre-create minds dir with one connected mind
        minds_dir = tmp_path / ".anton" / "minds"
        minds_dir.mkdir(parents=True)
        (minds_dir / "sales.md").write_text("Sales mind.")

        session = ChatSession(
            mock_llm,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()
        settings = MagicMock()
        state = {"llm_client": mock_llm}

        minds_list = [
            {"name": "sales", "datasources": ["sales_db"]},
            {"name": "hr", "datasources": ["hr_db"]},
        ]

        # "" = keep key, "https://mdb.ai" = keep URL, "sales" = toggle, "" = exit
        prompt_responses = iter(["", "https://mdb.ai", "sales", ""])

        with patch.dict(os.environ, {"MINDS_API_KEY": "test-key", "MINDS_BASE_URL": "https://mdb.ai"}), \
             patch.object(session._minds, "list_minds", new_callable=AsyncMock, return_value=minds_list), \
             patch("rich.prompt.Prompt") as mock_prompt_cls, \
             patch.object(session, "_handle_minds_disconnect") as mock_disconnect:
            mock_prompt_cls.ask.side_effect = lambda *a, **kw: next(prompt_responses)
            await _handle_minds_command(
                console, settings, workspace, state, None, session,
            )

        # sales was connected, so it should disconnect
        mock_disconnect.assert_called_once_with("sales", console)
        await session.close()

    async def test_toggle_connects_unconnected_mind(self, tmp_path):
        """User typing an unconnected mind name triggers connect."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Summary."))
        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        (tmp_path / ".anton" / "minds").mkdir(parents=True)

        session = ChatSession(
            mock_llm,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()
        settings = MagicMock()
        state = {"llm_client": mock_llm}

        minds_list = [{"name": "hr", "datasources": ["hr_db"]}]
        # "" = keep key, "https://mdb.ai" = keep URL, "hr" = toggle, "" = exit
        prompt_responses = iter(["", "https://mdb.ai", "hr", ""])

        with patch.dict(os.environ, {"MINDS_API_KEY": "test-key", "MINDS_BASE_URL": "https://mdb.ai"}), \
             patch.object(session._minds, "list_minds", new_callable=AsyncMock, return_value=minds_list), \
             patch("rich.prompt.Prompt") as mock_prompt_cls, \
             patch.object(session, "_handle_minds_connect", new_callable=AsyncMock) as mock_connect:
            mock_prompt_cls.ask.side_effect = lambda *a, **kw: next(prompt_responses)
            await _handle_minds_command(
                console, settings, workspace, state, None, session,
            )

        mock_connect.assert_awaited_once_with("hr", console)
        await session.close()
