"""Unit tests for MindsClient, Mind, and MindResponse (anton/minds.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.minds import MindsClient, Mind, MindResponse


class TestMindsClientAuth:
    def test_headers_contain_bearer_token(self):
        client = MindsClient(api_key="test-key-123")
        headers = client._headers()
        assert headers["Authorization"] == "Bearer test-key-123"
        assert headers["Content-Type"] == "application/json"

    def test_custom_base_url(self):
        client = MindsClient(api_key="k", base_url="https://custom.example.com")
        assert client.base_url == "https://custom.example.com"

    def test_default_base_url(self):
        client = MindsClient(api_key="k")
        assert client.base_url == "https://mdb.ai"


class TestMindsClientAsk:
    async def test_ask_sends_correct_payload_and_stores_ids(self):
        client = MindsClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "id": "msg_123",
            "conversation_id": "conv_456",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "The top customer is Acme Corp."}
                    ],
                }
            ],
        }

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await client.ask("Who is the top customer?", "sales")

        assert result == "The top customer is Acme Corp."
        assert client._last_conversation_id == "conv_456"
        assert client._last_message_id == "msg_123"

        # Verify the POST was called with correct payload
        mock_http_client.post.assert_awaited_once()
        call_args = mock_http_client.post.call_args
        assert call_args[0][0] == "/api/v1/responses"
        payload = call_args[1]["json"]
        assert payload["input"] == "Who is the top customer?"
        assert payload["model"] == "sales"

    async def test_ask_with_conversation_id(self):
        client = MindsClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "id": "msg_2",
            "conversation_id": "conv_existing",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Follow-up answer."}],
                }
            ],
        }

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await client.ask("And last quarter?", "sales", conversation_id="conv_existing")

        assert result == "Follow-up answer."
        payload = mock_http_client.post.call_args[1]["json"]
        assert payload["conversation_id"] == "conv_existing"

    async def test_ask_http_error_raises(self):
        client = MindsClient(api_key="bad-key")

        import httpx

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=MagicMock(),
            response=MagicMock(status_code=401),
        )

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            with pytest.raises(httpx.HTTPStatusError):
                await client.ask("test", "mind1")


class TestMindsClientData:
    async def test_data_returns_formatted_table(self):
        client = MindsClient(api_key="test-key")
        client._last_conversation_id = "conv_1"
        client._last_message_id = "msg_1"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "column_names": ["name", "revenue"],
            "data": [
                ["Acme Corp", 1000000],
                ["Globex", 750000],
            ],
        }

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await client.data(limit=10)

        assert "name" in result
        assert "revenue" in result
        assert "Acme Corp" in result
        assert "1000000" in result
        assert "Globex" in result
        # Should be a markdown table
        assert "|" in result
        assert "---" in result

        # Verify correct URL
        mock_http_client.get.assert_awaited_once()
        call_args = mock_http_client.get.call_args
        assert "/api/v1/conversations/conv_1/items/msg_1/result" in call_args[0][0]

    async def test_data_errors_without_prior_ask(self):
        client = MindsClient(api_key="test-key")
        # No prior ask — IDs are None

        with pytest.raises(ValueError, match="No prior ask"):
            await client.data()

    async def test_data_with_limit_and_offset(self):
        client = MindsClient(api_key="test-key")
        client._last_conversation_id = "c1"
        client._last_message_id = "m1"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"column_names": ["x"], "data": [[1]]}

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            await client.data(limit=50, offset=10)

        params = mock_http_client.get.call_args[1]["params"]
        assert params["limit"] == 50
        assert params["offset"] == 10


class TestMindsClientCatalog:
    async def test_catalog_returns_formatted_listing(self):
        client = MindsClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {
                "name": "customers",
                "columns": [
                    {"name": "id", "type": "integer"},
                    {"name": "company_name", "type": "varchar"},
                    {"name": "revenue", "type": "decimal"},
                ],
            },
            {
                "name": "orders",
                "columns": [
                    {"name": "order_id", "type": "integer"},
                    {"name": "customer_id", "type": "integer"},
                ],
            },
        ]

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await client.catalog("my_db")

        assert "customers" in result
        assert "orders" in result
        assert "id" in result
        assert "company_name" in result
        assert "integer" in result
        assert "varchar" in result

        # Verify correct URL
        call_args = mock_http_client.get.call_args
        assert "/api/v1/datasources/my_db/catalog" in call_args[0][0]

    async def test_catalog_dict_with_tables_key(self):
        client = MindsClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "tables": [
                {
                    "name": "users",
                    "columns": [{"name": "email", "type": "text"}],
                }
            ]
        }

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await client.catalog("db2")

        assert "users" in result
        assert "email" in result


class TestFormatHelpers:
    def test_format_table_empty_columns(self):
        result = MindsClient._format_table({"column_names": [], "data": []})
        assert result == "No data returned."

    def test_format_table_with_none_values(self):
        result = MindsClient._format_table({
            "column_names": ["a", "b"],
            "data": [[1, None], [None, "hello"]],
        })
        assert "| 1 |  |" in result
        assert "|  | hello |" in result

    def test_format_catalog_empty(self):
        result = MindsClient._format_catalog([])
        assert result == "No tables found."

    def test_format_catalog_no_columns(self):
        result = MindsClient._format_catalog([{"name": "empty_table", "columns": []}])
        assert "empty_table" in result
        assert "no column info" in result


class TestMindsClientExport:
    async def test_export_returns_csv_text(self):
        client = MindsClient(api_key="test-key")
        client._last_conversation_id = "conv_1"
        client._last_message_id = "msg_1"

        csv_text = "name,revenue\nAcme Corp,1000000\nGlobex,750000\n"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = csv_text

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await client.export()

        assert result == csv_text
        assert "name,revenue" in result

        # Verify correct URL
        mock_http_client.get.assert_awaited_once()
        call_args = mock_http_client.get.call_args
        assert "/api/v1/conversations/conv_1/items/msg_1/export" in call_args[0][0]

    async def test_export_errors_without_prior_ask(self):
        client = MindsClient(api_key="test-key")
        # No prior ask — IDs are None

        with pytest.raises(ValueError, match="No prior ask"):
            await client.export()


class TestMindsClientGetMind:
    async def test_get_mind_returns_parsed_json(self):
        client = MindsClient(api_key="test-key")

        mind_data = {
            "name": "sales",
            "datasources": ["sales_db"],
            "model_name": "gpt-4",
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mind_data

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await client.get_mind("sales")

        assert result == mind_data
        call_args = mock_http_client.get.call_args
        assert "/api/v1/minds/sales" in call_args[0][0]

    async def test_get_mind_raises_on_404(self):
        client = MindsClient(api_key="test-key")

        import httpx

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_mind("nonexistent")


class TestMindsClientCatalogWithMind:
    async def test_catalog_passes_mind_param(self):
        client = MindsClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {"name": "users", "columns": [{"name": "id", "type": "integer"}]},
        ]

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            await client.catalog("my_db", mind="sales")

        call_args = mock_http_client.get.call_args
        params = call_args[1]["params"]
        assert params["mind"] == "sales"

    async def test_catalog_omits_mind_when_none(self):
        client = MindsClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = []

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            await client.catalog("my_db")

        call_args = mock_http_client.get.call_args
        params = call_args[1]["params"]
        assert "mind" not in params


class TestMind:
    def test_mind_reads_env_vars(self):
        with patch.dict("os.environ", {"MINDS_API_KEY": "key-123", "MINDS_BASE_URL": "https://custom.example.com"}):
            mind = Mind("sales")
        assert mind.name == "sales"
        assert mind._api_key == "key-123"
        assert mind._base_url == "https://custom.example.com"

    def test_mind_default_base_url(self):
        with patch.dict("os.environ", {"MINDS_API_KEY": "key-123"}, clear=False):
            # Remove MINDS_BASE_URL if present
            import os
            env = os.environ.copy()
            env.pop("MINDS_BASE_URL", None)
            env["MINDS_API_KEY"] = "key-123"
            with patch.dict("os.environ", env, clear=True):
                mind = Mind("sales")
            assert mind._base_url == "https://mdb.ai"

    def test_mind_raises_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="MINDS_API_KEY"):
                Mind("sales")

    def test_mind_tracks_conversation_id(self):
        with patch.dict("os.environ", {"MINDS_API_KEY": "key-123"}):
            mind = Mind("sales")
        assert mind._conversation_id is None
        mind._conversation_id = "conv_123"
        assert mind._conversation_id == "conv_123"

    def test_ask_opens_streaming_request(self):
        with patch.dict("os.environ", {"MINDS_API_KEY": "key-123"}):
            mind = Mind("sales")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.build_request.return_value = MagicMock()
        mock_client.send.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            result = mind.ask("top customers?")

        assert isinstance(result, MindResponse)
        mock_client.send.assert_called_once()
        call_kwargs = mock_client.send.call_args
        assert call_kwargs[1]["stream"] is True

    def test_ask_sends_conversation_id_when_set(self):
        with patch.dict("os.environ", {"MINDS_API_KEY": "key-123"}):
            mind = Mind("sales")
            mind._conversation_id = "conv_456"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.build_request.return_value = MagicMock()
        mock_client.send.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            mind.ask("follow up?")

        build_args = mock_client.build_request.call_args
        payload = build_args[1]["json"]
        assert payload["conversation_id"] == "conv_456"

    def test_ask_closes_client_on_http_error(self):
        import httpx

        with patch.dict("os.environ", {"MINDS_API_KEY": "key-123"}):
            mind = Mind("sales")

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=MagicMock(status_code=401),
        )

        mock_client = MagicMock()
        mock_client.build_request.return_value = MagicMock()
        mock_client.send.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                mind.ask("test")

        mock_client.close.assert_called_once()


class TestMindResponse:
    def _make_response(self, lines, mind=None):
        """Helper to create a MindResponse with mock SSE lines."""
        if mind is None:
            with patch.dict("os.environ", {"MINDS_API_KEY": "key-123"}):
                mind = Mind("test")

        mock_http_response = MagicMock()
        mock_http_response.iter_lines.return_value = iter(lines)
        mock_http_response.close = MagicMock()

        mock_client = MagicMock()
        mock_client.close = MagicMock()

        return MindResponse(client=mock_client, response=mock_http_response, mind=mind)

    def test_iteration_yields_deltas(self):
        lines = [
            'data: {"type": "response.output_text.delta", "delta": "Hello"}',
            'data: {"type": "response.output_text.delta", "delta": " world"}',
            'data: {"type": "response.completed", "response": {"conversation_id": "c1", "id": "m1"}}',
        ]
        resp = self._make_response(lines)
        chunks = list(resp)
        assert chunks == ["Hello", " world"]

    def test_text_accumulates(self):
        lines = [
            'data: {"type": "response.output_text.delta", "delta": "Hello"}',
            'data: {"type": "response.output_text.delta", "delta": " world"}',
            'data: {"type": "response.completed", "response": {"conversation_id": "c1", "id": "m1"}}',
        ]
        resp = self._make_response(lines)
        list(resp)  # drain
        assert resp.text == "Hello world"

    def test_completed_flag_set(self):
        lines = [
            'data: {"type": "response.output_text.delta", "delta": "Hi"}',
            'data: {"type": "response.completed", "response": {"conversation_id": "c1", "id": "m1"}}',
        ]
        resp = self._make_response(lines)
        list(resp)
        assert resp.completed is True

    def test_ids_extracted_from_completed(self):
        lines = [
            'data: {"type": "response.completed", "response": {"conversation_id": "conv_99", "id": "msg_42"}}',
        ]
        resp = self._make_response(lines)
        list(resp)
        assert resp.conversation_id == "conv_99"
        assert resp.message_id == "msg_42"

    def test_completed_updates_parent_mind_conversation_id(self):
        with patch.dict("os.environ", {"MINDS_API_KEY": "key-123"}):
            mind = Mind("sales")
        assert mind._conversation_id is None

        lines = [
            'data: {"type": "response.completed", "response": {"conversation_id": "conv_new", "id": "m1"}}',
        ]
        resp = self._make_response(lines, mind=mind)
        list(resp)
        assert mind._conversation_id == "conv_new"

    def test_skips_non_data_lines(self):
        lines = [
            "event: ping",
            "",
            'data: {"type": "response.output_text.delta", "delta": "OK"}',
            'data: {"type": "response.completed", "response": {"conversation_id": "c1", "id": "m1"}}',
        ]
        resp = self._make_response(lines)
        chunks = list(resp)
        assert chunks == ["OK"]

    def test_skips_done_marker(self):
        lines = [
            'data: {"type": "response.output_text.delta", "delta": "Hi"}',
            "data: [DONE]",
            'data: {"type": "response.completed", "response": {"conversation_id": "c1", "id": "m1"}}',
        ]
        resp = self._make_response(lines)
        chunks = list(resp)
        assert chunks == ["Hi"]

    def test_get_data_no_limit_calls_export(self):
        """get_data() with no limit should GET the export endpoint."""
        with patch.dict("os.environ", {"MINDS_API_KEY": "key-123"}):
            mind = Mind("sales")

        lines = [
            'data: {"type": "response.completed", "response": {"conversation_id": "c1", "id": "m1"}}',
        ]
        resp = self._make_response(lines, mind=mind)
        list(resp)  # drain

        csv_text = "name,revenue\nAcme,1000\n"
        mock_get_response = MagicMock()
        mock_get_response.raise_for_status = MagicMock()
        mock_get_response.text = csv_text

        mock_data_client = MagicMock()
        mock_data_client.get.return_value = mock_get_response
        mock_data_client.__enter__ = MagicMock(return_value=mock_data_client)
        mock_data_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_data_client):
            result = resp.get_data()

        assert result == csv_text
        call_args = mock_data_client.get.call_args
        assert "/export" in call_args[0][0]

    def test_get_data_with_limit_calls_result(self):
        """get_data(limit=N) should GET the result endpoint."""
        with patch.dict("os.environ", {"MINDS_API_KEY": "key-123"}):
            mind = Mind("sales")

        lines = [
            'data: {"type": "response.completed", "response": {"conversation_id": "c1", "id": "m1"}}',
        ]
        resp = self._make_response(lines, mind=mind)
        list(resp)  # drain

        mock_get_response = MagicMock()
        mock_get_response.raise_for_status = MagicMock()
        mock_get_response.json.return_value = {"column_names": ["x"], "data": [[1]]}

        mock_data_client = MagicMock()
        mock_data_client.get.return_value = mock_get_response
        mock_data_client.__enter__ = MagicMock(return_value=mock_data_client)
        mock_data_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_data_client):
            result = resp.get_data(limit=100)

        assert "| x |" in result
        call_args = mock_data_client.get.call_args
        assert "/result" in call_args[0][0]
        assert call_args[1]["params"]["limit"] == 100

    def test_get_data_auto_drains_stream(self):
        """get_data() should auto-drain the stream if not yet iterated."""
        with patch.dict("os.environ", {"MINDS_API_KEY": "key-123"}):
            mind = Mind("sales")

        lines = [
            'data: {"type": "response.output_text.delta", "delta": "Hi"}',
            'data: {"type": "response.completed", "response": {"conversation_id": "c1", "id": "m1"}}',
        ]
        resp = self._make_response(lines, mind=mind)
        # NOT calling list(resp) — get_data should auto-drain

        csv_text = "a,b\n1,2\n"
        mock_get_response = MagicMock()
        mock_get_response.raise_for_status = MagicMock()
        mock_get_response.text = csv_text

        mock_data_client = MagicMock()
        mock_data_client.get.return_value = mock_get_response
        mock_data_client.__enter__ = MagicMock(return_value=mock_data_client)
        mock_data_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_data_client):
            result = resp.get_data()

        assert result == csv_text
        assert resp.text == "Hi"  # was accumulated during auto-drain

    def test_get_data_raises_if_not_completed(self):
        """get_data() should raise RuntimeError if stream didn't complete."""
        with patch.dict("os.environ", {"MINDS_API_KEY": "key-123"}):
            mind = Mind("sales")

        # No completed event
        lines = [
            'data: {"type": "response.output_text.delta", "delta": "partial"}',
        ]
        resp = self._make_response(lines, mind=mind)
        list(resp)  # drain without completed event

        with pytest.raises(RuntimeError, match="did not complete"):
            resp.get_data()
