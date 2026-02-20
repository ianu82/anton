"""Unit tests for MindsClient and SyncMindsClient (anton/minds.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.minds import MindsClient, SyncMindsClient


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


class TestSyncMindsClient:
    def test_sync_client_initializes_inner_client(self):
        client = SyncMindsClient(api_key="test-key", base_url="https://custom.example.com")
        assert client._client.api_key == "test-key"
        assert client._client.base_url == "https://custom.example.com"

    def test_sync_ask_wraps_async(self):
        client = SyncMindsClient(api_key="test-key")
        with patch.object(
            client._client, "ask", new_callable=AsyncMock, return_value="Answer text"
        ) as mock_ask:
            result = client.ask("question?", "my_mind")
        mock_ask.assert_awaited_once_with("question?", "my_mind", None)
        assert result == "Answer text"

    def test_sync_data_wraps_async(self):
        client = SyncMindsClient(api_key="test-key")
        with patch.object(
            client._client, "data", new_callable=AsyncMock, return_value="| a |\n| --- |"
        ) as mock_data:
            result = client.data(limit=50, offset=10)
        mock_data.assert_awaited_once_with(limit=50, offset=10)
        assert "| a |" in result

    def test_sync_export_wraps_async(self):
        client = SyncMindsClient(api_key="test-key")
        with patch.object(
            client._client, "export", new_callable=AsyncMock, return_value="a,b\n1,2\n"
        ) as mock_export:
            result = client.export()
        mock_export.assert_awaited_once()
        assert result == "a,b\n1,2\n"

    def test_sync_catalog_wraps_async(self):
        client = SyncMindsClient(api_key="test-key")
        with patch.object(
            client._client, "catalog", new_callable=AsyncMock, return_value="## users"
        ) as mock_catalog:
            result = client.catalog("my_db")
        mock_catalog.assert_awaited_once_with("my_db")
        assert result == "## users"
