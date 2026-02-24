"""MindsDB (Minds) HTTP client for natural language data access.

Translates natural language questions into SQL via MindsDB's REST API.
Data stays in MindsDB — only results come back.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class MindsClient:
    """Stateful HTTP client for the Minds REST API.

    Tracks conversation_id and message_id from the last ask() so that
    data() can fetch tabular results without the caller managing IDs.
    """

    api_key: str
    base_url: str = "https://mdb.ai"
    _last_conversation_id: str | None = field(default=None, repr=False)
    _last_message_id: str | None = field(default=None, repr=False)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def ask(
        self,
        question: str,
        mind: str,
        conversation_id: str | None = None,
    ) -> str:
        """Ask a natural language question to a mind.

        POSTs to /api/v1/responses and stores the returned conversation_id
        and message_id for subsequent data() calls.

        Returns the text answer from the mind.
        """
        import httpx

        payload: dict = {
            "input": question,
            "model": mind,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60, follow_redirects=True) as client:
            resp = await client.post(
                "/api/v1/responses",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # Store IDs for subsequent data() calls
        self._last_conversation_id = data.get("conversation_id")
        self._last_message_id = data.get("id")

        # Extract text output
        output_parts: list[str] = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        output_parts.append(content.get("text", ""))
        return "\n".join(output_parts) if output_parts else data.get("output_text", "")

    async def ask_stream(
        self,
        question: str,
        mind: str,
        conversation_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Streaming version of ask(). Yields text deltas as they arrive.

        POSTs to /api/v1/responses with ``"stream": true`` and parses the
        SSE event stream.  On ``response.output_text.delta`` events the
        delta text is yielded.  On ``response.completed`` the conversation
        and message IDs are stored so that subsequent data()/export() calls
        work exactly like after a non-streaming ask().
        """
        import httpx

        payload: dict = {
            "input": question,
            "model": mind,
            "stream": True,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id

        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=120, follow_redirects=True,
        ) as client:
            async with client.stream(
                "POST",
                "/api/v1/responses",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:"):].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    etype = event.get("type", "")
                    if etype == "response.output_text.delta":
                        delta = event.get("delta", "")
                        if delta:
                            yield delta
                    elif etype == "response.completed":
                        resp_obj = event.get("response", {})
                        self._last_conversation_id = resp_obj.get("conversation_id")
                        self._last_message_id = resp_obj.get("id")

    async def data(self, limit: int = 100, offset: int = 0) -> str:
        """Fetch raw tabular results from the last ask() call.

        GETs /api/v1/conversations/{conv_id}/items/{msg_id}/result and
        formats the response as a markdown table.

        Raises ValueError if no prior ask() has been made.
        """
        if not self._last_conversation_id or not self._last_message_id:
            raise ValueError(
                "No prior ask() call — use 'ask' first to get an answer, "
                "then 'data' to fetch raw results."
            )

        import httpx

        url = (
            f"/api/v1/conversations/{self._last_conversation_id}"
            f"/items/{self._last_message_id}/result"
        )
        params: dict = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60, follow_redirects=True) as client:
            resp = await client.get(url, headers=self._headers(), params=params)
            resp.raise_for_status()
            data = resp.json()

        return self._format_table(data)

    async def export(self) -> str:
        """Export the full result set from the last ask() as CSV.

        GETs /api/v1/conversations/{conv_id}/items/{msg_id}/export and
        returns the raw CSV text.

        Raises ValueError if no prior ask() has been made.
        """
        if not self._last_conversation_id or not self._last_message_id:
            raise ValueError(
                "No prior ask() call — use 'ask' first to get an answer, "
                "then 'export' to fetch CSV results."
            )

        import httpx

        url = (
            f"/api/v1/conversations/{self._last_conversation_id}"
            f"/items/{self._last_message_id}/export"
        )

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60, follow_redirects=True) as client:
            resp = await client.get(url, headers=self._headers())
            resp.raise_for_status()
            return resp.text

    async def list_minds(self) -> list[dict]:
        """Fetch all minds available to the authenticated user.

        GETs /api/v1/minds and returns the parsed JSON list of mind objects.
        """
        import httpx

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60, follow_redirects=True) as client:
            resp = await client.get(
                "/api/v1/minds",
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()

    async def get_mind(self, name: str) -> dict:
        """Fetch metadata for a single mind.

        GETs /api/v1/minds/{name} and returns the parsed JSON dict
        (name, datasources, model_name, etc.).
        """
        import httpx

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60, follow_redirects=True) as client:
            resp = await client.get(
                f"/api/v1/minds/{name}",
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()

    async def catalog(self, datasource: str, *, mind: str | None = None) -> str:
        """Discover tables and columns for a datasource.

        GETs /api/v1/datasources/{datasource}/catalog and returns a
        formatted listing of tables and their columns.

        When *mind* is provided, passes ``?mind={name}`` as a query param.
        """
        import httpx

        params: dict[str, str] = {}
        if mind is not None:
            params["mind"] = mind

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60, follow_redirects=True) as client:
            resp = await client.get(
                f"/api/v1/datasources/{datasource}/catalog",
                headers=self._headers(),
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

        return self._format_catalog(data)

    @staticmethod
    def _format_table(data: dict) -> str:
        """Format a result payload as a markdown table."""
        columns: list[str] = data.get("column_names", [])
        rows: list[list] = data.get("data", [])

        if not columns:
            return "No data returned."

        # Header
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        lines = [header, separator]

        for row in rows:
            cells = [str(cell) if cell is not None else "" for cell in row]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    @staticmethod
    def _format_catalog(data: dict | list) -> str:
        """Format a catalog payload as a readable listing."""
        # The API may return a list of tables or a dict with a tables key
        tables: list[dict] = []
        if isinstance(data, list):
            tables = data
        elif isinstance(data, dict):
            tables = data.get("tables", data.get("items", []))

        if not tables:
            return "No tables found."

        lines: list[str] = []
        for table in tables:
            name = table.get("name", table.get("table_name", "unknown"))
            lines.append(f"## {name}")
            columns = table.get("columns", [])
            if columns:
                for col in columns:
                    col_name = col.get("name", col.get("column_name", "?"))
                    col_type = col.get("type", col.get("data_type", ""))
                    type_str = f" ({col_type})" if col_type else ""
                    lines.append(f"  - {col_name}{type_str}")
            else:
                lines.append("  (no column info)")
            lines.append("")

        return "\n".join(lines)


class MindResponse:
    """Streaming response from a Mind.ask() call.

    Iterate to consume SSE text deltas. After iteration completes,
    ``.text`` holds the full accumulated answer and ``.get_data()``
    can fetch tabular results or CSV exports.
    """

    def __init__(
        self,
        client,  # httpx.Client — owned, closed on drain
        response,  # httpx.Response — streaming
        mind: Mind,
    ) -> None:
        self._client = client
        self._response = response
        self._mind = mind

        self.text: str = ""
        self.completed: bool = False
        self.conversation_id: str | None = None
        self.message_id: str | None = None
        self._drained: bool = False

    def __iter__(self):
        try:
            yield from self._iter_deltas()
        finally:
            self._close()

    def _iter_deltas(self):
        buf = ""
        for line in self._response.iter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[len("data:"):].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            etype = event.get("type", "")
            if etype == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    self.text += delta
                    yield delta
            elif etype == "response.completed":
                resp_obj = event.get("response", {})
                self.conversation_id = resp_obj.get("conversation_id")
                self.message_id = resp_obj.get("id")
                self.completed = True
                # Update parent Mind for multi-turn
                self._mind._conversation_id = self.conversation_id
        self._drained = True

    def _close(self) -> None:
        try:
            self._response.close()
        except Exception:
            pass
        try:
            self._client.close()
        except Exception:
            pass
        self._drained = True

    def _auto_drain(self) -> None:
        """Consume the stream if not yet iterated."""
        if self._drained:
            return
        for _ in self:
            pass

    def get_data(self, *, limit: int | None = None, offset: int = 0) -> str:
        """Fetch tabular results from this response.

        - ``limit=None`` → GET ``/export`` → returns CSV string.
          Falls back to ``/result`` (markdown table) if export fails.
        - ``limit=N`` → GET ``/result?limit=N&offset=M`` → returns markdown table.

        Auto-drains the stream if not yet iterated.
        Raises ``RuntimeError`` if the stream didn't complete successfully.
        """
        import httpx

        self._auto_drain()
        if not self.completed:
            raise RuntimeError(
                "Stream did not complete — cannot fetch data. "
                "The mind's text answer is available in response.text"
            )

        headers = {
            "Authorization": f"Bearer {self._mind._api_key}",
            "Content-Type": "application/json",
        }
        base = (
            f"/api/v1/conversations/{self.conversation_id}"
            f"/items/{self.message_id}"
        )

        with httpx.Client(
            base_url=self._mind._base_url, timeout=60, follow_redirects=True,
        ) as client:
            if limit is not None:
                params: dict = {"limit": limit}
                if offset:
                    params["offset"] = offset
                resp = client.get(f"{base}/result", headers=headers, params=params)
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"get_data(limit={limit}) failed (HTTP {resp.status_code}): "
                        f"{resp.text[:500]}\n"
                        f"The mind's text answer is available in response.text"
                    )
                return MindsClient._format_table(resp.json())

            # No limit → try export first, fall back to result
            resp = client.get(f"{base}/export", headers=headers)
            if resp.status_code < 400:
                return resp.text

            # Export failed — fall back to result endpoint (markdown table)
            fallback = client.get(
                f"{base}/result", headers=headers, params={"limit": 500},
            )
            if fallback.status_code < 400:
                return MindsClient._format_table(fallback.json())

            # Both failed
            raise RuntimeError(
                f"get_data() export failed (HTTP {resp.status_code}): "
                f"{resp.text[:500]}\n"
                f"get_data(limit=N) also failed (HTTP {fallback.status_code}): "
                f"{fallback.text[:500]}\n"
                f"The mind's text answer is available in response.text"
            )


class Mind:
    """Sync, streaming-first interface for querying a MindsDB mind.

    Usage::

        mind = Mind("sales")
        response = mind.ask("top customers?")
        for chunk in response:
            print(chunk, end="")

        csv = response.get_data()           # full CSV export
        table = response.get_data(limit=100) # paginated markdown table
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._api_key = os.environ.get("MINDS_API_KEY", "")
        self._base_url = os.environ.get("MINDS_BASE_URL", "https://mdb.ai")
        if not self._api_key:
            raise ValueError(
                "MINDS_API_KEY environment variable is not set. "
                "Configure it via /minds setup or set it in .anton/.env."
            )
        self._conversation_id: str | None = None

    def ask(self, question: str) -> MindResponse:
        """Ask a natural language question. Returns a streaming MindResponse."""
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "input": question,
            "model": self.name,
            "stream": True,
        }
        if self._conversation_id:
            payload["conversation_id"] = self._conversation_id

        client = httpx.Client(
            base_url=self._base_url, timeout=120, follow_redirects=True,
        )
        try:
            request = client.build_request(
                "POST", "/api/v1/responses", headers=headers, json=payload,
            )
            response = client.send(request, stream=True)
            response.raise_for_status()
        except Exception:
            client.close()
            raise

        return MindResponse(client=client, response=response, mind=self)
