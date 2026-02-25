from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request

from anton.connectors.base import ConnectorClient, ConnectorError, ConnectorInfo, ConnectorSchema, QueryResult


class HTTPConnectorClient(ConnectorClient):
    """HTTP bridge to an existing connector management service."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str | None = None,
        timeout_seconds: int = 30,
        path_prefix: str = "/v1",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout_seconds = timeout_seconds
        self._path_prefix = path_prefix.rstrip("/")

    async def list_connectors(self) -> list[ConnectorInfo]:
        payload = await self._request("GET", f"{self._path_prefix}/connectors")
        items = payload.get("connectors", payload)
        if not isinstance(items, list):
            raise ConnectorError("Connector service returned invalid connector list payload.")
        out: list[ConnectorInfo] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            out.append(
                ConnectorInfo(
                    connector_id=str(item.get("id", "")),
                    connector_type=str(item.get("type", "unknown")),
                    description=str(item.get("description", "")),
                )
            )
        return out

    async def describe_schema(self, connector_id: str) -> ConnectorSchema:
        payload = await self._request("GET", f"{self._path_prefix}/connectors/{connector_id}/schema")
        tables = payload.get("tables", {})
        if not isinstance(tables, dict):
            raise ConnectorError("Connector service returned invalid schema payload.")
        normalized: dict[str, list[str]] = {}
        for table, cols in tables.items():
            if isinstance(cols, list):
                normalized[str(table)] = [str(col) for col in cols]
        return ConnectorSchema(connector_id=connector_id, tables=normalized)

    async def run_query(
        self,
        connector_id: str,
        query: str,
        *,
        limit: int = 1000,
    ) -> QueryResult:
        payload = await self._request(
            "POST",
            f"{self._path_prefix}/connectors/{connector_id}/query",
            {"query": query, "limit": max(1, limit), "mode": "read"},
        )
        return self._to_query_result(connector_id, query, payload)

    async def sample(
        self,
        connector_id: str,
        table: str,
        *,
        limit: int = 100,
    ) -> QueryResult:
        payload = await self._request(
            "POST",
            f"{self._path_prefix}/connectors/{connector_id}/sample",
            {"table": table, "limit": max(1, limit)},
        )
        query = str(payload.get("query", f"SELECT * FROM {table} LIMIT {max(1, limit)}"))
        return self._to_query_result(connector_id, query, payload)

    async def write(self, connector_id: str, query: str) -> QueryResult:
        payload = await self._request(
            "POST",
            f"{self._path_prefix}/connectors/{connector_id}/query",
            {"query": query, "mode": "write"},
        )
        result = self._to_query_result(connector_id, query, payload)
        result.affected_rows = int(payload.get("affected_rows", 0))
        return result

    def _to_query_result(self, connector_id: str, query: str, payload: dict) -> QueryResult:
        columns = payload.get("columns", [])
        rows = payload.get("rows", [])
        if not isinstance(columns, list) or not isinstance(rows, list):
            raise ConnectorError("Connector service returned invalid query payload.")

        parsed_rows: list[list[object]] = []
        for row in rows:
            if isinstance(row, list):
                parsed_rows.append(row)
            elif isinstance(row, dict):
                parsed_rows.append([row.get(col) for col in columns])
            else:
                parsed_rows.append([row])

        row_count = int(payload.get("row_count", len(parsed_rows)))
        truncated = bool(payload.get("truncated", False))
        return QueryResult(
            connector_id=connector_id,
            query=query,
            columns=[str(col) for col in columns],
            rows=parsed_rows,
            row_count=row_count,
            truncated=truncated,
        )

    async def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        return await asyncio.to_thread(self._request_sync, method, path, body)

    def _request_sync(self, method: str, path: str, body: dict | None = None) -> dict:
        url = urllib.parse.urljoin(f"{self._base_url}/", path.lstrip("/"))
        headers = {"Accept": "application/json"}
        data: bytes | None = None
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ConnectorError(f"Connector service HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ConnectorError(f"Connector service unreachable: {exc}") from exc

        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConnectorError("Connector service returned non-JSON response.") from exc
        if not isinstance(parsed, dict):
            raise ConnectorError("Connector service returned invalid JSON payload type.")
        return parsed
