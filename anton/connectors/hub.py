from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anton.connectors.base import ConnectorAuthContext, ConnectorClient, ConnectorError, QueryResult
from anton.connectors.http_bridge import HTTPConnectorClient
from anton.connectors.local import LocalSQLiteConnectorClient


def format_query_result(result: QueryResult, *, max_rows: int = 20) -> str:
    if result.affected_rows is not None:
        return (
            f"Connector '{result.connector_id}' write executed successfully. "
            f"Affected rows: {result.affected_rows}."
        )

    lines: list[str] = [
        f"Connector '{result.connector_id}' query completed.",
        f"Rows returned: {len(result.rows)} (row_count={result.row_count}, truncated={result.truncated}).",
    ]

    if result.columns:
        lines.append("Columns: " + ", ".join(result.columns))

    if result.rows:
        preview_rows = result.rows[:max_rows]
        lines.append("Preview:")
        for idx, row in enumerate(preview_rows, start=1):
            lines.append(f"  {idx}: {row}")
        if len(result.rows) > max_rows:
            lines.append(f"  ... ({len(result.rows) - max_rows} more rows in payload)")

    return "\n".join(lines)


class ConnectorHub:
    """High-level helper around the configured connector backend."""

    def __init__(self, client: ConnectorClient, *, workspace: Path) -> None:
        self._client = client
        self._workspace = workspace

    @classmethod
    def from_settings(cls, settings, *, workspace: Path) -> ConnectorHub | None:
        mode = getattr(settings, "connector_mode", "none")
        if mode == "none":
            return None

        if mode == "http":
            base_url = getattr(settings, "connector_api_base_url", None)
            if not base_url:
                return None
            client = HTTPConnectorClient(
                base_url=base_url,
                token=getattr(settings, "connector_api_token", None),
                timeout_seconds=getattr(settings, "connector_timeout_seconds", 30),
                path_prefix=getattr(settings, "connector_api_path_prefix", "/v1"),
            )
            return cls(client, workspace=workspace)

        if mode == "local":
            raw = getattr(settings, "connector_local_sqlite_map", "{}")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {}
            mapping = {str(key): str(value) for key, value in parsed.items()} if isinstance(parsed, dict) else {}
            if not mapping:
                return None
            client = LocalSQLiteConnectorClient(mapping)
            return cls(client, workspace=workspace)

        return None

    @staticmethod
    def auth_context_from_dict(value: dict[str, Any] | None) -> ConnectorAuthContext | None:
        if not value:
            return None
        return ConnectorAuthContext(
            user_id=str(value.get("user_id", "")),
            org_id=str(value.get("org_id", "")),
            roles=[str(role) for role in value.get("roles", []) if isinstance(role, str)] if isinstance(value.get("roles"), list) else [],
            attributes=value.get("attributes", {}) if isinstance(value.get("attributes"), dict) else {},
        )

    async def list_connectors(self, *, auth_context: ConnectorAuthContext | None = None) -> str:
        connectors = await self._client.list_connectors(auth_context=auth_context)
        if not connectors:
            return "No connectors available."
        lines = ["Available connectors:"]
        for connector in connectors:
            suffix = f" - {connector.description}" if connector.description else ""
            lines.append(f"- {connector.connector_id} ({connector.connector_type}){suffix}")
        return "\n".join(lines)

    async def schema(
        self,
        connector_id: str,
        *,
        auth_context: ConnectorAuthContext | None = None,
    ) -> str:
        schema = await self._client.describe_schema(connector_id, auth_context=auth_context)
        if not schema.tables:
            return f"Connector '{connector_id}' has no discoverable tables."
        lines = [f"Schema for connector '{connector_id}':"]
        for table_name, cols in sorted(schema.tables.items()):
            lines.append(f"- {table_name}: {', '.join(cols) if cols else '(no columns)'}")
        return "\n".join(lines)

    async def query(
        self,
        connector_id: str,
        query: str,
        *,
        limit: int = 1000,
        auth_context: ConnectorAuthContext | None = None,
    ) -> str:
        result = await self._client.run_query(connector_id, query, limit=limit, auth_context=auth_context)
        return format_query_result(result)

    async def sample(
        self,
        connector_id: str,
        table: str,
        *,
        limit: int = 100,
        auth_context: ConnectorAuthContext | None = None,
    ) -> str:
        result = await self._client.sample(connector_id, table, limit=limit, auth_context=auth_context)
        return format_query_result(result)

    async def write(
        self,
        connector_id: str,
        query: str,
        *,
        auth_context: ConnectorAuthContext | None = None,
    ) -> str:
        result = await self._client.write(connector_id, query, auth_context=auth_context)
        return format_query_result(result)


__all__ = ["ConnectorHub", "ConnectorError", "format_query_result"]
