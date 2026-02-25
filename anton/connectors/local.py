from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from anton.connectors.base import ConnectorClient, ConnectorError, ConnectorInfo, ConnectorSchema, QueryResult


class LocalSQLiteConnectorClient(ConnectorClient):
    """Local SQLite connector backend for development/testing."""

    def __init__(self, connector_map: dict[str, str]) -> None:
        self._connector_map = connector_map

    def _db_path(self, connector_id: str) -> str:
        path = self._connector_map.get(connector_id)
        if path is None:
            raise ConnectorError(f"Unknown connector_id '{connector_id}'.")
        db_path = Path(path).expanduser().resolve()
        if not db_path.exists():
            raise ConnectorError(f"SQLite database for connector '{connector_id}' not found: {db_path}")
        return str(db_path)

    async def list_connectors(self) -> list[ConnectorInfo]:
        return [
            ConnectorInfo(connector_id=connector_id, connector_type="sqlite", description=path)
            for connector_id, path in sorted(self._connector_map.items())
        ]

    async def describe_schema(self, connector_id: str) -> ConnectorSchema:
        db_path = self._db_path(connector_id)
        return await asyncio.to_thread(self._describe_schema_sync, connector_id, db_path)

    async def run_query(
        self,
        connector_id: str,
        query: str,
        *,
        limit: int = 1000,
    ) -> QueryResult:
        db_path = self._db_path(connector_id)
        return await asyncio.to_thread(self._run_query_sync, connector_id, db_path, query, limit)

    async def sample(
        self,
        connector_id: str,
        table: str,
        *,
        limit: int = 100,
    ) -> QueryResult:
        query = f"SELECT * FROM {table} LIMIT {max(1, limit)}"
        return await self.run_query(connector_id, query, limit=limit)

    async def write(
        self,
        connector_id: str,
        query: str,
    ) -> QueryResult:
        db_path = self._db_path(connector_id)
        return await asyncio.to_thread(self._write_sync, connector_id, db_path, query)

    @staticmethod
    def _describe_schema_sync(connector_id: str, db_path: str) -> ConnectorSchema:
        tables: dict[str, list[str]] = {}
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            for (table_name,) in rows:
                col_rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                tables[table_name] = [str(col[1]) for col in col_rows]
        finally:
            conn.close()
        return ConnectorSchema(connector_id=connector_id, tables=tables)

    @staticmethod
    def _run_query_sync(
        connector_id: str,
        db_path: str,
        query: str,
        limit: int,
    ) -> QueryResult:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(query)
            columns = [desc[0] for desc in (cur.description or [])]
            rows = cur.fetchmany(max(1, limit)) if columns else []
            truncated = False
            row_count = len(rows)
            if columns:
                extra = cur.fetchmany(1)
                if extra:
                    truncated = True
                    row_count += 1
            return QueryResult(
                connector_id=connector_id,
                query=query,
                columns=columns,
                rows=[list(row) for row in rows],
                row_count=row_count,
                truncated=truncated,
            )
        except sqlite3.Error as exc:
            raise ConnectorError(f"SQLite query failed: {exc}") from exc
        finally:
            conn.close()

    @staticmethod
    def _write_sync(connector_id: str, db_path: str, query: str) -> QueryResult:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()
            affected = cur.rowcount if cur.rowcount != -1 else 0
            return QueryResult(
                connector_id=connector_id,
                query=query,
                columns=[],
                rows=[],
                row_count=0,
                affected_rows=affected,
            )
        except sqlite3.Error as exc:
            conn.rollback()
            raise ConnectorError(f"SQLite write failed: {exc}") from exc
        finally:
            conn.close()
