from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from anton.chat import ChatSession
from anton.config.settings import AntonSettings
from anton.connectors import ConnectorHub
from anton.tools import dispatch_tool


@pytest.fixture()
def sqlite_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE sales (id INTEGER, amount REAL)")
        conn.execute("INSERT INTO sales(id, amount) VALUES (1, 10.5), (2, 20.0)")
        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.mark.asyncio()
async def test_connector_tool_included_when_hub_present(sqlite_db: Path, tmp_path: Path):
    settings = AntonSettings(
        connector_mode="local",
        connector_local_sqlite_map=json.dumps({"localdb": str(sqlite_db)}),
    )
    settings.resolve_workspace(str(tmp_path))

    hub = ConnectorHub.from_settings(settings, workspace=tmp_path)
    assert hub is not None

    mock_llm = AsyncMock()
    session = ChatSession(mock_llm, connector_hub=hub)

    tools = session._build_tools()
    names = [tool["name"] for tool in tools]
    assert "connector" in names


@pytest.mark.asyncio()
async def test_connector_list_schema_query(sqlite_db: Path, tmp_path: Path):
    settings = AntonSettings(
        connector_mode="local",
        connector_local_sqlite_map=json.dumps({"localdb": str(sqlite_db)}),
    )
    settings.resolve_workspace(str(tmp_path))

    hub = ConnectorHub.from_settings(settings, workspace=tmp_path)
    assert hub is not None

    mock_llm = AsyncMock()
    session = ChatSession(mock_llm, connector_hub=hub)

    listed = await dispatch_tool(session, "connector", {"action": "list"})
    assert "localdb" in listed

    schema = await dispatch_tool(
        session,
        "connector",
        {"action": "schema", "connector_id": "localdb"},
    )
    assert "sales" in schema
    assert "amount" in schema

    queried = await dispatch_tool(
        session,
        "connector",
        {
            "action": "query",
            "connector_id": "localdb",
            "query": "SELECT * FROM sales ORDER BY id",
            "limit": 10,
        },
    )
    assert "Rows returned" in queried
    assert "10.5" in queried
