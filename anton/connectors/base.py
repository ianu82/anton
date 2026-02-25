from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class ConnectorError(Exception):
    """Raised when connector operations fail."""


@dataclass(slots=True)
class ConnectorInfo:
    connector_id: str
    connector_type: str
    description: str = ""


@dataclass(slots=True)
class ConnectorSchema:
    connector_id: str
    tables: dict[str, list[str]]


@dataclass(slots=True)
class QueryResult:
    connector_id: str
    query: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool = False
    affected_rows: int | None = None


class ConnectorClient(ABC):
    """Interface for connector backends used by Anton."""

    @abstractmethod
    async def list_connectors(self) -> list[ConnectorInfo]:
        raise NotImplementedError

    @abstractmethod
    async def describe_schema(self, connector_id: str) -> ConnectorSchema:
        raise NotImplementedError

    @abstractmethod
    async def run_query(
        self,
        connector_id: str,
        query: str,
        *,
        limit: int = 1000,
    ) -> QueryResult:
        raise NotImplementedError

    @abstractmethod
    async def sample(
        self,
        connector_id: str,
        table: str,
        *,
        limit: int = 100,
    ) -> QueryResult:
        raise NotImplementedError

    @abstractmethod
    async def write(
        self,
        connector_id: str,
        query: str,
    ) -> QueryResult:
        raise NotImplementedError
