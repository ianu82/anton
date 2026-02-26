from __future__ import annotations

import json
from dataclasses import dataclass

from anton.connectors.base import ConnectorAuthContext
from anton.connectors.http_bridge import HTTPConnectorClient


@dataclass
class _FakeResponse:
    payload: dict

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_http_bridge_attaches_auth_headers(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["headers"] = dict(req.headers)
        captured["method"] = req.get_method()
        captured["timeout"] = timeout
        captured["data"] = req.data
        return _FakeResponse({"connectors": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = HTTPConnectorClient(base_url="https://example.com", token="token-123")
    auth = ConnectorAuthContext(user_id="u-1", org_id="org-1", roles=["analyst"])

    result = client._request_sync("GET", "/v1/connectors", auth_context=auth)
    assert result == {"connectors": []}
    assert captured["method"] == "GET"
    assert captured["headers"].get("Authorization") == "Bearer token-123"
    assert captured["headers"].get("X-anton-user-id") == "u-1"
    assert captured["headers"].get("X-anton-org-id") == "org-1"


def test_http_bridge_includes_auth_context_in_query_body(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["data"] = req.data
        return _FakeResponse(
            {
                "columns": ["id"],
                "rows": [[1]],
                "row_count": 1,
                "truncated": False,
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = HTTPConnectorClient(base_url="https://example.com", token="token-123")
    auth = ConnectorAuthContext(
        user_id="u-2",
        org_id="org-2",
        roles=["admin"],
        attributes={"region": "us"},
    )

    payload = {
        "query": "SELECT 1",
        "limit": 50,
        "mode": "read",
        "auth_context": client._auth_payload(auth),
    }
    result = client._request_sync("POST", "/v1/connectors/warehouse/query", payload, auth_context=auth)

    assert result["row_count"] == 1
    parsed_body = json.loads(captured["data"].decode("utf-8"))
    assert parsed_body["auth_context"]["user_id"] == "u-2"
    assert parsed_body["auth_context"]["org_id"] == "org-2"
    assert parsed_body["auth_context"]["roles"] == ["admin"]
    assert parsed_body["auth_context"]["attributes"]["region"] == "us"
