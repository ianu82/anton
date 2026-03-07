from __future__ import annotations

import json
import ssl
import urllib.request


def _ssl_context(verify: bool) -> ssl.SSLContext | None:
    if verify:
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def list_datasources(base_url: str, api_key: str, *, verify: bool = True) -> list[dict]:
    """Fetch datasource list from a Minds server using stdlib urllib."""
    url = f"{base_url}/api/v1/datasources/"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "anton/1.0")

    with urllib.request.urlopen(req, context=_ssl_context(verify), timeout=30) as resp:
        data = json.loads(resp.read().decode())

    if isinstance(data, list):
        return data
    return data.get("datasources", data if isinstance(data, list) else [])


def query_datasource(
    base_url: str,
    api_key: str,
    query: str,
    *,
    datasource: str,
    verify: bool = True,
) -> dict:
    """Query a Minds datasource with SQL and return the JSON result payload."""
    url = f"{base_url}/api/v1/datasources/{datasource}/query"
    payload = json.dumps({"query": query, "native_query": True}).encode()

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "anton/1.0")

    try:
        with urllib.request.urlopen(req, context=_ssl_context(verify), timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as err:
        body = ""
        try:
            body = err.read().decode()
        except Exception:
            pass
        return {
            "type": "error",
            "data": None,
            "column_names": None,
            "error_message": f"HTTP {err.code}: {body or err.reason}",
        }
    except Exception as err:
        return {
            "type": "error",
            "data": None,
            "column_names": None,
            "error_message": str(err),
        }
