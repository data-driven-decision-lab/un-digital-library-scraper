"""Turso HTTP API compatibility layer.

Provides a connection-like interface using the Turso HTTP API (v2/pipeline),
allowing the dashboard pipeline to run on Windows where libsql-experimental
won't build.

Usage:
    from turso_http import get_turso_connection
    conn = get_turso_connection()
    cursor = conn.execute("SELECT * FROM annual_scores WHERE Year = ?", [2025])
    rows = cursor.fetchall()
    conn.commit()  # no-op for Turso (auto-commit)
"""

import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()


class TursoCursor:
    """Mimics a DB-API cursor using Turso HTTP results."""

    def __init__(self, columns, rows):
        self.description = [(col, None, None, None, None, None, None) for col in columns]
        self._rows = rows
        self._pos = 0

    def fetchall(self):
        return self._rows

    def fetchone(self):
        if self._pos < len(self._rows):
            row = self._rows[self._pos]
            self._pos += 1
            return row
        return None

    @property
    def rowcount(self):
        return len(self._rows)


class TursoHTTPConnection:
    """SQLite-like connection interface backed by Turso HTTP API."""

    def __init__(self, url, auth_token):
        self._url = url.replace("libsql://", "https://")
        self._token = auth_token
        self._api_url = f"{self._url}/v2/pipeline"

    def _send(self, sql, args=None):
        stmt = {"sql": sql}
        if args:
            stmt["args"] = [self._convert_arg(a) for a in args]

        payload = json.dumps({"requests": [{"type": "execute", "stmt": stmt}, {"type": "close"}]})
        req = urllib.request.Request(self._api_url, data=payload.encode(), headers={
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json"
        })
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())

        result = data["results"][0]
        if result["type"] == "error":
            raise Exception(f"Turso SQL error: {result.get('error', {}).get('message', 'unknown')}")

        exec_result = result["response"]["result"]
        columns = [col["name"] for col in exec_result.get("cols", [])]
        raw_rows = exec_result.get("rows", [])

        # Convert Turso typed values to Python values
        rows = []
        for raw_row in raw_rows:
            row = []
            for cell in raw_row:
                if cell["type"] == "null":
                    row.append(None)
                elif cell["type"] == "integer":
                    row.append(int(cell["value"]))
                elif cell["type"] == "float":
                    row.append(float(cell["value"]))
                else:
                    row.append(cell["value"])
            rows.append(tuple(row))

        return TursoCursor(columns, rows)

    def execute(self, sql, args=None):
        return self._send(sql, args or [])

    def _convert_arg(self, a):
        """Convert a Python value to Turso HTTP API typed value."""
        import math
        if a is None:
            return {"type": "null"}
        # Handle numpy/pandas NA types
        try:
            if hasattr(a, 'item'):  # numpy scalar
                a = a.item()
            if isinstance(a, float) and (math.isnan(a) or math.isinf(a)):
                return {"type": "null"}
        except (TypeError, ValueError):
            pass
        if isinstance(a, bool):
            return {"type": "integer", "value": str(int(a))}
        if isinstance(a, int):
            return {"type": "integer", "value": str(a)}
        if isinstance(a, float):
            return {"type": "float", "value": a}
        return {"type": "text", "value": str(a)}

    def executemany(self, sql, seq_of_args):
        """Execute SQL for each set of args. Uses batched HTTP requests."""
        import time
        batch_size = 500
        for start in range(0, len(seq_of_args), batch_size):
            batch = seq_of_args[start:start + batch_size]
            requests = []
            for args in batch:
                stmt = {"sql": sql, "args": [self._convert_arg(a) for a in args]}
                requests.append({"type": "execute", "stmt": stmt})
            requests.append({"type": "close"})

            payload = json.dumps({"requests": requests})
            for attempt in range(3):
                try:
                    req = urllib.request.Request(self._api_url, data=payload.encode(), headers={
                        "Authorization": f"Bearer {self._token}",
                        "Content-Type": "application/json"
                    })
                    resp = urllib.request.urlopen(req, timeout=120)
                    data = json.loads(resp.read())
                    # Check for errors in batch
                    for r in data.get("results", []):
                        if r.get("type") == "error":
                            raise Exception(f"Batch SQL error: {r.get('error', {}).get('message', 'unknown')}")
                    break
                except urllib.error.HTTPError as e:
                    body = e.read().decode('utf-8', errors='replace')[:500]
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        raise Exception(f"HTTP {e.code}: {body}\nSQL: {sql[:200]}\nFirst args sample: {[self._convert_arg(a) for a in batch[0]][:5]}")
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        raise

    def commit(self):
        pass  # Turso auto-commits

    def close(self):
        pass

    def sync(self):
        pass  # No-op for HTTP-based connection


def get_turso_connection():
    """Create a Turso HTTP connection using environment variables."""
    url = os.getenv("TURSO_DATABASE_URL", "")
    token = os.getenv("TURSO_AUTH_TOKEN", "")
    if not url:
        raise ValueError("TURSO_DATABASE_URL not set")
    if not token:
        raise ValueError("TURSO_AUTH_TOKEN not set")
    return TursoHTTPConnection(url, token)
