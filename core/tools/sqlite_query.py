from __future__ import annotations

import sqlite3
from pathlib import Path

from core.tools.registry import Tool


def build_sqlite_query_tool(database_path: str | Path) -> Tool:
    db_path = Path(database_path)

    def sqlite_query(query: str, params: list[str] | None = None) -> dict[str, object]:
        if not db_path.exists():
            raise FileNotFoundError(str(db_path))
        lowered = query.strip().lower()
        if not lowered.startswith("select") or ";" in query.rstrip(";"):
            raise ValueError("Only single read-only SELECT queries are allowed")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params or []).fetchall()
        finally:
            conn.close()
        return {"columns": list(rows[0].keys()) if rows else [], "rows": [dict(r) for r in rows]}

    return Tool(
        name="sqlite_query",
        description="Execute a read-only SELECT query against the research SQLite database.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "params": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
        func=sqlite_query,
    )
