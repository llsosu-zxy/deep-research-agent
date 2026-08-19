from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.tools.python_sandbox import build_python_sandbox_tool
from core.tools.registry import ToolRegistry
from core.tools.sqlite_query import build_sqlite_query_tool


class ToolsTest(unittest.TestCase):
    def test_registry_unknown_tool(self) -> None:
        registry = ToolRegistry()
        result = registry.call("nope")
        self.assertFalse(result.ok)
        self.assertIn("Unknown tool", result.error)

    def test_python_sandbox(self) -> None:
        tool = build_python_sandbox_tool()
        result = tool.invoke(code="print(sum(range(5)))")
        self.assertTrue(result.ok)
        self.assertIn("10", result.output)

    def test_sqlite_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "research.sqlite"
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE jobs (company TEXT)")
                conn.execute("INSERT INTO jobs VALUES ('Shopee')")
                conn.commit()
            finally:
                conn.close()
            tool = build_sqlite_query_tool(db)
            ok_result = tool.invoke(query="SELECT company FROM jobs")
            self.assertTrue(ok_result.ok)
            self.assertIn("Shopee", ok_result.output)
            bad_result = tool.invoke(query="DELETE FROM jobs")
            self.assertFalse(bad_result.ok)


if __name__ == "__main__":
    unittest.main()
