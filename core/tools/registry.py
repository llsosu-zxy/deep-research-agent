from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from core.models import ToolResult


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Any],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def invoke(self, **kwargs: Any) -> ToolResult:
        started = time.perf_counter()
        try:
            output = self.func(**kwargs)
            if isinstance(output, ToolResult):
                return output
            if isinstance(output, str):
                return ToolResult(
                    tool=self.name,
                    ok=True,
                    output=output,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
            serialized = json.dumps(output, ensure_ascii=False, default=str)[:8000]
            return ToolResult(
                tool=self.name,
                ok=True,
                output=serialized,
                data=output,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary must not crash the graph
            return ToolResult(
                tool=self.name,
                ok=False,
                output="",
                error=str(exc),
                duration_ms=(time.perf_counter() - started) * 1000,
            )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> ToolRegistry:
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(tool=name, ok=False, output="", error=f"Unknown tool: {name}")
        return tool.invoke(**kwargs)
