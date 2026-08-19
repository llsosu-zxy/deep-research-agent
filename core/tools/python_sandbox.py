from __future__ import annotations

import contextlib
import io
import time

from core.tools.registry import Tool

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def build_python_sandbox_tool(timeout_seconds: float = 5.0) -> Tool:
    def python_sandbox(code: str) -> dict[str, object]:
        namespace = {"__builtins__": SAFE_BUILTINS}
        output = io.StringIO()
        started = time.perf_counter()
        with contextlib.redirect_stdout(output):
            try:
                exec(compile(code, "<sandbox>", "exec"), namespace, namespace)  # noqa: S102 - intentional sandbox
            except Exception as exc:
                raise RuntimeError(f"sandbox error: {exc}") from exc
        duration = (time.perf_counter() - started) * 1000
        if duration > timeout_seconds * 1000:
            raise TimeoutError("sandbox timeout")
        return {"stdout": output.getvalue(), "duration_ms": round(duration, 3)}

    return Tool(
        name="python_sandbox",
        description="Run safe Python code for statistics and chart generation.",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        func=python_sandbox,
    )
