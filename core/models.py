from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass
class Source:
    """A retrieved or fetched evidence item with citation metadata."""

    id: str
    title: str
    url: str = ""
    snippet: str = ""
    retrieved_at: str = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "retrieved_at": self.retrieved_at,
            "metadata": self.metadata,
        }


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "tokens": self.tokens,
        }


@dataclass
class SubTask:
    id: str
    question: str
    tools: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | done | failed | skipped
    result: str = ""
    sources: list[Source] = field(default_factory=list)
    error: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "tools": self.tools,
            "status": self.status,
            "result": self.result,
            "sources": [s.to_dict() for s in self.sources],
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class Plan:
    objective: str
    subtasks: list[SubTask] = field(default_factory=list)
    rationale: str = ""
    dependencies: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "rationale": self.rationale,
            "dependencies": self.dependencies,
        }


@dataclass
class ToolResult:
    tool: str
    ok: bool
    output: str
    data: Any = None
    error: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class TraceSpan:
    name: str
    node: str
    duration_ms: float
    meta: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "node": self.node,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "meta": self.meta,
        }


@dataclass
class GraphState:
    question: str
    plan: Plan | None = None
    context: list[Source] = field(default_factory=list)
    tool_log: list[ToolResult] = field(default_factory=list)
    draft: str = ""
    feedback: str = ""
    passed: bool = False
    report: str = ""
    iterations: int = 0
    max_iterations: int = 2
    trace: list[TraceSpan] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def add_span(
        self,
        name: str,
        node: str,
        duration_ms: float,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.trace.append(
            TraceSpan(name=name, node=node, duration_ms=duration_ms, meta=meta or {})
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "plan": self.plan.to_dict() if self.plan else None,
            "context": [s.to_dict() for s in self.context],
            "tool_log": [t.to_dict() for t in self.tool_log],
            "draft": self.draft,
            "feedback": self.feedback,
            "passed": self.passed,
            "report": self.report,
            "iterations": self.iterations,
            "trace": [t.to_dict() for t in self.trace],
            "summary": self.summary,
        }
