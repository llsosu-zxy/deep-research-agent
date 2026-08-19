from __future__ import annotations

import time

from core.guardrails.output import validate_citations
from core.models import GraphState, ToolResult
from core.retrieval.index import RetrievalIndex


class SingleTurnRAG:
    """Retrieval-only baseline: top-k chunks in, template report out.

    Used for the plan's "single-turn RAG vs agent" comparison experiment.
    """

    def __init__(self, index: RetrievalIndex, top_k: int = 5) -> None:
        self.index = index
        self.top_k = top_k

    def _render(self, question: str, sources) -> str:
        lines = [f"# RAG Answer: {question}", "", "## Summary"]
        if not sources:
            lines.append("No evidence found in the current corpus.")
        else:
            for number, source in enumerate(sources, start=1):
                lines.append(f"- {source.title}: {source.snippet[:220]} [{number}]")
        lines.extend(["", "## Sources"])
        for number, source in enumerate(sources, start=1):
            if source.url:
                lines.append(f"{number}. [{source.title}]({source.url})")
            else:
                lines.append(f"{number}. {source.title}")
        return "\n".join(lines)

    def run(self, question: str) -> GraphState:
        started = time.perf_counter()
        ranked = self.index.search(question, top_k=self.top_k)
        sources = self.index.to_sources(ranked)
        report = self._render(question, sources)
        state = GraphState(question=question, context=sources, draft=report, report=report)
        ok, _ = validate_citations(report, sources)
        state.passed = ok or not sources
        state.tool_log.append(
            ToolResult(
                tool="retrieve",
                ok=bool(sources),
                output=str(len(sources)),
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        )
        return state
