from __future__ import annotations

from collections.abc import Callable

from agents.brain import AgentBrain
from agents.graph import ResearchGraph
from core.config import Settings
from core.models import GraphState, Plan
from core.retrieval.index import RetrievalIndex
from core.tools.pdf import build_pdf_parse_tool
from core.tools.python_sandbox import build_python_sandbox_tool
from core.tools.registry import ToolRegistry
from core.tools.sqlite_query import build_sqlite_query_tool
from core.tools.web import (
    build_arxiv_search_tool,
    build_fetch_url_tool,
    build_retrieve_tool,
    build_web_search_tool,
)
from core.tracing import BudgetTracker, TraceLogger


class ResearchAgent:
    """End-to-end agent: guardrails -> LangGraph/MiniGraph -> trace + budget."""

    def __init__(
        self,
        settings: Settings | None = None,
        index: RetrievalIndex | None = None,
        trace_logger: TraceLogger | None = None,
        approval_callback: Callable[[Plan], bool] | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.index = index or RetrievalIndex.from_corpus(self.settings.corpus_dir)
        self.registry = ToolRegistry()
        self.registry.register(
            build_retrieve_tool(self.index, top_k=self.settings.retrieval_top_k)
        )
        self.registry.register(build_web_search_tool())
        self.registry.register(build_fetch_url_tool())
        self.registry.register(build_arxiv_search_tool())
        self.registry.register(build_pdf_parse_tool())
        self.registry.register(build_python_sandbox_tool())
        self.registry.register(build_sqlite_query_tool(self.settings.cache_dir / "research.sqlite"))
        self.brain = AgentBrain(self.settings, self.registry)
        self.graph = ResearchGraph(self.brain, self.settings, approval_callback)
        self.trace_logger = trace_logger or TraceLogger(self.settings.trace_path)
        self.budget = BudgetTracker(self.settings.daily_budget_usd)

    def run(self, question: str) -> GraphState:
        ok, sanitized, issues = self.brain.sanitize(question)
        state = GraphState(question=sanitized, max_iterations=max(1, self.settings.max_critic_iterations))
        if not ok:
            state.report = "# Request blocked\n\n" + "\n".join(f"- {issue}" for issue in issues)
            state.passed = False
            self.trace_logger.log(
                {
                    "event": "research_blocked",
                    "question": sanitized,
                    "issues": issues,
                }
            )
            return state
        state = self.graph.invoke(sanitized)
        self.trace_logger.log(
            {
                "event": "research_complete",
                "engine": self.graph.engine,
                "state": state.to_dict(),
                "budget_remaining_usd": round(self.budget.remaining_usd(), 4),
            }
        )
        return state
