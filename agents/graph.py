from __future__ import annotations

import logging
from collections.abc import Callable

from agents.brain import AgentBrain
from agents.nodes import (
    critic_node,
    execute_node,
    finalize_node,
    plan_node,
    synthesize_node,
)
from core.config import Settings
from core.models import GraphState

try:  # langgraph is optional so the offline engine can run without it
    from langgraph.graph import END, START, StateGraph

    HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover
    HAS_LANGGRAPH = False

logger = logging.getLogger(__name__)


class MiniGraph:
    """Dependency-free fallback that runs the same nodes in a critic loop."""

    def __init__(
        self,
        brain: AgentBrain,
        settings: Settings,
        approval_callback: Callable[[object], bool] | None = None,
    ) -> None:
        self.brain = brain
        self.settings = settings
        self.approval_callback = approval_callback

    def invoke(self, question: str) -> GraphState:
        state = GraphState(
            question=question,
            max_iterations=max(1, self.settings.max_critic_iterations),
        )
        plan_node(state, self.brain)
        if self.approval_callback is not None and not self.approval_callback(state.plan):
            state.passed = False
            state.report = "# Plan pending approval\n\nResearch plan was not approved; execution skipped."
            finalize_node(state, self.brain)
            return state
        for _ in range(state.max_iterations + 1):
            execute_node(state, self.brain)
            synthesize_node(state, self.brain)
            critic_node(state, self.brain)
            if state.passed:
                break
        finalize_node(state, self.brain)
        return state


class LangGraphRunner:
    """LangGraph version of the same four-node pipeline."""

    def __init__(self, brain: AgentBrain, settings: Settings) -> None:
        self.brain = brain
        self.settings = settings
        self.app = self._build()

    def _to_dict(self, state: GraphState) -> dict:
        return {
            "question": state.question,
            "plan": state.plan,
            "context": state.context,
            "tool_log": state.tool_log,
            "draft": state.draft,
            "feedback": state.feedback,
            "passed": state.passed,
            "report": state.report,
            "iterations": state.iterations,
            "max_iterations": state.max_iterations,
            "trace": state.trace,
            "summary": state.summary,
        }

    def _from_dict(self, data: dict) -> GraphState:
        return GraphState(
            question=data["question"],
            plan=data.get("plan"),
            context=data.get("context") or [],
            tool_log=data.get("tool_log") or [],
            draft=data.get("draft") or "",
            feedback=data.get("feedback") or "",
            passed=bool(data.get("passed")),
            report=data.get("report") or "",
            iterations=int(data.get("iterations") or 0),
            max_iterations=int(data.get("max_iterations") or 2),
            trace=data.get("trace") or [],
            summary=data.get("summary") or {},
        )

    def _plan(self, data: dict) -> dict:
        state = self._from_dict(data)
        plan_node(state, self.brain)
        return self._to_dict(state)

    def _execute(self, data: dict) -> dict:
        state = self._from_dict(data)
        execute_node(state, self.brain)
        return self._to_dict(state)

    def _synthesize(self, data: dict) -> dict:
        state = self._from_dict(data)
        synthesize_node(state, self.brain)
        return self._to_dict(state)

    def _critic(self, data: dict) -> dict:
        state = self._from_dict(data)
        critic_node(state, self.brain)
        return self._to_dict(state)

    def _route(self, data: dict) -> str:
        if not data.get("passed") and int(data.get("iterations") or 0) <= int(
            data.get("max_iterations") or 2
        ):
            return "executor"
        return "end"

    def _build(self):
        workflow = StateGraph(dict)
        workflow.add_node("planner", self._plan)
        workflow.add_node("executor", self._execute)
        workflow.add_node("synthesizer", self._synthesize)
        workflow.add_node("critic", self._critic)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "synthesizer")
        workflow.add_edge("synthesizer", "critic")
        workflow.add_conditional_edges(
            "critic",
            self._route,
            {"executor": "executor", "end": END},
        )
        return workflow.compile()

    def invoke(self, question: str) -> GraphState:
        initial = GraphState(
            question=question,
            max_iterations=max(1, self.settings.max_critic_iterations),
        )
        data = self.app.invoke(self._to_dict(initial))
        state = self._from_dict(data)
        finalize_node(state, self.brain)
        return state


class ResearchGraph:
    def __init__(
        self,
        brain: AgentBrain,
        settings: Settings,
        approval_callback: Callable[[object], bool] | None = None,
    ) -> None:
        self.brain = brain
        self.settings = settings
        self.approval_callback = approval_callback
        self.mini = MiniGraph(brain, settings, approval_callback)
        self.lang: LangGraphRunner | None = None
        if HAS_LANGGRAPH and approval_callback is None:
            try:
                self.lang = LangGraphRunner(brain, settings)
            except Exception:  # noqa: BLE001 - fall back to the offline engine
                self.lang = None

    @property
    def engine(self) -> str:
        if self.approval_callback is not None:
            return "mini-approval"
        return "langgraph" if self.lang is not None else "mini"

    def invoke(self, question: str) -> GraphState:
        if self.lang is not None:
            try:
                return self.lang.invoke(question)
            except Exception:
                logger.warning("LangGraph invocation failed; falling back to MiniGraph", exc_info=True)
        return self.mini.invoke(question)
