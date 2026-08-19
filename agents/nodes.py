from __future__ import annotations

import time

from agents.brain import AgentBrain
from core.models import GraphState


def _rebuild_subtask_citations(subtask, context) -> None:
    lines = []
    for source in subtask.sources:
        number = next(
            (index + 1 for index, existing in enumerate(context) if existing.id == source.id),
            0,
        )
        lines.append(f"- {source.title} [{number}]: {source.snippet[:220]}")
    subtask.result = "\n".join(lines)


def plan_node(state: GraphState, brain: AgentBrain) -> GraphState:
    started = time.perf_counter()
    state.plan = brain.plan(state.question)
    state.add_span(
        "planner",
        "planner",
        (time.perf_counter() - started) * 1000,
        {"subtasks": len(state.plan.subtasks)},
    )
    return state


def execute_node(state: GraphState, brain: AgentBrain) -> GraphState:
    started = time.perf_counter()
    if not state.plan:
        state.plan = brain.plan(state.question)

    for subtask in state.plan.subtasks:
        if subtask.status not in {"pending", "failed"}:
            continue
        _, sources, tool_result = brain.execute_subtask(subtask)
        state.tool_log.append(tool_result)
        for source in sources:
            if not any(existing.id == source.id for existing in state.context):
                state.context.append(source)
        if sources:
            _rebuild_subtask_citations(subtask, state.context)

    state.add_span(
        "executor",
        "executor",
        (time.perf_counter() - started) * 1000,
        {
            "subtasks": len(state.plan.subtasks),
            "sources": len(state.context),
            "tool_calls": len(state.tool_log),
        },
    )
    return state


def synthesize_node(state: GraphState, brain: AgentBrain) -> GraphState:
    started = time.perf_counter()
    state.draft = brain.synthesize(state)
    state.add_span(
        "synthesizer",
        "synthesizer",
        (time.perf_counter() - started) * 1000,
        {"draft_chars": len(state.draft)},
    )
    return state


def critic_node(state: GraphState, brain: AgentBrain) -> GraphState:
    started = time.perf_counter()
    state.passed, state.feedback = brain.critique(state)
    state.iterations += 1
    state.add_span(
        "critic",
        "critic",
        (time.perf_counter() - started) * 1000,
        {"passed": state.passed, "feedback": state.feedback[:200]},
    )
    return state


def finalize_node(state: GraphState, brain: AgentBrain) -> GraphState:
    return brain.finalize(state)
