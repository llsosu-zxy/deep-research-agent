from __future__ import annotations

import statistics
from collections.abc import Iterable

from core.guardrails.output import validate_citations
from core.models import GraphState


def tool_success_rate(state: GraphState) -> float:
    if not state.tool_log:
        return 0.0
    return sum(1 for item in state.tool_log if item.ok) / len(state.tool_log)


def citation_accuracy(state: GraphState) -> float:
    if not state.context:
        return 1.0 if "blocked" in state.report.lower() else 0.0
    ok, _issues = validate_citations(state.report, state.context)
    if not ok:
        return 0.0
    return 1.0


def answer_coverage(state: GraphState, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    report_lower = state.report.lower()
    return sum(1 for keyword in expected_keywords if keyword.lower() in report_lower) / len(
        expected_keywords
    )


def multi_hop_synthesis(
    state: GraphState,
    expected_keywords: list[str],
    question_type: str,
) -> float:
    """Multi-hop answers must cover every entity and include an explicit comparison."""
    if question_type != "multi-hop":
        return 1.0
    report_lower = state.report.lower()
    has_comparison = "comparison" in report_lower
    has_entities = all(keyword.lower() in report_lower for keyword in expected_keywords)
    return 1.0 if has_comparison and has_entities else 0.0


def summarize(states: Iterable[tuple[dict, GraphState, float]]) -> dict:
    """Input: (golden_dict, state, duration_ms) tuples."""
    rows = list(states)
    if not rows:
        return {}
    coverage = [
        answer_coverage(state, golden.get("expected_keywords", []))
        for golden, state, _ in rows
    ]
    citations = [citation_accuracy(state) for _, state, _ in rows]
    tools = [tool_success_rate(state) for _, state, _ in rows]
    latencies = [duration for _, _, duration in rows]
    multi_hop_rows = [
        (golden, state)
        for golden, state, _ in rows
        if golden.get("question_type") == "multi-hop"
    ]
    multi_hop = [
        multi_hop_synthesis(state, golden.get("expected_keywords", []), "multi-hop")
        for golden, state in multi_hop_rows
    ]
    multi_hop_rate = round(sum(multi_hop) / len(multi_hop), 4) if multi_hop else 1.0
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
    return {
        "cases": len(rows),
        "mean_answer_coverage": round(sum(coverage) / len(coverage), 4),
        "multi_hop_synthesis_rate": multi_hop_rate,
        "citation_accuracy": round(sum(citations) / len(citations), 4),
        "tool_success_rate": round(sum(tools) / len(tools), 4),
        "passed_critique_rate": round(
            sum(1 for _, state, _ in rows if state.passed) / len(rows), 4
        ),
        "latency_p50_ms": round(p50, 1),
        "latency_p95_ms": round(p95, 1),
    }
