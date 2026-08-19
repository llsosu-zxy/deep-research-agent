from __future__ import annotations

from pathlib import Path

from core.models import GraphState
from eval.metrics import answer_coverage, citation_accuracy, summarize, tool_success_rate


def render_report(rows: list[tuple[dict, GraphState, float]]) -> str:
    summary = summarize(rows)
    lines = [
        "# Eval Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Per-case Results", "", "| ID | Type | Passed | Coverage | Citation | Tools | p50 ms |", "|---|---|---|---|---|---|---|"])
    for golden, state, duration in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {:.0f} |".format(
                golden["id"],
                golden["question_type"],
                state.passed,
                round(answer_coverage(state, golden["expected_keywords"]), 2),
                round(citation_accuracy(state), 2),
                round(tool_success_rate(state), 2),
                duration,
            )
        )
    lines.extend(["", "## Notes", "", "- The seed corpus is deterministic and offline; real API/embedding runs can be enabled via .env."])
    return "\n".join(lines)


def write_report(rows: list[tuple[dict, GraphState, float]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(rows), encoding="utf-8")
    return path
