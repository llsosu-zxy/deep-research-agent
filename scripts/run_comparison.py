from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.agent import ResearchAgent
from core.config import Settings
from eval.baseline_rag import SingleTurnRAG
from eval.golden_set import get_golden_set
from eval.metrics import summarize


def _golden_dict(question) -> dict:
    return asdict(question)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single-turn RAG against the multi-agent pipeline.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--baseline-k", type=int, default=3, help="Retrieval depth for the single-shot baseline")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "comparison_report.md")
    args = parser.parse_args()

    settings = Settings.from_env()
    agent = ResearchAgent(settings=settings)
    baseline = SingleTurnRAG(agent.index, top_k=args.baseline_k)

    agent_rows = []
    rag_rows = []
    for question in get_golden_set(args.limit):
        golden = _golden_dict(question)

        started = time.perf_counter()
        agent_state = agent.run(question.question)
        agent_rows.append((golden, agent_state, (time.perf_counter() - started) * 1000))

        started = time.perf_counter()
        rag_state = baseline.run(question.question)
        rag_rows.append((golden, rag_state, (time.perf_counter() - started) * 1000))

    agent_metrics = summarize(agent_rows)
    rag_metrics = summarize(rag_rows)

    lines = [
        "# Agent vs Single-Turn RAG",
        "",
        "## Summary",
        "",
        "| Metric | Agent | Single-Turn RAG | Delta |",
        "|---|---|---|---|",
    ]
    for key in sorted(agent_metrics):
        agent_value = agent_metrics.get(key, "-")
        rag_value = rag_metrics.get(key, "-")
        if isinstance(agent_value, (int, float)) and isinstance(rag_value, (int, float)):
            delta = f"{agent_value - rag_value:+.4f}"
        else:
            delta = "-"
        lines.append(f"| {key} | {agent_value} | {rag_value} | {delta} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Baseline: one retrieval call with a single top-k pass.",
            "- Agent: multiple entity-targeted retrieval calls plus critic reruns.",
            "- The agent trades latency for broader, failure-recoverable evidence.",
            "- Re-run after enabling real LLM and embedding models to get production numbers.",
        ]
    )
    report = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Compared {len(agent_rows)} cases; report written to {args.output}")
    print(report)


if __name__ == "__main__":
    main()
