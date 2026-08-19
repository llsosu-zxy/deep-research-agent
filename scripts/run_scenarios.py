from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.agent import ResearchAgent
from core.config import Settings
from eval.scenarios import SCENARIOS


def main() -> None:
    agent = ResearchAgent(settings=Settings.from_env())
    rows = []
    for scenario in SCENARIOS:
        started = time.perf_counter()
        state = agent.run(scenario.question)
        duration_ms = (time.perf_counter() - started) * 1000
        rows.append((scenario, state, duration_ms))

    lines = [
        "# E2E Scenario Report",
        "",
        "| ID | Expected | Actual | Blocked | Sources | ms |",
        "|---|---|---|---|---|---|",
    ]
    for scenario, state, duration in rows:
        actual = state.passed
        blocked = not state.passed and "blocked" in state.report.lower()
        lines.append(
            f"| {scenario.id} | {scenario.expected_pass} | {actual} | "
            f"{blocked} | {len(state.context)} | {duration:.1f} |"
        )
    output = ROOT / "docs" / "scenarios_report.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Ran {len(rows)} scenarios; report written to {output}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
