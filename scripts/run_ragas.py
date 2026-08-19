from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.agent import ResearchAgent
from core.config import Settings
from eval.golden_set import get_golden_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional RAGAS evaluation for the golden set.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "ragas_report.md")
    args = parser.parse_args()

    if os.getenv("RAGAS_ENABLED", "0") != "1":
        print("RAGAS is optional and needs ragas + an LLM provider. Set RAGAS_ENABLED=1 to run.")
        return

    from eval.ragas_adapter import run_ragas

    agent = ResearchAgent(settings=Settings.from_env())
    questions, answers, contexts = [], [], []
    for golden in get_golden_set(args.limit):
        state = agent.run(golden.question)
        questions.append(golden.question)
        answers.append(state.report)
        contexts.append([source.snippet for source in state.context])
    records = run_ragas(questions, answers, contexts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(str(records), encoding="utf-8")
    print(f"RAGAS report written to {args.output}")


if __name__ == "__main__":
    main()
