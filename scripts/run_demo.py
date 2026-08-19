from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.agent import ResearchAgent
from core.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the research agent on one question.")
    parser.add_argument(
        "question",
        nargs="?",
        default="Compare Shopee, TikTok and Grab AI internship opportunities and required skills in Singapore.",
    )
    parser.add_argument("--json", action="store_true", help="Print full state as JSON")
    args = parser.parse_args()

    settings = Settings.from_env()
    agent = ResearchAgent(settings=settings)
    state = agent.run(args.question)

    print(f"Engine: {agent.graph.engine}")
    print(f"Critique passed: {state.passed}")
    print(f"Iterations: {state.iterations}")
    print(f"Sources: {len(state.context)}")
    print()
    print(state.report)
    if args.json:
        import json

        print(json.dumps(state.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
