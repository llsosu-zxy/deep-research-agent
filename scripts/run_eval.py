from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.report import write_report
from eval.runner import EvalRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden-set regression.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "eval_report.md")
    args = parser.parse_args()

    runner = EvalRunner()
    rows = runner.run(limit=args.limit)
    path = write_report(rows, args.output)
    print(f"Ran {len(rows)} cases; report written to {path}")


if __name__ == "__main__":
    main()
