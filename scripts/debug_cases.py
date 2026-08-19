from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.agent import ResearchAgent
from core.config import Settings


def _settings(tmp: str, corpus: Path) -> Settings:
    return Settings(
        corpus_dir=corpus,
        cache_dir=Path(tmp) / "storage",
        trace_path=Path(tmp) / "storage" / "traces.jsonl",
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus"
        corpus.mkdir()
        (corpus / "shopee-a.md").write_text(
            "---\ntitle: Shopee Official\nsource_url: https://shopee-a\n---\n"
            "Shopee AI interns earn about S$2,500-4,000 per month.",
            encoding="utf-8",
        )
        (corpus / "shopee-b.md").write_text(
            "---\ntitle: Shopee Survey\nsource_url: https://shopee-b\n---\n"
            "A survey reports Shopee AI interns earn S$8,000-10,000 per month.",
            encoding="utf-8",
        )
        agent = ResearchAgent(settings=_settings(tmp, corpus))
        cases = [
            "What monthly compensation ranges are reported for Shopee AI interns?",
            "What is the exact monthly stipend for OpenAI AI interns in Singapore?",
            "If a Shopee intern works 8 hours per day, 5 days per week for 10 weeks, how many total hours do they work?",
        ]
        for question in cases:
            state = agent.run(question)
            print("====", question)
            print("passed:", state.passed)
            print("feedback:", state.feedback)
            print("subtasks:", [(s.id, s.status, s.error[:80]) for s in state.plan.subtasks])
            print("context:", [s.id for s in state.context])
            print("citations:", [t.name for t in state.trace if t.name == "critic"])
            print()


if __name__ == "__main__":
    main()
