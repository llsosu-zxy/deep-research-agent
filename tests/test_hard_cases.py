from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.agent import ResearchAgent
from core.config import Settings


class HardCasesTest(unittest.TestCase):
    @staticmethod
    def _settings(tmp: str, corpus_dir: Path) -> Settings:
        return Settings(
            corpus_dir=corpus_dir,
            cache_dir=Path(tmp) / "storage",
            trace_path=Path(tmp) / "storage" / "traces.jsonl",
        )

    @staticmethod
    def _single_doc_corpus(tmp: str) -> Path:
        corpus = Path(tmp) / "corpus"
        corpus.mkdir()
        (corpus / "shopee.md").write_text(
            "---\ntitle: Shopee AI Intern\nsource_url: https://shopee\n---\n"
            "Shopee AI interns use Python and PyTorch.",
            encoding="utf-8",
        )
        return corpus

    def test_calculation_subtask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = self._single_doc_corpus(tmp)
            agent = ResearchAgent(settings=self._settings(tmp, corpus))
            state = agent.run(
                "If a Shopee intern works 8 hours per day, 5 days per week for 10 weeks, "
                "how many total hours do they work?"
            )
            self.assertIn("400", state.report)
            self.assertTrue(state.passed)

    def test_no_answer_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = self._single_doc_corpus(tmp)
            agent = ResearchAgent(settings=self._settings(tmp, corpus))
            state = agent.run("What is the exact monthly stipend for OpenAI AI interns in Singapore?")
            self.assertIn("no direct evidence", state.report)
            self.assertTrue(state.passed)

    def test_conflicting_evidence_is_flagged(self) -> None:
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
            agent = ResearchAgent(settings=self._settings(tmp, corpus))
            state = agent.run("What monthly compensation ranges are reported for Shopee AI interns?")
            self.assertIn("Conflicting Evidence", state.report)
            self.assertIn("S$", state.report)
            self.assertTrue(state.passed)


if __name__ == "__main__":
    unittest.main()
