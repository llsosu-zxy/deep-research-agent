from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.agent import ResearchAgent
from core.config import Settings


class ApprovalTest(unittest.TestCase):
    @staticmethod
    def _agent(tmp: str):
        corpus = Path(tmp) / "corpus"
        corpus.mkdir()
        (corpus / "shopee.md").write_text(
            "---\ntitle: Shopee AI\nsource_url: https://shopee\n---\n"
            "Shopee AI interns use Python and PyTorch.",
            encoding="utf-8",
        )
        settings = Settings(
            corpus_dir=corpus,
            cache_dir=Path(tmp) / "storage",
            trace_path=Path(tmp) / "storage" / "traces.jsonl",
        )
        return settings

    def test_plan_approval_rejection_stops_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = ResearchAgent(
                settings=self._agent(tmp),
                approval_callback=lambda plan: False,
            )
            state = agent.run("What skills do Shopee AI interns need?")
            self.assertIn("pending approval", state.report)
            self.assertFalse(state.passed)
            self.assertEqual(len(state.tool_log), 0)

    def test_plan_approval_acceptance_runs_normally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = ResearchAgent(
                settings=self._agent(tmp),
                approval_callback=lambda plan: True,
            )
            state = agent.run("What skills do Shopee AI interns need?")
            self.assertTrue(state.passed)
            self.assertGreater(len(state.context), 0)


if __name__ == "__main__":
    unittest.main()
