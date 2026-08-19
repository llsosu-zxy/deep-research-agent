from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.agent import ResearchAgent
from core.config import Settings
from core.tools.python_sandbox import build_python_sandbox_tool


class FailureModesTest(unittest.TestCase):
    @staticmethod
    def _settings(tmp: str, **overrides) -> Settings:
        base = {
            "corpus_dir": Path(tmp) / "corpus",
            "cache_dir": Path(tmp) / "storage",
            "trace_path": Path(tmp) / "storage" / "traces.jsonl",
        }
        base.update(overrides)
        return Settings(**base)

    def test_empty_corpus_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            corpus.mkdir()
            agent = ResearchAgent(settings=self._settings(tmp))
            state = agent.run("What AI internships are available in Singapore?")
            self.assertTrue(state.report)
            self.assertFalse(state.passed)

    def test_llm_api_failure_falls_back_to_heuristics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            corpus.mkdir()
            (corpus / "shopee.md").write_text(
                "---\ntitle: Shopee AI\nsource_url: https://shopee\n---\n"
                "Shopee AI interns use Python and PyTorch.",
                encoding="utf-8",
            )
            agent = ResearchAgent(
                settings=self._settings(
                    tmp,
                    llm_provider="openai_compatible",
                    llm_base_url="http://127.0.0.1:1/v1",
                    llm_api_key="invalid",
                )
            )
            state = agent.run("What skills do Shopee AI interns need?")
            self.assertIn("Shopee", state.report)
            self.assertGreater(len(state.context), 0)

    def test_sandbox_rejects_dangerous_code(self) -> None:
        tool = build_python_sandbox_tool()
        result = tool.invoke(code="import os\nprint(os.getcwd())")
        self.assertFalse(result.ok)
        self.assertIn("sandbox error", result.error)


if __name__ == "__main__":
    unittest.main()
