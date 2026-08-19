from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.retrieval.index import RetrievalIndex


class RetrievalTest(unittest.TestCase):
    def test_hybrid_search_finds_relevant_doc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            corpus.mkdir()
            (corpus / "shopee.md").write_text(
                "---\ntitle: Shopee AI\nsource_url: https://shopee\n---\n"
                "# Roles\nShopee AI interns use Python and PyTorch for ranking.",
                encoding="utf-8",
            )
            (corpus / "tiktok.md").write_text(
                "---\ntitle: TikTok AI\nsource_url: https://tiktok\n---\n"
                "# Roles\nTikTok research interns work on LLMs and recommendations.",
                encoding="utf-8",
            )
            index = RetrievalIndex.from_corpus(corpus)
            results = index.search("Shopee PyTorch internship", top_k=3)
            self.assertTrue(results)
            self.assertEqual(results[0].chunk.metadata["title"], "Shopee AI")

    def test_empty_corpus_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = RetrievalIndex.from_corpus(Path(tmp))
            self.assertEqual(index.search("anything"), [])


if __name__ == "__main__":
    unittest.main()
