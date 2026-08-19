from __future__ import annotations

import unittest

from core.retrieval.chunker import chunk_markdown_document, parse_front_matter


class ChunkerTest(unittest.TestCase):
    def test_front_matter_parsed(self) -> None:
        markdown = "---\ntitle: Shopee AI\nsource_url: https://x\n---\nBody"
        metadata, body = parse_front_matter(markdown)
        self.assertEqual(metadata["title"], "Shopee AI")
        self.assertEqual(body.strip(), "Body")

    def test_heading_chunks_keep_metadata(self) -> None:
        markdown = (
            "---\ntitle: Test\nsource_url: https://x\n---\n"
            "# A\n" + "word " * 100 + "\n\n# B\n" + "token " * 100
        )
        _, chunks = chunk_markdown_document("doc1", markdown, max_tokens=80, overlap=10)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.metadata["title"] == "Test" for chunk in chunks))
        self.assertTrue(all(chunk.metadata["doc_id"] == "doc1" for chunk in chunks))

    def test_tokens_do_not_explode(self) -> None:
        markdown = "---\ntitle: X\n---\n# H\n" + "a b c " * 300
        _, chunks = chunk_markdown_document("d", markdown, max_tokens=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.tokens), 140)


if __name__ == "__main__":
    unittest.main()
