from __future__ import annotations

import unittest

from eval.judge import LLMJudge


class LLMJudgeTest(unittest.TestCase):
    def test_keyword_fallback(self) -> None:
        judge = LLMJudge()
        result = judge.score(
            "Compare Shopee and Grab",
            "Shopee uses Python [1]. Grab uses SQL [2].",
            ["Shopee", "Grab"],
        )
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["judge"], "keyword-fallback")
        self.assertGreaterEqual(result["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
