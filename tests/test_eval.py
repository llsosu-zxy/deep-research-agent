from __future__ import annotations

import unittest

from core.models import GraphState, Source
from eval.metrics import answer_coverage, citation_accuracy, tool_success_rate


class EvalMetricsTest(unittest.TestCase):
    def test_metrics(self) -> None:
        state = GraphState(
            question="Compare Shopee and Grab",
            context=[
                Source(id="s1", title="Shopee", snippet="Shopee AI intern skills include Python."),
                Source(id="s2", title="Grab", snippet="Grab data science interns use SQL."),
            ],
        )
        state.report = "Shopee uses Python [1]. Grab uses SQL [2]."
        state.tool_log = []
        self.assertEqual(citation_accuracy(state), 1.0)
        self.assertEqual(answer_coverage(state, ["Shopee", "Grab"]), 1.0)
        self.assertEqual(tool_success_rate(state), 0.0)


if __name__ == "__main__":
    unittest.main()
