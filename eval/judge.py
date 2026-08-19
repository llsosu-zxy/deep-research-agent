from __future__ import annotations

from agents.llm import OpenAICompatibleLLM
from core.models import GraphState
from eval.metrics import answer_coverage


class LLMJudge:
    """Optional strong-model judge with a deterministic offline fallback."""

    def __init__(self, llm: OpenAICompatibleLLM | None = None) -> None:
        self.llm = llm

    def score(
        self,
        question: str,
        report: str,
        expected_keywords: list[str],
    ) -> dict:
        if self.llm is None:
            state = GraphState(question=question, report=report)
            coverage = answer_coverage(state, expected_keywords)
            return {
                "score": coverage,
                "verdict": "pass" if coverage >= 0.5 else "fail",
                "judge": "keyword-fallback",
                "reason": "mock mode: expected keyword overlap",
            }
        try:
            prompt = (
                f"Question: {question}\n\nExpected facts: {expected_keywords}\n\n"
                f"Report:\n{report[:6000]}\n\n"
                "Return JSON with score (0-1), verdict (pass/fail) and reason."
            )
            raw = self.llm.complete_json(
                [
                    {
                        "role": "system",
                        "content": "You are a strict answer-quality judge.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            return {
                "score": float(raw.get("score", 0)),
                "verdict": str(raw.get("verdict", "fail")),
                "judge": "llm",
                "reason": str(raw.get("reason", "")),
            }
        except Exception as exc:  # noqa: BLE001 - judge must not break eval
            return {
                "score": 0.0,
                "verdict": "fail",
                "judge": "error",
                "reason": str(exc),
            }
