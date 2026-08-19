from __future__ import annotations

import time

from agents.agent import ResearchAgent
from core.config import Settings
from core.models import GraphState
from eval.golden_set import GoldenQuestion, get_golden_set


class EvalRunner:
    def __init__(self, agent: ResearchAgent | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.agent = agent or ResearchAgent(settings=self.settings)

    def run_question(self, question: GoldenQuestion) -> tuple[dict, GraphState, float]:
        started = time.perf_counter()
        state = self.agent.run(question.question)
        duration_ms = (time.perf_counter() - started) * 1000
        return (
            {
                "id": question.id,
                "question": question.question,
                "question_type": question.question_type,
                "expected_keywords": question.expected_keywords,
                "note": question.note,
            },
            state,
            duration_ms,
        )

    def run(self, limit: int | None = None) -> list[tuple[dict, GraphState, float]]:
        return [self.run_question(question) for question in get_golden_set(limit)]
