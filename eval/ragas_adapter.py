from __future__ import annotations

from typing import Any


def run_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truth: list[str] | None = None,
) -> dict[str, Any]:
    """Optional RAGAS evaluation. Requires ragas plus an LLM/embedding backend."""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ragas is not installed. Install with `pip install ragas` and configure an LLM provider."
        ) from exc

    from datasets import Dataset

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truth:
        data["ground_truth"] = ground_truth
    dataset = Dataset.from_dict(data)
    metrics = [answer_relevancy, faithfulness, answer_correctness, context_precision, context_recall]
    result = evaluate(dataset, metrics=metrics)
    return result.to_pandas().to_dict(orient="records")
