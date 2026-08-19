from __future__ import annotations

from dataclasses import dataclass

from core.models import Chunk


@dataclass
class RankedChunk:
    chunk: Chunk
    score: float


class IdentityReranker:
    def rerank(self, query: str, items: list[RankedChunk], top_k: int) -> list[RankedChunk]:
        return items[:top_k]


class CrossEncoderReranker:
    """Optional cross-encoder reranker backed by sentence-transformers."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError("sentence-transformers is not installed") from exc
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, items: list[RankedChunk], top_k: int) -> list[RankedChunk]:
        pairs = [(query, item.chunk.text) for item in items]
        scores = self.model.predict(pairs)
        scored = sorted(
            (RankedChunk(item.chunk, float(score)) for item, score in zip(items, scores)),
            key=lambda x: x.score,
            reverse=True,
        )
        return scored[:top_k]


def make_reranker(kind: str, model_name: str = "", device: str = "cpu") -> IdentityReranker | CrossEncoderReranker:
    if kind == "identity":
        return IdentityReranker()
    if kind == "cross-encoder":
        return CrossEncoderReranker(model_name or "BAAI/bge-reranker-base", device)
    raise ValueError(f"Unknown reranker kind: {kind}")
