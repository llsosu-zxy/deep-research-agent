from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from core.models import Chunk, Source
from core.retrieval.bm25 import BM25Okapi
from core.retrieval.chunker import chunk_markdown_document, load_markdown_corpus, tokenize
from core.retrieval.embeddings import Embedder, HashEmbedder
from core.retrieval.rerank import RankedChunk, make_reranker


class RetrievalIndex:
    """Hybrid BM25 + dense retrieval index with optional reranking."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder | None = None,
        alpha: float = 0.6,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder or HashEmbedder()
        self.alpha = alpha
        self.bm25 = BM25Okapi([chunk.tokens for chunk in self.chunks])
        if self.embedder is not None and all(chunk.embedding is None for chunk in self.chunks):
            embeddings = self.embedder.encode([chunk.text for chunk in self.chunks])
            for chunk, vector in zip(self.chunks, embeddings):
                chunk.embedding = vector

    @classmethod
    def from_corpus(
        cls,
        corpus_dir: Path,
        embedder: Embedder | None = None,
        max_tokens: int = 350,
        overlap: int = 60,
    ) -> RetrievalIndex:
        chunks: list[Chunk] = []
        for doc_id, markdown, _ in load_markdown_corpus(corpus_dir):
            _, doc_chunks = chunk_markdown_document(doc_id, markdown, max_tokens, overlap)
            chunks.extend(doc_chunks)
        return cls(chunks, embedder)

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return float(dot)

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float | None = None,
        reranker: str = "identity",
    ) -> list[RankedChunk]:
        if not self.chunks:
            return []
        query_tokens = tokenize(query)
        bm25_scores = self.bm25.scores(query_tokens)
        query_embedding = self.embedder.encode([query])[0]
        dense_scores = [
            self._cosine(query_embedding, chunk.embedding) if chunk.embedding else 0.0
            for chunk in self.chunks
        ]
        effective_alpha = self.alpha if alpha is None else alpha
        combined = [
            effective_alpha * bm + (1 - effective_alpha) * dense
            for bm, dense in zip(bm25_scores, dense_scores)
        ]
        ranked = sorted(
            (RankedChunk(chunk, score) for chunk, score in zip(self.chunks, combined)),
            key=lambda item: item.score,
            reverse=True,
        )[: max(top_k * 2, top_k)]
        reranker_obj = make_reranker(reranker)
        return reranker_obj.rerank(query, ranked, top_k)

    def to_sources(self, items: Sequence[RankedChunk]) -> list[Source]:
        sources: list[Source] = []
        for item in items:
            chunk = item.chunk
            sources.append(
                Source(
                    id=chunk.id,
                    title=str(chunk.metadata.get("title", chunk.doc_id)),
                    url=str(chunk.metadata.get("source_url", "")),
                    snippet=chunk.text[:500],
                    metadata={
                        "doc_id": chunk.doc_id,
                        "heading": chunk.metadata.get("heading", ""),
                        "score": round(item.score, 4),
                    },
                )
            )
        return sources
