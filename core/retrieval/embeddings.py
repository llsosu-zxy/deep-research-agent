from __future__ import annotations

import hashlib
import math
from typing import Protocol

from core.retrieval.chunker import tokenize


class Embedder(Protocol):
    dimension: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic hashed-token embedder used for offline demos and tests."""

    dimension = 256

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]


class SentenceTransformerEmbedder:
    """Optional local embedding backed by sentence-transformers."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError("sentence-transformers is not installed") from exc
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class FlagEmbeddingEmbedder:
    """BGE-M3 embedder backed by FlagEmbedding when available."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError("FlagEmbedding is not installed") from exc
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=device.startswith("cuda"),
            devices=device if device != "cpu" else None,
        )
        self.dimension = 1024

    def encode(self, texts: list[str]) -> list[list[float]]:
        output = self.model.encode(texts, return_dense=True, max_length=8192)
        return output["dense_vecs"].tolist()


def make_embedder(
    mode: str,
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
) -> Embedder:
    if mode == "hash":
        return HashEmbedder()
    if mode == "bge-m3":
        try:
            return FlagEmbeddingEmbedder(model_name, device)
        except RuntimeError:
            return SentenceTransformerEmbedder(model_name, device)
    if mode in {"sentence-transformers", "local"}:
        return SentenceTransformerEmbedder(model_name, device)
    raise ValueError(f"Unknown embedding mode: {mode}")
