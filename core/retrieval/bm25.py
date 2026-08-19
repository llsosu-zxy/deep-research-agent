from __future__ import annotations

import math
from collections import Counter


class BM25Okapi:
    """Small dependency-free BM25 implementation."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_count = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / max(self.doc_count, 1)
        self.doc_freqs: list[Counter[str]] = [Counter(doc) for doc in corpus]
        self.idf: dict[str, float] = {}
        df: Counter[str] = Counter()
        for doc in self.doc_freqs:
            df.update(doc.keys())
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        doc_len = len(self.corpus[doc_index])
        doc_freq = self.doc_freqs[doc_index]
        score = 0.0
        for term in set(query_tokens):
            if term not in doc_freq:
                continue
            tf = doc_freq[term]
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1.0))
            score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / denom
        return score

    def scores(self, query_tokens: list[str]) -> list[float]:
        return [self.score(query_tokens, i) for i in range(self.doc_count)]
