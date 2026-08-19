from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.retrieval.embeddings import make_embedder
from core.retrieval.index import RetrievalIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPU hybrid retrieval with reranking.")
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "storage" / "index-bge-m3.json")
    parser.add_argument("--embedding-mode", default="bge-m3")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query", default="Shopee LLM Agent and Prompt Engineering intern skills")
    args = parser.parse_args()

    embedder = make_embedder(args.embedding_mode, args.embedding_model, args.device)
    index = RetrievalIndex.from_json(args.index, embedder)
    started = time.perf_counter()
    results = index.search(
        args.query,
        top_k=5,
        reranker="cross-encoder",
        reranker_model=args.reranker_model,
        reranker_device=args.device,
    )
    elapsed = time.perf_counter() - started
    print(f"Query: {args.query}")
    print(f"Elapsed: {elapsed:.2f}s")
    for item in results:
        print("-", round(item.score, 4), item.chunk.doc_id, item.chunk.metadata.get("title", "")[:60])
    try:
        import torch

        print("GPU allocated MiB:", round(torch.cuda.memory_allocated() / 1024**2, 1))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
