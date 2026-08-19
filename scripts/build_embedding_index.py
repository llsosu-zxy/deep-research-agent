from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import Settings
from core.retrieval.embeddings import make_embedder
from core.retrieval.index import RetrievalIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and cache a dense embedding index on GPU or CPU.")
    parser.add_argument("--mode", default=None, help="hash | sentence-transformers | bge-m3")
    parser.add_argument("--model", default=None, help="Hugging Face model name")
    parser.add_argument("--device", default=None, help="cpu | cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "storage" / "index.json")
    args = parser.parse_args()

    settings = Settings.from_env()
    mode = args.mode or settings.embedding_mode
    model = args.model or settings.embedding_model
    device = args.device or settings.embedding_device
    embedder = make_embedder(mode, model, device)

    started = time.perf_counter()
    index = RetrievalIndex.from_corpus(settings.corpus_dir, embedder)
    index.to_json(args.output)
    elapsed = time.perf_counter() - started
    print(
        f"Built {len(index.chunks)} chunk vectors with mode={mode} model={model} "
        f"device={device} in {elapsed:.2f}s -> {args.output}"
    )
    sample = index.search("Shopee AI intern skills", top_k=3, reranker="identity")
    for item in sample:
        print("-", round(item.score, 4), item.chunk.doc_id)


if __name__ == "__main__":
    main()
