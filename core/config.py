from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime configuration loaded from environment variables."""

    llm_provider: str = "mock"  # mock | openai_compatible
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    embedding_mode: str = "hash"  # hash | sentence-transformers | bge-m3
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    reranker: str = "identity"  # identity | cross-encoder
    reranker_model: str = "BAAI/bge-reranker-base"
    max_tool_steps: int = 12
    max_critic_iterations: int = 2
    daily_budget_usd: float = 0.50
    require_plan_approval: bool = False
    retrieval_top_k: int = 5
    cache_dir: Path = field(default_factory=lambda: Path("data/storage"))
    corpus_dir: Path = field(default_factory=lambda: Path("data/corpus"))
    trace_path: Path = field(default_factory=lambda: Path("data/storage/traces.jsonl"))
    seed_dir: Path = field(default_factory=lambda: Path("data/corpus/seed"))

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = Path(os.getenv("DATA_DIR", "data"))
        cache = data_dir / "storage"
        corpus = data_dir / "corpus"
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "mock").strip().lower(),
            llm_base_url=os.getenv("LLM_BASE_URL", "").strip(),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "deepseek-chat").strip(),
            embedding_mode=os.getenv("EMBEDDING_MODE", "hash").strip().lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip(),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu").strip(),
            reranker=os.getenv("RERANKER", "identity").strip().lower(),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base").strip(),
            max_tool_steps=_env_int("MAX_TOOL_STEPS", 12),
            max_critic_iterations=_env_int("MAX_CRITIC_ITERATIONS", 2),
            daily_budget_usd=float(os.getenv("DAILY_BUDGET_USD", "0.50")),
            require_plan_approval=os.getenv("REQUIRE_PLAN_APPROVAL", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 5),
            cache_dir=cache,
            corpus_dir=corpus,
            trace_path=cache / "traces.jsonl",
            seed_dir=corpus / "seed",
        )
