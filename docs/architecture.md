# Architecture

```text
User question
      |
      v
Guardrails (injection / PII / length)
      |
      v
Planner -> research plan (subtasks + tools)
      |
      v
Executor loop -> Tool Registry
      |         - retrieve (BM25 + dense + rerank)
      |         - web_search / fetch_url
      |         - arxiv_search / pdf_parse
      |         - python_sandbox / sqlite_query
      v
Synthesizer -> cited Markdown draft
      |
      v
Critic -> citation check + coverage check
      |   (fail -> back to Executor, max N iterations)
      v
Final report + trace + budget record
      |
      v
FastAPI REST / WebSocket + Gradio UI
```

The graph is defined once in `agents/graph.py`. When `langgraph` is installed it
compiles the same nodes into a real `StateGraph`; otherwise `MiniGraph` runs the
same node functions in an equivalent loop so the project stays runnable offline.

Retrieval lives in `core/retrieval/`:

- `chunker.py` parses Markdown front-matter and creates heading-aware chunks.
- `bm25.py` is a dependency-free BM25 implementation.
- `embeddings.py` provides a deterministic hash embedder for offline mode and a
  `sentence-transformers` adapter for real dense retrieval.
- `index.py` combines BM25 and cosine scores, then applies an optional reranker.

Evaluation is intentionally separate from inference. `eval/golden_set.py` keeps
the 20-question set, `eval/runner.py` executes it, and `eval/metrics.py` reports
coverage, citations, tool success, critique pass rate and latency percentiles.
