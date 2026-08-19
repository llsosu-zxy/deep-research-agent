# Plan Roadmap Status

Mapping of the six-week plan in `LLM_Agent项目方案.md` to the current repository.

| Plan item | Status | Evidence |
|---|---|---|
| Week 1: GitHub repo structure and CI | Done | `.github/workflows/ci.yml`, `docker/` |
| Week 1: data collection, cleaning, parent-child chunking | Done (seed corpus) | `data/corpus/seed/`, `core/retrieval/chunker.py` |
| Week 1: BM25 + dense + reranker | Done (offline + optional GPU) | `core/retrieval/index.py`, `embeddings.py`, `rerank.py` |
| Week 1: QA UI + golden set + baseline | Done | `ui/gradio_app.py`, 103-question `eval/golden_set.py` |
| Week 2: LangGraph four nodes | Done | `agents/graph.py` runs real LangGraph when installed |
| Week 2: web_search, pdf_parse, sqlite_query tools | Done | `core/tools/web.py`, `pdf.py`, `sqlite_query.py` |
| Week 2: function calling and tool registry | Done | `core/tools/registry.py`, `agents/llm.py` |
| Week 3: critic feedback rerun | Done | `agents/nodes.py`, `agents/brain.py` |
| Week 3: multi-hop scenarios | Done | Multi-hop entries in golden set |
| Week 3: 60-question golden set | Done | `eval/golden_set.py` expands to 60+ |
| Week 3: single RAG vs agent comparison | Done | `scripts/run_comparison.py`, `docs/comparison_report.md` |
| Week 4: injection / PII filters | Done | `core/guardrails/` |
| Week 4: trace and cost/latency stats | Done | `core/tracing.py`, eval metrics |
| Week 4: budget, timeout, auto-degradation | Done | `core/tracing.py`, HTTP timeouts, LLM/graph fallback |
| Week 4: error injection tests | Done | `tests/test_failure_modes.py` |
| Week 5: FastAPI + Docker + HF Space demo | Done | `app/`, `docker/`, `hf_space_app.py`, `hf_space/README.md` |
| Week 5: 80-120 golden set + eval report | Done (103 questions) | `docs/eval_report.md` |
| Week 5: tech blog | Draft done | `docs/blog_draft.md` |
| LLM-as-Judge evaluation adapter | Done (optional) | `eval/judge.py` |
| Week 6: README + resume bullets + interview prep | Done | `README.md`, `docs/resume_notes.md` |
| GPU-heavy LoRA fine-tuning | Gated, intentionally not run | `scripts/train_lora.py` |
