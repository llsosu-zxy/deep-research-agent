# Deep Research Agent

A multi-agent research copilot built with LangGraph, hybrid RAG, tool calling,
guardrails and a quantitative eval harness. The demo scenario is AI internship
research in Singapore (Shopee / TikTok / Grab / NVIDIA / GovTech / A*STAR), so it
is directly useful while applying for AI internships.

## Features

- Four-node LangGraph pipeline: `Planner -> Executor -> Synthesizer -> Critic`,
  with critic feedback routed back to the executor.
- Hybrid retrieval: BM25 + dense embeddings + optional cross-encoder reranking.
- Tool registry with `retrieve`, `web_search`, `fetch_url`, `arxiv_search`,
  `pdf_parse`, `python_sandbox` and read-only `sqlite_query`.
- Guardrails: PII redaction, prompt-injection detection, citation validation.
- JSONL tracing and a daily budget tracker.
- 103-question golden set (single-hop, multi-hop, no-answer, calculation and
  contradictory cases) with answer coverage, multi-hop synthesis, citation
  accuracy, tool success rate, critique pass rate and latency metrics.
- Single-turn RAG vs agent comparison experiment with a generated report.
- FastAPI REST + WebSocket API and a Gradio UI.
- Docker Compose and GitHub Actions CI.
- Offline mock mode that runs without API keys, GPU or network.
- Verified GPU path: BGE-M3 embeddings, bge-reranker and Qwen LoRA fine-tuning.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python scripts\seed_corpus.py
python scripts\run_demo.py
python scripts\run_eval.py
python scripts\run_comparison.py
pytest
```

Start the API:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start the Gradio UI:

```powershell
python ui\gradio_app.py
```

On Windows you can also double-click `start-ui.bat` for the UI or
`start-api.bat` for the API.

## Configuration

Copy `.env.example` to `.env` to enable real LLM calls:

```text
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxxx
LLM_MODEL=deepseek-chat
```

The default `LLM_PROVIDER=mock` runs the full graph offline with deterministic
heuristics over the local corpus. `EMBEDDING_MODE=hash` also runs without model
downloads; switch to `sentence-transformers` or `bge-m3` when you want local
dense embeddings (CPU or GPU depending on `EMBEDDING_DEVICE`).

## Project Layout

```text
deep-research-agent/
├── agents/        # LangGraph graph, nodes, brain, LLM client
├── app/           # FastAPI + WebSocket
├── core/          # retrieval, tools, guardrails, tracing
├── eval/          # golden set, metrics, regression runner
├── ui/            # Gradio app
├── scripts/       # seed corpus, demo, eval, gated LoRA training
├── data/corpus/   # public-source seed corpus
├── docker/        # Dockerfile + compose
├── docs/          # architecture and eval report
└── tests/
```

## Offline Baseline

The offline golden regression on the 103-question corpus reports:

| Metric | Value |
|---|---|
| Cases | 103 |
| Mean answer coverage | 80.4% |
| Multi-hop synthesis rate | 100% |
| Citation accuracy | 100% |
| Tool success rate | 100% |
| Critique pass rate | 100% |
| p50 latency | 8.8 ms |
| p95 latency | 11.3 ms |

These numbers are the offline baseline with the mock LLM and hash embeddings.
The agent achieves 100% multi-hop synthesis on the same set while the
single-turn RAG baseline scores 0%, and mean answer coverage improves by 15.2
percentage points, because the planner decomposes questions by entity and the
critic forces synthesis. Full comparison:
`docs/comparison_report.md`. Metrics should be re-measured after enabling real
LLM/embedding/reranker models.

GPU results for BGE-M3, reranking and LoRA fine-tuning are recorded in
`docs/gpu_report.md`.

## GPU Path

Install CUDA-enabled PyTorch for your platform, then:

```powershell
pip install -e ".[gpu]"
python scripts\build_embedding_index.py --mode bge-m3 --device cuda
$env:EMBEDDING_MODE="bge-m3"
$env:EMBEDDING_DEVICE="cuda"
$env:RERANKER="cross-encoder"
$env:RERANKER_DEVICE="cuda"
python scripts\run_demo.py
```

The LoRA fine-tuning step is `python scripts\train_lora.py` with
`RUN_GPU_STEP=1`; the adapter is written to `data/lora/tool-calling-1.5b`.

## Corpus

`data/corpus/seed/` contains six curated scenario documents. The import script
`scripts/import_sg_jobs.py` parses the real Singapore AI internship opportunity
sheet (38 records with company, roles, compensation, hiring process and apply
links) into `data/corpus/imported/`, including a master list for broad queries.

## GPU-Heavy Steps Are Gated

`scripts/train_lora.py` contains the LoRA fine-tuning step for tool-calling
routing. It intentionally refuses to run unless `RUN_GPU_STEP=1` is set, because
fine-tuning a 1.5B model would consume significant GPU memory and time. The main
research pipeline is CPU-safe and can run completely before that step.
