# Resume and Interview Notes

## Suggested Resume Bullets

**Multi-Agent Research Copilot (LLM Agent + RAG) | Personal Project, 2026**

- Built a LangGraph four-node research copilot (Planner / Executor / Critic /
  Synthesizer) with hybrid BM25 + dense retrieval, a tool registry, guardrails
  and JSONL tracing; the offline golden regression passes 103 cases with 100%
  citation accuracy and 100% tool success rate.
- Designed a reproducible evaluation harness with answer coverage, citation
  accuracy, tool success, critique pass rate and p50/p95 latency metrics, plus
  a single-turn RAG vs agent comparison experiment to quantify the value of
  multi-agent reasoning.
- Implemented PII redaction, prompt-injection detection, citation validation,
  budget limits, API failure fallback, FastAPI REST/WebSocket, Gradio UI and
  Docker/CI packaging, keeping the demo runnable offline without GPU.

Update the numbers after the real LLM / BGE-M3 / reranker run with your API key.

## Interview Answers

### Why multi-agent instead of a single agent?

The four-node design separates planning, evidence gathering, critique and
synthesis. This makes failure isolation easier, lets the critic send targeted
feedback back to the executor, and produces traceable spans for each stage. For
simple questions a single agent is cheaper and faster, so the system should
route by difficulty.

### How do you reduce hallucination?

The report must cite every retrieved source, the critic validates citations
before release, and out-of-range or uncited sources fail the loop and trigger a
rerun. The offline mode also refuses to invent content when the corpus has no
evidence.

### How do you handle a stuck or looping agent?

The graph has a maximum critic iteration count, per-tool timeouts, a tool-step
budget, a daily cost budget, and a MiniGraph fallback if LangGraph fails. Traces
record each plan, tool call and critique so failures are attributable.

### How do you evaluate an agent?

The golden set covers single-hop, multi-hop and comparison questions. The
harness records answer coverage, citation accuracy, tool success, critique pass
rate and latency percentiles, then writes a Markdown regression report.

### How do you control cost?

The default demo uses mock mode and hash embeddings with zero API cost. Real
deployments route through a budget tracker, cache traces, cap retrieval results
and use OpenAI-compatible cheap models; the GPU-heavy LoRA step is explicitly
gated behind `RUN_GPU_STEP=1`.
