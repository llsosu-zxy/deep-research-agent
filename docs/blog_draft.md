# Technical Blog Draft

## Title

Building a Multi-Agent Research Copilot That You Can Actually Evaluate

## Draft

Most RAG demos stop at "query a vector store and paste chunks into a prompt."
That is useful, but it does not answer questions that need planning, tool
calling, self-checking or quantitative comparison. This project is a small
LangGraph pipeline that adds those layers without requiring a GPU cluster.

### Design decision: four nodes instead of one big prompt

Planner, Executor, Synthesizer and Critic each own one responsibility. The
critical change is that the Critic does not just grade the final answer; its
feedback is routed back to the Executor so missing evidence or uncited claims
cause a rerun. This is cheap in offline mode and easy to observe in the trace
log.

### Design decision: hybrid retrieval

BM25 handles exact keywords and entity names, dense embeddings handle
paraphrases, and an optional cross-encoder reranker improves precision when the
model is available. The offline mode uses a deterministic hash embedder, which
keeps the whole evaluation reproducible without downloading models.

### Design decision: evaluation before UI polish

The first deliverable was a 103-question golden set and a metrics runner, not a
prettier chat box. Metrics like citation accuracy and tool success rate expose
regressions that a demo answer cannot.

### Failure cases worth sharing

1. Long single paragraphs were not split into chunks, so one huge paragraph
   became one oversized chunk. The chunker now splits oversized paragraphs with
   word-level overlap.
2. Entity questions returned unrelated global top-5 results. The executor now
   filters results by the entity mentioned in the subtask, which made reports
   much cleaner.
3. The first SQLite tool left connections open on Windows, which broke temp
   directory cleanup. The tool now closes connections in a `finally` block.
4. API failures should not kill a demo. LLM planning and synthesis fall back to
   deterministic heuristics, and LangGraph falls back to MiniGraph.

### Next steps

Enable real LLM and embedding/reranker models, run the comparison experiment,
then unlock the gated LoRA fine-tuning step for tool-calling routing on GPU.
