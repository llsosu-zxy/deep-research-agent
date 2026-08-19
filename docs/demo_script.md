# 2-3 Minute Demo Script

## 0:00 - Hook

"This is a multi-agent research copilot. It takes an open-ended question,
plans the research, calls tools, checks its own citations, and produces a
cited report."

## 0:20 - Architecture

Show the LangGraph flow: Planner -> Executor -> Synthesizer -> Critic, with the
critic sending feedback back to the executor. Mention hybrid retrieval
(BM25 + dense + reranker) and the tool registry.

## 0:50 - Live Demo

Run this question:

> Compare Shopee, TikTok and Grab AI internship opportunities and required
> skills in Singapore.

Point out: the planner decomposed the question by company, the executor
collected entity-specific evidence, and the critic accepted only a fully cited
report.

## 1:30 - Evaluation

Show `docs/eval_report.md`: 103 offline cases, 100% citation accuracy, 100%
multi-hop synthesis, and the single-turn RAG comparison.

## 2:00 - GPU Work

Show `docs/gpu_report.md`: BGE-M3 embeddings and bge-reranker on the RTX 5060,
then the Qwen2.5-1.5B LoRA adapter with loss dropping from 5.9 to 0.18.

## 2:40 - Close

"The system is designed to be reproducible, measurable, and deployable from a
laptop to Docker or Hugging Face Spaces."

## Recording Tips

- Use the Gradio UI at `python ui\gradio_app.py`.
- Keep the eval report and GPU report open in the background.
- Record at 1080p and keep the demo under 3 minutes.
