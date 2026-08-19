# GPU Verification Report

## Environment

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM
- CUDA: 12.8, PyTorch 2.11.0+cu128
- Python: 3.12 (creative-t2i CUDA venv)

## BGE-M3 Embedding Index

- Model: `BAAI/bge-m3`
- Corpus: 201 chunks from the Singapore AI internship corpus
- Device: CUDA
- Build time: 5.03 s
- Output: `data/storage/index-bge-m3.json`

## Reranking

- Reranker: `BAAI/bge-reranker-base` (cross-encoder, CUDA)
- Query: `Shopee LLM Agent and Prompt Engineering intern skills`
- Top result: `Shopee（Sea Group）- LLM Agent & Prompt Engineering Intern`
- Rerank latency: 59.10 s including model load
- GPU allocated during inference: 3235.8 MiB

## LoRA Fine-Tuning

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- PEFT: LoRA r=16, alpha=32, fp16, gradient checkpointing
- Steps: 20, batch size 1, gradient accumulation 4
- Loss: 5.897 -> 0.1778, final train loss 1.256
- Runtime: 63.44 s
- GPU allocated during training: 3174.2 MiB
- Adapter: `data/lora/tool-calling-1.5b`

Post-training inference produced a JSON research plan:

```json
{"subtasks": [{"id": "1", "question": "What is the purpose of this research?"}
```

An end-to-end Agent demo was also run with BGE-M3 + bge-reranker on CUDA;
the LangGraph pipeline passed critique with 8 cited sources.

## Notes

All GPU steps were executed and completed. The project still defaults to mock
mode with hash embeddings for zero-cost offline demos; set `EMBEDDING_MODE`,
`EMBEDDING_DEVICE`, `RERANKER` and `RERANKER_DEVICE` to use the GPU path.
