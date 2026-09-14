# 第八章 GPU 路径与 LoRA 微调

## 8.1 GPU 环境

项目在 Windows 上使用 CUDA 环境运行 GPU 任务。已验证环境为：

- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8 GB VRAM。
- CUDA：12.8。
- PyTorch：2.11.0+cu128。
- Python：3.12。

项目默认 .venv 可以运行 CPU 与离线流程。GPU 任务使用已有 CUDA 虚拟环境，并在需要时安装 sentence-transformers、peft、accelerate、datasets 等依赖。

## 8.2 BGE-M3 向量索引

scripts/build_embedding_index.py 支持 --mode、--model、--device、--output。

```python
embedder = make_embedder(mode, model, device)
index = RetrievalIndex.from_corpus(settings.corpus_dir, embedder)
index.to_json(args.output)
sample = index.search("Shopee AI intern skills", top_k=3, reranker="identity")
```

实际运行命令：

```powershell
python scripts\build_embedding_index.py --mode bge-m3 --model BAAI/bge-m3 --device cuda --output data\storage\index-bge-m3.json
```

结果：

- 语料 chunk 数：201。
- 构建设备：CUDA。
- 构建时间：5.03 秒。
- GPU 显存占用约 3.2 GB。
- 输出：data/storage/index-bge-m3.json。

索引 JSON 中每个 chunk 保存 tokens 与 embedding。后续调用 RetrievalIndex.from_json 时，不再重新计算 corpus embedding，只需要加载 query embedding 模型。

## 8.3 bge-reranker 重排

scripts/run_gpu_retrieval.py 加载 BGE-M3 索引和 CrossEncoderReranker：

```python
embedder = make_embedder(args.embedding_mode, args.embedding_model, args.device)
index = RetrievalIndex.from_json(args.index, embedder)
results = index.search(
    args.query,
    top_k=5,
    reranker="cross-encoder",
    reranker_model=args.reranker_model,
    reranker_device=args.device,
)
```

实际验证：

- 重排模型：BAAI/bge-reranker-base。
- 查询：Shopee LLM Agent and Prompt Engineering intern skills。
- 第一名：Shopee（Sea Group）LLM Agent & Prompt Engineering Intern。
- 包含模型加载的耗时：59.10 秒。
- GPU 分配显存：3235.8 MiB。

另外，使用 BGE-M3 + bge-reranker 跑过一次端到端 Agent demo，LangGraph 流程通过 Critic，并生成 8 个带引用来源的报告。

## 8.4 LoRA 微调目标

LoRA 的目标是让 1.5B 小模型学习“研究计划”的 JSON 输出格式，而不是替代主流程中的 DeepSeek。微调样本覆盖：

- Shopee 单实体计划。
- TikTok 与 Grab 多实体对比计划。
- 计算子任务计划。
- OpenAI 无答案计划。
- Shopee 薪资矛盾计划。
- 2026 趋势总结计划。

## 8.5 LoRA 配置

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

## 8.6 训练参数

```python
training_args = TrainingArguments(
    output_dir=str(output_dir),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=20,
    learning_rate=2e-4,
    gradient_checkpointing=True,
    logging_steps=2,
    save_steps=10,
    save_total_limit=1,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
)
```

## 8.7 运行门控

train_lora.py 默认拒绝运行：

```python
if os.getenv("RUN_GPU_STEP", "0") != "1":
    print("GPU-heavy LoRA fine-tuning is intentionally gated. Set RUN_GPU_STEP=1 and confirm GPU memory before continuing.")
    sys.exit(0)
```

运行时命令：

```powershell
$env:RUN_GPU_STEP='1'
python scripts\train_lora.py
```

## 8.8 实际训练结果

@CAPTION: 表 8-1 LoRA 训练结果
:::table
项目|配置或结果
基础模型|Qwen/Qwen2.5-1.5B-Instruct
方法|LoRA r=16，alpha=32，fp16
训练步数|20
batch size|1
梯度累积|4
初始 loss|5.897
最终观察到 loss|0.1778
train_loss|1.256
训练耗时|63.44 秒
GPU 分配显存|3174.2 MiB
适配器目录|data/lora/tool-calling-1.5b
:::

训练过程日志中 loss 从 5.897 逐步下降到 0.1778，说明模型快速适应了小样本 JSON 输出格式。由于样本只有 6 条，这不是生产级微调，而是一次 PEFT 流程与显存验证。

## 8.9 LoRA 推理验证

scripts/test_lora.py 加载基础模型与 adapter：

```python
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
model = PeftModel.from_pretrained(model, adapter_dir)
prompt = "### Instruction\nPlan a research task comparing Shopee and TikTok AI internships.\n\n### Response\n"
outputs = model.generate(**inputs, max_new_tokens=160, do_sample=False)
```

实际输出已经开始生成 JSON 结构：

```json
{"subtasks": [{"id": "1", "question": "What is the purpose of this research?"}
```

这证明 adapter 已可加载，且模型学会了 JSON 起始结构。由于训练样本少，输出内容质量还不稳定，后续可以用真实 Planner 日志构造更大的工具调用数据集。

## 8.10 GPU 路径的限制

第一，BGE-M3 与 reranker 的显存占用会叠加。在 8 GB 显卡上，两者与 Qwen 1.5B 不应同时常驻。

第二，当前 LoRA 数据集只有 6 条，目标是验证流程，不是追求模型效果。

第三，训练脚本没有自动清理 checkpoint。save_total_limit=1 可以限制保留数量，但 data/lora 目录仍应加入 .gitignore。

第四，RAGAS 在当前 ragas 0.4.3 与 langchain-community 组合下无法导入，GPU 评测与 RAGAS 评测属于两条不同的可选路径。
