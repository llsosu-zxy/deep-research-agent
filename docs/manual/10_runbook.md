# 第十章 运行手册、代码索引与复现实验

## 10.1 环境准备

项目要求 Python 3.11 以上。当前使用 Python 3.12。

```powershell
cd C:\Users\29716\Documents\demo1\deep-research-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python scripts\seed_corpus.py
```

如果需要 GPU：

```powershell
pip install -e ".[gpu]"
```

同时必须安装与你显卡驱动匹配的 CUDA 版 PyTorch。当前验证环境为 torch 2.11.0+cu128。

## 10.2 离线 mock 模式

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py
```

默认问题为 Compare Shopee, TikTok and Grab AI internship opportunities and required skills in Singapore。输出包含 Engine、Critique passed、Iterations、Sources 和完整报告。

如果只想看 JSON：

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py "Compare Shopee and Grab" --json
```

## 10.3 真实 LLM 模式

在 .env 中配置：

```text
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-你的key
LLM_MODEL=deepseek-v4-flash
```

然后运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py "Compare Shopee and Grab AI internship compensation and hiring process in Singapore."
```

真实 DeepSeek 模式会调用两次主要 LLM：Planner 一次，Synthesizer 一次；如果 Critic 失败，Synthesizer 会再次调用。

## 10.4 索引与 GPU 检索

构建 BGE-M3 索引：

```powershell
python scripts\build_embedding_index.py --mode bge-m3 --model BAAI/bge-m3 --device cuda --output data\storage\index-bge-m3.json
```

运行 GPU 检索与重排：

```powershell
python scripts\run_gpu_retrieval.py --index data\storage\index-bge-m3.json --device cuda
```

用 GPU 配置跑完整 Agent：

```powershell
$env:EMBEDDING_MODE="bge-m3"
$env:EMBEDDING_DEVICE="cuda"
$env:RERANKER="cross-encoder"
$env:RERANKER_DEVICE="cuda"
python scripts\run_demo.py "Compare Shopee and Grab AI internship compensation."
```

## 10.5 LoRA 微调

```powershell
$env:RUN_GPU_STEP='1'
python scripts\train_lora.py
python scripts\test_lora.py
```

训练结果写入 data/lora/tool-calling-1.5b。该目录已加入 .gitignore，不进入 GitHub。

## 10.6 评测命令

```powershell
# 离线 103 题
.\.venv\Scripts\python.exe scripts\run_eval.py

# 真实 LLM 前 10 题，输出到独立文件
.\.venv\Scripts\python.exe scripts\run_eval.py --limit 10 --output docs\eval_report_real.md

# Agent vs 单轮 RAG
.\.venv\Scripts\python.exe scripts\run_comparison.py

# 12 个端到端场景
.\.venv\Scripts\python.exe scripts\run_scenarios.py
```

## 10.7 API 与界面

```powershell
# API
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# UI
.\.venv\Scripts\python.exe ui\gradio_app.py
```

也可以直接双击 start-api.bat 或 start-ui.bat。

## 10.8 Docker

```powershell
docker compose -f docker\docker-compose.yml up --build
```

API 地址 http://localhost:8000，UI 地址 http://localhost:7860。

## 10.9 代码索引

@CAPTION: 表 10-1 关键类与函数索引
:::table
文件|类或函数|职责
agents/agent.py|ResearchAgent.__init__|装配索引、工具、Brain、Graph、Trace、Budget
agents/agent.py|ResearchAgent.run|输入安全检查、图执行、trace 写入
agents/brain.py|AgentBrain.plan|启发式或 LLM 规划
agents/brain.py|AgentBrain.execute_subtask|执行计算或检索子任务
agents/brain.py|AgentBrain.critique|检查子任务、上下文和引用
agents/brain.py|AgentBrain.synthesize|LLM 或模板生成报告
agents/brain.py|_merge_by_doc|合并同一文档 chunk，优先薪资片段
agents/brain.py|_find_conflicts|检测同一实体多个薪资范围
agents/graph.py|MiniGraph.invoke|无依赖循环执行四节点
agents/graph.py|LangGraphRunner._build|构建 StateGraph 与条件边
agents/graph.py|ResearchGraph.invoke|选择 LangGraph 或 MiniGraph
agents/nodes.py|plan_node|执行 Planner 并记录 span
agents/nodes.py|execute_node|执行子任务并重建引用
agents/nodes.py|synthesize_node|生成 draft 并记录 span
agents/nodes.py|critic_node|执行 Critic 并递增 iterations
agents/llm.py|OpenAICompatibleLLM.complete|调用 OpenAI 兼容 chat completions
agents/llm.py|OpenAICompatibleLLM.complete_json|解析 JSON、处理 tool_calls、重试
core/config.py|Settings.from_env|读取 .env 与环境变量
core/models.py|GraphState|图执行唯一可变状态
core/retrieval/chunker.py|chunk_markdown_document|Markdown 标题感知切块
core/retrieval/bm25.py|BM25Okapi.score|BM25 单文档打分
core/retrieval/embeddings.py|make_embedder|选择 Hash、SentenceTransformer、FlagEmbedding
core/retrieval/rerank.py|make_reranker|选择 Identity、CrossEncoder、FlagReranker
core/retrieval/index.py|RetrievalIndex.search|BM25 + dense + rerank
core/tools/registry.py|Tool.invoke|工具边界与异常捕获
core/tools/python_sandbox.py|build_python_sandbox_tool|受限 Python 计算
core/guardrails/input.py|validate_input|PII、长度、注入检查
core/guardrails/output.py|validate_citations|引用越界与未引用来源检查
eval/metrics.py|summarize|汇总覆盖、引用、工具、延迟指标
eval/golden_set.py|get_golden_set|返回 103 题全集或前 limit 题
eval/baseline_rag.py|SingleTurnRAG.run|单轮检索基线
app/api.py|create_app|REST、WebSocket、健康检查
ui/gradio_app.py|build_ui|Gradio 界面
:::

## 10.10 复现实验清单

### 10.10.1 复现离线 103 题

1. 运行 python scripts\seed_corpus.py。
2. 确认 data/corpus/seed 有 7 个文件。
3. 运行 python scripts\run_eval.py。
4. 检查 docs/eval_report.md 中 cases=103、citation_accuracy=1.0。

### 10.10.2 复现真实 LLM 10 题

1. 在 .env 配置 DeepSeek v4-flash。
2. 运行 python scripts\run_eval.py --limit 10 --output docs\eval_report_real.md。
3. 预期 p50 为十几到二十秒级，citation_accuracy 可能在 0.8 左右。

### 10.10.3 复现 GPU 检索

1. 确认 torch.cuda.is_available() 为 True。
2. 运行 build_embedding_index.py --mode bge-m3 --device cuda。
3. 运行 run_gpu_retrieval.py。
4. 检查输出第一名是否为 Shopee LLM Agent & Prompt Engineering Intern。

### 10.10.4 复现 LoRA

1. 设置 RUN_GPU_STEP=1。
2. 运行 train_lora.py。
3. 检查 data/lora/tool-calling-1.5b/adapter_model.safetensors。
4. 运行 test_lora.py。

## 10.11 面试讲解建议

讲解项目时，不要从“我用了 LangGraph”开始，而要按问题、决策、结果组织：

1. 问题：普通 RAG 不能处理多跳、计算、矛盾证据和无答案场景。
2. 方案：四节点图、混合检索、工具注册表、严格引用校验。
3. 结果：103 题离线评测、Agent 对比单轮 RAG 覆盖率 +15.2 个百分点、真实 DeepSeek 10 题评测。
4. GPU：BGE-M3、bge-reranker、Qwen LoRA 微调。
5. 复盘：引用校验过严、预算未完全接入、沙箱不是强隔离、RAGAS 版本兼容问题。

这样能同时展示实现能力、评测意识和工程边界意识。

## 10.12 后续可扩展方向

- 把 LLMResponse 的 token usage 接入 BudgetTracker，实现真实成本统计。
- 在 GraphState 中增加 tool_steps 计数器，强制执行 max_tool_steps。
- 用子进程或容器替换 python_sandbox，实现可中断的真实沙箱。
- 把 web_search 解析成结构化 Source，并加入来源可信度。
- 对真实 LLM 报告采用分层引用校验，区分核心来源和补充来源。
- 用真实 Planner 日志扩充 LoRA 数据集，训练更稳定的工具调用模型。
- 增加 RAGAS 兼容版本约束，或在适配器中显式配置 embedding provider。
- 将 HF Space 上线，加入在线 demo 与演示视频链接。
