# 第一章 项目定位与总体架构

## 1.1 项目定位

deep-research-agent 是一个基于 LangGraph 的多智能体研究助手。它接收一个开放式研究问题，先做计划，再执行检索和工具调用，随后生成带引用的报告，最后用 Critic 检查引用与覆盖情况。项目的默认演示场景是新加坡 AI 实习机会调研，语料同时包含人工整理的种子文档和从真实实习清单 docx 导入的 38 条岗位记录。

项目目标不是做一个只会聊天的 RAG Demo，而是把以下环节做成可运行、可评测、可观测的闭环：

- 计划：把开放式问题拆成子问题，并决定每个子问题使用哪些工具。
- 检索：BM25 与向量检索混合，可选交叉编码器重排。
- 工具：统一注册表管理本地检索、网页搜索、网页抓取、arXiv、PDF、Python 沙箱和 SQLite。
- 校验：检查引用是否真实、是否引用全部来源、是否遗漏子任务。
- 评测：103 道 golden set、单轮 RAG 基线、12 个端到端场景、真实 DeepSeek 评测。
- 可观测：每个节点记录 TraceSpan，运行结束写入 JSONL。
- GPU：BGE-M3 向量、bge-reranker 重排、Qwen2.5-1.5B LoRA 微调。

## 1.2 核心能力矩阵

@CAPTION: 表 1-1 项目能力与代码位置
:::table
能力|实现位置|关键类或函数
四节点 Agent 编排|agents/graph.py、agents/nodes.py|MiniGraph、LangGraphRunner、plan_node、execute_node、synthesize_node、critic_node
规划与推理|agents/brain.py|AgentBrain.plan、_plan_with_llm
工具执行|agents/brain.py、core/tools/|AgentBrain.execute_subtask、ToolRegistry
混合检索|core/retrieval/index.py|RetrievalIndex.search
BM25|core/retrieval/bm25.py|BM25Okapi.score、BM25Okapi.scores
向量模型|core/retrieval/embeddings.py|HashEmbedder、SentenceTransformerEmbedder、FlagEmbeddingEmbedder
重排|core/retrieval/rerank.py|IdentityReranker、CrossEncoderReranker、FlagReranker
文本切块|core/retrieval/chunker.py|chunk_markdown_document、_split_paragraphs
输入安全|core/guardrails/input.py|redact_pii、detect_injection、validate_input
引用校验|core/guardrails/output.py|extract_citations、validate_citations
LLM 客户端|agents/llm.py|OpenAICompatibleLLM.complete、complete_json
API|app/api.py|create_app、/api/research、/ws/research
界面|ui/gradio_app.py|build_ui
评测|eval/|GoldenQuestion、EvalRunner、summarize、SingleTurnRAG
GPU|scripts/build_embedding_index.py、scripts/train_lora.py|BGE-M3 建索引、Qwen LoRA 微调
:::

## 1.3 总体架构

系统的静态结构可以用下面的调用链表示：

```text
用户问题
  |
  v
ResearchAgent.run
  |
  +--> Brain.sanitize
  |      +--> redact_pii
  |      +--> detect_injection
  |      +--> validate_input
  |
  +--> ResearchGraph.invoke
         |
         +--> Planner  (AgentBrain.plan)
         +--> Executor (AgentBrain.execute_subtask + ToolRegistry)
         +--> Synthesizer (AgentBrain.synthesize)
         +--> Critic (AgentBrain.critique)
         +--> Finalize (AgentBrain.finalize)
  |
  +--> TraceLogger.log
  |
  v
GraphState.report
```

这条调用链对应四个核心文件：agents/agent.py 负责入口与依赖装配，agents/graph.py 负责图执行，agents/nodes.py 负责节点适配，agents/brain.py 负责真正的计划、执行、校验与综合逻辑。

## 1.4 一次请求的完整执行顺序

下面按 time 顺序描述一次问句从进入到返回的路径。每一步都对应真实代码。

1. 调用方执行 ResearchAgent.run(question)，入口位于 agents/agent.py 的 ResearchAgent.run。
2. run 首先调用 self.brain.sanitize(question)。sanitize 内部先执行 redact_pii，再执行 validate_input。
3. 如果 validate_input 返回 ok=False，run 直接构造 GraphState，写入 "# Request blocked" 报告，记录 research_blocked trace，然后返回，不进入图执行。
4. 如果输入合法，run 调用 self.graph.invoke(sanitized)。graph 可能是 LangGraphRunner，也可能是 MiniGraph。
5. 图先执行 Planner。启发式 Planner 用 KNOWN_ENTITIES 提取实体，生成 ev-1、ev-2 等子任务，再追加 compare 子任务；如果识别出计算题，再追加 calc 子任务。
6. Executor 遍历状态为 pending 或 failed 的子任务。calc 子任务调用 python_sandbox；普通子任务调用 retrieve。
7. retrieve 返回候选 chunk。Executor 先按实体过滤，再调用 _merge_by_doc 合并同一文档的多个 chunk，最后最多保留 3 个文档作为证据来源。
8. Executor 把来源追加到 GraphState.context，并重建子任务中的引用编号。
9. Synthesizer 生成 draft。若配置了真实 LLM，则调用 _synthesize_with_llm；否则使用模板生成 Markdown。
10. Critic 检查子任务状态、证据是否存在、draft 是否存在，并调用 validate_citations 校验引用。
11. 如果 Critic 不通过且还有迭代次数，LangGraph 的条件边回到 Executor；MiniGraph 则在循环中重新执行。
12. 最终 finalize 写入 state.report 与 state.summary，run 再把完整 state 写入 TraceLogger。

## 1.5 双引擎设计

agents/graph.py 同时包含两套执行引擎。

MiniGraph 是零依赖执行器。它不依赖 langgraph，只用普通 Python 循环执行 plan_node、execute_node、synthesize_node、critic_node。这个执行器保证项目在离线或依赖不完整时仍然可运行。

LangGraphRunner 是 LangGraph 执行器。它把同一个 GraphState 在 dict 与 dataclass 之间转换，然后用 StateGraph(dict) 构建四节点图。节点顺序为 planner -> executor -> synthesizer -> critic；critic 的条件边根据 passed 与 iterations 决定回到 executor 还是结束。

ResearchGraph 负责选择引擎。如果环境中安装了 langgraph 且没有 approval_callback，它会尝试构造 LangGraphRunner；如果构造或调用失败，会记录 warning 并回退到 MiniGraph。engine 属性会返回 "langgraph"、"mini" 或 "mini-approval"，便于 API 健康检查和调试。

## 1.6 运行模式

@CAPTION: 表 1-2 项目运行模式
:::table
模式|配置|行为|适用场景
离线 mock|LLM_PROVIDER=mock，EMBEDDING_MODE=hash|不联网、不下载模型、不花钱，使用启发式规划和模板综合|本地开发、CI、面试演示
真实 LLM|LLM_PROVIDER=openai_compatible，配置 base_url、api_key、model|Planner 与 Synthesizer 调用真实模型，工具与检索仍在本地|真实报告、真实评测
GPU 检索|EMBEDDING_MODE=bge-m3，EMBEDDING_DEVICE=cuda，RERANKER=cross-encoder|BGE-M3 向量 + 交叉编码器重排|检索质量演示、显存验证
LoRA 微调|RUN_GPU_STEP=1|Qwen2.5-1.5B + LoRA 工具调用规划微调|展示 GPU 训练与 PEFT 经验
:::

## 1.7 目录结构

```text
deep-research-agent/
  agents/
    agent.py        入口 ResearchAgent
    brain.py        规划、执行、批判、综合
    graph.py        LangGraph 与 MiniGraph
    llm.py          OpenAI 兼容客户端
    nodes.py        LangGraph 节点适配
  app/
    api.py          FastAPI + WebSocket
    main.py         uvicorn 入口
  core/
    config.py       Settings
    models.py       Source/Chunk/Plan/GraphState 等
    tracing.py      TraceLogger/BudgetTracker
    guardrails/     输入与输出安全检查
    retrieval/      切块、BM25、向量、重排、索引
    tools/          工具注册表与工具实现
  eval/
    golden_set.py   103 题评测集
    metrics.py      指标计算
    runner.py       评测执行器
    report.py       Markdown 报告生成
    baseline_rag.py 单轮 RAG 基线
    judge.py        LLM-as-Judge
    scenarios.py    12 个端到端场景
    ragas_adapter.py 可选 RAGAS 适配
  scripts/
    seed_corpus.py          生成种子语料
    import_sg_jobs.py       导入真实实习清单
    run_demo.py             运行单题
    run_eval.py             运行 golden set
    run_comparison.py       对比 Agent 与单轮 RAG
    run_scenarios.py        运行 12 个场景
    build_embedding_index.py 构建 BGE-M3 索引
    run_gpu_retrieval.py     GPU 检索与重排
    train_lora.py            LoRA 微调
    test_lora.py             LoRA 推理
  ui/gradio_app.py  Gradio 界面
  tests/            103 个自动化测试
  docker/           Dockerfile 与 compose
  docs/             架构、评测、GPU、部署文档
```

## 1.8 当前量化结果

@CAPTION: 表 1-3 离线 103 题评测
:::table
指标|结果
用例数|103
平均答案覆盖|0.8042（80.42%）
多跳综合率|1.0（100%）
引用准确率|1.0（100%）
工具成功率|1.0（100%）
Critic 通过率|1.0（100%）
延迟 p50|8.8 ms
延迟 p95|11.3 ms
:::

@CAPTION: 表 1-4 DeepSeek v4-flash 真实 10 题评测
:::table
指标|结果
用例数|10
平均答案覆盖|0.85
多跳综合率|0.50
引用准确率|0.80
工具成功率|1.0
Critic 通过率|0.80
延迟 p50|20992.4 ms
延迟 p95|70259.5 ms
:::

## 1.9 项目的主要设计取舍

第一，检索层保持可替换。HashEmbedder 用于零成本离线，SentenceTransformerEmbedder 与 FlagEmbeddingEmbedder 用于真实向量模型，检索接口不变。

第二，执行图保持可降级。LangGraph 不可用或运行失败时，MiniGraph 继续执行相同节点，避免演示环境被框架依赖卡住。

第三，引用校验放在 Critic，而不是只写在提示词里。validate_citations 会检查越界引用和未引用来源，因此模型不能只写一个看似合理的答案。

第四，评测与推理分离。eval 目录只依赖 GraphState 和标准指标，不侵入 agents 与 core。这样可以替换 LLM、embedding 和 reranker，而不必重写评测。

第五，代价是当前实现没有把预算、工具步数和沙箱隔离做到生产级。BudgetTracker 和 max_tool_steps 已经定义，但还没有完全接入所有调用路径；python_sandbox 也不是操作系统级沙箱。这些限制会在第十二章逐条说明。
