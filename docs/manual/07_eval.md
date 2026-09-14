# 第七章 评测体系

## 7.1 评测体系结构

评测代码位于 eval/，与推理代码解耦。它只依赖 agents/agent.py 的 ResearchAgent.run 和 core/models.py 的 GraphState。这样可以在不修改评测逻辑的情况下替换 LLM、embedding、reranker 或检索参数。

@CAPTION: 表 7-1 评测模块职责
:::table
文件|职责|关键对象
golden_set.py|定义 103 题 golden set 与动态题目生成|GoldenQuestion、get_golden_set
metrics.py|计算覆盖、引用、工具成功率、多跳综合率、延迟|tool_success_rate、citation_accuracy、answer_coverage、multi_hop_synthesis、summarize
runner.py|执行 golden set 并返回 (golden, state, duration)|EvalRunner
report.py|把评测结果渲染为 Markdown|render_report、write_report
baseline_rag.py|单轮 RAG 基线|SingleTurnRAG
judge.py|LLM-as-Judge 与关键词 fallback|LLMJudge
scenarios.py|12 个端到端场景|Scenario、SCENARIOS
ragas_adapter.py|可选 RAGAS 适配|run_ragas
:::

## 7.2 GoldenQuestion

```python
@dataclass
class GoldenQuestion:
    id: str
    question: str
    question_type: str
    expected_keywords: list[str] = field(default_factory=list)
    note: str = ""
```

question_type 包括 single-hop、multi-hop、contradictory、no-answer、calculation。expected_keywords 用于自动覆盖评分；note 记录出题依据。

## 7.3 103 题的组成

### 7.3.1 第一批：sg-01 到 sg-20

这 20 题是人工设计的核心场景，覆盖：

- Shopee、TikTok、Grab、NVIDIA、GovTech、A*STAR 的单跳问题。
- Shopee/TikTok/Grab、Shopee/Grab、NVIDIA/Shopee 的多跳对比。
- 招聘趋势、技能、return offer、面试主题。

### 7.3.2 第二批：sg-21 到 sg-60

_build_generated_questions 根据 6 个实体模板生成 36 题，再加 4 个趋势题。每个实体模板包含 teams、skills、process、duration、after、unique 六组关键词。题目仍然是确定性的，不调用 LLM。

```python
templates = [
    ("single-hop", f"What AI teams do {entity} interns join?", spec["teams"]),
    ("single-hop", f"What skills matter most for {entity} AI interns?", spec["skills"]),
    ("single-hop", f"What is the hiring process for {entity} AI interns?", spec["process"]),
    ("single-hop", f"How long is the {entity} AI internship?", spec["duration"]),
    ("single-hop", f"What opportunities can {entity} AI interns expect after the program?", spec["after"]),
    ("single-hop", f"What makes the {entity} AI internship technically unique?", spec["unique"]),
]
```

### 7.3.3 第三批：sg-61 到 sg-100

_build_real_data_questions 从导入的真实岗位文档中选择 20 家公司，每家公司生成两题：

```python
questions.append(GoldenQuestion(
    id=f"sg-{index:02d}",
    question=f"What AI or data internship roles does {company} offer in Singapore?",
    question_type="single-hop",
    expected_keywords=[company, "AI"],
))
questions.append(GoldenQuestion(
    id=f"sg-{index:02d}",
    question=f"What compensation range is listed for {company} internships in Singapore?",
    question_type="single-hop",
    expected_keywords=[company, "S$"],
))
```

涉及公司包括 Tencent、Huawei、Alibaba、Amazon、Apple、SAP、PayPal、Visa、Salesforce、DBS、OCBC、UOB、JPMorgan、GIC、ST Engineering、Singtel、Razer、Micron、Cynapse、ESGPedia。

### 7.3.4 第四批：sg-101 到 sg-103

三题覆盖困难类型：

- sg-101 calculation：8 小时/天、5 天/周、10 周，总小时数。期望关键词 400。
- sg-102 no-answer：OpenAI 新加坡实习的精确月薪。期望报告包含 no、evidence。
- sg-103 contradictory：Shopee 实习薪资范围。期望报告包含 Shopee、S$、Conflicting。

## 7.4 指标实现

### 7.4.1 工具成功率

```python
def tool_success_rate(state):
    if not state.tool_log:
        return 0.0
    return sum(1 for item in state.tool_log if item.ok) / len(state.tool_log)
```

### 7.4.2 引用准确率

```python
def citation_accuracy(state):
    if not state.context:
        return 1.0 if "blocked" in state.report.lower() else 0.0
    ok, _issues = validate_citations(state.report, state.context)
    if not ok:
        return 0.0
    return 1.0
```

这是二元指标：只要有一个来源未引用或引用越界，整题记为 0。

### 7.4.3 答案覆盖

```python
def answer_coverage(state, expected_keywords):
    if not expected_keywords:
        return 0.0
    report_lower = state.report.lower()
    return sum(1 for keyword in expected_keywords if keyword.lower() in report_lower) / len(expected_keywords)
```

覆盖是关键词级指标，优点是确定性、零成本，缺点是只能衡量词是否出现，不能判断语义是否正确。

### 7.4.4 多跳综合率

```python
def multi_hop_synthesis(state, expected_keywords, question_type):
    if question_type != "multi-hop":
        return 1.0
    report_lower = state.report.lower()
    has_comparison = "comparison" in report_lower
    has_entities = all(keyword.lower() in report_lower for keyword in expected_keywords)
    return 1.0 if has_comparison and has_entities else 0.0
```

这个指标要求报告同时出现 comparison 字样与全部实体。单轮 RAG 模板没有 Comparison 段落，因此在这个指标上为 0；Agent 模板有 Comparison Notes，因此为 1。

### 7.4.5 summarize

summarize 接收 (golden_dict, state, duration_ms) 元组列表，计算平均值与延迟分位数：

```python
latencies.sort()
p50 = statistics.median(latencies)
p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
return {
    "cases": len(rows),
    "mean_answer_coverage": round(sum(coverage) / len(coverage), 4),
    "multi_hop_synthesis_rate": multi_hop_rate,
    "citation_accuracy": round(sum(citations) / len(citations), 4),
    "tool_success_rate": round(sum(tools) / len(tools), 4),
    "passed_critique_rate": round(sum(1 for _, state, _ in rows if state.passed) / len(rows), 4),
    "latency_p50_ms": round(p50, 1),
    "latency_p95_ms": round(p95, 1),
}
```

## 7.5 EvalRunner 与报告

EvalRunner.run_question 调用 agent.run，并返回 golden、state、duration。run 方法按 get_golden_set(limit) 顺序执行。

report.render_report 输出 Markdown 摘要表和逐题表。write_report 写入指定路径。

运行命令：

```powershell
.\.venv\Scripts\python.exe scripts\run_eval.py
.\.venv\Scripts\python.exe scripts\run_eval.py --limit 10 --output docs\eval_report_real.md
```

## 7.6 单轮 RAG 基线

SingleTurnRAG 只做一次检索：

```python
ranked = self.index.search(question, top_k=self.top_k)
sources = self.index.to_sources(ranked)
report = self._render(question, sources)
state = GraphState(question=question, context=sources, draft=report, report=report)
ok, _ = validate_citations(report, sources)
state.passed = ok or not sources
```

它没有 Planner、没有 Critic、没有多轮检索，因此是评估多智能体增益的直接对照。

scripts/run_comparison.py 对同一批 golden set 分别执行 Agent 与 SingleTurnRAG，默认 baseline-k=3。结果写入 docs/comparison_report.md。

## 7.7 LLM-as-Judge

LLMJudge 有两种模式：

- 无 LLM：用 answer_coverage 作为 fallback，verdict 阈值为 0.5。
- 有 LLM：要求模型返回 JSON，包含 score、verdict、reason。

```python
raw = self.llm.complete_json([
    {"role": "system", "content": "You are a strict answer-quality judge."},
    {"role": "user", "content": prompt},
])
return {"score": float(raw.get("score", 0)), "verdict": str(raw.get("verdict", "fail")), ...}
```

任何异常都会被捕获并返回 judge="error"，避免评测因为 judge 失败而中断。

## 7.8 12 个端到端场景

eval/scenarios.py 定义 e2e-01 到 e2e-12，覆盖多公司对比、单跳技能、无答案、计算、矛盾、注入攻击、PII、空证据、真实公司、趋势、创业公司。scripts/run_scenarios.py 运行后写 docs/scenarios_report.md。

当前场景结果：12 个场景全部符合预期，其中 e2e-06 注入攻击被正确拦截。

## 7.9 RAGAS 适配

eval/ragas_adapter.py 调用 ragas.evaluate，使用 answer_relevancy、faithfulness、answer_correctness、context_precision、context_recall 五个指标。

当前环境中安装了 ragas 0.4.3，但它依赖 langchain_community.chat_models.vertexai，当前 langchain-community 版本没有该模块，因此真实运行在导入阶段失败。这个问题已经在 docs/roadmap_status.md 中标注为可选依赖限制。RAGAS 不是项目核心路径，离线指标与 DeepSeek 真实指标已能覆盖主要评测需求。

## 7.10 当前评测结果

### 7.10.1 离线 103 题

@CAPTION: 表 7-2 离线评测结果
:::table
指标|结果
用例数|103
平均答案覆盖|0.8042
多跳综合率|1.0
引用准确率|1.0
工具成功率|1.0
Critic 通过率|1.0
延迟 p50|8.8 ms
延迟 p95|11.3 ms
:::

### 7.10.2 Agent 与单轮 RAG 对比

@CAPTION: 表 7-3 Agent vs SingleTurnRAG
:::table
指标|Agent|SingleTurnRAG|差值
平均答案覆盖|0.8042|0.6521|+0.1521
多跳综合率|1.0|0.0|+1.0
引用准确率|1.0|1.0|0.0
工具成功率|1.0|1.0|0.0
p50|8.0 ms|2.6 ms|+5.4 ms
p95|10.2 ms|3.1 ms|+7.1 ms
:::

### 7.10.3 DeepSeek v4-flash 真实 10 题

@CAPTION: 表 7-4 真实 LLM 评测
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

真实 LLM 评测的延迟远高于离线，因为每次 Planner 与 Synthesizer 都会调用远程 API。多跳综合率和引用准确率下降，原因是 LLM 综合阶段可能漏引来源或结构化程度不稳定。这正是 Critic 重试机制存在的原因，也是后续要改进的方向。
