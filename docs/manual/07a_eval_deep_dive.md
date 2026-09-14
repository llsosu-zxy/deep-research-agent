# 第七章补充 评测逐指标与逐脚本走读

## 7A.1 GoldenQuestion 的字段语义

id 是稳定标识，当前为 sg-01 到 sg-103。question 是实际发送给 Agent 的问题。question_type 决定多跳综合率是否参与计算。expected_keywords 是覆盖指标的关键词。note 用于人工审阅，不参与自动计算。

## 7A.2 覆盖指标的计算细节

```python
report_lower = state.report.lower()
return sum(1 for keyword in expected_keywords if keyword.lower() in report_lower) / len(expected_keywords)
```

它做的是子串匹配，不是分词匹配。例如 expected_keywords=["S$"]，只要报告里出现 S$ 就算命中。expected_keywords 为空时返回 0.0。

这个指标的优点是稳定、便宜、可解释；缺点是同义词不算命中，错误上下文也可能命中。

## 7A.3 引用指标的计算细节

```python
if not state.context:
    return 1.0 if "blocked" in state.report.lower() else 0.0
ok, _issues = validate_citations(state.report, state.context)
return 1.0 if ok else 0.0
```

没有 context 时，如果报告包含 blocked，则引用指标为 1；否则为 0。有 context 时，只要 validate_citations 失败，整题为 0。

## 7A.4 工具成功率

```python
if not state.tool_log:
    return 0.0
return sum(1 for item in state.tool_log if item.ok) / len(state.tool_log)
```

无工具调用的题目工具成功率为 0。这在 no-answer 或 blocked 场景可能影响整体均值，因此当前汇总时把每题的 0.0 也计入平均。后续可以把 no-tool 题排除。

## 7A.5 多跳综合率的分母

```python
multi_hop_rows = [(golden, state) for golden, state, _ in rows if golden.get("question_type") == "multi-hop"]
multi_hop_rate = round(sum(multi_hop) / len(multi_hop), 4) if multi_hop else 1.0
```

分母只包含 multi-hop 题，不包含单跳题。这样 Agent 的 1.0 与单轮 RAG 的 0.0 是可比的。

## 7A.6 p50 与 p95 的计算

```python
latencies.sort()
p50 = statistics.median(latencies)
p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
```

p50 使用中位数。p95 使用索引 int(n*0.95)，并用 len-1 截断，避免越界。对于 10 个样本，int(10*0.95)=9，取第 10 个值。

## 7A.7 EvalRunner 的执行循环

```python
def run_question(self, question):
    started = time.perf_counter()
    state = self.agent.run(question.question)
    duration_ms = (time.perf_counter() - started) * 1000
    return ({...golden fields...}, state, duration_ms)

def run(self, limit=None):
    return [self.run_question(q) for q in get_golden_set(limit)]
```

评测是串行执行。真实 LLM 模式下，如果 10 题，每题 20 秒，整体大约 3-4 分钟。

## 7A.8 报告的逐题表如何生成

```python
for golden, state, duration in rows:
    lines.append("| {} | {} | {} | {} | {} | {} | {:.0f} |".format(
        golden["id"],
        golden["question_type"],
        state.passed,
        round(answer_coverage(state, golden["expected_keywords"]), 2),
        round(citation_accuracy(state), 2),
        round(tool_success_rate(state), 2),
        duration,
    ))
```

逐题表没有输出 report 正文，只输出指标。要检查具体报告，需要看 state.report 或 traces。

## 7A.9 单轮 RAG 基线的构造

```python
ranked = self.index.search(question, top_k=self.top_k)
sources = self.index.to_sources(ranked)
report = self._render(question, sources)
state = GraphState(question=question, context=sources, draft=report, report=report)
ok, _ = validate_citations(report, sources)
state.passed = ok or not sources
```

它没有 Planner、Critic、多轮检索。它的 tool_log 只有一条 retrieve 记录。默认 top_k=3，由 run_comparison.py 的 --baseline-k 控制。

## 7A.10 为什么 Agent 在覆盖指标上优于单轮 RAG

单轮 RAG 只执行一次 top-3 检索，可能漏掉部分实体。Agent 对每个实体执行独立检索，检索深度至少 20，再按实体过滤和按文档合并。因此 Agent 的 context 覆盖更多实体，报告覆盖也更高。

对比结果：Agent 0.8042，SingleTurnRAG 0.6521，提升 0.1521。

## 7A.11 真实 LLM 评测的失败模式

真实 10 题中，sg-05 与 sg-10 为 multi-hop，Critic 未通过。原因是 LLM 综合报告没有按 validate_citations 的要求引用全部来源。这个结果不是检索失败，而是生成格式不合格。

可能的改进：

1. 在 Synthesizer prompt 中列出每个来源编号，要求逐条引用。
2. 在 Critic 失败时把 citation issues 作为 feedback 传回 Synthesizer。
3. 对真实 LLM 模式采用分层引用校验。
4. 在最终报告中自动追加未引用来源列表。

## 7A.12 LLM-as-Judge 的输入

```python
prompt = (
    f"Question: {question}\n\nExpected facts: {expected_keywords}\n\n"
    f"Report:\n{report[:6000]}\n\n"
    "Return JSON with score (0-1), verdict (pass/fail) and reason."
)
```

它要求模型返回 JSON，并解析 score、verdict、reason。没有 LLM 时，用关键词覆盖作为 fallback，阈值 0.5。

## 7A.13 场景测试与 golden set 的区别

golden set 关注指标，每个问题有 expected_keywords。场景测试关注行为是否符合预期，例如注入攻击必须被拦截、计算题必须输出 400、矛盾题必须出现 Conflicting Evidence。场景测试不计算平均值，只输出每场景的 Expected、Actual、Blocked、Sources、ms。

## 7A.14 评测体系当前缺口

第一，没有 LLM-as-Judge 的实际批量运行结果，只有适配器。

第二，没有 RAGAS 实际指标，因为依赖不兼容。

第三，没有人工抽检 20%。可以在 docs 中增加人工评审模板。

第四，没有把 token 成本纳入逐题指标。BudgetTracker 未接入。

第五，覆盖指标是关键词级，不是语义级。真实 LLM 报告可能包含同义表达但关键词未命中。
