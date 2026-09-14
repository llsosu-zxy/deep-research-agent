# 第五章 Agent 核心：规划、执行、综合与批判

## 5.1 AgentBrain 的职责

AgentBrain 位于 agents/brain.py，是项目的决策中枢。它不负责 HTTP、不负责 UI、不负责图调度，只做四件事：

1. plan：把问题变成 Plan。
2. execute_subtask：执行单个子任务，返回证据与来源。
3. critique：检查报告是否满足引用和覆盖要求。
4. synthesize：生成最终 Markdown 报告。

构造函数根据 Settings 决定是否创建真实 LLM 客户端：

```python
self.llm = None
if settings.llm_provider == "openai_compatible" and settings.llm_api_key:
    self.llm = OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
```

因此，没有 API Key 时不会创建 LLM，plan 和 synthesize 自动走启发式与模板路径。

## 5.2 已知实体列表 KNOWN_ENTITIES

KNOWN_ENTITIES 保存当前语料中出现的公司、机构和项目名称，包括 Shopee、TikTok、ByteDance、Grab、Google、Meta、NVIDIA、A*STAR、GovTech、OpenAI、Microsoft、Alibaba、Lazada、Tencent、Huawei、Sea、Garena、Amazon、Apple、SAP、PayPal、Visa、Salesforce、DBS、OCBC、UOB、Standard Chartered、JPMorgan、GIC、ST Engineering、Singtel、Razer、Micron、Cynapse、Guidesify、AIPilot、ESGPedia、X Star、Hong Ye、YY Circle、LinkWave。

这个列表有两个作用：

- Planner 用它提取问题中的实体，决定生成几个 ev-* 子任务。
- Executor 用它判断子任务是否需要实体过滤和更深召回。

## 5.3 计算题识别

CALC_PATTERNS 定义了三类模式：小时-天-周、天-周、加法。

```python
CALC_PATTERNS = [
    (re.compile(r"(\d+)\s*hours?\D+(\d+)\s*days?\D+(\d+)\s*weeks?", re.IGNORECASE),
     lambda h, d, w: f"print({h} * {d} * {w})"),
    (re.compile(r"(\d+)\s*days?\D+(\d+)\s*weeks?", re.IGNORECASE),
     lambda d, w: f"print({d} * {w})"),
    (re.compile(r"(\d+)\s*\+\s*(\d+)"), lambda a, b: f"print({a} + {b})"),
]
```

_calculation_code 返回要执行的 Python 代码。_is_calculation_question 同时要求问题中有 calculate、how many、total hours 或 sum of 等标记，并且能匹配到数字模式。

```python
def _is_calculation_question(question: str) -> bool:
    has_marker = bool(re.search(r"calculate|how many|total hours|sum of", question, re.IGNORECASE))
    return has_marker and _calculation_code(question) is not None
```

## 5.4 证据合并 _merge_by_doc

一次检索会返回多个 chunk。如果不合并，同一个公司的 Roles、Compensation、Hiring 会被当成三条独立来源，报告会重复。_merge_by_doc 解决这个问题。

```python
groups = {}
for item in results:
    doc_id = str(item.get("doc_id") or item.get("id") or "")
    if doc_id not in groups:
        groups[doc_id] = {"item": dict(item), "snippets": []}
    groups[doc_id]["snippets"].append(str(item.get("snippet", "")))
```

合并时优先保留包含薪资信息的 snippet：

```python
salary_snippets = [
    snippet for snippet in snippets
    if "S$" in snippet or "薪资" in snippet or "compensation" in snippet.lower()
]
ordered = salary_snippets + [snippet for snippet in snippets if snippet not in salary_snippets]
raw = "\n".join(ordered[:6])
```

然后逐行去重，最多保留 2400 字符：

```python
seen = set()
clean_lines = []
for line in raw.splitlines():
    stripped = line.strip()
    if stripped and stripped not in seen:
        seen.add(stripped)
        clean_lines.append(stripped)
item["snippet"] = "\n".join(clean_lines)[:2400]
```

这个改动直接解决了一个演示问题：Shopee 的官方岗位文档与社区 survey 文档同时存在时，如果不优先合并薪资 chunk，矛盾检测无法看到 S$2,500-4,000 与 S$8,000-10,000 两个范围。

## 5.5 矛盾检测 _find_conflicts

SALARY_RE 匹配 S$2,500-4,000 这类范围：

```python
SALARY_RE = re.compile(r"S\$\s*[\d,]+\s*(?:–|-)\s*[\d,]+")
```

_find_conflicts 遍历 state.context，对每个 Source 尝试找到实体名，并收集所有薪资范围。如果同一实体有超过一个不同范围，就返回冲突列表。

```python
return [(entity, sorted(ranges)) for entity, ranges in by_entity.items() if len(ranges) > 1]
```

模板综合阶段会把这个列表渲染成 "## Conflicting Evidence" 段落。

## 5.6 Planner：plan 与 _plan_with_llm

### 5.6.1 启发式 plan

```python
entities = self._extract_entities(question)
subtasks = [
    SubTask(
        id=f"ev-{index}",
        question=f"What are the AI internship roles, required skills and hiring signals at {entity}?",
        tools=["retrieve", "web_search"],
    )
    for index, entity in enumerate(entities, start=1)
]
subtasks.append(
    SubTask(
        id="compare",
        question="Compare the AI internship opportunities across companies, including skills and hiring signals.",
        tools=["retrieve", "python_sandbox"],
    )
)
if _is_calculation_question(question):
    subtasks.append(SubTask(id="calc", question=question, tools=["python_sandbox"]))
```

启发式 Planner 会为每个识别出的实体生成一个证据子任务，再追加一个 compare 任务。compare 任务用于强制报告结构包含对比；calc 任务只在计算题中出现。

注意：compare 子任务的 tools 列表中包含 python_sandbox，但 Executor 只在 subtask.id == "calc" 时走计算分支。因此 compare 仍然执行 retrieve，不会被误判为计算任务。

### 5.6.2 LLM Planner

真实 LLM 模式调用 _plan_with_llm：

```python
system = (
    "You are a research planner. Return JSON with keys objective, rationale, "
    "subtasks (array of id, question, tools). Tools: retrieve, web_search, arxiv_search, "
    "pdf_parse, python_sandbox, sqlite_query."
)
raw = self.llm.complete_json([
    {"role": "system", "content": system},
    {"role": "user", "content": question},
])
```

这里刻意不传 tools 参数。早期版本把 ToolRegistry.schemas() 传给规划请求，DeepSeek 会把它解释为 function calling，返回一个工具调用而不是 JSON 计划，导致 raw.get("subtasks", []) 为空。修复方式是把工具名称写在 system prompt 中，规划阶段只要求 JSON，不触发 function calling。

解析后生成 SubTask：

```python
subtasks = [
    SubTask(
        id=str(item.get("id", f"st-{index}")),
        question=str(item.get("question", "")),
        tools=[str(t) for t in item.get("tools", ["retrieve"])],
    )
    for index, item in enumerate(raw.get("subtasks", []))
]
if not subtasks:
    raise ValueError("LLM returned an empty plan")
```

如果 LLM 失败或返回空计划，plan 会捕获异常并回退到启发式 Planner。

## 5.7 Executor：execute_subtask

### 5.7.1 计算分支

```python
if subtask.id == "calc":
    code = _calculation_code(subtask.question)
    if code is None:
        subtask.status = "failed"
        subtask.error = "no calculation pattern found"
        return "", [], ToolResult(tool="python_sandbox", ok=False, output="", error="no calculation pattern found")
    tool_result = self.registry.call("python_sandbox", code=code)
    if not tool_result.ok:
        subtask.status = "failed"
        subtask.error = tool_result.error
        return "", [], tool_result
    subtask.status = "done"
    subtask.result = f"Calculation result: {tool_result.output.strip()}"
    return subtask.result, [], tool_result
```

计算子任务不产生 Source。Critic 会识别 python_sandbox 子任务，允许它没有来源，但要求 result 非空。

### 5.7.2 检索分支

```python
entities = [entity for entity in KNOWN_ENTITIES if entity.lower() in subtask.question.lower()]
retrieve_k = max(self.settings.retrieval_top_k, 20) if entities else self.settings.retrieval_top_k
result = self.registry.call("retrieve", query=subtask.question, k=retrieve_k)
```

如果子任务包含实体，召回深度至少 20，避免实体文档排名靠后时被 top_k 截断。

如果工具失败或没有结果：

```python
if not result.ok or not result.data or not result.data.get("results"):
    subtask.status = "failed"
    subtask.error = result.error or "no evidence returned"
    return "", [], result
```

实体过滤：

```python
if entities:
    entity = entities[0]
    filtered = [
        item for item in results
        if entity.lower() in str(item.get("title", "")).lower()
        or entity.lower() in str(item.get("snippet", "")).lower()
    ]
    if not filtered:
        subtask.status = "failed"
        subtask.error = f"no evidence found for {entity}"
        return "", [], result
    results = filtered[: min(12, len(filtered))]
```

这里的行为很关键：已知实体但语料中没有证据时，子任务直接失败，不退回全量结果。这使 OpenAI 无答案问题可以正确产生 "no evidence found for OpenAI"，而不是引用无关来源。

合并与截断：

```python
results = _merge_by_doc(results)
results = results[: min(3, len(results))]
```

最后构造 Source：

```python
source = Source(
    id=str(item["id"]),
    title=str(item.get("title", item.get("id", ""))),
    url=str(item.get("url", "")),
    snippet=str(item.get("snippet", "")),
    metadata={"doc_id": item.get("id", ""), "heading": item.get("heading", ""), "score": item.get("score", 0.0)},
)
```

子任务 evidence_lines 先用 candidate-1、candidate-2 占位，稍后在 execute_node 中用 _rebuild_subtask_citations 替换成全局引用编号。

## 5.8 Critic：critique

critique 返回 (passed, feedback)。检查项包括：

1. plan 是否存在且非空。
2. 每个子任务是否 failed 或未执行。
3. 计算子任务是否有 result。
4. 是否有需要证据的子任务但没有 context。
5. draft 是否存在。
6. validate_citations 是否通过。

关键片段：

```python
for subtask in state.plan.subtasks:
    if subtask.status == "failed":
        if "no evidence" in subtask.error and "no direct evidence" in state.draft:
            continue
        issues.append(f"subtask {subtask.id} failed: {subtask.error}")
    elif subtask.status != "done":
        issues.append(f"subtask {subtask.id} not executed")
    elif "python_sandbox" in subtask.tools and not subtask.result:
        issues.append(f"calculation subtask {subtask.id} has no result")
```

无答案例外：如果子任务失败原因是 no evidence，且 draft 明确写了 no direct evidence，则 Critic 不把它算作失败。这样无答案拒答可以通过批判。

证据子任务判定：

```python
evidence_subtasks = [s for s in state.plan.subtasks if "python_sandbox" not in s.tools]
if evidence_subtasks and not state.context:
    issues.append("no evidence collected")
```

引用校验：

```python
ok, citation_issues = validate_citations(state.draft, state.context)
if not ok:
    issues.extend(citation_issues[:5])
```

## 5.9 Synthesizer：synthesize

如果 self.llm 存在，先调用 _synthesize_with_llm。失败时记录 warning 并回到模板路径。

模板报告结构：

- # Research Report: 问题
- ## Executive Summary
- ## Evidence
- ## Comparison Notes
- ## Conflicting Evidence（只有检测到冲突时）
- ## Sources

摘要部分会对每个子任务生成一行。计算子任务没有 source，但有 result：

```python
if "python_sandbox" in subtask.tools and subtask.result:
    lines.append(f"- {subtask.question}: {subtask.result}")
else:
    lines.append(f"- {subtask.question}: no direct evidence was found in the current corpus.")
```

有来源的子任务使用 _citation_number 生成全局引用：

```python
citations = "".join(f"[{self._citation_number(state, source.id)}]" for source in subtask.sources)
lines.append(f"- {subtask.question} Evidence: {subtask.result.splitlines()[0][:180]} {citations}")
```

来源列表：

```python
for index, source in enumerate(state.context, start=1):
    if source.url:
        lines.append(f"{index}. [{title}]({source.url})")
    else:
        lines.append(f"{index}. {title}")
```

## 5.10 LLM 综合 _synthesize_with_llm

真实 LLM 综合时，先把 context 拼成带编号的证据块：

```python
context = "\n\n".join(
    f"[{index}] {source.title} ({source.url})\n{source.snippet}"
    for index, source in enumerate(state.context, start=1)
)
```

system prompt 要求输出 Executive Summary、Evidence、Comparison Notes、Sources，并要求每条来源都用 [n] 引用、不得编造事实。prompt 为：

```text
Question: {state.question}

Retrieved evidence:
{context}
```

返回值直接作为 state.draft，随后交给 Critic。真实 LLM 可能出现漏引来源的情况，因此 Critic 会重试，重试仍失败时最终报告 passed=False，但报告内容仍然生成。

## 5.11 LLM 客户端 agents/llm.py

OpenAICompatibleLLM 是一个轻量 OpenAI 兼容客户端。

complete 构造 payload：

```python
payload = {
    "model": self.model,
    "messages": messages,
    "temperature": temperature,
    "max_tokens": max_tokens if max_tokens is not None else 2048,
}
if tools:
    payload["tools"] = tools
headers = {"Authorization": f"Bearer {self.api_key}"}
response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
```

默认 timeout 为 120 秒。解析时读取 choices[0].message.content、usage、tool_calls。tool_calls 的 arguments 会尝试 JSON 解析，失败时置为空 dict。

complete_json 用于 Planner 和 LLM Judge。它最多重试 2 次，每次使用 max_tokens=4096：

```python
for _ in range(retries):
    response = self.complete(messages, tools=tools, max_tokens=4096)
    if response.tool_calls:
        arguments = response.tool_calls[0].get("arguments", {})
        if isinstance(arguments, dict) and arguments:
            return arguments
    text = response.content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    return json.loads(text)
```

如果 JSON 截断或为空，循环会重试。这个修复是针对 DeepSeek v4-flash 规划请求偶发截断问题的。

## 5.12 节点层 agents/nodes.py

plan_node 调用 brain.plan，记录 planner span。

execute_node 遍历 pending 或 failed 子任务，调用 brain.execute_subtask，追加 ToolResult，追加 Source，并重建引用。

```python
_, sources, tool_result = brain.execute_subtask(subtask)
state.tool_log.append(tool_result)
for source in sources:
    if not any(existing.id == source.id for existing in state.context):
        state.context.append(source)
if sources:
    _rebuild_subtask_citations(subtask, state.context)
```

synthesize_node 调用 brain.synthesize，写入 draft，并加 synthesizer span。

critic_node 调用 brain.critique，写入 passed/feedback，iterations 加一，并加 critic span。

finalize_node 调用 brain.finalize。

## 5.13 图执行 agents/graph.py

### 5.13.1 MiniGraph

```python
plan_node(state, self.brain)
if self.approval_callback is not None and not self.approval_callback(state.plan):
    state.passed = False
    state.report = "# Plan pending approval\n\nResearch plan was not approved; execution skipped."
    finalize_node(state, self.brain)
    return state
for _ in range(state.max_iterations + 1):
    execute_node(state, self.brain)
    synthesize_node(state, self.brain)
    critic_node(state, self.brain)
    if state.passed:
        break
finalize_node(state, self.brain)
```

### 5.13.2 LangGraphRunner

LangGraphRunner 在 dict 与 GraphState 之间转换。_to_dict 把 dataclass 展平为字典；_from_dict 再构造 GraphState。

```python
workflow = StateGraph(dict)
workflow.add_node("planner", self._plan)
workflow.add_node("executor", self._execute)
workflow.add_node("synthesizer", self._synthesize)
workflow.add_node("critic", self._critic)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "synthesizer")
workflow.add_edge("synthesizer", "critic")
workflow.add_conditional_edges("critic", self._route, {"executor": "executor", "end": END})
```

_route 的条件：

```python
if not data.get("passed") and int(data.get("iterations") or 0) <= int(data.get("max_iterations") or 2):
    return "executor"
return "end"
```

### 5.13.3 ResearchGraph

研究图负责引擎选择与回退：

```python
self.mini = MiniGraph(brain, settings, approval_callback)
self.lang = None
if HAS_LANGGRAPH and approval_callback is None:
    try:
        self.lang = LangGraphRunner(brain, settings)
    except Exception:
        self.lang = None
```

engine 属性：

```python
if self.approval_callback is not None:
    return "mini-approval"
return "langgraph" if self.lang is not None else "mini"
```

invoke 优先使用 LangGraph，失败时记录 warning 并回退 MiniGraph。

## 5.14 入口 agents/agent.py

ResearchAgent 构造函数负责装配：

1. 读取或创建 index。
2. 注册工具。
3. 创建 AgentBrain。
4. 创建 ResearchGraph。
5. 创建 TraceLogger 与 BudgetTracker。

索引选择逻辑：

```python
if index is not None:
    self.index = index
elif self.settings.embedding_mode == "hash":
    self.index = RetrievalIndex.from_corpus(self.settings.corpus_dir)
else:
    embedder = make_embedder(settings.embedding_mode, settings.embedding_model, settings.embedding_device)
    cache_path = settings.cache_dir / "index.json"
    self.index = RetrievalIndex.load_or_build(settings.corpus_dir, cache_path, embedder)
```

run 方法：

```python
ok, sanitized, issues = self.brain.sanitize(question)
state = GraphState(question=sanitized, max_iterations=max(1, self.settings.max_critic_iterations))
if not ok:
    state.report = "# Request blocked\n\n" + "\n".join(f"- {issue}" for issue in issues)
    state.passed = False
    self.trace_logger.log({"event": "research_blocked", "question": sanitized, "issues": issues})
    return state
state = self.graph.invoke(sanitized)
self.trace_logger.log({
    "event": "research_complete",
    "engine": self.graph.engine,
    "state": state.to_dict(),
    "budget_remaining_usd": round(self.budget.remaining_usd(), 4),
})
return state
```

## 5.15 Agent 核心的准确边界

第一，Planner 的启发式路线是确定性的，真实 LLM 路线是 JSON 计划，不是自由 ReAct 循环。Executor 目前按 Subtask.id 和 tools 字段执行，不解析任意 tool_calls。

第二，Critic 的引用策略是严格全覆盖：context 中每个来源都必须被 draft 引用。真实 LLM 可能因此失败，这是当前真实评测 citation_accuracy=0.8 的直接原因之一。

第三，max_tool_steps 目前没有全局计数器强制限制工具调用次数，实际重试上限由 max_critic_iterations 控制。

第四，BudgetTracker.add 已实现，但 OpenAICompatibleLLM 没有自动调用它，所以 trace 中的 budget_remaining_usd 目前只是初始预算，不会随 token 消费更新。
