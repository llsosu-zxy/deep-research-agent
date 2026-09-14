# 第五章补充 问题拆解原理与代码实现

## 5A.1 为什么需要拆子问题

开放式研究问题通常不能靠一次检索解决。以 “Compare Shopee, TikTok and Grab AI internship opportunities and required skills in Singapore” 为例，这句话至少包含三个独立实体和一个比较动作：

- Shopee 的 AI 实习机会、技能要求、招聘信号。
- TikTok 的 AI 实习机会、技能要求、招聘信号。
- Grab 的 AI 实习机会、技能要求、招聘信号。
- 三家公司之间的差异、共同点和取舍。

如果只把整句问题丢给一次 top-k 检索，可能出现三类问题。第一，检索结果偏向某一个实体，另外两个实体的证据不足。第二，检索结果里只有岗位方向，没有补偿、招聘流程等字段。第三，综合阶段没有强制结构，模型可能只写一段概述，不生成对比。

拆子问题的目的，是把一个开放式问题转换成一小组边界清晰、可以分别执行、可以分别校验的 SubTask。每个 SubTask 都有自己的问题和工具列表，执行结果再汇总成 Plan。

## 5A.2 两种拆解路径

项目同时支持两种拆解方式，由 AgentBrain 的构造函数决定走哪条。

```python
self.llm = None
if settings.llm_provider == "openai_compatible" and settings.llm_api_key:
    self.llm = OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
```

如果 self.llm 为空，plan 使用启发式拆解；如果 self.llm 存在，plan 先尝试调用 LLM 拆解，失败时回退到启发式。两条路径最终都必须返回同一个 Plan 对象，因此 Executor 不需要知道计划是由谁生成的。

@CAPTION: 表 5A-1 两种拆解路径对比
:::table
维度|启发式拆解|LLM 拆解
触发条件|LLM_PROVIDER=mock 或没有 API Key|LLM_PROVIDER=openai_compatible 且有 API Key
拆解依据|KNOWN_ENTITIES + 固定模板|system prompt + 用户问题
输出|ev-1、ev-2、compare、calc|LLM 返回的 subtasks JSON
优点|零成本、确定性、可测试|可以处理开放实体与复杂依赖
缺点|实体词表有限、模板固定|可能返回空计划、引用不稳定、需要 API
回退|不需要回退|异常时回退启发式
:::

## 5A.3 启发式拆解：实体识别

启发式拆解的第一步是从问题中找出已知实体。实现位于 agents/brain.py 的 AgentBrain._extract_entities。

```python
def _extract_entities(self, question: str) -> list[str]:
    found = [entity for entity in KNOWN_ENTITIES if entity.lower() in question.lower()]
    return found or [question.strip()[:80]]
```

这段代码的逻辑是：

1. 遍历 KNOWN_ENTITIES。
2. 对每个实体做小写包含判断。
3. 保留在问题文本中出现的实体。
4. 如果没有任何已知实体，就把问题本身截断到 80 字符，作为一个兜底实体。

KNOWN_ENTITIES 是显式词表，包含 Shopee、TikTok、ByteDance、Grab、Google、Meta、NVIDIA、A*STAR、GovTech、OpenAI、Microsoft、Alibaba、Lazada、Tencent、Huawei、Sea、Garena、Amazon、Apple、SAP、PayPal、Visa、Salesforce、DBS、OCBC、UOB、Standard Chartered、JPMorgan、GIC、ST Engineering、Singtel、Razer、Micron、Cynapse、Guidesify、AIPilot、ESGPedia、X Star、Hong Ye、YY Circle、LinkWave。

这个方法的优点是简单、可预测、不需要模型。缺点是它只能识别词表里的实体。如果用户问一个没在词表中的公司，代码会把整句问题当成一个实体，后续实体过滤就找不到精确来源。

## 5A.4 启发式拆解：生成证据子任务

确认实体之后，plan 为每个实体生成一个 ev-* 子任务。

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
```

这里的 SubTask 不是自然语言中的一个“步骤”，而是一个可执行对象。它包含：

- id：ev-1、ev-2 这样的稳定编号。
- question：执行时真正发送给检索工具的问题。
- tools：这个子任务允许使用的工具。
- status：初始为 pending。

特别注意 question 的构造方式。代码没有把原始问题直接复制给每个实体，而是使用模板：

```text
What are the AI internship roles, required skills and hiring signals at {entity}?
```

这样做有两个目的。第一，把所有实体的子问题归一化成同一类检索任务，便于批量执行。第二，把实体名放在问题里，使 Executor 能够通过 KNOWN_ENTITIES 识别出子任务对应的实体，从而启用实体过滤和更深召回。

## 5A.5 启发式拆解：生成 compare 子任务

实体证据子任务之后，plan 追加一个固定 id 为 compare 的子任务。

```python
subtasks.append(
    SubTask(
        id="compare",
        question="Compare the AI internship opportunities across companies, including skills and hiring signals.",
        tools=["retrieve", "python_sandbox"],
    )
)
```

compare 子任务的作用不是再执行一次普通检索，而是强制报告层存在一个跨实体综合步骤。它带来两个结果：

第一，Synthesizer 会为 compare 生成一个独立的 Evidence 小节。

第二，multi_hop_synthesis 指标要求报告包含 comparison 相关结构，compare 子任务保证了这一点。

需要说明的是，compare 的 tools 列表中有 python_sandbox，但 Executor 只在 subtask.id == "calc" 时走计算分支：

```python
if subtask.id == "calc":
    ...
```

因此 compare 不会被误判为计算任务，它仍然调用 retrieve。

## 5A.6 启发式拆解：计算子任务

计算问题需要额外的 calc 子任务。判断逻辑由 CALC_PATTERNS 与 _is_calculation_question 完成。

```python
CALC_PATTERNS = [
    (re.compile(r"(\d+)\s*hours?\D+(\d+)\s*days?\D+(\d+)\s*weeks?", re.IGNORECASE),
     lambda h, d, w: f"print({h} * {d} * {w})"),
    (re.compile(r"(\d+)\s*days?\D+(\d+)\s*weeks?", re.IGNORECASE),
     lambda d, w: f"print({d} * {w})"),
    (re.compile(r"(\d+)\s*\+\s*(\d+)"), lambda a, b: f"print({a} + {b})"),
]
```

```python
def _calculation_code(question: str) -> str | None:
    for pattern, builder in CALC_PATTERNS:
        match = pattern.search(question)
        if match:
            numbers = [int(value) for value in match.groups()]
            return builder(*numbers)
    return None

def _is_calculation_question(question: str) -> bool:
    has_marker = bool(re.search(r"calculate|how many|total hours|sum of", question, re.IGNORECASE))
    return has_marker and _calculation_code(question) is not None
```

plan 中对应逻辑：

```python
if _is_calculation_question(question):
    subtasks.append(
        SubTask(
            id="calc",
            question=question,
            tools=["python_sandbox"],
        )
    )
```

计算子任务的 question 直接使用原始问题，因为 Executor 需要从中提取数字。例如问题包含 8 hours、5 days、10 weeks，_calculation_code 会返回 `print(8 * 5 * 10)`，python_sandbox 执行后得到 400。

## 5A.7 启发式拆解完整示例

问题：

```text
Compare Shopee, TikTok and Grab AI internship opportunities and required skills in Singapore.
```

KNOWN_ENTITIES 命中 Shopee、TikTok、Grab，_is_calculation_question 返回 False。生成的 Plan 为：

```json
{
  "objective": "Compare Shopee, TikTok and Grab ...",
  "subtasks": [
    {"id": "ev-1", "question": "What are the AI internship roles ... at Shopee?", "tools": ["retrieve", "web_search"]},
    {"id": "ev-2", "question": "What are the AI internship roles ... at TikTok?", "tools": ["retrieve", "web_search"]},
    {"id": "ev-3", "question": "What are the AI internship roles ... at Grab?", "tools": ["retrieve", "web_search"]},
    {"id": "compare", "question": "Compare the AI internship opportunities ...", "tools": ["retrieve", "python_sandbox"]}
  ]
}
```

问题：

```text
If a Shopee intern works 8 hours per day, 5 days per week for 10 weeks, how many total hours do they work?
```

KNOWN_ENTITIES 命中 Shopee，_is_calculation_question 返回 True。生成的 Plan 为 ev-1、compare、calc 三个子任务。

## 5A.8 LLM 拆解：System Prompt

LLM 拆解由 _plan_with_llm 实现。

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

这里有一个关键设计：规划阶段不把 ToolRegistry.schemas() 传给 complete_json。原因是 DeepSeek 等 OpenAI 兼容模型在收到 tools 参数时可能返回 tool_calls，而不是返回 JSON 计划。早期版本正是因为这个原因出现 raw.get("subtasks", []) 为空。修复后，工具名称只出现在 system prompt 中，模型的任务是输出计划 JSON，不是调用工具。

## 5A.9 LLM 拆解：JSON 解析与容错

complete_json 会做以下处理：

1. 调用 complete，max_tokens=4096。
2. 如果响应包含 tool_calls，尝试取第一个 tool_call 的 arguments。
3. 去掉 ```json 代码围栏。
4. 找到第一个 { 和最后一个 }，截取 JSON 主体。
5. 解析失败或空内容时重试，默认 retries=2。

这一步对真实模型很重要。DeepSeek v4-flash 的规划回复偶尔会带 Markdown 代码围栏，偶尔会被截断；重试和 JSON 主体截取可以显著提高可用性。

## 5A.10 LLM 拆解：SubTask 构造

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

LLM 返回的每个 item 至少需要 id、question、tools。如果 tools 缺失，默认使用 retrieve。如果 subtasks 为空，抛出 ValueError，外层 plan 捕获后回退到启发式拆解。

## 5A.11 拆解结果如何被执行

Plan 生成后，Executor 不关心是谁生成的计划，只按 SubTask 执行。

```python
for subtask in state.plan.subtasks:
    if subtask.status not in {"pending", "failed"}:
        continue
    _, sources, tool_result = brain.execute_subtask(subtask)
    state.tool_log.append(tool_result)
    for source in sources:
        if not any(existing.id == source.id for existing in state.context):
            state.context.append(source)
    if sources:
        _rebuild_subtask_citations(subtask, state.context)
```

执行时有两个关键分支：

```python
if subtask.id == "calc":
    tool_result = self.registry.call("python_sandbox", code=code)
```

```python
entities = [entity for entity in KNOWN_ENTITIES if entity.lower() in subtask.question.lower()]
retrieve_k = max(self.settings.retrieval_top_k, 20) if entities else self.settings.retrieval_top_k
result = self.registry.call("retrieve", query=subtask.question, k=retrieve_k)
```

这意味着拆解不是纯文本操作，而是直接影响执行参数。实体子任务会获得更深的候选池，然后按实体过滤；compare 子任务没有实体，因此只取配置的 retrieval_top_k。

## 5A.12 为什么这种拆解能提升多跳问题

多跳问题的关键不是“检索更多”，而是“分别检索，再合并”。以三家公司对比为例，单轮 RAG 用一次 top-k 检索，可能只返回 Shopee 和 TikTok，漏掉 Grab。拆解后：

1. ev-1 专门检索 Shopee。
2. ev-2 专门检索 TikTok。
3. ev-3 专门检索 Grab。
4. 每个子任务独立进行实体过滤与文档合并。
5. compare 子任务确保最终报告有跨实体综合段落。

因此，多跳综合率从单轮 RAG 的 0 提升到 1.0，平均答案覆盖从 0.6521 提升到 0.8042。

## 5A.13 当前拆解策略的局限

第一，实体识别依赖固定词表。出现词表外公司时，无法精确拆出实体子任务。

第二，工具选择是模板化的。ev-* 固定带 retrieve 与 web_search，compare 固定带 retrieve，calc 固定带 python_sandbox。真实 LLM 虽然可以返回 tools，但 Executor 仍主要按 id 判断，不会自动执行任意工具。

第三，dependencies 字段已保留，但当前没有按依赖关系做 DAG 调度。所有子任务按列表顺序执行。

第四，LLM 计划可能不稳定。空计划、截断 JSON、工具字段缺失都需要 fallback 或默认值。

第五，Critic 目前按子任务是否失败来判断，不会重新规划。它只能重试 Executor，不能改变 Plan 结构。

## 5A.14 可以怎样改进

1. 用 NER 或小模型做实体识别，替代固定词表。
2. 在 Plan 中显式建模依赖关系，例如 ev-1、ev-2、ev-3 并行，compare 依赖三者。
3. 按问题难度路由：简单问题走单 Agent，多跳问题走完整四节点。
4. 让 LLM 返回工具调用意图，并由 Executor 解析执行，而不是只按 id 分支。
5. 增加 Plan 校验，检查子任务是否覆盖所有实体、是否包含 compare、是否有非法工具名。
6. 将拆解质量纳入评测，例如统计“实体覆盖率”和“子任务冗余率”。
