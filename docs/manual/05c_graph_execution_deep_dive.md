# 第五章补充 图执行逐节点走读

## 5C.1 为什么同时保留 MiniGraph 与 LangGraphRunner

LangGraph 是项目方案指定的编排框架，但演示环境可能出现未安装、版本不兼容、状态对象序列化失败等问题。MiniGraph 使用完全相同的节点函数，只是把条件边换成 Python 循环。这样，项目在真实 LangGraph 和纯 Python 环境下的行为一致。

## 5C.2 plan_node 的输入输出

输入：GraphState，其中至少包含 question。

输出：同一个 GraphState，但 plan 字段被填充，trace 追加 planner span。

```python
def plan_node(state, brain):
    started = time.perf_counter()
    state.plan = brain.plan(state.question)
    state.add_span("planner", "planner", (time.perf_counter() - started) * 1000,
                   {"subtasks": len(state.plan.subtasks)})
    return state
```

plan_node 不处理异常，因为 brain.plan 已经内置了 LLM 失败回退。只有启发式 Planner 自身出错时才会抛异常，这种情况属于代码错误。

## 5C.3 execute_node 的循环

```python
if not state.plan:
    state.plan = brain.plan(state.question)
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

关键点：

- 只处理 pending 与 failed，因此 Critic 重试不会重复执行已 done 的子任务。
- 每个子任务都追加一个 ToolResult，即使失败也记录。
- Source 按 id 去重，避免多个子任务重复引用同一 chunk。
- _rebuild_subtask_citations 把 candidate-n 占位符换成全局 [n]。

## 5C.4 synthesize_node

```python
def synthesize_node(state, brain):
    started = time.perf_counter()
    state.draft = brain.synthesize(state)
    state.add_span("synthesizer", "synthesizer", (time.perf_counter() - started) * 1000,
                   {"draft_chars": len(state.draft)})
    return state
```

draft 是中间报告。它可能被 Critic 拒绝，因此不等于最终 report。finalize 才把 draft 或提前写入的 report 固化到 state.report。

## 5C.5 critic_node

```python
def critic_node(state, brain):
    started = time.perf_counter()
    state.passed, state.feedback = brain.critique(state)
    state.iterations += 1
    state.add_span("critic", "critic", (time.perf_counter() - started) * 1000,
                   {"passed": state.passed, "feedback": state.feedback[:200]})
    return state
```

iterations 在 Critic 执行后递增，因此第一次 Critic 后 iterations=1。MiniGraph 的循环范围是 max_iterations + 1，默认 2 时最多执行三轮。

## 5C.6 finalize_node

```python
def finalize_node(state, brain):
    return brain.finalize(state)
```

brain.finalize 的逻辑：

```python
if not state.report:
    state.report = state.draft
state.summary = {
    "question": state.question,
    "iterations": state.iterations,
    "passed_critique": state.passed,
    "subtasks": len(state.plan.subtasks) if state.plan else 0,
    "sources": len(state.context),
    "tool_calls": len(state.tool_log),
    "report_chars": len(state.report),
}
```

report 优先保留已经写入的内容，例如 plan approval 被拒绝时写入的 "# Plan pending approval"。

## 5C.7 LangGraphRunner 的 dict 适配

LangGraph 的 StateGraph(dict) 接收 dict 状态。GraphState 是 dataclass，因此需要双向转换。

_to_dict 把 dataclass 字段展开；_from_dict 再把 dict 还原为 GraphState。

```python
def _from_dict(self, data):
    return GraphState(
        question=data["question"],
        plan=data.get("plan"),
        context=data.get("context") or [],
        tool_log=data.get("tool_log") or [],
        draft=data.get("draft") or "",
        feedback=data.get("feedback") or "",
        passed=bool(data.get("passed")),
        report=data.get("report") or "",
        iterations=int(data.get("iterations") or 0),
        max_iterations=int(data.get("max_iterations") or 2),
        trace=data.get("trace") or [],
        summary=data.get("summary") or {},
    )
```

转换时必须用 get 与默认值，因为 LangGraph 首次调用时某些字段尚未出现。

## 5C.8 LangGraph 的边与条件路由

```python
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "synthesizer")
workflow.add_edge("synthesizer", "critic")
workflow.add_conditional_edges("critic", self._route, {"executor": "executor", "end": END})
```

流程是严格顺序：Planner 必须先生成计划，Executor 才能执行；Synthesizer 必须先生成 draft，Critic 才能校验。

路由函数：

```python
def _route(self, data):
    if not data.get("passed") and int(data.get("iterations") or 0) <= int(data.get("max_iterations") or 2):
        return "executor"
    return "end"
```

条件解释：

- passed=True：结束。
- passed=False 且 iterations <= max_iterations：回到 executor。
- passed=False 且 iterations > max_iterations：结束。

## 5C.9 approval_callback 的作用

如果传入 approval_callback，ResearchGraph 不会构造 LangGraphRunner，engine 返回 "mini-approval"。MiniGraph 在 plan_node 后立即调用回调：

```python
if self.approval_callback is not None and not self.approval_callback(state.plan):
    state.passed = False
    state.report = "# Plan pending approval\n\nResearch plan was not approved; execution skipped."
    finalize_node(state, self.brain)
    return state
```

拒绝时不会执行任何工具，tool_log 长度为 0。tests/test_approval.py 对此有专门断言。

## 5C.10 图的完整时序图

```text
ResearchAgent.run
  |
  +-- Brain.sanitize
  |
  +-- Graph.invoke
        |
        +-- plan_node ------------> state.plan
        |
        +-- execute_node ---------> state.context, state.tool_log
        |
        +-- synthesize_node ------> state.draft
        |
        +-- critic_node ----------> state.passed, state.feedback, iterations++
        |
        +-- route:
              passed? end
              not passed and iterations <= max? execute_node
              else end
        |
        +-- finalize_node --------> state.report, state.summary
  |
  +-- TraceLogger.log
```

## 5C.11 图执行的边界

第一，LangGraph 版本变化可能影响 StateGraph(dict) 的行为。ResearchGraph 有 try/except 回退，但没有版本白名单。

第二，GraphState 中的 dataclass 对象在 dict 中按引用传递，没有深拷贝。节点修改 plan 或 context 时会影响同一对象。

第三，没有 checkpoint。LangGraph 的持久化能力没有被使用，进程中断后不能从中间状态恢复。

第四，没有并行执行子任务。所有 ev-* 与 compare 都是顺序执行。对于 I/O 型检索，可以并行化提升速度。
