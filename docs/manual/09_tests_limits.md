# 第九章 测试体系、边界与故障处理

## 9.1 测试规模与结构

项目当前有 103 个 pytest 用例。测试分布在 tests/ 下，既有 unittest 风格类，也有 pytest 参数化用例。

@CAPTION: 表 9-1 测试文件与覆盖范围
:::table
测试文件|用例数|覆盖范围
test_agent.py|2|端到端离线流程、注入拦截
test_approval.py|2|规划拒绝、规划通过
test_chunker.py|3|front matter、标题分块、token 控制
test_eval.py|1|引用与覆盖指标
test_failure_modes.py|3|空语料、LLM API 失败回退、危险沙箱代码
test_guardrails.py|4|PII、注入、输入校验、引用校验
test_hard_cases.py|3|计算、无答案、矛盾证据
test_judge.py|1|LLM Judge 关键词 fallback
test_llm_local.py|2|本地 mock OpenAI 服务、LLM 规划与综合路径
test_param_cases.py|77|分词、切块、PII、注入、BM25、引用、指标、沙箱、SQLite、混合检索
test_retrieval.py|2|混合检索命中、空语料
test_tools.py|3|工具注册、沙箱、SQLite 只读
:::

## 9.2 端到端测试

test_agent.test_end_to_end_offline 创建临时语料，运行 ResearchAgent，并检查：

- state.context 非空。
- 报告包含 Research Report。
- validate_citations 通过。
- trace 文件已创建。

test_agent.test_injection_is_blocked 验证 prompt injection 不会进入图，而是被 ResearchAgent.run 拦截。

## 9.3 失败模式测试

### 9.3.1 空语料

```python
agent = ResearchAgent(settings=self._settings(tmp))
state = agent.run("What AI internships are available in Singapore?")
self.assertTrue(state.report)
self.assertFalse(state.passed)
```

空语料不会让程序崩溃，Critic 会判定不通过，但报告仍然生成。

### 9.3.2 LLM API 失败回退

测试把 llm_base_url 指向 http://127.0.0.1:1/v1，api_key 设置为 invalid。Planner 捕获异常后回退到启发式，最终报告仍然包含 Shopee 证据。

### 9.3.3 危险沙箱代码

```python
result = tool.invoke(code="import os\nprint(os.getcwd())")
self.assertFalse(result.ok)
self.assertIn("sandbox error", result.error)
```

SAFE_BUILTINS 没有 __import__，因此 import os 会失败。

## 9.4 困难场景测试

test_hard_cases.py 覆盖三个最容易出错的场景。

计算题：问题包含 8 小时/天、5 天/周、10 周，检查报告包含 400。

无答案：问 OpenAI 新加坡实习精确月薪，检查报告包含 no direct evidence，并且 passed=True。这说明 Critic 接受了“正确拒答”。

矛盾证据：两个临时文档分别给出 S$2,500-4,000 与 S$8,000-10,000，检查报告包含 Conflicting Evidence 与 S$。

## 9.5 参数化测试

test_param_cases.py 用 pytest.mark.parametrize 展开 77 个用例，重点包括：

- tokenize 对中文、英文、金额、混合文本的分词。
- parse_front_matter 对有无 front matter 的处理。
- chunker 对长段落、中文长文本、纯标题文档的处理。
- PII 对邮箱、手机号、国际号码、身份证的脱敏。
- injection 对多种提示注入变体的识别。
- validate_input 对正常问题、注入、PII、超长文本的判断。
- BM25 对多文档排序和未知词的处理。
- 引用抽取与越界校验。
- 覆盖指标与多跳综合指标。
- python_sandbox 的算术、字符串、循环输出。
- sqlite_query 的 SELECT、DELETE、多语句、缺失表。
- 混合检索在不同语料下的命中。

## 9.6 本地 OpenAI 兼容服务测试

test_llm_local.py 起一个 ThreadingHTTPServer，模拟 /v1/chat/completions：

```python
class MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.loads(self.rfile.read(length))
        system = next(m["content"] for m in payload["messages"] if m["role"] == "system")
        if "planner" in system:
            content = json.dumps({"objective": "local objective", "subtasks": [...]})
        else:
            content = "# Local LLM Report\nShopee evidence [1].\n\n## Sources\n1. Shopee source 1"
```

这个测试验证了 OpenAICompatibleLLM.complete_json 的解析路径，以及 ResearchAgent 在真实 LLM 分支下仍然可以使用 langgraph 引擎。

## 9.7 当前实现边界

### 9.7.1 max_tool_steps 尚未强制生效

Settings.max_tool_steps 默认 12，但当前 Executor 没有全局 step counter。实际重试由 max_critic_iterations 控制。如果要实现严格工具步数限制，需要在 GraphState 中增加 tool_steps，并在 ToolRegistry 调用前后递减或检查。

### 9.7.2 BudgetTracker 尚未接入 LLM 消费

BudgetTracker.add 已实现，但 OpenAICompatibleLLM 没有调用它。因此 trace 中的 budget_remaining_usd 目前等于 daily_limit_usd，没有反映真实 token 消费。要修复这个问题，可以在 LLMResponse 返回 prompt_tokens 与 completion_tokens 后，由 AgentBrain 或 ResearchAgent 调用 budget.add。

### 9.7.3 python_sandbox 不是强隔离

python_sandbox 通过移除 __import__ 等 builtins 限制代码，但它仍然运行在同一 Python 进程。timeout_seconds 是在 exec 完成后检查耗时，不能中断死循环。生产环境应改为子进程 + 资源限制 + 容器隔离。

### 9.7.4 web_search 不解析结果

web_search 目前只返回 DuckDuckGo HTML 长度和状态码，不返回结构化搜索结果。它更像网络可用性探针。真正的网页搜索应接入搜索 API 或解析 HTML 结果并生成 Source。

### 9.7.5 引用校验过于严格

validate_citations 要求 context 中每个来源都必须被 draft 引用。真实 LLM 可能只引用部分来源，因此真实评测 citation_accuracy 只有 0.8。后续可以区分“核心来源必须引用”和“补充来源可选引用”，或者让 Synthesizer 明确列出未使用来源。

### 9.7.6 RAGAS 版本不兼容

当前安装 ragas 0.4.3 后，导入阶段报 ModuleNotFoundError: langchain_community.chat_models.vertexai。要恢复 RAGAS，需要调整 langchain-community 版本或安装额外 provider 包，并确认 embedding provider。

### 9.7.7 HF Space 未实际上线

项目已生成 dist/hf_space.zip 与 hf_space_app.py，但按用户决定未实际上线 Space。上线需要 Hugging Face Token 与 Space 创建权限。

## 9.8 故障处理策略总结

@CAPTION: 表 9-2 常见故障与处理路径
:::table
故障|检测点|当前处理
Prompt injection|validate_input|直接返回 blocked 报告，不进入图
PII|redact_pii|脱敏为 [EMAIL_REDACTED] 等占位符，继续执行
空语料|RetrievalIndex.search 返回空|子任务 failed，报告仍生成，Critic 不通过
实体无证据|实体过滤 filtered 为空|子任务 failed，报告写 no direct evidence
LLM Planner 失败|_plan_with_llm 抛异常|回退到启发式 Planner
LLM Synthesizer 失败|_synthesize_with_llm 抛异常|回退到模板 Synthesizer
LangGraph 失败|ResearchGraph.invoke 捕获异常|回退 MiniGraph
工具异常|Tool.invoke except|返回 ToolResult(ok=False)
引用漏写|validate_citations|Critic 失败并触发重试
:::
