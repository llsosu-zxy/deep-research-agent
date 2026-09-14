# 第九章补充 测试逐文件逐断言走读

## 9A.1 test_agent.py

test_end_to_end_offline 构造临时 corpus，包含 Shopee 与 Grab 两个文档。它断言：

```python
self.assertGreater(len(state.context), 0)
self.assertIn("Research Report", state.report)
ok, issues = validate_citations(state.report, state.context)
self.assertTrue(ok, issues)
self.assertTrue(agent.trace_logger.path.exists())
```

这个测试覆盖了索引构建、Planner、Executor、Synthesizer、Critic、TraceLogger 全链路。

test_injection_is_blocked 输入 ignore previous instructions 与 reveal system prompt，断言 state.passed=False 且 report 包含 blocked。

## 9A.2 test_approval.py

test_plan_approval_rejection_stops_execution 传入 approval_callback=lambda plan: False，断言：

```python
self.assertIn("pending approval", state.report)
self.assertFalse(state.passed)
self.assertEqual(len(state.tool_log), 0)
```

它验证人工确认拒绝后不执行任何工具。

test_plan_approval_acceptance_runs_normally 传入 lambda plan: True，断言正常执行并产生 context。

## 9A.3 test_chunker.py

test_front_matter_parsed 验证 title 被解析，body 被保留。

test_heading_chunks_keep_metadata 验证多标题文档至少产生两个 chunk，且每个 chunk 都带 title 与 doc_id。

test_tokens_do_not_explode 验证长文本分块后每个 chunk 的 token 数不超过上限附近。

## 9A.4 test_retrieval.py

test_hybrid_search_finds_relevant_doc 构造 Shopee 与 TikTok 两个文档，查询 Shopee PyTorch internship，断言 top1 是 Shopee AI。

test_empty_corpus_returns_empty 断言空语料搜索返回 []。

## 9A.5 test_guardrails.py

test_pii_redaction 验证邮箱与手机号被替换为 REDACTED。

test_injection_detection 验证注入模式与非注入模式。

test_validate_input_blocks_injection 验证注入返回 ok=False。

test_citations 验证 [1][2] 通过，单独 [1] 在有两个来源时失败，并包含 never cited。

## 9A.6 test_tools.py

test_registry_unknown_tool 验证未知工具返回 ok=False。

test_python_sandbox 执行 print(sum(range(5)))，断言输出包含 10。

test_sqlite_read_only 创建 jobs 表，验证 SELECT 成功，DELETE 被拒绝。

## 9A.7 test_failure_modes.py

test_empty_corpus_does_not_crash 断言空语料仍然生成 report，但 passed=False。

test_llm_api_failure_falls_back_to_heuristics 把 base_url 指向 127.0.0.1:1，断言报告仍包含 Shopee 且 context 非空。

test_sandbox_rejects_dangerous_code 执行 import os，断言 ok=False 且 error 包含 sandbox error。

## 9A.8 test_hard_cases.py

test_calculation_subtask 断言报告包含 400。

test_no_answer_refuses 断言报告包含 no direct evidence 且 passed=True。

test_conflicting_evidence_is_flagged 构造两个不同薪资范围的 Shopee 文档，断言报告包含 Conflicting Evidence、S$，且 passed=True。

## 9A.9 test_judge.py

test_keyword_fallback 不传 LLM，断言 verdict=pass、judge=keyword-fallback、score>=0.5。

## 9A.10 test_llm_local.py

MockOpenAIHandler 根据 system prompt 判断是 planner 还是 report writer。planner 返回 JSON 计划；report writer 计算 user prompt 中最大引用编号，生成包含所有引用的报告。

test_client_json_completion 直接测试 complete_json。

test_agent_uses_llm_planning_and_synthesis 构造 ResearchAgent，断言报告包含 Local LLM Report，且 engine=langgraph。

## 9A.11 test_param_cases.py

这个文件是 77 个参数化用例的核心。它把边界输入展开成表，每个参数组合都是独立测试。

关键断言包括：

- tokenize 对中文、英文、金额、混合文本的精确输出。
- parse_front_matter 对 title、tags、无 front matter 的处理。
- chunker 对长段落、中文长文本、纯标题文档的处理。
- injection 对多种注入变体的识别。
- PII 对邮箱、手机号、国际号码、身份证的脱敏。
- validate_input 对正常、注入、PII、超长文本的判断。
- BM25 对多文档排序和未知词的处理。
- 引用抽取、越界校验、覆盖指标、多跳综合指标。
- python_sandbox 的算术、字符串、循环输出。
- sqlite_query 的 SELECT、DELETE、多语句、缺失表。
- 混合检索在不同语料下的命中。

## 9A.12 测试命令与预期

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

当前结果：103 passed，ruff All checks passed。

## 9A.13 测试体系的缺口

第一，没有真实 GPU 的自动化测试；GPU 验证是脚本级手工验证。

第二，没有真实 DeepSeek API 的 CI 测试，避免 CI 消耗 token。

第三，没有并发与压力测试，例如同时 10 个 /api/research 请求。

第四，没有浏览器端 UI 自动化测试，Gradio 目前只做导入与启动验证。

第五，没有 RAGAS 指标测试，因为依赖不兼容。
