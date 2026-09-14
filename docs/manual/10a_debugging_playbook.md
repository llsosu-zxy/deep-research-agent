# 第十章补充 调试与故障排查手册

## 10A.1 问题：ModuleNotFoundError: No module named agents

症状：直接执行 python ui\gradio_app.py 时报错。

原因：Python 的 sys.path[0] 是脚本所在目录 ui，项目根目录不在 sys.path。

修复：

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
```

该修复已在 ui/gradio_app.py 中提交。

## 10A.2 问题：GitHub 作者显示 Codex

原因：提交时使用了 `-c user.name="Codex" -c user.email="codex@local"`。

修复分两步：

1. 设置本地身份。
2. 用 git-filter-repo 与 .mailmap 重写历史，再 force push。

当前本地与远端作者均为 zhaoxinyu，邮箱为 llsosu-zxy@users.noreply.github.com。

## 10A.3 问题：DeepSeek Planner 返回空计划

症状：日志出现 LLM returned an empty plan，随后回退启发式。

原因：规划请求带了 tools 参数，DeepSeek 返回 tool_calls，而不是 JSON 计划。

修复：_plan_with_llm 不再把 registry.schemas() 传给 complete_json，工具名称只写在 system prompt 中。

## 10A.4 问题：DeepSeek JSON 截断

症状：json.decoder.JSONDecodeError: Unterminated string。

原因：max_tokens 过小或模型输出被截断。

修复：

- complete_json 使用 max_tokens=4096。
- 去掉代码围栏。
- 截取第一个 { 到最后一个 }。
- 失败后重试，默认 2 次。

## 10A.5 问题：DeepSeek 综合超时

症状：httpx.ReadTimeout。

原因：默认 timeout=30 秒不足，完整报告生成可能超过 30 秒。

修复：OpenAICompatibleLLM 默认 timeout 改为 120 秒。

## 10A.6 问题：真实 LLM Critic 不通过

症状：passed=False，feedback 包含 source [n] never cited。

原因：LLM 报告没有引用全部 context 来源。validate_citations 要求全覆盖。

当前处理：Critic 触发重试；重试仍失败时保留报告，但 passed=False。

改进方向：在 Synthesizer prompt 中列出所有来源编号；或对真实 LLM 模式采用分层引用校验。

## 10A.7 问题：矛盾证据没有触发 Conflicting Evidence

症状：两个来源有不同薪资范围，但报告没有冲突段。

原因：同一个文档的 Compensation chunk 没有进入合并后的 snippet。

修复：

- _merge_by_doc 优先保留含 S$、薪资或 compensation 的 snippet。
- 合并前保留最多 12 个候选 chunk，再按文档合并。
- 合并后最多保留 3 个文档。

修复后，Shopee 的 S$2,500-4,000 与 S$8,000-10,000 可以被 _find_conflicts 同时看到。

## 10A.8 问题：Tencent 等导入文档检索不到

症状：问 Tencent 时返回 Shopee/TikTok 结果。

原因一：分词器只支持 ASCII，中文正文没有 token。

修复：TOKEN_RE 增加 [\u4e00-\u9fff]+。

原因二：公司名只在 Markdown 标题中，不在正文 chunk 中。

修复：每个 chunk 前置 document_title 与 heading。

原因三：实体文档排名在 top_k 之外。

修复：实体子任务的 retrieve_k 提高到 max(top_k, 20)。

## 10A.9 问题：Gradio 无法启动

检查顺序：

1. 是否在项目根目录执行。
2. .venv 是否存在。
3. gradio 是否安装。
4. 7860 端口是否被占用。

启动命令：

```powershell
.\.venv\Scripts\python.exe ui\gradio_app.py --host 127.0.0.1 --port 7860
```

## 10A.10 问题：FastAPI 启动后 /api/health 正常，但 /api/research 很慢

原因：

- 真实 LLM 模式需要远程 API。
- 每个实体子任务都会检索。
- Critic 可能触发多轮综合。

排查：

1. 看 /api/health 返回的 engine 与 corpus_chunks。
2. 看 data/storage/traces.jsonl 中 research_complete 的 trace。
3. 看 state.iterations 与 tool_log。
4. 如果是 LLM 超时，检查网络与 120 秒 timeout。

## 10A.11 问题：GPU 不可用

检查：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
```

如果 torch 是 +cpu，需要安装 CUDA 版本。PyPI 默认 Windows 轮子可能是 CPU 版，必须使用对应 CUDA 索引或已有 CUDA 环境。

## 10A.12 问题：显存不足

当前 RTX 5060 Laptop 只有 8 GB。BGE-M3 与 reranker 同时加载约占 3.2 GB，Qwen 1.5B LoRA 约占 3.17 GB。若同时加载三者，可能 OOM。

处理：

- 分步骤运行，不把 embedding、reranker、训练模型同时常驻。
- 降低 batch size。
- 使用 gradient checkpointing。
- 使用 fp16。
- 训练结束后释放模型。

## 10A.13 问题：RAGAS 导入失败

症状：ModuleNotFoundError: langchain_community.chat_models.vertexai。

处理：

- 调整 langchain-community 版本。
- 或安装对应 provider 包。
- 或暂时跳过 RAGAS，使用现有 103 题指标与 DeepSeek 真实 10 题报告。

## 10A.14 通用排障顺序

1. 先跑离线 demo，确认核心链路。
2. 再跑 pytest，确认代码回归。
3. 再跑 run_eval，确认指标。
4. 再切换到真实 LLM，单独验证一次 run_demo。
5. 最后再跑真实评测，避免一次消耗大量 token。
6. 任何配置变更后，清理或重建 data/storage/index.json。
