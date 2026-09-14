# 第二章 配置系统与数据模型

## 2.1 配置入口 core/config.py

项目所有运行参数集中在 core/config.py 的 Settings 类。Settings 是 dataclass，字段默认值保证不配置任何环境变量也能启动。

```python
@dataclass
class Settings:
    llm_provider: str = "mock"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    embedding_mode: str = "hash"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    reranker: str = "identity"
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cpu"
    max_tool_steps: int = 12
    max_critic_iterations: int = 2
    daily_budget_usd: float = 0.50
    require_plan_approval: bool = False
    retrieval_top_k: int = 5
    cache_dir: Path = Path("data/storage")
    corpus_dir: Path = Path("data/corpus")
    trace_path: Path = Path("data/storage/traces.jsonl")
    seed_dir: Path = Path("data/corpus/seed")
```

Settings.from_env 负责把环境变量转换成字段。它先调用 load_dotenv，因此项目根目录的 .env 会自动加载。没有 python-dotenv 时，ImportError 被捕获，程序仍然可以靠操作系统环境变量运行。

@CAPTION: 表 2-1 Settings 字段与用途
:::table
字段|环境变量|默认值|作用
llm_provider|LLM_PROVIDER|mock|选择 mock 或 openai_compatible
llm_base_url|LLM_BASE_URL|空|OpenAI 兼容 API 地址，例如 https://api.deepseek.com/v1
llm_api_key|LLM_API_KEY|空|API Key，只应放在本地 .env 或环境变量中
llm_model|LLM_MODEL|deepseek-chat|模型名称；当前验证使用 deepseek-v4-flash
embedding_mode|EMBEDDING_MODE|hash|hash、sentence-transformers 或 bge-m3
embedding_model|EMBEDDING_MODEL|BAAI/bge-m3|向量模型名称
embedding_device|EMBEDDING_DEVICE|cpu|cpu 或 cuda
reranker|RERANKER|identity|identity、cross-encoder、bge-reranker 或 flag
reranker_model|RERANKER_MODEL|BAAI/bge-reranker-base|重排模型名称
reranker_device|RERANKER_DEVICE|cpu|重排设备
max_tool_steps|MAX_TOOL_STEPS|12|预留的工具步数上限
max_critic_iterations|MAX_CRITIC_ITERATIONS|2|Critic 失败后最多重试次数
daily_budget_usd|DAILY_BUDGET_USD|0.50|BudgetTracker 的日预算
require_plan_approval|REQUIRE_PLAN_APPROVAL|0|是否启用人工确认配置
retrieval_top_k|RETRIEVAL_TOP_K|5|默认检索条数
cache_dir|DATA_DIR|data/storage|索引与 trace 缓存目录
corpus_dir|DATA_DIR|data/corpus|语料目录
trace_path|DATA_DIR|data/storage/traces.jsonl|JSONL trace 文件
seed_dir|DATA_DIR|data/corpus/seed|种子文档目录
:::

## 2.2 环境变量示例

项目根目录的 .env.example 是完整模板。当前真实 DeepSeek 运行时使用以下配置：

```text
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-你的key
LLM_MODEL=deepseek-v4-flash
```

.env 已在 .gitignore 中，不会进入 Git。代码和文档中不应写入真实 key。若需要切换回离线模式，把 LLM_PROVIDER 改回 mock 即可。

## 2.3 数据模型 core/models.py

core/models.py 定义项目所有跨模块传递的数据结构。它们全部是 dataclass，并提供 to_dict，用于 API 返回、trace 写入和评测归档。

### 2.3.1 Source：证据来源

```python
@dataclass
class Source:
    id: str
    title: str
    url: str = ""
    snippet: str = ""
    retrieved_at: str = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)
```

Source 是报告引用的最小单位。id 通常是 chunk id；title 用于生成来源列表；url 是原始链接；snippet 是进入上下文的正文；retrieved_at 记录抓取时间；metadata 保存 doc_id、heading、score 等检索信息。

### 2.3.2 Chunk：索引单元

```python
@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
```

Chunk 是 chunker 的输出，也是 RetrievalIndex 的索引对象。tokens 供 BM25 使用，embedding 供 dense retrieval 使用。一个 Chunk 同时携带两种表示，因此混合检索不需要维护两套 chunk 表。

### 2.3.3 SubTask：子任务

```python
@dataclass
class SubTask:
    id: str
    question: str
    tools: list[str] = field(default_factory=list)
    status: str = "pending"
    result: str = ""
    sources: list[Source] = field(default_factory=list)
    error: str = ""
    duration_ms: float = 0.0
```

status 的取值包括 pending、running、done、failed、skipped。Executor 只处理 pending 或 failed 的子任务，因此 Critic 重试时不会重复执行已经完成的子任务。

### 2.3.4 Plan：研究计划

```python
@dataclass
class Plan:
    objective: str
    subtasks: list[SubTask] = field(default_factory=list)
    rationale: str = ""
    dependencies: dict[str, list[str]] = field(default_factory=dict)
```

Plan 由 Planner 生成。dependencies 字段保留依赖关系，当前启发式 Planner 生成的依赖为空，真实 LLM Planner 可以返回依赖结构。

### 2.3.5 ToolResult：工具结果

```python
@dataclass
class ToolResult:
    tool: str
    ok: bool
    output: str
    data: Any = None
    error: str = ""
    duration_ms: float = 0.0
```

Tool.invoke 永远返回 ToolResult，而不是把异常抛给图。这样单个工具失败不会让整个 Agent 崩溃，Critic 可以根据 tool_log 和子任务状态决定下一步。

### 2.3.6 TraceSpan：节点级追踪

```python
@dataclass
class TraceSpan:
    name: str
    node: str
    duration_ms: float
    meta: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=_now)
```

每个节点执行完都会追加一个 TraceSpan。meta 中记录该节点的关键计数，例如 planner 的 subtasks 数量、executor 的 sources 与 tool_calls、critic 的 passed 与 feedback。

### 2.3.7 GraphState：图状态

```python
@dataclass
class GraphState:
    question: str
    plan: Plan | None = None
    context: list[Source] = field(default_factory=list)
    tool_log: list[ToolResult] = field(default_factory=list)
    draft: str = ""
    feedback: str = ""
    passed: bool = False
    report: str = ""
    iterations: int = 0
    max_iterations: int = 2
    trace: list[TraceSpan] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
```

GraphState 是图中的唯一可变状态。LangGraphRunner 在节点边界把它转换成 dict；MiniGraph 直接传递同一个对象。GraphState.add_span 追加 TraceSpan，to_dict 生成 API 与 trace 所需的 JSON 结构。

## 2.4 数据流中的对象变化

@CAPTION: 表 2-2 一次请求中各对象的生命周期
:::table
阶段|输入对象|输出对象|代码位置
输入安全检查|question: str|sanitized: str、issues: list[str]|AgentBrain.sanitize
规划|question|Plan(SubTask 列表)|AgentBrain.plan
执行单个子任务|SubTask|result: str、sources: list[Source]、ToolResult|AgentBrain.execute_subtask
合并证据|多个检索结果 dict|合并后的 results|_merge_by_doc
写入状态|sources、ToolResult|GraphState.context、GraphState.tool_log|agents/nodes.py execute_node
生成草稿|GraphState|state.draft: str|AgentBrain.synthesize
批判|GraphState|passed: bool、feedback: str|AgentBrain.critique
收尾|GraphState|report、summary|AgentBrain.finalize
:::

## 2.5 配置与数据模型的设计含义

第一，所有跨模块对象都有明确的 dataclass，而不是传裸 dict。只有在工具边界和 LangGraph 状态转换处才使用 dict。

第二，Source 与 Chunk 分离。Chunk 是索引内部对象，Source 是报告外部对象；Source 可以来自本地检索，也可以来自网页或 PDF，因此报告层不需要知道证据的内部来源。

第三，GraphState 同时包含流程字段与观测字段。process 字段是 plan、context、draft、passed；observability 字段是 tool_log、trace、summary。trace 不影响决策，summary 不影响图执行，便于后续替换观测方案。

第四，Plan 与 SubTask 的依赖关系被保留，但当前启发式管道只使用线性执行。后续可以在 Executor 中按照 dependencies 做真实 DAG 调度。
