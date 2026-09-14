# 第二章补充 配置系统逐行走读

## 2A.1 import 阶段发生了什么

core/config.py 在模块 import 时先执行：

```python
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
```

这意味着只要任何代码 import core.config，项目根目录的 .env 就会被读取并写入 os.environ。include 顺序很重要：如果进程启动前已经设置了同名环境变量，python-dotenv 默认不会覆盖它，因此操作系统环境变量的优先级高于 .env。

如果 python-dotenv 未安装，ImportError 被捕获，Settings.from_env 仍然可以读取 os.getenv。也就是说，dotenv 是便利项，不是硬依赖。

## 2A.2 _env_int 的容错

```python
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
```

这个函数处理三类情况：

- 环境变量不存在或为空字符串：返回默认值。
- 环境变量是合法整数：返回 int。
- 环境变量不是整数：吞掉 ValueError，返回默认值。

因此 MAX_TOOL_STEPS=abc 不会让程序启动失败，而是回退到 12。这个策略适合演示项目，但在生产环境中应该记录 warning，避免配置错误被静默忽略。

## 2A.3 from_env 的路径推导

```python
data_dir = Path(os.getenv("DATA_DIR", "data"))
cache = data_dir / "storage"
corpus = data_dir / "corpus"
```

如果 DATA_DIR=data，则：

- cache_dir=data/storage
- corpus_dir=data/corpus
- trace_path=data/storage/traces.jsonl
- seed_dir=data/corpus/seed

如果用户设置 DATA_DIR=D:\research-data，所有路径都会迁移到该目录。这样可以把语料与代码分离，适合 Docker volume。

## 2A.4 布尔值解析

```python
require_plan_approval=os.getenv("REQUIRE_PLAN_APPROVAL", "0")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"},
```

支持的 true 形式为 1、true、yes、on，大小写不敏感。其他值一律为 False。注意：这个字段只表示配置意图，ResearchAgent 的人工确认仍然需要显式传入 approval_callback；仅设置 REQUIRE_PLAN_APPROVAL=1 不会自动创建交互式确认流程。

## 2A.5 Settings 的默认值与运行后果

@CAPTION: 表 2A-1 默认值如何影响一次请求
:::table
字段|默认值|直接影响
llm_provider|mock|不创建 OpenAICompatibleLLM，Planner 与 Synthesizer 走离线路径
embedding_mode|hash|RetrievalIndex.from_corpus 不下载模型，HashEmbedder 生成 256 维向量
embedding_device|cpu|不会访问 CUDA
reranker|identity|search 只截取前 top_k，不加载 CrossEncoder
max_critic_iterations|2|MiniGraph 最多执行 3 轮 execute/synthesize/critic
retrieval_top_k|5|普通子任务 retrieve k=5；实体子任务至少 k=20
daily_budget_usd|0.50|BudgetTracker 的 daily_limit_usd
cache_dir|data/storage|索引缓存、trace、SQLite 默认位置
:::

## 2A.6 真实 DeepSeek 配置的完整路径

当 .env 配置：

```text
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=deepseek-v4-flash
```

运行时发生：

1. core.config import 时 load_dotenv 读取 .env。
2. Settings.from_env 把 LLM_PROVIDER 变为 "openai_compatible"。
3. AgentBrain.__init__ 检测 provider 与 api_key，创建 OpenAICompatibleLLM。
4. Planner 调用 _plan_with_llm，LLM 客户端 POST /chat/completions。
5. Synthesizer 调用 _synthesize_with_llm，再次 POST。
6. Critic 根据引用校验决定是否重试；如果重试，Synthesizer 再调用一次。
7. TraceLogger 写入 research_complete，state 中包含完整 trace。

## 2A.7 配置错误会导致什么

@CAPTION: 表 2A-2 配置错误与症状
:::table
错误配置|症状|处理
LLM_PROVIDER=openai_compatible 但没有 API Key|AgentBrain.llm 仍为 None，走离线路径|检查 llm_api_key 是否为空
LLM_BASE_URL 缺少 /v1|OpenAICompatibleLLM POST 404|使用 https://api.deepseek.com/v1
LLM_MODEL 名称错误|API 返回 400/404|先用 /v1/models 查询可用模型
EMBEDDING_MODE=bge-m3 但未装模型依赖|make_embedder 抛错或回退 SentenceTransformer|先运行 build_embedding_index.py 验证
RERANKER=cross-encoder 但未装 sentence-transformers|CrossEncoderReranker 抛 RuntimeError|安装 GPU extras 或改回 identity
DATA_DIR 指向不存在盘符|Path 操作失败|创建目录或修正路径
:::

## 2A.8 配置系统的边界

第一，没有 Pydantic Settings 校验，非法 embedding_mode 会在 make_embedder 时才报错，而不是启动时。

第二，没有配置热更新。Settings 在进程启动时读取，修改 .env 需要重启服务。

第三，没有区分 secret 与普通配置。LLM_API_KEY 与普通字段放在同一个 dataclass 中，依赖 .gitignore 保护它。生产环境应接入 Secret Manager 或环境变量注入。

第四，没有配置审计。trace 中没有记录本次运行使用的模型、embedding 模式和 reranker，只记录 engine。若要做严格复现实验，应把 Settings 的非敏感字段写入 trace。
