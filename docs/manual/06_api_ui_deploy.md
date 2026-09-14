# 第六章 API、界面与部署

## 6.1 FastAPI 应用 app/api.py

app/api.py 定义 ResearchRequest 和 create_app。ResearchRequest 是 Pydantic BaseModel：

```python
class ResearchRequest(BaseModel):
    question: str
```

get_agent 使用 lru_cache(maxsize=1)，保证一个进程只构造一次 ResearchAgent，避免每次请求都重建索引。

```python
@lru_cache(maxsize=1)
def get_agent() -> ResearchAgent:
    settings = Settings.from_env()
    agent = ResearchAgent(settings=settings)
    return agent
```

### 6.1.1 GET /api/health

```python
@app.get("/api/health")
def health() -> dict:
    agent = get_agent()
    return {
        "status": "ok",
        "engine": agent.graph.engine,
        "corpus_chunks": len(agent.index.chunks),
        "tools": agent.registry.names(),
    }
```

这个接口适合 Docker 健康检查，也可以观察当前是 langgraph 还是 mini 引擎。

### 6.1.2 POST /api/research

```python
@app.post("/api/research")
def research(payload: ResearchRequest) -> dict:
    agent = get_agent()
    state = agent.run(payload.question)
    return {"state": state.to_dict()}
```

请求体示例：

```json
{"question": "Compare Shopee and Grab AI internship compensation."}
```

返回的 state 包含 question、plan、context、tool_log、draft、feedback、passed、report、iterations、trace、summary。

### 6.1.3 GET /api/traces

```python
@app.get("/api/traces")
def traces(limit: int = 20) -> dict:
    agent = get_agent()
    return {"traces": agent.trace_logger.recent(limit)}
```

它读取 data/storage/traces.jsonl 的最后 limit 条记录。

### 6.1.4 WebSocket /ws/research

```python
@app.websocket("/ws/research")
async def ws_research(websocket: WebSocket) -> None:
    await websocket.accept()
    agent = get_agent()
    try:
        while True:
            payload = await websocket.receive_json()
            question = str(payload.get("question", "")).strip()
            if not question:
                await websocket.send_json({"event": "error", "message": "empty question"})
                continue
            await websocket.send_json({"event": "started", "question": question})
            state = await asyncio.to_thread(agent.run, question)
            await websocket.send_json({"event": "complete", "state": state.to_dict()})
    except Exception as exc:
        await websocket.close(code=1011, reason=str(exc))
```

asyncio.to_thread 把同步 agent.run 放到线程池，避免阻塞事件循环。当前 WebSocket 发送 started 和 complete 两个事件；如果要实时展示每个节点，需要在图执行过程中增加回调或事件队列。

## 6.2 Gradio 界面 ui/gradio_app.py

ui/gradio_app.py 直接运行时需要把项目根目录加入 sys.path：

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.agent import ResearchAgent
from core.config import Settings
```

这个路径修复解决了一个真实报错：直接执行 python ui\gradio_app.py 时，Python 的 sys.path[0] 是 ui 目录，不会自动找到 agents 包。

build_ui 创建 Gradio Blocks：

```python
settings = Settings.from_env()
agent = ResearchAgent(settings=settings)

def answer(question: str):
    state = agent.run(question)
    meta = (
        f"Engine: {agent.graph.engine} | Critique passed: {state.passed} | "
        f"Iterations: {state.iterations} | Sources: {len(state.context)}"
    )
    return state.report, meta

with gr.Blocks(title="Deep Research Agent") as demo:
    gr.Markdown("# Deep Research Agent")
    question = gr.Textbox(label="Research question", value="Compare Shopee, TikTok and Grab ...")
    run_button = gr.Button("Run", variant="primary")
    status = gr.Markdown()
    report = gr.Markdown()
    run_button.click(answer, inputs=question, outputs=[report, status])
```

start-ui.bat 和 start-api.bat 提供一键启动：

```bat
@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" ui\gradio_app.py
```

```bat
@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 6.3 Docker

docker/Dockerfile 使用 python:3.12-slim：

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/seed_corpus.py
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

docker/docker-compose.yml 提供 api 与 ui 两个服务，默认使用 mock LLM 与 hash embedding，避免镜像启动时下载模型。

```yaml
services:
  api:
    build: { context: .., dockerfile: docker/Dockerfile }
    ports: ["8000:8000"]
    environment:
      LLM_PROVIDER: mock
      EMBEDDING_MODE: hash
    volumes: ["../data:/app/data"]

  ui:
    build: { context: .., dockerfile: docker/Dockerfile }
    command: ["python", "ui/gradio_app.py", "--host", "0.0.0.0", "--port", "7860"]
    ports: ["7860:7860"]
    environment:
      LLM_PROVIDER: mock
      EMBEDDING_MODE: hash
```

## 6.4 GitHub Actions CI

.github/workflows/ci.yml 在 push 和 pull_request 时执行：

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: python -m pip install --upgrade pip
      - run: pip install -e ".[dev]"
      - run: python scripts/seed_corpus.py
      - run: pytest
      - run: python scripts/run_eval.py --limit 5
```

当前仓库在 GitHub 上已经跑过多次 CI，最新一次为 success。

## 6.5 Hugging Face Space

hf_space_app.py 是 Space 入口：

```python
from ui.gradio_app import build_ui

if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
```

scripts/package_hf_space.py 生成 dist/hf_space.zip，包含 agents、app、core、data/corpus、docs、eval、scripts、ui、hf_space_app.py、requirements.txt、README.md 和 .env.example。默认 mock 模式可以在免费 CPU 层运行；GPU 只用于 BGE-M3、reranker 或 LoRA。

当前项目按用户决定不实际上线 Space，但打包与文档已完成。

## 6.6 本地运行命令

```powershell
cd C:\Users\29716\Documents\demo1\deep-research-agent

# 离线 demo
.\.venv\Scripts\python.exe scripts\run_demo.py "Compare Shopee and Grab AI internships"

# 103 题评测
.\.venv\Scripts\python.exe scripts\run_eval.py

# Agent vs 单轮 RAG
.\.venv\Scripts\python.exe scripts\run_comparison.py

# 12 个端到端场景
.\.venv\Scripts\python.exe scripts\run_scenarios.py

# API
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# UI
.\.venv\Scripts\python.exe ui\gradio_app.py
```
