# 第四章 工具层与 Guardrails

## 4.1 工具注册表 core/tools/registry.py

工具层的核心是 Tool 与 ToolRegistry。所有工具都实现为 Tool 对象，包含名称、描述、JSON Schema 参数和实际函数。

```python
class Tool:
    def __init__(self, name, description, parameters, func):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
```

schema 方法把 Tool 转换成 OpenAI function calling 格式：

```python
{
  "type": "function",
  "function": {
    "name": self.name,
    "description": self.description,
    "parameters": self.parameters
  }
}
```

invoke 是工具边界，负责三件事：计时、执行、捕获异常。

```python
def invoke(self, **kwargs):
    started = time.perf_counter()
    try:
        output = self.func(**kwargs)
        if isinstance(output, ToolResult):
            return output
        if isinstance(output, str):
            return ToolResult(tool=self.name, ok=True, output=output, duration_ms=...)
        serialized = json.dumps(output, ensure_ascii=False, default=str)[:8000]
        return ToolResult(tool=self.name, ok=True, output=serialized, data=output, duration_ms=...)
    except Exception as exc:
        return ToolResult(tool=self.name, ok=False, output="", error=str(exc), duration_ms=...)
```

这意味着工具抛出的任何异常都会被转换为 ToolResult(ok=False)，Agent 不会因为一次网页超时或 SQL 错误而崩溃。

ToolRegistry 维护 name -> Tool 的字典：

```python
def register(self, tool): self._tools[tool.name] = tool; return self
def get(self, name): return self._tools.get(name)
def names(self): return sorted(self._tools)
def schemas(self): return [tool.schema() for tool in self._tools.values()]
def call(self, name, **kwargs):
    tool = self._tools.get(name)
    if tool is None:
        return ToolResult(tool=name, ok=False, output="", error=f"Unknown tool: {name}")
    return tool.invoke(**kwargs)
```

当前注册的工具包括 retrieve、web_search、fetch_url、arxiv_search、pdf_parse、python_sandbox、sqlite_query。

## 4.2 retrieve 工具 core/tools/web.py

build_retrieve_tool 把 RetrievalIndex 包装成工具。它接收 query 和可选 k，调用 index.search，返回 JSON 友好的 results。

```python
def retrieve(query: str, k: int = top_k):
    ranked = index.search(
        query,
        top_k=int(k),
        reranker=reranker,
        reranker_model=reranker_model,
        reranker_device=reranker_device,
    )
    return {
        "query": query,
        "results": [
            {
                "id": item.chunk.id,
                "doc_id": item.chunk.metadata.get("doc_id", item.chunk.doc_id),
                "title": item.chunk.metadata.get("title", item.chunk.doc_id),
                "url": item.chunk.metadata.get("source_url", ""),
                "heading": item.chunk.metadata.get("heading", ""),
                "snippet": item.chunk.text[:600],
                "score": round(item.score, 4),
            }
            for item in ranked
        ],
    }
```

这是项目默认最常用的工具。Executor 对每个实体子任务调用 retrieve；当子任务包含已知实体时，retrieve_k 会提高到 max(top_k, 20)，以便实体过滤有足够候选。

## 4.3 web_search

web_search 使用 httpx 请求 DuckDuckGo HTML 端点：

```python
endpoint = "https://html.duckduckgo.com/html/"
with httpx.Client(timeout=timeout, follow_redirects=True) as client:
    response = client.get(endpoint, params={"q": query})
    response.raise_for_status()
    return {"query": query, "status": response.status_code, "html_length": len(response.text)}
```

当前实现只返回状态码和 HTML 长度，不解析搜索结果列表。它适合做网络可用性验证或后续扩展的占位工具，不是主要检索路径。若要做生产级 web search，应增加结果解析、来源去重、可信度排序和缓存。

## 4.4 fetch_url

fetch_url 请求指定 URL，并用 BeautifulSoup 去掉 script、style、nav、footer，再提取 readable text，最多 6000 字符。

```python
soup = BeautifulSoup(response.text, "html.parser")
for tag in soup(["script", "style", "nav", "footer"]):
    tag.decompose()
text = " ".join(soup.get_text(" ", strip=True).split())[:6000]
```

如果没有 BeautifulSoup，则退化为 response.text[:6000]。

## 4.5 arxiv_search

arxiv_search 调用 arXiv Atom API，解析 XML，返回 title、url、summary，最多 max_results 条。

```python
params = {
    "search_query": f"all:{query}",
    "start": 0,
    "max_results": max_results,
    "sortBy": "relevance",
}
response = client.get("https://export.arxiv.org/api/query", params=params)
root = ET.fromstring(response.text)
ns = {"a": "http://www.w3.org/2005/Atom"}
```

返回的 summary 截断到 500 字符。这个工具用于论文场景，不属于默认实习调研路径。

## 4.6 pdf_parse

pdf_parse 使用 pypdf.PdfReader：

```python
reader = PdfReader(str(pdf_path))
pages = []
for page in reader.pages[:max_pages]:
    pages.append((page.extract_text() or "").strip())
return {"path": path, "pages": len(reader.pages), "extracted_pages": pages}
```

它不负责 OCR，也不做版面重建。扫描版 PDF 需要额外的 OCR 工具。

## 4.7 python_sandbox

python_sandbox 是计算类子任务使用的工具。它用受限 builtins 执行代码，并捕获 stdout。

```python
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool,
    "dict": dict, "enumerate": enumerate, "float": float,
    "int": int, "len": len, "list": list, "max": max,
    "min": min, "print": print, "range": range, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "zip": zip,
}
```

执行时：

```python
namespace = {"__builtins__": SAFE_BUILTINS}
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(code, "<sandbox>", "exec"), namespace, namespace)
return {"stdout": output.getvalue(), "duration_ms": round(duration, 3)}
```

危险代码如 import os 会失败，因为 SAFE_BUILTINS 中没有 __import__。计算题示例会把 "8 hours, 5 days, 10 weeks" 转成 `print(8 * 5 * 10)`，输出 `400`。

需要明确的是，python_sandbox 不是操作系统级沙箱。它的 timeout_seconds 参数是在 exec 结束后检查 duration，不能中断死循环；真正的生产环境应改为子进程、容器或 seccomp/Job Object 级别的隔离。

## 4.8 sqlite_query

sqlite_query 只允许单条 SELECT：

```python
lowered = query.strip().lower()
if not lowered.startswith("select") or ";" in query.rstrip(";"):
    raise ValueError("Only single read-only SELECT queries are allowed")
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
try:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(query, params or []).fetchall()
finally:
    conn.close()
```

它使用 SQLite URI 的 mode=ro 打开只读连接，显式关闭连接，返回 columns 与 rows。限制包括：不支持 WITH、不支持 PRAGMA、不支持多语句。

## 4.9 Guardrails 输入侧 core/guardrails/input.py

### 4.9.1 PII 正则

```python
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"(?<!\d)(?:(?:\+?\d{1,3}[-.\s]?)?\d{4}[-.\s]?\d{4}|"
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})(?!\d)"
)
ID_RE = re.compile(r"\b[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b")
```

redact_pii 依次替换 EMAIL_RE、PHONE_RE、ID_RE：

```python
redacted = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
redacted = PHONE_RE.sub("[PHONE_REDACTED]", redacted)
return ID_RE.sub("[ID_REDACTED]", redacted)
```

### 4.9.2 Prompt Injection 检测

INJECTION_PATTERNS 包含四类模式：

- ignore previous instructions
- disregard system prompt
- you are now ...
- reveal your system prompt

detect_injection 返回命中的正则字符串或 None。

### 4.9.3 validate_input

```python
def validate_input(text: str, max_chars: int = 4000) -> tuple[bool, str, list[str]]:
    issues = []
    sanitized = redact_pii(text)
    if len(text) > max_chars:
        issues.append(f"input exceeds {max_chars} chars")
    injected = detect_injection(text)
    if injected:
        issues.append("prompt injection pattern detected")
        return False, sanitized, issues
    return not issues, sanitized, issues
```

注意：PII 会被脱敏但不会阻止请求；只有超长或 prompt injection 会导致 ok=False。这个行为在 tests/test_param_cases.py 中有对应测试。

## 4.10 Guardrails 输出侧 core/guardrails/output.py

CITATION_RE 匹配 [1]、[2]、[12] 这类引用。

```python
CITATION_RE = re.compile(r"\[(\d{1,3})\]")

def extract_citations(text: str) -> list[int]:
    return [int(match) for match in CITATION_RE.findall(text)]
```

validate_citations 做两类检查：

- 引用越界：citation number < 1 或 > len(sources)。
- 来源未引用：1..len(sources) 中任何一个没有出现在 cited 里。

```python
for number in cited:
    if number < 1 or number > len(sources):
        issues.append(f"citation [{number}] out of range")
for idx in range(1, len(sources) + 1):
    if idx not in cited:
        issues.append(f"source [{idx}] never cited")
```

这个策略非常严格。它保证了没有无来源断言，但对真实 LLM 也更苛刻：只要模型漏引任何一个检索来源，Critic 就会失败并触发重试。

## 4.11 工具层与 Guardrails 的实现边界

第一，工具调用目前主要由 AgentBrain 直接选择，不是由 LLM 自由 function calling 决定。LLM 在 Planner 阶段返回 tools 字段，Executor 按子任务类型调用 retrieve 或 python_sandbox。

第二，web_search 与 arxiv_search 的返回结构还未统一为 Source，因此它们更适合作为后续扩展工具。当前主链路主要使用 retrieve。

第三，所有网络工具都有 8 秒超时，失败会以 ToolResult(ok=False) 返回，不会让图崩溃。

第四，输入 Guardrails 是正则级别的轻量防护，不是工业级内容安全系统。生产环境应增加更全面的注入检测、内容分类和权限控制。
