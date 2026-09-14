# 第四章补充 工具层逐工具走读

## 4A.1 Tool.invoke 的返回契约

Tool.invoke 的契约是永远返回 ToolResult，不抛异常。它支持三种返回值：

- 返回 ToolResult：原样返回。
- 返回 str：包装为 ToolResult(ok=True, output=str)。
- 返回其他对象：json.dumps 后截断到 8000 字符，data 保存原对象。

```python
if isinstance(output, ToolResult):
    return output
if isinstance(output, str):
    return ToolResult(tool=self.name, ok=True, output=output, duration_ms=...)
serialized = json.dumps(output, ensure_ascii=False, default=str)[:8000]
return ToolResult(tool=self.name, ok=True, output=serialized, data=output, duration_ms=...)
```

异常分支：

```python
except Exception as exc:
    return ToolResult(tool=self.name, ok=False, output="", error=str(exc), duration_ms=...)
```

这个契约让 Executor 不需要写 try/except，只需要检查 result.ok。

## 4A.2 retrieve 的完整数据形状

retrieve 返回：

```json
{
  "query": "Shopee AI intern skills",
  "results": [
    {
      "id": "02_shopee-sea-group-abc123",
      "doc_id": "02_shopee-sea-group",
      "title": "Shopee（Sea Group） - LLM Agent & Prompt Engineering Intern",
      "url": "https://careers.shopee.sg/jobs",
      "heading": "Roles",
      "snippet": "...",
      "score": 8.42
    }
  ]
}
```

AgentBrain.execute_subtask 读取 result.data["results"]。如果 data 缺失或 results 为空，子任务直接 failed。

## 4A.3 web_search 的控制流

web_search 的关键点是它没有解析 HTML：

```python
response = client.get(endpoint, params={"q": query})
response.raise_for_status()
return {"query": query, "status": response.status_code, "html_length": len(response.text)}
```

它适合验证网络是否可达，不适合作为主要来源。要升级为生产工具，需要：

1. 解析结果标题与 URL。
2. 去重域名。
3. 抓取正文。
4. 转成 Source。
5. 增加可信度排序与缓存。

## 4A.4 fetch_url 的正文清洗

```python
soup = BeautifulSoup(response.text, "html.parser")
for tag in soup(["script", "style", "nav", "footer"]):
    tag.decompose()
text = " ".join(soup.get_text(" ", strip=True).split())[:6000]
```

它删除四类标签，再把剩余文本按空白切分并重新拼接。这样会丢失段落结构，但能压缩长度。对于招聘页面，这种清洗足够做初步证据。

## 4A.5 arxiv_search 的 XML 解析

```python
root = ET.fromstring(response.text)
ns = {"a": "http://www.w3.org/2005/Atom"}
for entry in root.findall("a:entry", ns)[:max_results]:
    results.append({
        "title": " ".join(entry.findtext("a:title", default="", namespaces=ns).split()),
        "url": entry.findtext("a:id", default="", namespaces=ns),
        "summary": entry.findtext("a:summary", default="", namespaces=ns)[:500],
    })
```

namespace 必须显式传入，否则 findall 找不到 Atom 元素。summary 容易被截断，适合做候选筛选而不是最终引用。

## 4A.6 pdf_parse 的限制

```python
reader = PdfReader(str(pdf_path))
for page in reader.pages[:max_pages]:
    pages.append((page.extract_text() or "").strip())
```

它只做文本层提取。扫描版 PDF 没有文本层时，extract_text() 返回空字符串。此时需要 OCR，例如 PaddleOCR、Tesseract 或云 OCR。

## 4A.7 python_sandbox 的安全边界逐条说明

安全措施：

- __builtins__ 只包含白名单函数。
- 没有 __import__，import 语句无法执行。
- 没有 open、exec、eval、compile 等危险内建。
- 输出被重定向到 StringIO。

不安全的地方：

- 仍然使用 exec，运行在当前进程。
- 没有资源限制，没有文件系统隔离。
- timeout_seconds 是事后检查，不能中断死循环。
- 对象可以访问 __class__ 等属性，理论上存在逃逸风险。

因此它适合演示统计与算术，不适合执行不可信代码。

## 4A.8 sqlite_query 的 SQL 注入防御

```python
rows = conn.execute(query, params or []).fetchall()
```

params 通过 SQLite 参数绑定传入，不是字符串拼接。只允许单条 SELECT，且连接以 mode=ro 打开。这样即使 query 被模型控制，也不能写数据库。

限制是它不支持 WITH、PRAGMA、ATTACH，也不支持分号结尾。对于研究数据查询来说，这些限制可以接受。

## 4A.9 工具错误如何向上传播

工具错误不会抛给图，而是变成 ToolResult(ok=False)。Executor 看到失败后会：

```python
if not result.ok or not result.data or not result.data.get("results"):
    subtask.status = "failed"
    subtask.error = result.error or "no evidence returned"
    return "", [], result
```

如果错误是 no evidence，Critic 在 draft 明确写 no direct evidence 时可以接受；其他错误会导致 Critic 失败并触发重试。

## 4A.10 工具层的扩展方式

新增工具只需要三步：

1. 写一个返回 dict 或 str 的函数。
2. 用 Tool(name, description, parameters, func) 包装。
3. 在 ResearchAgent.__init__ 中注册。

如果要让真实 LLM 自主调用工具，还需要把 registry.schemas() 传给 chat completions，并在 Executor 中解析 tool_calls。当前项目在 Planner 阶段刻意不传 tools，以避免 DeepSeek 返回工具调用而不是 JSON 计划。
