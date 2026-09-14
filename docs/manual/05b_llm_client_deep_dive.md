# 第五章补充 LLM 客户端逐行走读

## 5B.1 LLMResponse 的字段

```python
@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
```

content 是模型正文；tool_calls 是标准化后的工具调用；prompt_tokens 与 completion_tokens 用于未来接入预算统计。

## 5B.2 构造函数与超时

```python
def __init__(self, base_url, api_key, model, timeout=120.0):
    self.base_url = base_url.rstrip("/")
    self.api_key = api_key
    self.model = model
    self.timeout = timeout
```

rstrip("/") 避免 base_url 末尾斜杠导致 //chat/completions。默认 120 秒是因为 DeepSeek v4-flash 生成完整报告可能超过 30 秒，实测曾出现 30 秒超时。

## 5B.3 complete 的 payload

```python
payload = {
    "model": self.model,
    "messages": messages,
    "temperature": temperature,
    "max_tokens": max_tokens if max_tokens is not None else 2048,
}
if tools:
    payload["tools"] = tools
headers = {"Authorization": f"Bearer {self.api_key}"}
```

temperature 默认 0.2，偏确定性。Planner 使用 complete_json 时 max_tokens=4096；Synthesizer 使用默认 2048。

## 5B.4 HTTP 调用与错误

```python
with httpx.Client(timeout=self.timeout) as client:
    response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
```

raise_for_status 会把 4xx/5xx 转成 httpx.HTTPStatusError。调用方 AgentBrain 捕获异常并回退。网络超时会抛 httpx.ReadTimeout，同样回退。

## 5B.5 响应解析

```python
choice = data["choices"][0]["message"]
usage = data.get("usage", {})
tool_calls = []
for call in choice.get("tool_calls") or []:
    arguments = call.get("function", {}).get("arguments", "{}")
    try:
        arguments = json.loads(arguments)
    except json.JSONDecodeError:
        arguments = {}
    tool_calls.append({
        "id": call.get("id", ""),
        "name": call.get("function", {}).get("name", ""),
        "arguments": arguments,
    })
```

如果 arguments 不是合法 JSON，就退化为空 dict，而不是让整个调用失败。

## 5B.6 complete_json 的重试逻辑

```python
for _ in range(retries):
    response = self.complete(messages, tools=tools, max_tokens=4096)
    if response.tool_calls:
        arguments = response.tool_calls[0].get("arguments", {})
        if isinstance(arguments, dict) and arguments:
            return arguments
    text = response.content.strip()
    ...
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        last_error = exc
        continue
```

这段代码解决三类真实问题：

- content 为空，没有 tool_calls。
- content 有 Markdown 代码围栏。
- content 是截断 JSON，json.loads 失败。

默认 retries=2，意味着最多调用两次。两次都失败时抛出最后一次异常，由 AgentBrain 回退启发式 Planner。

## 5B.7 DeepSeek v4-flash 的两个真实修复

修复一：规划请求不传 tools。早期 _plan_with_llm 把 registry.schemas() 传给 complete_json，DeepSeek 返回 tool_calls，而不是 JSON 计划，导致 raw.get("subtasks", []) 为空。现在规划阶段的 tools 只出现在 system prompt 文本中，不进入 payload。

修复二：complete_json 增加重试与 JSON 主体截取。DeepSeek 偶尔返回 ```json 代码围栏或截断 JSON，重试与首尾大括号截取提高了成功率。

## 5B.8 真实 LLM 的调用次数与成本

一次普通请求：

- Planner 1 次。
- Synthesizer 1 次。

如果 Critic 失败：

- 每次迭代再调用 Synthesizer 1 次。

max_critic_iterations=2 时，最多 1 次 Planner + 3 次 Synthesizer。真实 10 题评测的 p95 达到 70 秒，主要来自多次远程调用与重试。

## 5B.9 LLM 客户端当前的缺口

第一，没有把 usage 写回 BudgetTracker。

第二，没有重试 HTTP 层面的 429/5xx，只在 complete_json 层面重试解析失败。

第三，没有流式输出，因此 WebSocket 不能逐 token 推送。

第四，没有并发控制，批量评测是串行调用。

第五，没有 prompt 版本号与模型版本记录，trace 无法完整复现一次真实 LLM 运行。
