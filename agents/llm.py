from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAICompatibleLLM:
    """Small OpenAI-compatible chat completions client (DeepSeek/Qwen/Ollama)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("httpx is not installed") from exc
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens is not None else 2048,
        }
        if tools:
            payload["tools"] = tools
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        choice = data["choices"][0]["message"]
        usage = data.get("usage", {})
        tool_calls = []
        for call in choice.get("tool_calls") or []:
            arguments = call.get("function", {}).get("arguments", "{}")
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                {
                    "id": call.get("id", ""),
                    "name": call.get("function", {}).get("name", ""),
                    "arguments": arguments,
                }
            )
        return LLMResponse(
            content=choice.get("content") or "",
            tool_calls=tool_calls,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        retries: int = 2,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(retries):
            response = self.complete(messages, tools=tools, max_tokens=4096)
            if response.tool_calls:
                arguments = response.tool_calls[0].get("arguments", {})
                if isinstance(arguments, dict) and arguments:
                    return arguments
            text = response.content.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:].strip()
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
            if not text:
                last_error = ValueError("LLM returned empty JSON")
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise ValueError("LLM returned empty JSON")
