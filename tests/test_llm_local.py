from __future__ import annotations

import json
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agents.agent import ResearchAgent
from agents.llm import OpenAICompatibleLLM
from core.config import Settings


class MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        messages = payload.get("messages", [])
        system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        if "planner" in system:
            content = json.dumps(
                {
                    "objective": "local objective",
                    "rationale": "local",
                    "subtasks": [
                        {
                            "id": "st-1",
                            "question": "Shopee AI intern skills",
                            "tools": ["retrieve"],
                        }
                    ],
                }
            )
        else:
            user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            numbers = [int(value) for value in re.findall(r"\[(\d+)\]", user)]
            max_number = max(numbers) if numbers else 1
            citations = "".join(f"[{i}]" for i in range(1, max_number + 1))
            content = f"# Local LLM Report\nShopee evidence {citations}.\n\n## Sources\n" + "\n".join(
                f"{i}. Shopee source {i}" for i in range(1, max_number + 1)
            )
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [],
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


class LLMLocalServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def _settings(self, tmp: str) -> Settings:
        corpus = Path(tmp) / "corpus"
        corpus.mkdir()
        (corpus / "shopee.md").write_text(
            "---\ntitle: Shopee AI\nsource_url: https://shopee\n---\n"
            "Shopee AI interns use Python and PyTorch.",
            encoding="utf-8",
        )
        return Settings(
            corpus_dir=corpus,
            cache_dir=Path(tmp) / "storage",
            trace_path=Path(tmp) / "storage" / "traces.jsonl",
            llm_provider="openai_compatible",
            llm_base_url=f"http://127.0.0.1:{self.port}/v1",
            llm_api_key="test-key",
            llm_model="mock-model",
        )

    def test_client_json_completion(self) -> None:
        client = OpenAICompatibleLLM(
            base_url=f"http://127.0.0.1:{self.port}/v1",
            api_key="test-key",
            model="mock-model",
        )
        result = client.complete_json(
            [
                {"role": "system", "content": "You are a research planner."},
                {"role": "user", "content": "Plan something."},
            ]
        )
        self.assertEqual(result["objective"], "local objective")

    def test_agent_uses_llm_planning_and_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = ResearchAgent(settings=self._settings(tmp))
            state = agent.run("What skills do Shopee AI interns need?")
            self.assertIn("Local LLM Report", state.report)
            self.assertEqual(agent.graph.engine, "langgraph")


if __name__ == "__main__":
    unittest.main()
