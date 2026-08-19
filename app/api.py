from __future__ import annotations

import asyncio
from functools import lru_cache

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

from agents.agent import ResearchAgent
from core.config import Settings


class ResearchRequest(BaseModel):
    question: str


@lru_cache(maxsize=1)
def get_agent() -> ResearchAgent:
    settings = Settings.from_env()
    agent = ResearchAgent(settings=settings)
    return agent


def create_app() -> FastAPI:
    app = FastAPI(title="Deep Research Agent", version="0.1.0")

    @app.get("/api/health")
    def health() -> dict:
        agent = get_agent()
        return {
            "status": "ok",
            "engine": agent.graph.engine,
            "corpus_chunks": len(agent.index.chunks),
            "tools": agent.registry.names(),
        }

    @app.post("/api/research")
    def research(payload: ResearchRequest) -> dict:
        agent = get_agent()
        state = agent.run(payload.question)
        return {"state": state.to_dict()}

    @app.get("/api/traces")
    def traces(limit: int = 20) -> dict:
        agent = get_agent()
        return {"traces": agent.trace_logger.recent(limit)}

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
        except Exception as exc:  # noqa: BLE001 - client disconnect should not kill the server
            await websocket.close(code=1011, reason=str(exc))

    return app
