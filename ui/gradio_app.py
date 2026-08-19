from __future__ import annotations

import argparse

from agents.agent import ResearchAgent
from core.config import Settings


def build_ui():
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("gradio is not installed") from exc

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
        question = gr.Textbox(
            label="Research question",
            value="Compare Shopee, TikTok and Grab AI internship opportunities and required skills in Singapore.",
        )
        run_button = gr.Button("Run", variant="primary")
        status = gr.Markdown()
        report = gr.Markdown()
        run_button.click(answer, inputs=question, outputs=[report, status])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
