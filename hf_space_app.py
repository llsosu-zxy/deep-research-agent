from __future__ import annotations

from ui.gradio_app import build_ui

if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
