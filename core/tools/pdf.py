from __future__ import annotations

from pathlib import Path

from core.tools.registry import Tool


def build_pdf_parse_tool() -> Tool:
    def pdf_parse(path: str, max_pages: int = 20) -> dict[str, object]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pypdf is not installed") from exc
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(path)
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages[:max_pages]:
            pages.append((page.extract_text() or "").strip())
        return {"path": path, "pages": len(reader.pages), "extracted_pages": pages}

    return Tool(
        name="pdf_parse",
        description="Parse a PDF file into structured page texts.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to PDF"},
                "max_pages": {"type": "integer"},
            },
            "required": ["path"],
        },
        func=pdf_parse,
    )
