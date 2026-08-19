from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from core.models import Chunk

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def parse_front_matter(markdown: str) -> tuple[dict[str, Any], str]:
    """Parse a simple YAML-like front-matter block from a Markdown file."""
    if not markdown.startswith("---"):
        return {}, markdown
    lines = markdown.splitlines()
    if len(lines) < 3 or lines[1].strip() != "---" and "---" not in lines[1]:
        # Front matter ends at the second '---' line.
        end = 1
        while end < len(lines) and lines[end].strip() != "---":
            end += 1
        if end >= len(lines):
            return {}, markdown
        meta: dict[str, Any] = {}
        for line in lines[1:end]:
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip("'\"")
            if value.startswith("[") and value.endswith("]"):
                value = [v.strip().strip("'\"") for v in value[1:-1].split(",")]
            meta[key] = value
        return meta, "\n".join(lines[end + 1 :])
    end = 1
    while end < len(lines) and lines[end].strip() != "---":
        end += 1
    meta = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip("'\"")
        if value.startswith("[") and value.endswith("]"):
            value = [v.strip().strip("'\"") for v in value[1:-1].split(",")]
        meta[key] = value
    return meta, "\n".join(lines[end + 1 :])


def _chunk_id(doc_id: str, text: str) -> str:
    digest = hashlib.sha1(f"{doc_id}:{text}".encode()).hexdigest()[:12]
    return f"{doc_id}-{digest}"


def _make_chunk(doc_id: str, text: str, metadata: dict[str, Any]) -> Chunk:
    clean = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    return Chunk(
        id=_chunk_id(doc_id, clean),
        doc_id=doc_id,
        text=clean,
        metadata=dict(metadata),
        tokens=tokenize(clean),
    )


def _split_long_paragraph(
    paragraph: str,
    doc_id: str,
    metadata: dict[str, Any],
    max_tokens: int,
    overlap: int,
    heading: str,
    document_title: str,
) -> list[Chunk]:
    words = paragraph.split()
    step = max(1, max_tokens - overlap)
    chunks: list[Chunk] = []
    index = 0
    while index < len(words):
        end = min(len(words), index + max_tokens)
        text = f"{document_title}\n{heading}\n\n" + " ".join(words[index:end])
        chunks.append(_make_chunk(doc_id, text, metadata))
        if end == len(words):
            break
        index += step
    return chunks


def _split_paragraphs(
    paragraphs: list[str],
    doc_id: str,
    metadata: dict[str, Any],
    max_tokens: int,
    overlap: int,
    heading: str,
    document_title: str,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    if not paragraphs:
        return [_make_chunk(doc_id, f"{document_title}\n{heading}", metadata)]
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        para_tokens = count_tokens(paragraph)
        if para_tokens > max_tokens:
            if current:
                chunks.append(
                    _make_chunk(doc_id, f"{document_title}\n{heading}\n\n" + "\n\n".join(current), metadata)
                )
                current = []
                current_tokens = 0
            chunks.extend(
                _split_long_paragraph(
                    paragraph,
                    doc_id,
                    metadata,
                    max_tokens,
                    overlap,
                    heading,
                    document_title,
                )
            )
            continue
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append(
                _make_chunk(doc_id, f"{document_title}\n{heading}\n\n" + "\n\n".join(current), metadata)
            )
            keep = 0
            kept = 0
            while keep < len(current) and kept < overlap:
                kept += count_tokens(current[keep])
                keep += 1
            current = current[-keep:] if keep else []
            current_tokens = kept
        current.append(paragraph)
        current_tokens += para_tokens

    if current:
        chunks.append(
            _make_chunk(doc_id, f"{document_title}\n{heading}\n\n" + "\n\n".join(current), metadata)
        )
    return chunks


def chunk_markdown_document(
    doc_id: str,
    markdown: str,
    max_tokens: int = 350,
    overlap: int = 60,
) -> tuple[dict[str, Any], list[Chunk]]:
    """Split a Markdown document into heading-aware, overlapping chunks."""
    front_matter, body = parse_front_matter(markdown)
    metadata: dict[str, Any] = dict(front_matter)
    metadata.setdefault("doc_id", doc_id)

    sections: list[tuple[str, list[str]]] = []
    current_heading = metadata.get("title", doc_id)
    current_paragraphs: list[str] = []

    for line in body.splitlines():
        if line.startswith("#"):
            if current_paragraphs:
                sections.append((current_heading, current_paragraphs))
            current_heading = line.lstrip("#").strip()
            current_paragraphs = []
        elif line.strip():
            current_paragraphs.append(line.strip())

    if current_paragraphs:
        sections.append((current_heading, current_paragraphs))
    if not sections and current_heading:
        sections.append((current_heading, []))

    document_title = str(metadata.get("title") or current_heading or doc_id)
    chunks: list[Chunk] = []
    for heading, paragraphs in sections:
        meta = dict(metadata)
        meta["heading"] = heading
        chunks.extend(
            _split_paragraphs(
                paragraphs,
                doc_id,
                meta,
                max_tokens,
                overlap,
                heading,
                document_title,
            )
        )
    return metadata, chunks


def load_markdown_corpus(corpus_dir: Path) -> list[tuple[str, str, dict[str, Any]]]:
    """Load all Markdown files under corpus_dir as (doc_id, text, metadata)."""
    documents: list[tuple[str, str, dict[str, Any]]] = []
    if not corpus_dir.exists():
        return documents
    for path in sorted(corpus_dir.rglob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8")
        front_matter, _ = parse_front_matter(text)
        doc_id = front_matter.get("doc_id") or path.stem
        documents.append((doc_id, text, front_matter))
    return documents
