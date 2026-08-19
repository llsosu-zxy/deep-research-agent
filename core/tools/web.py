from __future__ import annotations

from typing import Any

from core.retrieval.index import RetrievalIndex
from core.tools.registry import Tool


def build_retrieve_tool(index: RetrievalIndex, top_k: int = 5) -> Tool:
    def retrieve(query: str, k: int = top_k) -> dict[str, Any]:
        ranked = index.search(query, top_k=int(k))
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

    return Tool(
        name="retrieve",
        description="Hybrid BM25 + dense retrieval over the local research corpus.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "k": {"type": "integer", "description": "Number of results"},
            },
            "required": ["query"],
        },
        func=retrieve,
    )


def build_web_search_tool(timeout: float = 8.0) -> Tool:
    def web_search(query: str) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("httpx is not installed") from exc
        endpoint = "https://html.duckduckgo.com/html/"
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(endpoint, params={"q": query})
            response.raise_for_status()
            return {"query": query, "status": response.status_code, "html_length": len(response.text)}

    return Tool(
        name="web_search",
        description="Search the public web for a query.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        func=web_search,
    )


def build_fetch_url_tool(timeout: float = 8.0) -> Tool:
    def fetch_url(url: str) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("httpx is not installed") from exc
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = " ".join(soup.get_text(" ", strip=True).split())[:6000]
        except ImportError:
            text = response.text[:6000]
        return {"url": url, "status": response.status_code, "text": text}

    return Tool(
        name="fetch_url",
        description="Fetch a public URL and extract readable text.",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
        func=fetch_url,
    )


def build_arxiv_search_tool(timeout: float = 8.0) -> Tool:
    def arxiv_search(query: str, max_results: int = 5) -> dict[str, Any]:
        try:
            import xml.etree.ElementTree as ET

            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("httpx is not installed") from exc
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get("https://export.arxiv.org/api/query", params=params)
            response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        results = []
        for entry in root.findall("a:entry", ns)[:max_results]:
            results.append(
                {
                    "title": " ".join((entry.findtext("a:title", default="", namespaces=ns) or "").split()),
                    "url": entry.findtext("a:id", default="", namespaces=ns),
                    "summary": entry.findtext("a:summary", default="", namespaces=ns)[:500],
                }
            )
        return {"query": query, "results": results}

    return Tool(
        name="arxiv_search",
        description="Search arXiv papers by relevance.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
        func=arxiv_search,
    )
