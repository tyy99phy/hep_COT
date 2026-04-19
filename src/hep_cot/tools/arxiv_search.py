"""arXiv search tool via the public Atom-feed API."""

from __future__ import annotations

import hashlib
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from ..agent.tool_registry import ToolSchema


ARXIV_BASE = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}
CACHE_TTL_SECONDS = 24 * 3600


def _fetch(url: str, cache_dir: str) -> str | None:
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    cache_path = Path(cache_dir) / f"arxiv_search_{key}.xml"

    if cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age < CACHE_TTL_SECONDS:
                return cache_path.read_text(encoding="utf-8")
        except OSError:
            pass

    req = Request(url, headers={"User-Agent": "hep-copilot/0.2"})
    try:
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    try:
        cache_path.write_text(body, encoding="utf-8")
    except OSError:
        pass
    return body


def _parse_feed(xml: str) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []

    entries = []
    for entry in root.findall("atom:entry", NS):
        title_el = entry.find("atom:title", NS)
        summary_el = entry.find("atom:summary", NS)
        id_el = entry.find("atom:id", NS)
        pub_el = entry.find("atom:published", NS)

        authors = [
            (a.find("atom:name", NS).text or "")  # type: ignore[union-attr]
            for a in entry.findall("atom:author", NS)
            if a.find("atom:name", NS) is not None
        ]

        arxiv_id = ""
        if id_el is not None and id_el.text:
            m = re.search(r"arxiv\.org/abs/([\w\-.]+/?[\w.]+?)(v\d+)?$", id_el.text)
            if m:
                arxiv_id = m.group(1)

        title = (title_el.text or "").strip() if title_el is not None else ""
        title = re.sub(r"\s+", " ", title)
        abstract = (summary_el.text or "").strip() if summary_el is not None else ""
        abstract = re.sub(r"\s+", " ", abstract)

        entries.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors[:3] + (["..."] if len(authors) > 3 else []),
                "author_count": len(authors),
                "published": (pub_el.text or "").strip() if pub_el is not None else "",
                "abstract": abstract[:600],
            }
        )
    return entries


def _tool_arxiv_search(
    query: str,
    limit: int = 5,
    cache_dir: str = "./paper_cache",
) -> dict[str, Any]:
    url = (
        f"{ARXIV_BASE}?search_query={quote(query)}"
        f"&start=0&max_results={max(1, min(limit, 50))}"
        f"&sortBy=relevance&sortOrder=descending"
    )
    body = _fetch(url, cache_dir)
    if body is None:
        return {"error": "arXiv request failed"}
    results = _parse_feed(body)
    return {
        "query": query,
        "total": len(results),
        "results": results,
    }


def build_arxiv_search_tool(cache_dir: str = "./paper_cache") -> list[ToolSchema]:
    def search(query: str, limit: int = 5) -> dict[str, Any]:
        return _tool_arxiv_search(query, limit, cache_dir)

    return [
        ToolSchema(
            name="arxiv_search",
            description=(
                "Free-text search over arXiv. Useful for finding related "
                "analyses, theory papers, or follow-up measurements. Accepts "
                "arXiv advanced-search syntax (e.g. 'cat:hep-ex AND abs:\"same-sign WW\"')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query. Can be plain text or arXiv advanced syntax "
                            "like 'all:<term>', 'ti:<title>', 'abs:<abstract phrase>', "
                            "'cat:hep-ex'."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (max 50).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            func=search,
        )
    ]
