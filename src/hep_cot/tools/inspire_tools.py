"""INSPIRE-HEP tools: references + citations.

All calls hit the JSON REST endpoint ``https://inspirehep.net/api``
with local on-disk caching to respect rate limits and stay fast
across re-runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..agent.tool_registry import ToolSchema


INSPIRE_BASE = "https://inspirehep.net/api"
CACHE_TTL_SECONDS = 7 * 24 * 3600  # 1 week


def _fetch_json(url: str, cache_dir: str) -> dict[str, Any] | None:
    """HTTP GET with simple file-cache; returns parsed JSON or None on failure."""
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    cache_path = Path(cache_dir) / f"inspire_{key}.json"

    if cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age < CACHE_TTL_SECONDS:
                with open(cache_path, encoding="utf-8") as f:
                    return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    try:
        import httpx  # type: ignore
    except ImportError:
        # Fallback to stdlib
        from urllib.request import Request, urlopen

        req = Request(
            url,
            headers={
                "User-Agent": "hep-copilot/0.2",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
        except Exception:
            return None
    else:
        try:
            with httpx.Client(
                timeout=30.0,
                headers={
                    "User-Agent": "hep-copilot/0.2",
                    "Accept": "application/json",
                },
            ) as c:
                r = c.get(url)
                r.raise_for_status()
                data = r.json()
        except Exception:
            return None

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass

    return data


def _resolve_arxiv_to_inspire_id(arxiv_id: str, cache_dir: str) -> str | None:
    """Look up the numeric INSPIRE literature id for an arXiv id."""
    url = (
        f"{INSPIRE_BASE}/literature?q=arxiv:{quote(arxiv_id)}"
        f"&fields=control_number&size=1"
    )
    data = _fetch_json(url, cache_dir)
    if not data:
        return None
    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        return None
    cn = hits[0].get("metadata", {}).get("control_number")
    if cn is None:
        return None
    return str(cn)


def _brief_reference(rec: dict[str, Any]) -> dict[str, Any]:
    """Flatten an INSPIRE literature record to a citation-friendly dict."""
    md = rec.get("metadata") or rec  # may already be metadata
    titles = md.get("titles") or []
    title = titles[0].get("title", "") if titles else ""

    authors = md.get("authors") or []
    author_names = [a.get("full_name", "") for a in authors[:3] if a.get("full_name")]
    n_authors = len(authors)

    arxiv_eprints = md.get("arxiv_eprints") or []
    arxiv_id = arxiv_eprints[0].get("value", "") if arxiv_eprints else ""

    pub_info = md.get("publication_info") or []
    journal = ""
    year = ""
    if pub_info:
        p0 = pub_info[0]
        parts = [
            p0.get("journal_title", ""),
            p0.get("journal_volume", ""),
            str(p0.get("year", "")) if p0.get("year") else "",
        ]
        journal = " ".join(x for x in parts if x).strip()
        year = str(p0.get("year", ""))

    dois = md.get("dois") or []
    doi = dois[0].get("value", "") if dois else ""

    return {
        "inspire_id": rec.get("id") or md.get("control_number") or "",
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": author_names + (["..."] if n_authors > 3 else []),
        "author_count": n_authors,
        "journal": journal,
        "year": year,
        "doi": doi,
        "citation_count": md.get("citation_count"),
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _tool_inspire_references(
    arxiv_id: str, limit: int = 30, cache_dir: str = "./paper_cache"
) -> dict[str, Any]:
    """List papers referenced BY the given paper."""
    inspire_id = _resolve_arxiv_to_inspire_id(arxiv_id, cache_dir)
    if not inspire_id:
        return {"error": f"no INSPIRE record for arxiv:{arxiv_id}"}

    # Ask directly for references + resolve each to a lookup
    url = (
        f"{INSPIRE_BASE}/literature/{inspire_id}"
        f"?fields=references"
    )
    data = _fetch_json(url, cache_dir)
    if not data:
        return {"error": "INSPIRE request failed"}

    refs = (data.get("metadata") or {}).get("references") or []
    if not refs:
        return {
            "inspire_id": inspire_id,
            "arxiv_id": arxiv_id,
            "total_references": 0,
            "references": [],
        }

    # Each reference has a nested ``reference`` object (partial metadata)
    # plus an optional ``record`` link to the fully-indexed record.
    out: list[dict[str, Any]] = []
    for r in refs[: max(1, limit)]:
        ref = r.get("reference") or {}
        rec_link = r.get("record", {}).get("$ref", "") if r.get("record") else ""
        title = ""
        titles = ref.get("titles") or []
        if titles:
            title = titles[0].get("title", "")

        arxiv_eprint = ref.get("arxiv_eprint", "")
        authors = ref.get("authors") or []
        author_names = [a.get("full_name", "") for a in authors[:2]]

        pub_info = ref.get("publication_info") or {}
        if isinstance(pub_info, list):
            pub_info = pub_info[0] if pub_info else {}
        journal = " ".join(
            x
            for x in [
                pub_info.get("journal_title", ""),
                str(pub_info.get("year", "")) if pub_info.get("year") else "",
            ]
            if x
        )

        out.append(
            {
                "arxiv_id": arxiv_eprint,
                "title": title,
                "authors": author_names,
                "journal_year": journal,
                "inspire_record_url": rec_link,
                "misc": ref.get("misc") or "",
            }
        )

    return {
        "inspire_id": inspire_id,
        "arxiv_id": arxiv_id,
        "total_references": len(refs),
        "returned": len(out),
        "references": out,
    }


def _tool_inspire_citations(
    arxiv_id: str, limit: int = 20, cache_dir: str = "./paper_cache"
) -> dict[str, Any]:
    """List papers that cite this paper (up to ``limit`` most recent)."""
    inspire_id = _resolve_arxiv_to_inspire_id(arxiv_id, cache_dir)
    if not inspire_id:
        return {"error": f"no INSPIRE record for arxiv:{arxiv_id}"}

    # refersto:recid:<id> returns works that reference the given record.
    url = (
        f"{INSPIRE_BASE}/literature?q=refersto%3Arecid%3A{inspire_id}"
        f"&sort=mostrecent&size={max(1, min(limit, 50))}"
        f"&fields=titles,authors,arxiv_eprints,publication_info,dois,citation_count,control_number"
    )
    data = _fetch_json(url, cache_dir)
    if not data:
        return {"error": "INSPIRE request failed"}

    hits = data.get("hits", {}).get("hits", [])
    total = data.get("hits", {}).get("total", {}).get("value", len(hits))

    out = [_brief_reference(h) for h in hits]
    return {
        "inspire_id": inspire_id,
        "arxiv_id": arxiv_id,
        "total_citations": total,
        "returned": len(out),
        "citations": out,
    }


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def build_inspire_tools(cache_dir: str = "./paper_cache") -> list[ToolSchema]:
    def refs(arxiv_id: str, limit: int = 30) -> dict[str, Any]:
        return _tool_inspire_references(arxiv_id, limit, cache_dir)

    def cits(arxiv_id: str, limit: int = 20) -> dict[str, Any]:
        return _tool_inspire_citations(arxiv_id, limit, cache_dir)

    return [
        ToolSchema(
            name="inspire_references",
            description=(
                "List the papers referenced by the given arXiv paper, via "
                "INSPIRE-HEP. Each entry includes arxiv id (if any), title, "
                "first two authors, and journal/year. Use this to resolve "
                "numeric citations or find related analyses."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv id of the paper whose references to list.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum references to return (INSPIRE may hold hundreds).",
                        "default": 30,
                    },
                },
                "required": ["arxiv_id"],
            },
            func=refs,
        ),
        ToolSchema(
            name="inspire_citations",
            description=(
                "List the most recent papers that cite the given arXiv paper, "
                "via INSPIRE-HEP. Use this to find follow-up analyses, "
                "reinterpretations, or theoretical discussions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv id of the paper to find citers for.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum citing papers to return (max 50).",
                        "default": 20,
                    },
                },
                "required": ["arxiv_id"],
            },
            func=cits,
        ),
    ]
