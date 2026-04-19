"""Paper loading + full-text search tools for the agent loop.

A :class:`PaperContext` holds the currently-loaded paper (fetched via
the existing :mod:`hep_cot.paper_fetcher`) and exposes tool callables
that share that state. The same object is fed to :mod:`.figure_tools`.

When a semantic table-of-contents has been built for the paper (via
:mod:`hep_cot.tools.semantic_segmentation`), search/anchor lookups
prefer semantic section labels like ``§(event_selection)`` over raw
offsets or absent LaTeX section markers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..agent.tool_registry import ToolSchema
from ..llm.base import Provider
from ..paper_fetcher import PaperInfo, fetch_paper, normalize_arxiv_id
from .semantic_segmentation import SemanticTOC, segment_paper


MAX_SNIPPET_WINDOW = 400  # chars before/after a match in search_text


@dataclass
class PaperContext:
    """Shared state for paper-related tools."""

    cache_dir: str = "./paper_cache"
    paper: PaperInfo | None = None
    figures: list[Any] = field(default_factory=list)  # avoid circular import
    source_dir: str = ""
    toc: SemanticTOC | None = None
    segmenter_provider: Provider | None = None
    skip_segmentation: bool = False

    @property
    def is_loaded(self) -> bool:
        return self.paper is not None and bool(self.paper.text_content)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _tool_fetch_paper(ctx: PaperContext, identifier: str) -> dict[str, Any]:
    """Load a paper into the context. Subsequent tools operate on it."""
    import os

    try:
        info = fetch_paper(identifier, cache_dir=ctx.cache_dir)
    except Exception as e:
        return {"error": f"fetch_paper failed: {type(e).__name__}: {e}"}

    ctx.paper = info
    arxiv_id = info.arxiv_id or (normalize_arxiv_id(identifier) or "")
    ctx.source_dir = os.path.join(
        ctx.cache_dir, f"{arxiv_id.replace('/', '_')}_source"
    )

    # Reset figures for the new paper — figure_tools will lazy-load.
    ctx.figures = []

    # Build or load the semantic table-of-contents.
    ctx.toc = None
    if not ctx.skip_segmentation and info.text_content:
        ctx.toc = segment_paper(
            paper_text=info.text_content,
            arxiv_id=arxiv_id,
            title=info.title,
            cache_dir=ctx.cache_dir,
            segmenter_provider=ctx.segmenter_provider,
        )

    result = {
        "arxiv_id": info.arxiv_id,
        "title": info.title,
        "authors_count": len(info.authors),
        "abstract": info.abstract,
        "doi": info.doi,
        "categories": info.categories,
        "text_length_chars": len(info.text_content),
        "extraction_method": info.publication_info,
        "has_latex_source": info.publication_info == "latex_source",
    }
    if ctx.toc is not None:
        result["semantic_toc"] = {
            "source": ctx.toc.source,
            "segmenter_provider": ctx.toc.segmenter_provider,
            "segments": [
                {
                    "label": s.label,
                    "title": s.title,
                    "key_points": s.key_points,
                }
                for s in ctx.toc.segments
            ],
        }
    return result


def _nearest_anchor(ctx: PaperContext, offset: int) -> dict[str, str]:
    """Return the best cite-friendly anchor for ``offset``.

    Priority:
      1. Semantic TOC segment label (e.g. ``§(event_selection)``)
      2. LaTeX ``\\section{...}`` scan (rarely present in PRL short papers)
      3. ``(no-anchor)`` fallback
    """

    if ctx.toc is not None and ctx.toc.segments:
        seg = ctx.toc.nearest(offset)
        if seg is not None:
            return {
                "section_title": seg.title,
                "section_label": seg.label,
                "section_marker": f"§({seg.label})",
            }

    # LaTeX fallback — only relevant if the paper really has sections
    # AND the TOC wasn't built from them (rare edge case).
    if ctx.paper is not None:
        text = ctx.paper.text_content
        sec_re = re.compile(r"\\(sub)*section\*?\{([^}]+)\}", re.IGNORECASE)
        last: re.Match | None = None
        for m in sec_re.finditer(text, 0, offset + 1):
            last = m
        if last is not None:
            title = re.sub(r"\s+", " ", last.group(2)).strip()
            short = " ".join(title.split()[:6])
            return {
                "section_title": title,
                "section_label": "",
                "section_marker": f"§({short})",
            }

    return {
        "section_title": "",
        "section_label": "",
        "section_marker": "§(no-anchor)",
    }


def _tool_get_paper_info(ctx: PaperContext) -> dict[str, Any]:
    if not ctx.is_loaded:
        return {"error": "no paper loaded — call fetch_paper first"}
    p = ctx.paper
    assert p is not None
    return {
        "arxiv_id": p.arxiv_id,
        "title": p.title,
        "authors": p.authors[:5] + (["..."] if len(p.authors) > 5 else []),
        "abstract": p.abstract,
        "doi": p.doi,
        "categories": p.categories,
    }


def _nearest_section(text: str, offset: int) -> dict[str, str]:
    """DEPRECATED: use :func:`_nearest_anchor` (PaperContext-aware). Kept
    only so callers outside this module don't break."""
    sec_re = re.compile(r"\\(sub)*section\*?\{([^}]+)\}", re.IGNORECASE)
    last: re.Match | None = None
    for m in sec_re.finditer(text, 0, offset + 1):
        last = m
    if last is None:
        return {"section_title": "", "section_marker": "§(no-section)"}
    title = re.sub(r"\s+", " ", last.group(2)).strip()
    short = " ".join(title.split()[:6])
    return {"section_title": title, "section_marker": f"§({short})"}


def _tool_get_paper_section(
    ctx: PaperContext, section_pattern: str, max_chars: int = 8000
) -> dict[str, Any]:
    """Extract a section by semantic TOC label/title or LaTeX heading."""
    if not ctx.is_loaded:
        return {"error": "no paper loaded — call fetch_paper first"}
    text = ctx.paper.text_content  # type: ignore[union-attr]
    pattern_lc = section_pattern.lower().strip()

    # --- Semantic TOC lookup first ---
    if ctx.toc is not None and ctx.toc.segments:
        seg = ctx.toc.by_label_or_title(section_pattern)
        if seg is not None:
            body = text[seg.start_offset : seg.end_offset].strip()
            truncated = False
            if len(body) > max_chars:
                body = body[:max_chars]
                truncated = True
            return {
                "source": "semantic_toc",
                "section_label": seg.label,
                "section_title": seg.title,
                "section_marker": f"§({seg.label})",
                "body": body,
                "char_count": len(body),
                "truncated": truncated,
                "key_points": seg.key_points,
            }

    # --- LaTeX \section fallback ---
    sec_re = re.compile(r"\\(sub)*section\*?\{([^}]+)\}", re.IGNORECASE)
    matches = list(sec_re.finditer(text))
    if matches:
        target_idx = None
        for i, m in enumerate(matches):
            if pattern_lc in m.group(2).lower():
                target_idx = i
                break
        if target_idx is not None:
            start = matches[target_idx].end()
            end = (
                matches[target_idx + 1].start()
                if target_idx + 1 < len(matches)
                else len(text)
            )
            body = text[start:end].strip()
            truncated = False
            if len(body) > max_chars:
                body = body[:max_chars]
                truncated = True
            return {
                "source": "latex_section",
                "section_title": matches[target_idx].group(2),
                "section_marker": f"§({matches[target_idx].group(2)})",
                "body": body,
                "char_count": len(body),
                "truncated": truncated,
            }

    # --- No match ---
    available: list[str] = []
    if ctx.toc is not None and ctx.toc.segments:
        available = [f"{s.label} — {s.title}" for s in ctx.toc.segments]
    elif matches:
        available = [m.group(2) for m in matches[:30]]
    return {
        "error": f"no section matching '{section_pattern}'",
        "available_sections": available,
    }


def _tool_search_text(
    ctx: PaperContext,
    query: str,
    max_results: int = 8,
    window: int = MAX_SNIPPET_WINDOW,
) -> dict[str, Any]:
    """Grep-style search over the loaded paper's LaTeX text.

    Each hit includes the nearest semantic section label (or LaTeX
    section title fallback) as a cite-friendly ``section_marker``,
    plus a surrounding snippet. Do NOT cite raw character offsets —
    use ``section_marker`` / ``section_label`` instead.
    """
    if not ctx.is_loaded:
        return {"error": "no paper loaded — call fetch_paper first"}
    text = ctx.paper.text_content  # type: ignore[union-attr]

    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        # Fall back to literal if the user input isn't valid regex.
        pattern = re.compile(re.escape(query), re.IGNORECASE)

    hits: list[dict[str, Any]] = []
    for m in pattern.finditer(text):
        if len(hits) >= max_results:
            break
        start = max(0, m.start() - window)
        end = min(len(text), m.end() + window)
        snippet = text[start:end].replace("\n", " ").strip()
        sec = _nearest_anchor(ctx, m.start())
        hit = {
            "match": m.group(0),
            "section_marker": sec["section_marker"],
            "snippet": snippet,
        }
        if sec.get("section_label"):
            hit["section_label"] = sec["section_label"]
        if sec.get("section_title"):
            hit["section_title"] = sec["section_title"]
        hits.append(hit)

    has_toc = ctx.toc is not None and bool(ctx.toc.segments)
    return {
        "query": query,
        "total_hits": len(hits),
        "hits": hits,
        "note": (
            "Cite hits via `section_marker` (e.g. '§(event_selection)'). "
            + (
                "A semantic TOC is active — prefer semantic labels over raw "
                "figure/table numbers unless citing a specific Fig/Table."
                if has_toc
                else "No semantic TOC available; markers use LaTeX section "
                "titles or '§(no-anchor)' as fallback."
            )
        ),
    }


def _tool_get_bibliography(ctx: PaperContext, max_chars: int = 6000) -> dict[str, Any]:
    """Return the bibliography block from the LaTeX source, if present."""
    if not ctx.is_loaded:
        return {"error": "no paper loaded — call fetch_paper first"}
    text = ctx.paper.text_content  # type: ignore[union-attr]

    marker = "% === Bibliography ==="
    if marker not in text:
        return {"error": "no bibliography block recovered in extracted text"}
    bbl = text.split(marker, 1)[1].strip()
    truncated = False
    if len(bbl) > max_chars:
        bbl = bbl[:max_chars]
        truncated = True
    return {"bibliography": bbl, "truncated": truncated}


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def build_paper_tools(ctx: PaperContext) -> list[ToolSchema]:
    """Return tool schemas bound to a :class:`PaperContext`."""

    def fetch(identifier: str) -> dict[str, Any]:
        return _tool_fetch_paper(ctx, identifier)

    def info() -> dict[str, Any]:
        return _tool_get_paper_info(ctx)

    def section(section_pattern: str, max_chars: int = 8000) -> dict[str, Any]:
        return _tool_get_paper_section(ctx, section_pattern, max_chars)

    def search(query: str, max_results: int = 8) -> dict[str, Any]:
        return _tool_search_text(ctx, query, max_results)

    def biblio(max_chars: int = 6000) -> dict[str, Any]:
        return _tool_get_bibliography(ctx, max_chars)

    return [
        ToolSchema(
            name="fetch_paper",
            description=(
                "Load a HEP paper by arXiv ID, arXiv URL, or INSPIRE-HEP "
                "URL/texkey. Replaces any previously loaded paper. Returns "
                "basic metadata including title, authors count, and abstract."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": (
                            "arXiv id like '2206.08956' / 'hep-ex/0612015', "
                            "an arxiv.org URL, or an INSPIRE-HEP URL/texkey."
                        ),
                    }
                },
                "required": ["identifier"],
            },
            func=fetch,
        ),
        ToolSchema(
            name="get_paper_info",
            description=(
                "Return metadata about the currently loaded paper: arxiv id, "
                "title, first few authors, abstract, DOI, categories."
            ),
            parameters={"type": "object", "properties": {}},
            func=info,
        ),
        ToolSchema(
            name="get_paper_section",
            description=(
                "Return the raw LaTeX body of a section whose title contains "
                "the given substring (case-insensitive). Use this for focused "
                "reading of one part of the paper. If not found, a list of "
                "available section titles is returned."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "section_pattern": {
                        "type": "string",
                        "description": (
                            "Substring to match against LaTeX section titles, "
                            "e.g. 'Systematic' or 'Event selection'."
                        ),
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Truncate the section body at this many characters.",
                        "default": 8000,
                    },
                },
                "required": ["section_pattern"],
            },
            func=section,
        ),
        ToolSchema(
            name="search_text",
            description=(
                "Full-text search over the loaded paper's LaTeX source. "
                "Returns up to N hits; each hit includes `section_marker` "
                "(a cite-friendly tag like '§(Systematic uncertainties)'), "
                "`section_title`, `match`, and a surrounding `snippet`. "
                "The query is regex, falling back to literal. "
                "ALWAYS cite hits using `section_marker` or `section_title`, "
                "never a raw character offset."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Regex or literal search term. E.g. 'JES', "
                            "'signal region', 'm_{ll}', 'pileup reweight'."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of hits to return.",
                        "default": 8,
                    },
                },
                "required": ["query"],
            },
            func=search,
        ),
        ToolSchema(
            name="get_bibliography",
            description=(
                "Return the paper's bibliography block (contents of the .bbl "
                "file if recovered during extraction). Useful for resolving "
                "numeric citations to referenced works."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "max_chars": {
                        "type": "integer",
                        "description": "Truncate at this many characters.",
                        "default": 6000,
                    }
                },
            },
            func=biblio,
        ),
    ]
