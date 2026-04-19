"""Semantic paper segmentation.

Produces a table-of-contents (TOC) for a HEP paper even when the LaTeX
source has no ``\\section{}`` markers (common in PRL short papers).

Priority chain used by :mod:`hep_cot.tools.paper_tools`:
  1. Cached TOC from ``paper_cache/<arxiv_id>_semantic_toc.json``
  2. Extract from ``\\section{...}`` / ``\\subsection{...}`` if the
     paper has ≥3 such markers (structured long paper)
  3. LLM-based semantic segmentation (this module)

The TOC is cached per paper and re-used across runs and providers so
teacher/student see the *same* segmentation — critical for fair
comparison in the study pipeline.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..llm.base import Message, Provider, TextDelta, ThinkingDelta, TurnEnd


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SegmentEntry:
    """One semantic section of a paper."""

    label: str  # snake_case, e.g. "event_selection"
    title: str  # human-readable, e.g. "Event selection and categorisation"
    start_offset: int  # char offset in the paper text
    end_offset: int  # exclusive
    key_points: list[str] = field(default_factory=list)


@dataclass
class SemanticTOC:
    """Ordered list of :class:`SegmentEntry` with an offset lookup."""

    segments: list[SegmentEntry] = field(default_factory=list)
    source: str = ""  # "latex_sections" | "semantic_llm" | "cache"
    segmenter_provider: str = ""
    segmenter_model: str = ""
    paper_char_length: int = 0
    generated_at: float = 0.0

    def nearest(self, offset: int) -> SegmentEntry | None:
        """Return the segment containing ``offset`` (or the last segment
        if offset exceeds all ranges)."""
        if not self.segments:
            return None
        for seg in self.segments:
            if seg.start_offset <= offset < seg.end_offset:
                return seg
        # Fall back to closest by distance
        return min(
            self.segments,
            key=lambda s: min(abs(offset - s.start_offset), abs(offset - s.end_offset)),
        )

    def by_label_or_title(self, query: str) -> SegmentEntry | None:
        """Fuzzy match by label or title (case-insensitive substring)."""
        q = query.strip().lower()
        for seg in self.segments:
            if seg.label.lower() == q:
                return seg
        for seg in self.segments:
            if q in seg.label.lower() or q in seg.title.lower():
                return seg
        # Fuzzier: any key_point mention
        for seg in self.segments:
            if any(q in (kp.lower()) for kp in seg.key_points):
                return seg
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "segmenter_provider": self.segmenter_provider,
            "segmenter_model": self.segmenter_model,
            "paper_char_length": self.paper_char_length,
            "generated_at": self.generated_at,
            "segments": [asdict(s) for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SemanticTOC:
        segs = [SegmentEntry(**s) for s in d.get("segments", [])]
        return cls(
            segments=segs,
            source=d.get("source", ""),
            segmenter_provider=d.get("segmenter_provider", ""),
            segmenter_model=d.get("segmenter_model", ""),
            paper_char_length=d.get("paper_char_length", 0),
            generated_at=d.get("generated_at", 0.0),
        )


# ---------------------------------------------------------------------------
# LaTeX section extraction (cheap path for long papers)
# ---------------------------------------------------------------------------


_SECTION_RE = re.compile(
    r"\\(section|subsection|subsubsection|chapter|part)\*?\{([^}]+)\}"
)


def _latex_sections_to_toc(text: str) -> SemanticTOC | None:
    """Build a TOC from native ``\\section{}`` markers. Returns None if
    the paper has fewer than 3 sections (structure too thin to be useful)."""
    matches = list(_SECTION_RE.finditer(text))
    if len(matches) < 3:
        return None
    segments: list[SegmentEntry] = []
    for i, m in enumerate(matches):
        title = re.sub(r"\s+", " ", m.group(2)).strip()
        label = re.sub(r"[^\w]+", "_", title.lower()).strip("_")[:60]
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segments.append(
            SegmentEntry(
                label=label or f"section_{i+1}",
                title=title,
                start_offset=start,
                end_offset=end,
                key_points=[],
            )
        )
    return SemanticTOC(
        segments=segments,
        source="latex_sections",
        paper_char_length=len(text),
        generated_at=time.time(),
    )


# ---------------------------------------------------------------------------
# LLM-based semantic segmentation
# ---------------------------------------------------------------------------


_SEGMENTER_SYSTEM = """\
You are segmenting a HEP experimental paper (e.g. from ATLAS or CMS)
into meaningful logical sections that a physicist would use when citing
the paper. The input may lack explicit `\\section{}` markers (e.g. PRL
short papers) — your job is to identify the semantic segments.

Produce a JSON array of 6-12 entries, each describing one segment of
the paper in reading order. The output MUST be a JSON array only, no
preamble.

For each segment provide:
  - `label`: a short snake_case identifier (e.g. "intro_motivation",
    "event_selection", "signal_region", "syst_uncertainty",
    "stat_analysis", "results_limits", "summary").
  - `title`: a human-readable title (6 words max, English, physics
    terminology preserved as in paper).
  - `anchor_phrase`: a unique verbatim ~40-100 character substring
    that occurs at or near the START of that segment in the paper text
    (used to locate the segment's offset reliably).
  - `key_points`: 2-3 short bullet strings summarising what the segment
    contains.

Your segments MUST collectively cover the paper from start to end in
reading order with no gaps. Do NOT segment individual tables or
figures — they are content INSIDE a segment.

The segmentation schema should match typical HEP analysis structure:
introduction & motivation → samples & MC → event selection →
categorisation / signal regions → background estimation →
systematic uncertainties → statistical framework & results →
summary / conclusions. Not every paper has all of these; include only
those that apply.

Example output shape:

```json
[
  {
    "label": "intro_motivation",
    "title": "Introduction and physics motivation",
    "anchor_phrase": "The Standard Model (SM) of particle physics",
    "key_points": ["LNV motivation", "prior constraints", "VBF topology"]
  },
  {
    "label": "samples_and_mc",
    "title": "Data samples and MC simulation",
    "anchor_phrase": "Events are collected with single-muon triggers",
    "key_points": ["138 fb^-1 Run 2 data", "MadGraph+Pythia", "PDF sets"]
  }
]
```
"""


_SEGMENTER_USER_TEMPLATE = """\
Segment the paper below. Return only the JSON array.

Paper title: {title}
arXiv id: {arxiv_id}

=== Paper text (possibly truncated) ===
{body}
"""


def _parse_segments_json(raw: str) -> list[dict[str, Any]] | None:
    """Extract a JSON array from the LLM's output (fenced or bare)."""
    m = re.search(r"```(?:json)?\s*(\[[\s\S]+?\])\s*```", raw)
    candidates: list[str] = []
    if m:
        candidates.append(m.group(1))
    # Bare array: first '[' to last ']'
    first = raw.find("[")
    last = raw.rfind("]")
    if first >= 0 and last > first:
        candidates.append(raw[first : last + 1])

    for c in candidates:
        try:
            out = json.loads(c)
            if isinstance(out, list):
                return out
        except json.JSONDecodeError:
            continue
    return None


def _offset_of_anchor(text: str, anchor: str) -> int | None:
    """Find the offset of the anchor phrase in ``text``. Uses case-insensitive
    match with whitespace normalisation."""
    if not anchor:
        return None
    # Normalise whitespace in both.
    t_norm = re.sub(r"\s+", " ", text)
    a_norm = re.sub(r"\s+", " ", anchor).strip()

    idx = t_norm.lower().find(a_norm.lower())
    if idx < 0:
        # Try first 40 chars only (anchor may have minor paraphrase)
        short = a_norm[: min(40, len(a_norm))]
        if len(short) >= 15:
            idx = t_norm.lower().find(short.lower())
            if idx < 0:
                return None
        else:
            return None

    # Map back to the original text offset by counting non-whitespace
    # characters up to idx in t_norm (approximate but good enough).
    return _unnormalise_offset(text, idx)


def _unnormalise_offset(orig: str, norm_offset: int) -> int:
    """Convert an offset in whitespace-normalised text to the original."""
    count = 0
    i = 0
    while i < len(orig) and count < norm_offset:
        if orig[i].isspace():
            # Collapse runs of whitespace to one in normalised form.
            j = i
            while j < len(orig) and orig[j].isspace():
                j += 1
            # One logical whitespace char consumed.
            count += 1
            i = j
        else:
            count += 1
            i += 1
    return i


def _llm_segment(
    text: str,
    title: str,
    arxiv_id: str,
    provider: Provider,
    max_body_chars: int = 50_000,
    **provider_kwargs: Any,
) -> SemanticTOC | None:
    """Call the provider once to produce a semantic TOC."""

    body = text[:max_body_chars]
    if len(text) > max_body_chars:
        # Include both head and tail so the segmenter can see the
        # summary / conclusions section too.
        tail = text[-8000:]
        body = body + "\n\n[... mid-paper truncated ...]\n\n" + tail

    messages = [
        Message(
            role="user",
            content=_SEGMENTER_USER_TEMPLATE.format(
                title=title or "(unknown)",
                arxiv_id=arxiv_id or "(unknown)",
                body=body,
            ),
        )
    ]

    # Lower reasoning effort for OpenAI to save tokens on a one-shot task.
    provider_kwargs.setdefault("reasoning_effort", "low")
    provider_kwargs.setdefault("summary", "concise")

    text_buf: list[str] = []
    try:
        for ev in provider.stream_turn(
            messages=messages,
            tools=None,
            system=_SEGMENTER_SYSTEM,
            **provider_kwargs,
        ):
            if isinstance(ev, TextDelta):
                text_buf.append(ev.text)
            # ignore thinking + end for this one-shot
    except Exception as e:  # pragma: no cover
        print(f"[segmenter] provider call failed: {type(e).__name__}: {e}")
        return None

    raw = "".join(text_buf).strip()
    if not raw:
        print("[segmenter] provider returned empty text")
        return None

    parsed = _parse_segments_json(raw)
    if not parsed:
        print("[segmenter] could not parse JSON array from output")
        return None

    # Resolve anchors to offsets.
    segs: list[SegmentEntry] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        label = (item.get("label") or f"section_{i+1}").strip()
        title_ = (item.get("title") or label).strip()
        anchor = (item.get("anchor_phrase") or "").strip()
        kp = item.get("key_points") or []
        if not isinstance(kp, list):
            kp = []
        offset = _offset_of_anchor(text, anchor)
        if offset is None:
            # Skip segments whose anchor we cannot locate — they'd
            # destabilise the TOC.
            print(f"[segmenter] dropping segment '{label}' (anchor not found)")
            continue
        segs.append(
            SegmentEntry(
                label=label,
                title=title_,
                start_offset=offset,
                end_offset=len(text),  # provisional; fixed below
                key_points=[str(x) for x in kp if x],
            )
        )

    if not segs:
        return None

    # Sort by offset and set each segment's end to next segment's start.
    segs.sort(key=lambda s: s.start_offset)
    for i in range(len(segs) - 1):
        segs[i].end_offset = segs[i + 1].start_offset
    segs[-1].end_offset = len(text)

    return SemanticTOC(
        segments=segs,
        source="semantic_llm",
        segmenter_provider=provider.name,
        segmenter_model=getattr(provider, "model", ""),
        paper_char_length=len(text),
        generated_at=time.time(),
    )


# ---------------------------------------------------------------------------
# Public entry: segment_paper (with cache + fallbacks)
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: str, arxiv_id: str) -> Path:
    safe = (arxiv_id or "unknown").replace("/", "_")
    return Path(cache_dir) / f"{safe}_semantic_toc.json"


def _load_cache(cache_dir: str, arxiv_id: str) -> SemanticTOC | None:
    p = _cache_path(cache_dir, arxiv_id)
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        toc = SemanticTOC.from_dict(data)
        toc.source = "cache"
        return toc
    except (OSError, json.JSONDecodeError):
        return None


def _save_cache(cache_dir: str, arxiv_id: str, toc: SemanticTOC) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    p = _cache_path(cache_dir, arxiv_id)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(toc.to_dict(), f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def segment_paper(
    paper_text: str,
    arxiv_id: str,
    title: str = "",
    cache_dir: str = "./paper_cache",
    segmenter_provider: Provider | None = None,
    force_refresh: bool = False,
    **provider_kwargs: Any,
) -> SemanticTOC | None:
    """Produce a semantic TOC for the given paper text.

    Priority:
      1. Cached TOC at ``paper_cache/<arxiv_id>_semantic_toc.json``
      2. LaTeX ``\\section{}`` extraction (if the paper has ≥3 sections)
      3. LLM semantic segmentation (requires ``segmenter_provider``)
      4. None (callers should gracefully degrade)
    """

    if not paper_text or not paper_text.strip():
        return None

    if not force_refresh:
        cached = _load_cache(cache_dir, arxiv_id)
        if cached is not None and cached.segments:
            return cached

    # Try native LaTeX first — cheap and accurate when available.
    latex_toc = _latex_sections_to_toc(paper_text)
    if latex_toc is not None and len(latex_toc.segments) >= 3:
        _save_cache(cache_dir, arxiv_id, latex_toc)
        return latex_toc

    if segmenter_provider is None:
        print(
            "[segmenter] no provider available for LLM segmentation; "
            "falling back to no TOC"
        )
        return None

    print(
        f"[segmenter] running LLM segmentation with provider "
        f"'{segmenter_provider.name}' ({getattr(segmenter_provider, 'model', '?')})..."
    )
    t0 = time.time()
    toc = _llm_segment(
        paper_text,
        title=title,
        arxiv_id=arxiv_id,
        provider=segmenter_provider,
        **provider_kwargs,
    )
    dt = time.time() - t0
    if toc is None:
        print(f"[segmenter] failed after {dt:.1f}s")
        return None
    print(
        f"[segmenter] produced {len(toc.segments)} segments in {dt:.1f}s: "
        + ", ".join(s.label for s in toc.segments)
    )
    _save_cache(cache_dir, arxiv_id, toc)
    return toc
