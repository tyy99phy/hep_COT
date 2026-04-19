"""Tests for the offline paper_tools using the cached 2206.08956 paper."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hep_cot.tools.paper_tools import PaperContext, build_paper_tools

HEP_COT_ROOT = Path(__file__).resolve().parents[1]
CACHE = HEP_COT_ROOT / "paper_cache"
FIXTURE_PAPER = "2206.08956"


def _cached_source_available() -> bool:
    return (CACHE / f"{FIXTURE_PAPER}_source").is_dir()


@pytest.mark.skipif(
    not _cached_source_available(),
    reason=f"cached LaTeX source for {FIXTURE_PAPER} not present",
)
def test_paper_tools_load_and_search_cached():
    ctx = PaperContext(cache_dir=str(CACHE))
    tools = {t.name: t for t in build_paper_tools(ctx)}

    # The LaTeX source is already extracted; fetch_paper should recover
    # tex_content from it without hitting the network.
    info_result = tools["fetch_paper"].func(identifier=FIXTURE_PAPER)
    assert info_result.get("arxiv_id")
    assert info_result.get("text_length_chars", 0) > 1000

    # Search for something definitely in the paper text.
    search_out = tools["search_text"].func(query="selection", max_results=3)
    assert search_out["total_hits"] >= 1
    assert all("snippet" in h for h in search_out["hits"])

    # Section extraction.
    section_out = tools["get_paper_section"].func(section_pattern="Event")
    if "error" not in section_out:
        assert section_out["body"]
