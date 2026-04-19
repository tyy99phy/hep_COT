"""Teacher runner: run the full 6-phase protocol with GPT-5.4 (xhigh)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..llm.base import build_provider
from ..prompts import TEACHER_SYSTEM_PROMPT
from .protocol import make_runner


def run_teacher_protocol(
    paper_identifier: str,
    provider_name: str = "openai",
    model: str | None = None,
    reasoning_effort: str = "xhigh",
    cache_dir: str = "./paper_cache",
    output_dir: str = "./study_sessions",
    max_tool_iterations: int = 10,
    paper_text_truncate: int = 60_000,
    segmenter_provider: Any = None,
    skip_segmentation: bool = False,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any] | None]:
    """Run the teacher (GPT-5.4 xhigh) across all 6 phases.

    Returns ``(phase_results_list, run_dir, paper_info)``.
    """
    provider_kwargs: dict[str, Any] = {"reasoning_effort": reasoning_effort}
    if model:
        provider_kwargs["model"] = model
    provider = build_provider(provider_name, **provider_kwargs)

    stamp = int(time.time())
    run_dir = (
        Path(output_dir)
        / f"teacher_{paper_identifier.replace('/', '_')}_{stamp}"
    )
    runner, _reg, paper_info = make_runner(
        provider=provider,
        paper_identifier=paper_identifier,
        base_system_prompt=TEACHER_SYSTEM_PROMPT,
        output_dir=run_dir,
        label="teacher",
        cache_dir=cache_dir,
        max_tool_iterations=max_tool_iterations,
        paper_text_truncate=paper_text_truncate,
        segmenter_provider=segmenter_provider,
        skip_segmentation=skip_segmentation,
    )
    results = runner.run_all()
    print(f"  teacher run dir: {run_dir}")
    return results, run_dir, paper_info
