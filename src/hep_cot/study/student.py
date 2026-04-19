"""Student native-pass runner: DeepSeek-reasoner without feedback.

Runs the full 6-phase analysis protocol with the STUDENT persona and
no per-phase autoloop corrections — this produces the **native** trace
that characterises DeepSeek's unassisted reasoning. The aligned trace
(with feedback) is produced separately by :mod:`hep_cot.study.autoloop`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..llm.base import build_provider
from ..prompts import STUDENT_SYSTEM_PROMPT
from .protocol import make_runner


def run_student_protocol_native(
    paper_identifier: str,
    provider_name: str = "deepseek",
    model: str | None = None,
    cache_dir: str = "./paper_cache",
    output_dir: str = "./study_sessions",
    max_tool_iterations: int = 10,
    paper_text_truncate: int = 60_000,
    segmenter_provider: Any = None,
    skip_segmentation: bool = False,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any] | None]:
    provider_kwargs: dict[str, Any] = {}
    if model:
        provider_kwargs["model"] = model
    provider = build_provider(provider_name, **provider_kwargs)

    stamp = int(time.time())
    run_dir = (
        Path(output_dir)
        / f"student_native_{paper_identifier.replace('/', '_')}_{stamp}"
    )
    runner, _reg, paper_info = make_runner(
        provider=provider,
        paper_identifier=paper_identifier,
        base_system_prompt=STUDENT_SYSTEM_PROMPT,
        output_dir=run_dir,
        label="student_native",
        cache_dir=cache_dir,
        max_tool_iterations=max_tool_iterations,
        paper_text_truncate=paper_text_truncate,
        segmenter_provider=segmenter_provider,
        skip_segmentation=skip_segmentation,
    )
    results = runner.run_all()
    print(f"  student native run dir: {run_dir}")
    return results, run_dir, paper_info
