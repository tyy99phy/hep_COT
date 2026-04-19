"""Re-run student_aligned (autoloop) for R4 paper, reusing teacher + native from R4.

Outputs to study_sessions_round5/. Validates the two patches:
  1. autoloop best-of-rounds selection + FINAL sync
  2. protocol._submit empty-findings rejection
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from hep_cot.cli import _load_phase_results
from hep_cot.llm.base import build_provider
from hep_cot.study import (
    aggregate_protocol_metrics,
    build_hep_registry,
    run_protocol_autoloop,
    save_metrics,
    total_figure_count,
)

PAPER = "2206.08956"
# R4 was merged into R5 as the source of teacher+native; R5 is now self-contained.
OUT_DIR = Path("/home/tyyang/hep_COT/study_sessions_round5")
TEACHER_DIR = OUT_DIR / "teacher_2206.08956_1776507832"
NATIVE_DIR = OUT_DIR / "student_native_2206.08956_1776509738"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"=== R5 student_aligned rerun ({PAPER}) ===", flush=True)
    print(f"  teacher reuse:  {TEACHER_DIR}", flush=True)
    print(f"  native reuse:   {NATIVE_DIR}", flush=True)
    print(f"  output dir:     {OUT_DIR}", flush=True)

    teacher_results = _load_phase_results(str(TEACHER_DIR))
    native_results = _load_phase_results(str(NATIVE_DIR))
    print(
        f"  loaded teacher={len(teacher_results)} phases, "
        f"native={len(native_results)} phases",
        flush=True,
    )

    segmenter = build_provider("openai", reasoning_effort="low")
    print("  segmenter: openai (low) — toc likely cached, cheap fallback", flush=True)

    print(
        "\n=== student autoloop: deepseek (max_rounds_per_phase=3) ===",
        flush=True,
    )
    aligned_results, aligned_verdicts, aligned_dir, _paper_info = run_protocol_autoloop(
        paper_identifier=PAPER,
        teacher_results=teacher_results,
        provider_name="deepseek",
        cache_dir="/home/tyyang/hep_COT/paper_cache",
        output_dir=str(OUT_DIR),
        max_rounds_per_phase=3,
        max_tool_iterations=10,
        segmenter_provider=segmenter,
        skip_segmentation=False,
    )
    print(f"\n  aligned run dir: {aligned_dir}", flush=True)

    # Metrics: compare teacher vs native vs aligned (fresh with new autoloop).
    reg, _ctx, _info = build_hep_registry(PAPER, "/home/tyyang/hep_COT/paper_cache")
    total_figures = total_figure_count(reg)
    metrics = aggregate_protocol_metrics(
        paper_identifier=PAPER,
        teacher_results=teacher_results,
        student_native_results=native_results,
        student_aligned_results=aligned_results,
        per_phase_aligned_verdicts=aligned_verdicts,
        total_figures=total_figures,
    )
    metrics_path = save_metrics(metrics, str(OUT_DIR), PAPER)
    print(f"  metrics: {metrics_path}", flush=True)

    # Quick regression summary.
    print("\n=== patch-fix regression summary ===", flush=True)
    phase_json_dir = Path(aligned_dir)
    for phase_file in sorted(phase_json_dir.glob("phase_??_*.json")):
        if "_round" in phase_file.stem:
            continue
        with open(phase_file, encoding="utf-8") as f:
            d = json.load(f)
        ans = d.get("answer_parsed") or {}
        findings = ans.get("findings") or {}
        nonempty = sum(
            1 for v in findings.values() if isinstance(v, str) and v.strip()
        )
        total = len(findings)
        print(
            f"  {phase_file.name:38s}  findings={nonempty}/{total}  "
            f"rounds={d.get('autoloop_rounds', 0)}",
            flush=True,
        )

    print(f"\n=== done (elapsed {time.time() - t0:.0f}s) ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
