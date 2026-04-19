"""Aggregate gap metrics across a protocol-driven study run."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .figure_coverage import compute_coverage
from .phase_judge import PhaseJudgeReport, judge_phase
from .phases import PROTOCOL_PHASES, get_phase


def _hepdata_used(result: dict[str, Any]) -> bool:
    """Did this phase result use any HEPData tool?"""
    tools = (
        (result.get("answer_parsed") or {}).get("tools_used_this_phase") or []
    )
    for t in tools:
        if isinstance(t, str) and "hepdata" in t.lower():
            return True
    # Also scan the recorded tool_calls at the outer level
    for tc in result.get("tool_calls") or []:
        name = (tc.get("name") or "") if isinstance(tc, dict) else ""
        if "hepdata" in name.lower():
            return True
    return False


def _citations_of(result: dict[str, Any]) -> set[str]:
    parsed = result.get("answer_parsed") or {}
    return set(str(c) for c in (parsed.get("citations") or []) if c)


def _phase_continuity_score(results: list[dict[str, Any]]) -> float:
    """Fraction of phase-1-to-5 citations that reappear in phase 6 (assessment).

    A rough proxy for 'does the final assessment draw on earlier
    findings'. Ranges 0..1; higher means more continuity.
    """
    if not results:
        return 0.0
    by_key = {r.get("question_id"): r for r in results}
    assessment = by_key.get("assessment")
    if not assessment:
        return 0.0
    tail_cits = _citations_of(assessment)
    if not tail_cits:
        return 0.0

    earlier_cits: set[str] = set()
    for key in ("overview", "motivation", "strategy", "syst", "stat_result"):
        r = by_key.get(key)
        if r:
            earlier_cits |= _citations_of(r)
    if not earlier_cits:
        return 0.0
    overlap = tail_cits & earlier_cits
    return round(len(overlap) / len(tail_cits), 3)


def _collect_figures_rate(
    phase_results: list[dict[str, Any]], total_figures: int
) -> float:
    rpt = compute_coverage(phase_results, total_figures)
    return rpt.coverage_rate


def _hepdata_utilization(phase_results: list[dict[str, Any]]) -> float:
    """Fraction of hepdata-hint phases that actually called a HEPData tool."""
    hint_phases = [p.key for p in PROTOCOL_PHASES if p.hepdata_hint]
    if not hint_phases:
        return 1.0
    hit = 0
    for key in hint_phases:
        match = next(
            (r for r in phase_results if r.get("question_id") == key), None
        )
        if match and _hepdata_used(match):
            hit += 1
    return round(hit / len(hint_phases), 3)


def aggregate_protocol_metrics(
    paper_identifier: str,
    teacher_results: list[dict[str, Any]],
    student_native_results: list[dict[str, Any]],
    student_aligned_results: list[dict[str, Any]] | None,
    per_phase_aligned_verdicts: list[PhaseJudgeReport] | None,
    total_figures: int,
) -> dict[str, Any]:
    """Build a comprehensive protocol-run metrics dict."""

    teacher_by_key = {r.get("question_id"): r for r in teacher_results}
    native_by_key = {r.get("question_id"): r for r in student_native_results}
    aligned_by_key = (
        {r.get("question_id"): r for r in (student_aligned_results or [])}
        if student_aligned_results
        else {}
    )
    aligned_verdict_by_key: dict[str, PhaseJudgeReport] = {}
    if per_phase_aligned_verdicts:
        for v in per_phase_aligned_verdicts:
            aligned_verdict_by_key[v.phase_key] = v

    per_phase: list[dict[str, Any]] = []
    native_pass_count = 0
    aligned_pass_count = 0
    phases_with_verdict = 0

    for phase in PROTOCOL_PHASES:
        t = teacher_by_key.get(phase.key)
        n = native_by_key.get(phase.key)
        a = aligned_by_key.get(phase.key)

        if t is None or n is None:
            per_phase.append(
                {
                    "phase_key": phase.key,
                    "skipped": True,
                    "reason": "teacher or native result missing",
                }
            )
            continue

        native_verdict = judge_phase(phase, t, n)
        aligned_verdict = aligned_verdict_by_key.get(phase.key)
        if aligned_verdict is None and a is not None:
            aligned_verdict = judge_phase(phase, t, a)

        per_phase.append(
            {
                "phase_key": phase.key,
                "native": native_verdict.to_dict(),
                "aligned": aligned_verdict.to_dict() if aligned_verdict else None,
            }
        )
        phases_with_verdict += 1
        if native_verdict.overall_pass:
            native_pass_count += 1
        if aligned_verdict and aligned_verdict.overall_pass:
            aligned_pass_count += 1

    teacher_fig_rate = _collect_figures_rate(teacher_results, total_figures)
    native_fig_rate = _collect_figures_rate(student_native_results, total_figures)
    aligned_fig_rate = (
        _collect_figures_rate(student_aligned_results, total_figures)
        if student_aligned_results
        else 0.0
    )

    return {
        "generated_at": time.time(),
        "paper_identifier": paper_identifier,
        "totals": {
            "phases_judged": phases_with_verdict,
            "native_overall_pass": native_pass_count,
            "native_pass_rate": round(
                native_pass_count / max(1, phases_with_verdict), 3
            ),
            "aligned_overall_pass": aligned_pass_count,
            "aligned_pass_rate": round(
                aligned_pass_count / max(1, phases_with_verdict), 3
            ),
        },
        "figure_coverage_rate": {
            "teacher": teacher_fig_rate,
            "student_native": native_fig_rate,
            "student_aligned": aligned_fig_rate,
            "total_figures": total_figures,
        },
        "hepdata_utilization_rate": {
            "teacher": _hepdata_utilization(teacher_results),
            "student_native": _hepdata_utilization(student_native_results),
            "student_aligned": _hepdata_utilization(
                student_aligned_results or []
            ),
        },
        "phase_continuity": {
            "teacher": _phase_continuity_score(teacher_results),
            "student_native": _phase_continuity_score(student_native_results),
            "student_aligned": _phase_continuity_score(
                student_aligned_results or []
            ),
        },
        "per_phase": per_phase,
    }


def save_metrics(
    metrics: dict[str, Any], output_dir: str, paper_identifier: str
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = (
        Path(output_dir)
        / f"metrics_{paper_identifier.replace('/', '_')}_{int(time.time())}.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return str(path)


# Backwards-compat shim used by older callers (smoke tests etc).
def aggregate_metrics(**kwargs: Any) -> dict[str, Any]:
    """Deprecated flat-QA aggregator. Protocol users should call
    :func:`aggregate_protocol_metrics` directly."""
    return kwargs  # noqa: RET504  (keeps shape for old tests)
