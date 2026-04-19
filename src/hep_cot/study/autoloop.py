"""Phase-level autoloop: redo individual failing phases until L1+L4 pass.

Unlike the earlier flat-QA autoloop, the protocol autoloop only redoes
the *phase that failed*, preserving the summaries of earlier phases
that already converged. Each redo is saved separately as
``phase_NN_<key>_roundR.json`` so the trajectory is auditable.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..llm.base import build_provider
from ..prompts import STUDENT_SYSTEM_PROMPT
from .phase_judge import PhaseJudgeReport, build_phase_feedback, judge_phase
from .phases import PROTOCOL_PHASES
from .protocol import Phase6ProtocolRunner, extract_phase_summary, make_runner


def _verdict_score(v: PhaseJudgeReport) -> tuple[int, int, int, float]:
    """Rank a verdict — higher is better.

    Used by the autoloop to pick the **best** round instead of blindly
    accepting the last one (which can regress if a retry produces an
    empty / flaky submit). Ordering:
    1. overall_pass (1 > 0)
    2. number of sub-gates that passed (L1 + L4)
    3. L4 required-keys populated
    4. L1 number-hit + citation-hit rates (negatives clipped to 0)
    """

    return (
        1 if v.overall_pass else 0,
        (1 if v.l1_pass else 0) + (1 if v.l4_pass else 0),
        v.l4_required_keys_present,
        max(0.0, v.l1_number_hit_rate) + max(0.0, v.l1_citation_hit_rate),
    )


def run_protocol_autoloop(
    paper_identifier: str,
    teacher_results: list[dict[str, Any]],
    provider_name: str = "deepseek",
    model: str | None = None,
    cache_dir: str = "./paper_cache",
    output_dir: str = "./study_sessions",
    max_rounds_per_phase: int = 3,
    max_tool_iterations: int = 10,
    paper_text_truncate: int = 60_000,
    segmenter_provider: Any = None,
    skip_segmentation: bool = False,
) -> tuple[list[dict[str, Any]], list[PhaseJudgeReport], Path, dict[str, Any] | None]:
    """Produce the aligned student trace with per-phase feedback rounds.

    Returns ``(final_results_per_phase, per_phase_verdicts, run_dir, paper_info)``.
    ``final_results_per_phase`` is the best-scoring round for each phase
    (native if it already passed, otherwise the highest-ranked round by
    :func:`_verdict_score`).
    """

    # Index teacher results by phase key.
    teacher_by_phase: dict[str, dict[str, Any]] = {}
    for tr in teacher_results:
        qid = tr.get("question_id") or ""
        teacher_by_phase[qid] = tr

    provider_kwargs: dict[str, Any] = {}
    if model:
        provider_kwargs["model"] = model
    provider = build_provider(provider_name, **provider_kwargs)

    stamp = int(time.time())
    run_dir = (
        Path(output_dir)
        / f"student_aligned_{paper_identifier.replace('/', '_')}_{stamp}"
    )
    runner: Phase6ProtocolRunner
    runner, registry, paper_info = make_runner(
        provider=provider,
        paper_identifier=paper_identifier,
        base_system_prompt=STUDENT_SYSTEM_PROMPT,
        output_dir=run_dir,
        label="student_aligned",
        cache_dir=cache_dir,
        max_tool_iterations=max_tool_iterations,
        paper_text_truncate=paper_text_truncate,
        segmenter_provider=segmenter_provider,
        skip_segmentation=skip_segmentation,
    )

    per_phase_verdicts: list[PhaseJudgeReport] = []
    accepted_results: list[dict[str, Any]] = []

    for i, phase in enumerate(PROTOCOL_PHASES, start=1):
        print(f"  [student_aligned] phase {i}/6 · {phase.name}")

        # Round 0: native-style first attempt (no feedback)
        native_dict, _qr = runner.run_phase(phase)
        runner._save_phase(i, phase.key, native_dict)  # noqa: SLF001

        teacher_phase = teacher_by_phase.get(phase.key)
        if teacher_phase is None:
            print(f"    ⚠ no teacher reference for phase '{phase.key}', accepting native")
            accepted = native_dict
            verdict = PhaseJudgeReport(phase_key=phase.key)
            verdict.notes.append("no teacher reference available")
        else:
            verdict = judge_phase(phase, teacher_phase, native_dict)
            # Track the best-scoring (result, verdict) across all rounds.
            # Round 0 (native-style, no feedback) is the initial candidate;
            # subsequent feedback rounds only replace it when strictly
            # better by :func:`_verdict_score`. This guards against a
            # flaky late round (e.g. DeepSeek submitting an empty payload
            # under a transient connection error) erasing an earlier good
            # result.
            best_verdict = verdict
            best_accepted = native_dict
            best_round = 0

            rounds = 0
            while not best_verdict.overall_pass and rounds < max_rounds_per_phase:
                rounds += 1
                print(
                    f"    round {rounds}: L1={verdict.l1_pass} L4={verdict.l4_pass} "
                    f"→ retrying with feedback"
                )
                feedback = build_phase_feedback(phase, best_verdict, rounds)
                redo_dict, _qr = runner.run_phase(phase, extra_user_prefix=feedback)
                runner._save_phase(i, phase.key, redo_dict, round_number=rounds)  # noqa: SLF001
                verdict = judge_phase(phase, teacher_phase, redo_dict)
                if _verdict_score(verdict) > _verdict_score(best_verdict):
                    best_verdict = verdict
                    best_accepted = redo_dict
                    best_round = rounds

            accepted = best_accepted
            verdict = best_verdict
            # Overwrite the FINAL file (phase_NN_<key>.json, saved at
            # round 0) with whichever round we actually accepted, so the
            # on-disk FINAL matches downstream metrics + ANALYSIS.md.
            runner._save_phase(i, phase.key, accepted)  # noqa: SLF001
            print(
                f"    final: L1={verdict.l1_pass} L4={verdict.l4_pass} "
                f"overall={verdict.overall_pass} rounds={rounds} "
                f"accepted_from_round={best_round}"
            )

        per_phase_verdicts.append(verdict)
        accepted_results.append(accepted)

        # Only the accepted result feeds forward into prior_summaries.
        summary = extract_phase_summary(phase, accepted)
        # Replace the last prior_summary (runner may have appended the
        # native version during run_phase) with the accepted one, then
        # ensure we have exactly i entries.
        runner.prior_summaries = runner.prior_summaries[: i - 1]
        runner.prior_summaries.append(summary)
        # runner.phase_results mirrors prior_summaries semantically for
        # downstream figure coverage.
        runner.phase_results = accepted_results[:]

        if phase.figure_coverage_after:
            runner._run_figure_coverage_forced_pass(phase_index=i)  # noqa: SLF001
            # Whatever was just generated by the forced pass lives in
            # runner.phase_results; make sure accepted_results stays in
            # sync (it already appends into runner.phase_results via
            # _run_figure_coverage_forced_pass).
            accepted_results = runner.phase_results[:]

    runner._save_manifest()  # noqa: SLF001
    runner._write_analysis_markdown()  # noqa: SLF001
    print(f"  student aligned run dir: {run_dir}")
    return accepted_results, per_phase_verdicts, run_dir, paper_info
