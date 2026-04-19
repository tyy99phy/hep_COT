"""CLI for hep-copilot / hep-cot.

New (MVP-1) subcommands:
  * ``paper <arxiv_id>``   — interactive REPL around one paper
  * ``ask <arxiv_id> <q>`` — one-shot single question, print structured answer
  * ``study <arxiv_id>``   — comparative reasoning study (teacher + student)

Legacy Codex-based subcommands (preserved for backwards compat):
  * ``legacy-chat``, ``legacy-paper``, ``legacy-batch``, ``legacy-export``
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="hep-copilot",
        description="HEP paper literature reasoning copilot",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- new framework --------------------------------------------------

    paper_p = subparsers.add_parser(
        "paper", help="Interactive REPL around a single HEP paper"
    )
    _add_copilot_args(paper_p)
    paper_p.add_argument(
        "identifier",
        nargs="?",
        default=None,
        help="arXiv id / URL or INSPIRE id / URL. Optional — can load later.",
    )

    ask_p = subparsers.add_parser(
        "ask", help="One-shot question about a paper (non-interactive)"
    )
    _add_copilot_args(ask_p)
    ask_p.add_argument("identifier", help="Paper identifier (arxiv/inspire)")
    ask_p.add_argument("question", help="The question to ask")

    study_p = subparsers.add_parser(
        "study",
        help="Protocol-driven comparative reasoning study: teacher (GPT) vs "
        "student (DeepSeek) across 6 analysis phases",
    )
    study_p.add_argument("identifier", help="Paper identifier (arxiv/inspire)")
    study_p.add_argument(
        "--teacher",
        default="openai",
        help="Teacher provider name (default: openai)",
    )
    study_p.add_argument(
        "--student",
        default="deepseek",
        help="Student provider name (default: deepseek)",
    )
    study_p.add_argument(
        "--teacher-effort",
        default="xhigh",
        help="Reasoning effort for the teacher provider (default: xhigh)",
    )
    study_p.add_argument(
        "--autoloop",
        action="store_true",
        help="Run per-phase autoloop refinement of the student",
    )
    study_p.add_argument(
        "--max-rounds-per-phase",
        type=int,
        default=3,
        help="Max autoloop rounds per failing phase (default: 3)",
    )
    study_p.add_argument(
        "--max-tool-iterations",
        type=int,
        default=10,
        help="Max tool-call iterations per phase (default: 10)",
    )
    study_p.add_argument(
        "--paper-text-truncate",
        type=int,
        default=60_000,
        help="Truncate paper LaTeX text to this many characters (default: 60000)",
    )
    study_p.add_argument(
        "--segmenter",
        default="openai",
        help="Provider for one-time semantic paper segmentation (default: openai)",
    )
    study_p.add_argument(
        "--segmenter-effort",
        default="low",
        help="Reasoning effort for the segmenter provider (default: low — "
        "segmentation is a cheap one-shot task; high effort wastes tokens)",
    )
    study_p.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Disable semantic segmentation (fall back to LaTeX \\section "
        "extraction or '§(no-anchor)')",
    )
    study_p.add_argument(
        "--cache-dir", default="./paper_cache", help="Cache directory"
    )
    study_p.add_argument(
        "--output-dir", default="./study_sessions", help="Output directory"
    )
    study_p.add_argument(
        "--skip-teacher",
        action="store_true",
        help="Reuse existing teacher run (supply --teacher-run-dir)",
    )
    study_p.add_argument(
        "--teacher-run-dir",
        default=None,
        help="Existing teacher run dir to reuse (skip teacher pass)",
    )

    # ---- hep-copilot export (regenerate ANALYSIS.md/.pdf from phase JSONs) ----
    export_p = subparsers.add_parser(
        "export",
        help="Regenerate ANALYSIS.md and ANALYSIS.pdf from an existing run "
        "directory's phase_*.json files (no LLM calls)",
    )
    export_p.add_argument(
        "run_dir",
        help="Either a single run dir (teacher_*/student_native_*/...) OR a "
        "parent directory containing many such sub-runs — in which case all "
        "are exported.",
    )
    export_p.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Exclude the per-iteration reasoning callouts from output",
    )
    export_p.add_argument(
        "--no-pdf", action="store_true", help="Skip PDF generation"
    )

    # -- legacy Codex path ----------------------------------------------

    legacy_chat = subparsers.add_parser(
        "legacy-chat", help="(legacy) free-form Codex chat"
    )
    _add_legacy_args(legacy_chat)

    legacy_paper = subparsers.add_parser(
        "legacy-paper", help="(legacy) structured Codex-driven paper analysis"
    )
    legacy_paper.add_argument("identifier")
    _add_legacy_args(legacy_paper)
    legacy_paper.add_argument("--cache-dir", default="./paper_cache")

    legacy_batch = subparsers.add_parser(
        "legacy-batch", help="(legacy) non-interactive Codex batch mode"
    )
    legacy_batch.add_argument("task_file")
    _add_legacy_args(legacy_batch)

    legacy_export = subparsers.add_parser(
        "legacy-export", help="Export a session .json to Markdown"
    )
    legacy_export.add_argument("session_json")
    legacy_export.add_argument("-o", "--output")
    legacy_export.add_argument("--translate", action="store_true")
    legacy_export.add_argument("--translate-model", default="gpt-5.4-mini")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "paper":
        _run_paper(args)
    elif args.command == "ask":
        _run_ask(args)
    elif args.command == "study":
        _run_study(args)
    elif args.command == "export":
        _run_export_analysis(args)
    elif args.command == "legacy-chat":
        _run_legacy_chat(args)
    elif args.command == "legacy-paper":
        _run_legacy_paper(args)
    elif args.command == "legacy-batch":
        _run_legacy_batch(args)
    elif args.command == "legacy-export":
        _run_legacy_export(args)


# ---------------------------------------------------------------------------
# New-framework runners
# ---------------------------------------------------------------------------


def _add_copilot_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "gpt", "deepseek", "deepseek-reasoner"],
        help="LLM backend (default: openai)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model name (provider default if omitted)",
    )
    p.add_argument(
        "--effort",
        default=None,
        choices=["low", "medium", "high", "xhigh", "none"],
        help="Reasoning effort for OpenAI Responses (default: xhigh via ytyfree)",
    )
    p.add_argument("--cache-dir", default="./paper_cache")
    p.add_argument("--output-dir", default="./cot_sessions")
    p.add_argument(
        "--max-tool-iterations",
        type=int,
        default=8,
        help="Safety cap on tool calls per user round",
    )
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="Hide streaming thinking text in the terminal",
    )


def _run_paper(args: argparse.Namespace) -> None:
    from .session.copilot_repl import CopilotRepl

    repl = CopilotRepl(
        provider_name=args.provider,
        model=args.model,
        paper_identifier=args.identifier,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        max_tool_iterations=args.max_tool_iterations,
        show_thinking=not args.no_thinking,
        reasoning_effort=args.effort or "xhigh",
    )
    repl.run()


def _run_ask(args: argparse.Namespace) -> None:
    from .session.copilot_repl import CopilotRepl

    repl = CopilotRepl(
        provider_name=args.provider,
        model=args.model,
        paper_identifier=args.identifier,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        max_tool_iterations=args.max_tool_iterations,
        show_thinking=not args.no_thinking,
        reasoning_effort=args.effort or "xhigh",
    )
    # Minimal non-interactive round.
    print(f"=== hep-copilot ask ===")
    print(f"provider: {args.provider}    paper: {args.identifier}")

    # Pre-load paper silently.
    repl.registry.execute("fetch_paper", {"identifier": args.identifier})
    repl.loop.run_round(args.question)
    path = repl._save_session()  # noqa: SLF001
    print(f"\n[session saved] {path}")


def _run_study(args: argparse.Namespace) -> None:
    from .llm.base import build_provider
    from .study import (
        aggregate_protocol_metrics,
        run_protocol_autoloop,
        run_student_protocol_native,
        run_teacher_protocol,
        save_metrics,
        total_figure_count,
        build_hep_registry,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Build the segmenter provider (shared between teacher and student so
    # both see the same semantic TOC — critical for fair comparison).
    segmenter_provider = None
    if not args.skip_segmentation:
        try:
            segmenter_provider = build_provider(
                args.segmenter,
                reasoning_effort=args.segmenter_effort,
            )
            print(
                f"[segmenter] provider={args.segmenter} "
                f"effort={args.segmenter_effort} "
                f"(shared between teacher and student)"
            )
        except Exception as e:
            print(
                f"[segmenter] could not build provider '{args.segmenter}': "
                f"{type(e).__name__}: {e}. Continuing without segmentation."
            )
            segmenter_provider = None

    # ------------------------------------------------------------------
    # Teacher pass
    # ------------------------------------------------------------------
    if args.skip_teacher and args.teacher_run_dir:
        print(f"=== teacher: reusing run dir {args.teacher_run_dir} ===")
        teacher_results = _load_phase_results(args.teacher_run_dir)
        teacher_run_dir = args.teacher_run_dir
    else:
        print(f"=== teacher: {args.teacher} (effort={args.teacher_effort}) ===")
        teacher_results, teacher_run_dir, _info = run_teacher_protocol(
            paper_identifier=args.identifier,
            provider_name=args.teacher,
            reasoning_effort=args.teacher_effort,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            max_tool_iterations=args.max_tool_iterations,
            paper_text_truncate=args.paper_text_truncate,
            segmenter_provider=segmenter_provider,
            skip_segmentation=args.skip_segmentation,
        )

    # ------------------------------------------------------------------
    # Student native pass
    # ------------------------------------------------------------------
    print(f"\n=== student native: {args.student} ===")
    native_results, native_run_dir, _info2 = run_student_protocol_native(
        paper_identifier=args.identifier,
        provider_name=args.student,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        max_tool_iterations=args.max_tool_iterations,
        paper_text_truncate=args.paper_text_truncate,
        segmenter_provider=segmenter_provider,
        skip_segmentation=args.skip_segmentation,
    )

    # ------------------------------------------------------------------
    # Student autoloop pass (optional)
    # ------------------------------------------------------------------
    aligned_results: list = []
    aligned_verdicts: list = []
    if args.autoloop:
        print(
            f"\n=== student autoloop: {args.student} "
            f"(max_rounds_per_phase={args.max_rounds_per_phase}) ==="
        )
        aligned_results, aligned_verdicts, _, _ = run_protocol_autoloop(
            paper_identifier=args.identifier,
            teacher_results=teacher_results,
            provider_name=args.student,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            max_rounds_per_phase=args.max_rounds_per_phase,
            max_tool_iterations=args.max_tool_iterations,
            paper_text_truncate=args.paper_text_truncate,
            segmenter_provider=segmenter_provider,
            skip_segmentation=args.skip_segmentation,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    # Re-derive total figures via a fresh registry on the same paper.
    _reg, _ctx, _info = build_hep_registry(args.identifier, args.cache_dir)
    total_figures = total_figure_count(_reg)

    metrics = aggregate_protocol_metrics(
        paper_identifier=args.identifier,
        teacher_results=teacher_results,
        student_native_results=native_results,
        student_aligned_results=aligned_results if args.autoloop else None,
        per_phase_aligned_verdicts=aligned_verdicts if args.autoloop else None,
        total_figures=total_figures,
    )
    metrics_path = save_metrics(metrics, args.output_dir, args.identifier)

    print("\n=== done ===")
    print(f"teacher:  {teacher_run_dir}")
    print(f"student native:  {native_run_dir}")
    print(f"metrics: {metrics_path}")
    print(
        f"native_pass_rate={metrics['totals']['native_pass_rate']}  "
        f"aligned_pass_rate={metrics['totals'].get('aligned_pass_rate')}  "
        f"figure_coverage(student_native)={metrics['figure_coverage_rate']['student_native']}"
    )


def _load_phase_results(run_dir: str) -> list[dict]:
    """Load previously-saved phase JSON files from a run directory."""
    from pathlib import Path

    p = Path(run_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"teacher run dir not found: {run_dir}")
    results: list[dict] = []
    for fp in sorted(p.glob("phase_??_*.json")):
        # Skip autoloop per-round files; only keep the accepted version.
        import re as _re
        if _re.search(r"_round\d+\.json$", fp.name):
            continue
        with open(fp, encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def _run_export_analysis(args: argparse.Namespace) -> None:
    """Regenerate ANALYSIS.md/.pdf from a run dir (or many)."""
    from pathlib import Path

    from .session import export_all_under, export_run_dir

    p = Path(args.run_dir)
    if not p.is_dir():
        print(f"run_dir not found: {args.run_dir}")
        sys.exit(2)

    include_reasoning = not args.no_reasoning
    write_pdf = not args.no_pdf

    # Detect: is this a single run dir (has phase_*.json) or a parent
    # dir (contains teacher_*/student_*/...)?
    is_single = any(p.glob("phase_??_*.json"))

    if is_single:
        res = export_run_dir(
            p, include_reasoning=include_reasoning, write_pdf=write_pdf
        )
        print(f"exported: {res.get('md')}")
        if write_pdf:
            print(f"          {res.get('pdf')}")
    else:
        out = export_all_under(
            p, include_reasoning=include_reasoning, write_pdf=write_pdf
        )
        print(f"\nexported {len(out)} run dirs under {p}")


# ---------------------------------------------------------------------------
# Legacy Codex runners (unchanged)
# ---------------------------------------------------------------------------


def _add_legacy_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument(
        "--effort",
        default="xhigh",
        choices=["none", "low", "medium", "high", "xhigh"],
    )
    p.add_argument("--output-dir", default="./cot_sessions")
    p.add_argument("--codex-binary", default="codex")
    p.add_argument(
        "--sandbox",
        default="read-only",
        choices=["read-only", "workspace-write", "danger-full-access"],
    )


def _run_legacy_chat(args: argparse.Namespace) -> None:
    from .repl import CotRepl

    CotRepl(
        codex_binary=args.codex_binary,
        model=args.model,
        cwd=os.getcwd(),
        reasoning_effort=args.effort,
        output_dir=args.output_dir,
        sandbox=args.sandbox,
    ).run()


def _run_legacy_paper(args: argparse.Namespace) -> None:
    from .paper_session import PaperAnalysisSession

    PaperAnalysisSession(
        paper_identifier=args.identifier,
        codex_binary=args.codex_binary,
        model=args.model,
        reasoning_effort=args.effort,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        sandbox=args.sandbox,
    ).run()


def _run_legacy_batch(args: argparse.Namespace) -> None:
    from .batch import run_batch_from_file

    path = run_batch_from_file(
        task_file=args.task_file,
        codex_binary=args.codex_binary,
        model=args.model,
        reasoning_effort=args.effort,
        output_dir=args.output_dir,
    )
    print(f"Batch complete: {path}")


def _run_legacy_export(args: argparse.Namespace) -> None:
    from .export_markdown import export_session

    path = export_session(
        json_path=args.session_json,
        output_path=args.output,
        translate_reasoning=args.translate,
        translation_model=args.translate_model,
    )
    print(f"Exported: {path}")


if __name__ == "__main__":
    main()
