"""Phase 6 protocol runner.

Drives a single provider through all six analysis phases, persisting
each phase's structured result to disk. State accumulation between
phases is achieved by injecting compact prior-phase summaries into the
prompt of each subsequent phase (we deliberately do NOT keep full
messages in the AgentState across phases — that would blow the
DeepSeek 128K context budget quickly).

Figure coverage between phases 5 and 6 is handled as an extra pass:
if the student missed any figures, an additional forced-coverage JSON
result is persisted as ``phase_05b_figure_coverage.json`` before the
assessment phase runs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ..agent.loop import AgentLoop
from ..agent.state import AgentState
from ..agent.tool_registry import ToolRegistry
from ..llm.base import Provider
from .figure_coverage import (
    FigureCoverageReport,
    build_force_pass_question,
    compute_coverage,
    total_figure_count,
)
from .phases import (
    PHASE_JSON_SCHEMA_HINT,
    PROTOCOL_PHASES,
    ProtocolPhase,
    build_submit_tool_schema,
    submit_tool_description,
    submit_tool_name,
)
from ..agent.tool_registry import ToolSchema
from .runner import (
    QuestionResult,
    augment_system_prompt_with_paper,
    build_hep_registry,
    parse_structured_answer,
    run_single_question,
)


# ---------------------------------------------------------------------------
# Budget watchdog
# ---------------------------------------------------------------------------

_SOFT_TOKEN_LIMIT = 100_000  # chars / 4 ≈ tokens; conservative watchdog
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _compact_summaries(
    summaries: list[str], max_total_chars: int = 4000
) -> list[str]:
    """If the cumulative summary is too long, halve each entry."""
    total = sum(len(s) for s in summaries)
    if total <= max_total_chars:
        return summaries
    factor = max_total_chars / total
    return [s[: max(100, int(len(s) * factor))] + " ..." for s in summaries]


# ---------------------------------------------------------------------------
# Phase summary extraction
# ---------------------------------------------------------------------------


def extract_phase_summary(
    phase: ProtocolPhase, result_dict: dict[str, Any], max_chars: int = 900
) -> str:
    """Condense a phase result to a compact string for downstream injection."""
    parsed = result_dict.get("answer_parsed") if isinstance(result_dict, dict) else None
    if isinstance(parsed, dict):
        short = (parsed.get("short_answer") or "").strip()
        findings = parsed.get("findings") or {}
        bullets: list[str] = []
        if isinstance(findings, dict):
            for k, v in findings.items():
                if not v:
                    continue
                vs = str(v).strip()
                if len(vs) > 240:
                    vs = vs[:240] + "..."
                bullets.append(f"- {k}: {vs}")
        cit = parsed.get("citations") or []
        if isinstance(cit, list) and cit:
            bullets.append(
                "- citations: " + ", ".join(str(c) for c in cit[:8])
            )
        text = (
            f"[{phase.key} · {phase.name}] {short}\n"
            + "\n".join(bullets)
        )
    else:
        raw = (result_dict.get("answer_raw", "") or "").strip()
        text = f"[{phase.key}] (no JSON) {raw[:max_chars]}"

    if len(text) > max_chars:
        text = text[:max_chars] + " ..."
    return text


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class Phase6ProtocolRunner:
    """Runs the full 6-phase protocol on one paper with one provider.

    Instantiated per (paper, provider) pair. Call :meth:`run_all` to
    execute every phase; intermediate results are persisted as they go,
    so killing the process mid-run still leaves recoverable state.
    """

    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        paper_info: dict[str, Any] | None,
        base_system_prompt: str,
        output_dir: Path,
        paper_identifier: str,
        label: str,  # "teacher" | "student_native" | "student_aligned"
        max_tool_iterations: int = 10,
        paper_text_truncate: int = 60_000,
    ):
        self.provider = provider
        self.registry = registry
        self.paper_info = paper_info or {}
        self.base_system_prompt = base_system_prompt
        self.output_dir = Path(output_dir)
        self.paper_identifier = paper_identifier
        self.label = label
        self.max_tool_iterations = max_tool_iterations
        self.paper_text_truncate = paper_text_truncate

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prior_summaries: list[str] = []
        self.phase_results: list[dict[str, Any]] = []
        self._started_at = time.time()

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _base_system(self) -> str:
        """Assemble the full system prompt for this runner.

        The prompt stays constant across phases — each phase's specific
        task is delivered in the *user* message, not the system prompt,
        so providers that key off system-prompt changes don't thrash.

        We deliberately DO NOT inject :data:`PHASE_JSON_SCHEMA_HINT`
        here anymore: in Option B the submit tool's own JSON Schema
        (enforced at the API level) is the contract — asking the model
        to additionally write JSON as text would create a double-output
        hazard.
        """
        sys = augment_system_prompt_with_paper(
            self.base_system_prompt, self.paper_info
        )
        return sys

    def _phase_user_prompt(self, phase: ProtocolPhase) -> str:
        prior = _compact_summaries(self.prior_summaries)
        prior_block = ""
        if prior:
            prior_block = (
                "## 前序阶段结论（精简摘要）\n\n"
                + "\n\n".join(prior)
                + "\n\n---\n"
            )
        hepdata_note = ""
        if phase.hepdata_hint:
            hepdata_note = (
                "\n\nHEPData 提醒：如果你还没有调用过 `fetch_hepdata`，"
                "请立即对本论文调用一次；并根据返回结果，用 "
                "`get_hepdata_table` 拉取任何相关表格 —— "
                "结构化 HEPData 数据优于从 PDF 抽取的表格。\n"
            )
        submit_name = submit_tool_name(phase)
        req_keys = ", ".join(f"`{k}`" for k in phase.required_outputs)
        return (
            prior_block
            + phase.prompt_template
            + hepdata_note
            + f"\n\n## 本阶段结束方式（重要）\n\n"
            f"信息收集完成后，**调用 `{submit_name}` 工具一次**就结束本阶段。\n"
            f"- 工具的 arguments 就是 harness 抓取的结构化答案（无需在文本里再写 JSON）。\n"
            f"- `findings` 必须包含全部必填键：{req_keys}。\n"
            f"- 所有自然语言值用中文；物理术语、数值单位、引用标签"
            f"（§/Fig/Table/HEPData/arXiv）保持英文原文。\n"
            f"- 若某个 required 键论文确实未覆盖，value 填 `[uncertain]` "
            f"并在 `caveats` 里说明原因。\n"
            f"- **工具预算建议**：先调 3 个 discovery 工具（`get_paper_info`、"
            f"`list_figures`、`fetch_hepdata`），再按需做 3-5 次 search/section "
            f"检索，然后立即 `{submit_name}`。**尽早收敛，不要把工具预算用"
            f"满才 submit。**"
        )

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    def _watchdog_compact_if_needed(self, user_text: str, sys_text: str) -> None:
        total = _estimate_tokens(user_text) + _estimate_tokens(sys_text)
        if total > _SOFT_TOKEN_LIMIT:
            # Halve each prior summary until we fit.
            self.prior_summaries = [
                s[: max(80, len(s) // 2)] + " ..." for s in self.prior_summaries
            ]

    # ------------------------------------------------------------------
    # Per-phase dispatch
    # ------------------------------------------------------------------

    def run_phase(
        self,
        phase: ProtocolPhase,
        extra_user_prefix: str = "",
    ) -> tuple[dict[str, Any], QuestionResult]:
        """Run a single phase. Returns ``(serialised_result, QuestionResult)``.

        Implements Option B: structured output via a per-phase
        ``submit_<phase>_answer`` function-call tool. The model's call
        arguments ARE the answer — no JSON text parsing. If the model
        somehow ends the round without calling submit, we fall back to
        :func:`parse_structured_answer` on the raw text.
        """
        sys_prompt = self._base_system()
        user_prompt = self._phase_user_prompt(phase)
        if extra_user_prefix:
            user_prompt = extra_user_prefix + "\n\n---\n\n" + user_prompt

        self._watchdog_compact_if_needed(user_prompt, sys_prompt)

        # --- Per-phase registry (base HEP tools + submit_<phase>_answer) ---
        submit_captured: dict[str, Any] = {}
        submit_rejects: list[str] = []

        def _submit(**kwargs: Any) -> dict[str, Any]:
            # Defensive gate: some providers (notably DeepSeek under
            # flaky connections) occasionally bypass the tool-schema's
            # required fields and call submit with an empty / stub
            # payload. Rather than silently recording an empty phase
            # answer, reject clearly so the model retries within its
            # remaining tool-iteration budget.
            findings = kwargs.get("findings") or {}
            required = list(phase.required_outputs)
            if required and isinstance(findings, dict):
                filled = [
                    k
                    for k in required
                    if isinstance(findings.get(k), str)
                    and findings.get(k, "").strip()
                ]
            else:
                filled = []
            empty_submit = (not required and not kwargs) or (
                required and len(filled) == 0
            )
            # Allow up to 2 rejections before giving in, so the model
            # still terminates even if it keeps producing empty submits.
            if empty_submit and len(submit_rejects) < 2:
                missing = [k for k in required if k not in filled]
                submit_rejects.append("empty_submit")
                return {
                    "status": "rejected",
                    "reason": "empty_findings",
                    "missing_required_keys": missing,
                    "note": (
                        "提交的 `findings` 为空或缺少全部必填键："
                        f"{', '.join(missing) if missing else '(N/A)'}。"
                        "请在调用论文/HEPData 工具获取内容后，重新调用"
                        f"`{submit_tool_name(phase)}`，确保每个必填 key 的值为非空字符串"
                        "（若论文确实未覆盖，填 `[uncertain]` 并在 caveats 里说明）。"
                    ),
                }
            # Last wins if (mistakenly) called multiple times.
            submit_captured.clear()
            submit_captured.update(kwargs)
            return {
                "status": "accepted",
                "note": "结构化回答已接收；本阶段可以结束，不需要再输出任何文本。",
            }

        phase_registry = ToolRegistry()
        for name, schema in self.registry._tools.items():  # noqa: SLF001
            phase_registry.register_schema(schema)
        phase_registry.register(
            name=submit_tool_name(phase),
            description=submit_tool_description(phase),
            parameters=build_submit_tool_schema(phase),
            func=_submit,
        )

        # Fresh AgentState per phase — context across phases flows via
        # prior_summaries, not via raw message history.
        state = AgentState(system_prompt=sys_prompt)
        qr, _state = run_single_question(
            provider=self.provider,
            registry=phase_registry,
            system_prompt=sys_prompt,
            question=user_prompt,
            max_tool_iterations=self.max_tool_iterations,
            prior_state=state,
        )

        # Structured answer preference: tool-captured args over text parse.
        if submit_captured:
            qr.answer_parsed = dict(submit_captured)
            qr.answer_parsed["phase"] = phase.key  # stamp for downstream
        elif qr.answer_parsed is None and qr.answer_raw:
            # Fallback: best-effort parse of whatever text came back.
            from .runner import parse_structured_answer

            fallback = parse_structured_answer(qr.answer_raw)
            if fallback:
                qr.answer_parsed = fallback

        qr.question_id = phase.key
        qr.category = phase.key
        return qr.to_dict(), qr

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_phase(
        self,
        phase_index: int,
        phase_key: str,
        result_dict: dict[str, Any],
        round_number: int | None = None,
    ) -> Path:
        suffix = f"_round{round_number}" if round_number else ""
        fname = f"phase_{phase_index:02d}_{phase_key}{suffix}.json"
        path = self.output_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        return path

    def _save_manifest(self) -> Path:
        path = self.output_dir / "session_manifest.json"
        manifest = {
            "label": self.label,
            "provider": self.provider.name,
            "model": getattr(self.provider, "model", "n/a"),
            "paper_identifier": self.paper_identifier,
            "paper_info": self.paper_info,
            "started_at": self._started_at,
            "completed_at": time.time(),
            "phases_completed": [r.get("question_id") for r in self.phase_results],
            "paper_text_truncate": self.paper_text_truncate,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return path

    # ------------------------------------------------------------------
    # Public: run the full protocol (no autoloop)
    # ------------------------------------------------------------------

    def run_all(self) -> list[dict[str, Any]]:
        for i, phase in enumerate(PROTOCOL_PHASES, start=1):
            print(f"  [{self.label}] phase {i}/6 · {phase.name}")
            result_dict, _qr = self.run_phase(phase)
            self._save_phase(i, phase.key, result_dict)

            # Update prior summaries (even if JSON failed to parse, we
            # still want the raw block noted for downstream context).
            summary = extract_phase_summary(phase, result_dict)
            self.prior_summaries.append(summary)
            self.phase_results.append(result_dict)

            if phase.figure_coverage_after:
                # Run figure coverage forced pass before the next phase.
                self._run_figure_coverage_forced_pass(phase_index=i)

        self._save_manifest()
        self._write_analysis_markdown()
        return self.phase_results

    # ------------------------------------------------------------------
    # Figure coverage forced pass
    # ------------------------------------------------------------------

    def _run_figure_coverage_forced_pass(self, phase_index: int) -> None:
        total = total_figure_count(self.registry)
        if total == 0:
            return
        report = compute_coverage(self.phase_results, total)
        if not report.missing:
            print(
                f"    (figure coverage OK: {len(report.discussed)}/"
                f"{report.total_figures})"
            )
            return
        print(
            f"    forcing coverage pass on {len(report.missing)} missing "
            f"figures: {report.missing}"
        )

        # Build a synthetic phase for this pass.
        forced_phase = ProtocolPhase(
            key="figure_coverage",
            name="Figure Coverage Forced Pass",
            description="Analyse figures not discussed in any prior phase.",
            prompt_template=build_force_pass_question(
                report.missing, self.registry
            ),
            required_outputs=[f"fig_{n}" for n in report.missing],
            figure_coverage_after=False,
        )
        result_dict, _qr = self.run_phase(forced_phase)
        self._save_phase(
            phase_index, "figure_coverage", result_dict, round_number=None
        )
        # Also update summaries + add to phase_results so coverage
        # aggregation picks up newly-analysed figures.
        summary = extract_phase_summary(forced_phase, result_dict)
        self.prior_summaries.append(summary)
        self.phase_results.append(result_dict)

    # ------------------------------------------------------------------
    # Human-readable export for Claude Code review
    # ------------------------------------------------------------------

    def _write_analysis_markdown(self) -> Path:
        path = self.output_dir / "ANALYSIS.md"
        lines: list[str] = [
            f"# Protocol Analysis — {self.label}",
            "",
            f"**Paper:** {self.paper_info.get('arxiv_id', '')} — "
            f"{self.paper_info.get('title', '')}",
            f"**Provider:** {self.provider.name} "
            f"(`{getattr(self.provider, 'model', 'n/a')}`)",
            "",
            "---",
            "",
        ]
        for pr in self.phase_results:
            key = pr.get("question_id", "?")
            parsed = pr.get("answer_parsed") or {}
            short = parsed.get("short_answer", "")
            findings = parsed.get("findings") or {}
            citations = parsed.get("citations") or []
            key_values = parsed.get("key_values") or []
            figs = parsed.get("figures_discussed") or []
            tools = parsed.get("tools_used_this_phase") or []
            caveats = parsed.get("caveats") or []
            confidence = parsed.get("confidence", "")

            lines += [
                f"## Phase: `{key}`",
                "",
                f"**TL;DR:** {short}",
                "",
            ]
            if findings:
                lines += ["### Findings", ""]
                for k, v in findings.items():
                    vs = str(v).replace("\n", " ")
                    lines.append(f"- **{k}** — {vs}")
                lines.append("")
            if key_values:
                lines += ["### Key quantitative values", ""]
                for kv in key_values:
                    if isinstance(kv, dict):
                        lines.append(
                            f"- {kv.get('quantity','?')}: **{kv.get('value','?')}** "
                            f"[{kv.get('source','?')}]"
                        )
                lines.append("")
            if figs:
                lines += [f"- **Figures discussed:** {', '.join(str(x) for x in figs)}", ""]
            if tools:
                lines += [f"- **Tools used:** {', '.join(tools)}", ""]
            if citations:
                lines += [
                    f"- **Citations:** {', '.join(str(c) for c in citations[:20])}",
                    "",
                ]
            if caveats:
                lines += ["### Caveats", ""]
                for c in caveats:
                    lines.append(f"- {c}")
                lines.append("")
            if confidence:
                lines += [f"- **Confidence:** {confidence}", ""]
            lines += ["---", ""]

        path.write_text("\n".join(lines), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Helper: factory for a runner tied to a paper
# ---------------------------------------------------------------------------


def make_runner(
    provider: Provider,
    paper_identifier: str,
    base_system_prompt: str,
    output_dir: Path,
    label: str,
    cache_dir: str = "./paper_cache",
    max_tool_iterations: int = 10,
    paper_text_truncate: int = 60_000,
    segmenter_provider: Provider | None = None,
    skip_segmentation: bool = False,
) -> tuple[Phase6ProtocolRunner, ToolRegistry, dict[str, Any] | None]:
    """Construct a runner + its registry for one (paper, provider, label) triple."""
    registry, _ctx, paper_info = build_hep_registry(
        paper_identifier,
        cache_dir,
        segmenter_provider=segmenter_provider,
        skip_segmentation=skip_segmentation,
    )
    runner = Phase6ProtocolRunner(
        provider=provider,
        registry=registry,
        paper_info=paper_info,
        base_system_prompt=base_system_prompt,
        output_dir=output_dir,
        paper_identifier=paper_identifier,
        label=label,
        max_tool_iterations=max_tool_iterations,
        paper_text_truncate=paper_text_truncate,
    )
    return runner, registry, paper_info
