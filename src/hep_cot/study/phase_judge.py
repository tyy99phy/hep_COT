"""Phase-level judge: L1 + L2 + L4 (completeness) per phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .judge import (
    JudgeReport,
    _collect_citations,
    _collect_numbers,
    _number_match,
    _tool_sequence,
    _levenshtein,
)
from .phases import ProtocolPhase


@dataclass
class PhaseJudgeReport:
    """Per-phase verdict combining L1 (numbers+citations), L2 (tool diff), L4 (required-keys)."""

    phase_key: str = ""
    # L1
    l1_pass: bool = False
    l1_number_hit_rate: float = 0.0
    l1_citation_hit_rate: float = 0.0
    l1_missing_numbers: list[str] = field(default_factory=list)
    l1_missing_citations: list[str] = field(default_factory=list)
    # L2
    l2_tool_edit_distance: int = 0
    l2_tool_sequences: dict[str, list[str]] = field(default_factory=dict)
    # L4
    l4_required_keys_present: int = 0
    l4_required_keys_total: int = 0
    l4_missing_keys: list[str] = field(default_factory=list)
    l4_pass: bool = False
    # combined
    overall_pass: bool = False
    diff_summary: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_key": self.phase_key,
            "overall_pass": self.overall_pass,
            "l1": {
                "pass": self.l1_pass,
                "number_hit_rate": self.l1_number_hit_rate,
                "citation_hit_rate": self.l1_citation_hit_rate,
                "missing_numbers": self.l1_missing_numbers,
                "missing_citations": self.l1_missing_citations,
            },
            "l2": {
                "tool_edit_distance": self.l2_tool_edit_distance,
                "sequences": self.l2_tool_sequences,
            },
            "l4": {
                "present": self.l4_required_keys_present,
                "total": self.l4_required_keys_total,
                "missing_keys": self.l4_missing_keys,
                "pass": self.l4_pass,
            },
            "diff_summary": self.diff_summary,
            "notes": self.notes,
        }


def judge_phase(
    phase: ProtocolPhase,
    teacher_result: dict[str, Any],
    student_result: dict[str, Any],
    number_tol: float = 0.05,
    l1_number_threshold: float = 0.80,
    l1_citation_threshold: float = 0.70,
) -> PhaseJudgeReport:
    """Compare a single student phase result to the teacher reference phase result.

    L1 (numbers + citations) is judged *relative* to the teacher; if the
    teacher produced no parseable JSON, L1 is skipped but L4 is still
    judged independently against :attr:`ProtocolPhase.required_outputs`.
    """

    rpt = PhaseJudgeReport(phase_key=phase.key)

    t_ans = teacher_result.get("answer_parsed") if isinstance(teacher_result, dict) else None
    s_ans = student_result.get("answer_parsed") if isinstance(student_result, dict) else None

    teacher_ok = isinstance(t_ans, dict)
    student_ok = isinstance(s_ans, dict)

    if not student_ok:
        rpt.notes.append("student produced no parseable JSON")
        rpt.diff_summary = "Student did not return a parseable JSON object for this phase."
        rpt.overall_pass = False
        # Still record L4 missing keys for feedback.
        rpt.l4_required_keys_total = len(phase.required_outputs)
        rpt.l4_missing_keys = list(phase.required_outputs)
        return rpt

    if not teacher_ok:
        rpt.notes.append("teacher produced no parseable JSON — L1 skipped, L4 independent")

    # ------------------------------------------------------------------
    # L1 numbers (only if teacher parseable)
    # ------------------------------------------------------------------
    if teacher_ok:
        t_nums = _collect_numbers(t_ans)
        s_nums = _collect_numbers(s_ans)
        if t_nums:
            matched = [
                rn
                for rn in t_nums
                if any(_number_match(rn, sn, number_tol) for sn in s_nums)
            ]
            missing = [n for n in t_nums if n not in matched]
            rpt.l1_number_hit_rate = round(len(matched) / len(t_nums), 3)
            rpt.l1_missing_numbers = missing
        else:
            rpt.l1_number_hit_rate = 1.0  # nothing to match

        # ------------------------------------------------------------------
        # L1 citations
        # ------------------------------------------------------------------
        t_cits = _collect_citations(t_ans)
        s_cits = _collect_citations(s_ans)
        if t_cits:
            rpt.l1_citation_hit_rate = round(len(t_cits & s_cits) / len(t_cits), 3)
            rpt.l1_missing_citations = sorted(t_cits - s_cits)
        else:
            rpt.l1_citation_hit_rate = 1.0

        rpt.l1_pass = (
            rpt.l1_number_hit_rate >= l1_number_threshold
            and rpt.l1_citation_hit_rate >= l1_citation_threshold
        )

        # L2 tool sequence
        t_seq = _tool_sequence(teacher_result)
        s_seq = _tool_sequence(student_result)
        rpt.l2_tool_edit_distance = _levenshtein(t_seq, s_seq)
        rpt.l2_tool_sequences = {"teacher": t_seq, "student": s_seq}
    else:
        # No teacher reference: L1 gets a neutral "skipped" status.
        # overall_pass will fall back to L4 alone.
        rpt.l1_pass = True  # neutral; only L4 decides
        rpt.l1_number_hit_rate = -1.0  # sentinel: "not judged"
        rpt.l1_citation_hit_rate = -1.0
        rpt.l2_tool_edit_distance = -1

    # ------------------------------------------------------------------
    # L4 required-keys completeness — ALWAYS judged independently
    # ------------------------------------------------------------------
    s_findings = s_ans.get("findings") or {}
    rpt.l4_required_keys_total = len(phase.required_outputs)
    if isinstance(s_findings, dict):
        present = [
            k
            for k in phase.required_outputs
            if isinstance(s_findings.get(k), str)
            and s_findings.get(k, "").strip()
            and s_findings.get(k, "").strip() != "[uncertain]"
        ]
        missing = [k for k in phase.required_outputs if k not in present]
        rpt.l4_required_keys_present = len(present)
        rpt.l4_missing_keys = missing
    else:
        rpt.l4_missing_keys = list(phase.required_outputs)

    if rpt.l4_required_keys_total == 0:
        rpt.l4_pass = True
    else:
        rpt.l4_pass = (
            rpt.l4_required_keys_present / rpt.l4_required_keys_total >= 0.80
        )

    # Overall pass: when teacher was missing, rely on L4; otherwise both.
    if teacher_ok:
        rpt.overall_pass = bool(rpt.l1_pass and rpt.l4_pass)
    else:
        rpt.overall_pass = bool(rpt.l4_pass)

    # Diff summary
    bits: list[str] = []
    if rpt.l4_missing_keys:
        bits.append(
            "Required `findings` keys missing or empty: "
            + ", ".join(rpt.l4_missing_keys[:6])
            + (
                f" (+{len(rpt.l4_missing_keys) - 6} more)"
                if len(rpt.l4_missing_keys) > 6
                else ""
            )
        )
    if teacher_ok and rpt.l1_missing_numbers:
        bits.append(
            "Numbers present in the reference but missing from your answer: "
            + ", ".join(rpt.l1_missing_numbers[:5])
            + (
                f" (+{len(rpt.l1_missing_numbers) - 5} more)"
                if len(rpt.l1_missing_numbers) > 5
                else ""
            )
        )
    if teacher_ok and rpt.l1_missing_citations:
        bits.append(
            "Citations in the reference not present in your answer: "
            + ", ".join(rpt.l1_missing_citations[:5])
        )
    if teacher_ok and rpt.l2_tool_edit_distance > 0:
        bits.append(
            f"Tool-call path differs from reference "
            f"(edit distance {rpt.l2_tool_edit_distance})."
        )
    rpt.diff_summary = "\n".join(bits)
    return rpt


def build_phase_feedback(
    phase: ProtocolPhase, verdict: PhaseJudgeReport, round_number: int
) -> str:
    """Build the feedback message for a per-phase autoloop round."""

    hints: list[str] = []
    if verdict.l4_missing_keys:
        hints.append(
            f"请确保 `findings` 中包含 **全部** 以下 key，且每个 key 的"
            f"值非空：{', '.join(phase.required_outputs)}。"
            f"当前缺失：{', '.join(verdict.l4_missing_keys)}。"
        )
    if verdict.l1_missing_numbers:
        hints.append(
            "参考答案中的一些关键数值在你的回答中缺失。请重新核对"
            "论文 / HEPData / 图表，并在回答中给出具体数字（含单位"
            "与来源）。"
        )
    if verdict.l1_missing_citations:
        hints.append(
            "你的 citations 缺少参考答案中的关键锚点："
            + ", ".join(verdict.l1_missing_citations[:5])
            + "。请确保每一个数值都引用到具体的 section / figure / table。"
        )

    return (
        f"请重做 Phase `{phase.key}`（`{phase.name}`）。第 {round_number} 轮。\n\n"
        f"裁判给出的差异摘要：\n{verdict.diff_summary or '（无具体差异描述）'}\n\n"
        "修正建议：\n- "
        + "\n- ".join(hints or ["整体提升回答质量。"])
        + "\n\n重新思考，如有必要调用额外工具，然后按相同 JSON schema "
        "输出新的 JSON 对象。**不要盲目采纳假设的参考数值** —— 请通过"
        "工具调用亲自在论文中验证。"
    )


# Re-export for easy import by metrics/autoloop.
__all__ = ["PhaseJudgeReport", "judge_phase", "build_phase_feedback"]
