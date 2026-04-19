"""Judge: compare student answer against teacher reference.

Three levels (only L1 + L2 implemented for MVP-1):
  * **L1** — hard correctness on numeric values + citations
  * **L2** — tool-call sequence diff (which tools, which order)
  * **L3** — CoT semantic similarity (TODO: deferred to M2)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# L1: numeric + citation
# ---------------------------------------------------------------------------


_NUMBER_RE = re.compile(
    r"(?<![\w])"  # not preceded by a word char
    r"(?:[-+]?)"
    r"(?:\d+\.?\d*|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
    r"(?:\s*%|\s*(?:GeV|TeV|MeV|fb|pb|ab|fb\^?-?1|pb\^?-?1))?"
)


def _normalise_num_token(tok: str) -> str:
    return tok.strip().rstrip(".").rstrip(",")


def _extract_numbers(text: str) -> list[str]:
    return [_normalise_num_token(m.group(0)) for m in _NUMBER_RE.finditer(text or "")]


def _canonicalise_citation(tag: str) -> str:
    """Normalise §6.2 / Sec 6.2 / Section 6.2 → 'sec:6.2'; Fig 8 / Figure 8 → 'fig:8'."""
    t = tag.strip().lower()
    # Normalise § into a consistent 'sec ' token before stripping punctuation.
    t = t.replace("§", " sec ")
    t = re.sub(r"[^\w.\-:/]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # Paper-level special sections — canonicalise to a fixed short tag so
    # "Abstract" / "abstract" / "Introduction" / "Conclusion" / "Summary"
    # etc. all collapse to the same bucket.
    paper_level = {
        "abstract": "sec:abstract",
        "introduction": "sec:intro",
        "intro": "sec:intro",
        "conclusion": "sec:conclusion",
        "conclusions": "sec:conclusion",
        "summary": "sec:summary",
        "results": "sec:results",
        "discussion": "sec:discussion",
        "bibliography": "sec:bib",
        "references": "sec:bib",
    }
    first = t.split()[0] if t else ""
    if first in paper_level:
        return paper_level[first]

    m = re.match(r"sec(?:tion)?\s*([\d.]+)", t)
    if m:
        return f"sec:{m.group(1).rstrip('.')}"
    m = re.match(r"(?:fig(?:\.|ure)?)\s*([\d.]+)", t)
    if m:
        return f"fig:{m.group(1).rstrip('.')}"
    m = re.match(r"(?:tab(?:\.|le)?)\s*([\d.]+)", t)
    if m:
        return f"tab:{m.group(1).rstrip('.')}"
    m = re.match(r"(?:eq(?:\.|uation)?)\s*([\d.]+)", t)
    if m:
        return f"eq:{m.group(1).rstrip('.')}"
    m = re.match(r"arxiv\s*:?\s*([\d.]+|\w+/\d+)", t)
    if m:
        return f"arxiv:{m.group(1)}"
    return t


def _collect_citations(answer: dict[str, Any]) -> set[str]:
    tags: set[str] = set()
    for c in answer.get("citations") or []:
        if isinstance(c, str):
            tags.add(_canonicalise_citation(c))
    for kv in answer.get("key_values") or []:
        src = kv.get("source", "")
        if isinstance(src, str) and src.strip():
            tags.add(_canonicalise_citation(src))
    return {t for t in tags if t}


def _collect_numbers(answer: dict[str, Any]) -> list[str]:
    """Gather the numeric tokens the answer claims."""
    nums: list[str] = []
    for kv in answer.get("key_values") or []:
        v = str(kv.get("value", ""))
        nums.extend(_extract_numbers(v))
    return nums


def _number_match(ref: str, pred: str, tol: float = 0.05) -> bool:
    """Match two numeric tokens with relative tolerance."""
    try:
        rf = float(re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ref).group(0))  # type: ignore
        pf = float(re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", pred).group(0))  # type: ignore
    except (AttributeError, ValueError):
        return ref.strip() == pred.strip()

    if rf == 0 and pf == 0:
        return True
    if rf == 0:
        return abs(pf) < 1e-9
    return abs(rf - pf) / max(abs(rf), 1e-12) <= tol


# ---------------------------------------------------------------------------
# L2: tool call sequence
# ---------------------------------------------------------------------------


def _tool_sequence(result: dict[str, Any]) -> list[str]:
    return [tc.get("name", "") for tc in (result.get("tool_calls") or [])]


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                cur[j - 1] + 1,  # insert
                prev[j] + 1,  # delete
                prev[j - 1] + cost,  # sub
            )
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# Judge output
# ---------------------------------------------------------------------------


@dataclass
class JudgeReport:
    """Result of comparing one student answer to one teacher reference."""

    question_id: str = ""
    l1_pass: bool = False
    l1_numbers: dict[str, Any] = field(default_factory=dict)
    l1_citations: dict[str, Any] = field(default_factory=dict)
    l2_tool_edit_distance: int = 0
    l2_tool_sequences: dict[str, list[str]] = field(default_factory=dict)
    diff_summary: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "l1_pass": self.l1_pass,
            "l1_numbers": self.l1_numbers,
            "l1_citations": self.l1_citations,
            "l2_tool_edit_distance": self.l2_tool_edit_distance,
            "l2_tool_sequences": self.l2_tool_sequences,
            "diff_summary": self.diff_summary,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Main judge function
# ---------------------------------------------------------------------------


def judge_answer(
    teacher_result: dict[str, Any],
    student_result: dict[str, Any],
    number_tol: float = 0.05,
) -> JudgeReport:
    """Compare a single student result against the teacher reference."""

    rpt = JudgeReport(question_id=teacher_result.get("question_id", ""))

    t_ans = teacher_result.get("answer_parsed") or {}
    s_ans = student_result.get("answer_parsed") or {}

    if not t_ans:
        rpt.notes.append("teacher produced no parseable JSON — cannot judge L1")
        rpt.l1_pass = False
        return rpt
    if not s_ans:
        rpt.notes.append("student produced no parseable JSON")
        rpt.l1_pass = False
        rpt.diff_summary = "Student did not return a parseable structured answer."
        return rpt

    # --- L1 numbers ---
    t_nums = _collect_numbers(t_ans)
    s_nums = _collect_numbers(s_ans)

    matched: list[str] = []
    missing: list[str] = []
    for rn in t_nums:
        if any(_number_match(rn, sn, number_tol) for sn in s_nums):
            matched.append(rn)
        else:
            missing.append(rn)

    n_num_total = len(t_nums)
    n_num_hit = len(matched)
    rpt.l1_numbers = {
        "total": n_num_total,
        "matched": n_num_hit,
        "missing": missing,
        "hit_rate": (n_num_hit / n_num_total) if n_num_total else 1.0,
    }

    # --- L1 citations ---
    t_cits = _collect_citations(t_ans)
    s_cits = _collect_citations(s_ans)

    cit_hit = len(t_cits & s_cits)
    rpt.l1_citations = {
        "reference_set": sorted(t_cits),
        "student_set": sorted(s_cits),
        "hit": cit_hit,
        "total": len(t_cits),
        "hit_rate": (cit_hit / len(t_cits)) if t_cits else 1.0,
        "missing": sorted(t_cits - s_cits),
        "extra": sorted(s_cits - t_cits),
    }

    # --- L1 pass: numbers at >= 80%, citations at >= 70% ---
    num_ok = rpt.l1_numbers["hit_rate"] >= 0.80
    cit_ok = rpt.l1_citations["hit_rate"] >= 0.70
    rpt.l1_pass = bool(num_ok and cit_ok)

    # --- L2 tool sequence ---
    t_seq = _tool_sequence(teacher_result)
    s_seq = _tool_sequence(student_result)
    rpt.l2_tool_edit_distance = _levenshtein(t_seq, s_seq)
    rpt.l2_tool_sequences = {"teacher": t_seq, "student": s_seq}

    # --- diff summary for feedback ---
    bits: list[str] = []
    if missing:
        bits.append(
            f"Numbers missing/incorrect vs reference: {', '.join(missing[:5])}"
            + (f" (+{len(missing)-5} more)" if len(missing) > 5 else "")
        )
    if rpt.l1_citations["missing"]:
        bits.append(
            "Citations not present in your answer: "
            + ", ".join(rpt.l1_citations["missing"][:5])
        )
    if rpt.l2_tool_edit_distance > 0:
        bits.append(
            f"Tool-call path differs from reference "
            f"(edit distance {rpt.l2_tool_edit_distance})."
        )
    rpt.diff_summary = "\n".join(bits)
    return rpt
