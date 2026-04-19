"""Figure coverage validator.

Between phase 5 (``stat_result``) and phase 6 (``assessment``), we
aggregate the ``figures_discussed`` lists from every prior phase,
compare against the full figure inventory (from the ``list_figures``
tool), and if any figure has not been discussed, force a mini-pass
that makes the model analyse each missing figure before moving on.

This mirrors the ``_ensure_figure_coverage`` logic from the legacy
:mod:`hep_cot.paper_session` but operates on structured phase JSON
rather than free-text reasoning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..agent.tool_registry import ToolRegistry


@dataclass
class FigureCoverageReport:
    """Result of a coverage check across phase results."""

    total_figures: int = 0
    discussed: set[int] = field(default_factory=set)
    missing: list[int] = field(default_factory=list)
    coverage_rate: float = 0.0


def collect_figures_discussed(phase_results: list[dict[str, Any]]) -> set[int]:
    """Union the ``figures_discussed`` integer lists across phase JSON payloads.

    Tolerates the parsed-answer dict being absent (e.g. if a phase
    failed to produce valid JSON) — such phases contribute nothing.
    Also scans the raw answer text for ``Fig N`` / ``Figure N`` as a
    belt-and-braces fallback.
    """
    out: set[int] = set()

    for pr in phase_results:
        parsed = pr.get("answer_parsed") if isinstance(pr, dict) else None
        raw = pr.get("answer_raw", "") if isinstance(pr, dict) else ""

        if isinstance(parsed, dict):
            fd = parsed.get("figures_discussed") or []
            for x in fd:
                try:
                    out.add(int(x))
                except (TypeError, ValueError):
                    continue

        # Backstop: pattern-match on the free-text answer so a model that
        # described Fig 3 in prose but forgot to list it gets credit.
        if isinstance(raw, str) and raw:
            for m in re.finditer(
                r"(?:Fig(?:ure)?\.?)\s*(\d{1,3})", raw, flags=re.IGNORECASE
            ):
                try:
                    out.add(int(m.group(1)))
                except ValueError:
                    continue

    return out


def compute_coverage(
    phase_results: list[dict[str, Any]], total_figures: int
) -> FigureCoverageReport:
    discussed = collect_figures_discussed(phase_results)
    # Discard numbers out of range (e.g. matched "Figure 20" when paper
    # only has 9 figures — likely an external reference).
    in_range = {n for n in discussed if 1 <= n <= total_figures}
    missing = sorted(set(range(1, total_figures + 1)) - in_range)
    rate = (len(in_range) / total_figures) if total_figures > 0 else 1.0
    return FigureCoverageReport(
        total_figures=total_figures,
        discussed=in_range,
        missing=missing,
        coverage_rate=round(rate, 3),
    )


def figure_list_from_registry(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Call ``list_figures`` via the registry and return the raw figure list."""
    if "list_figures" not in registry:
        return []
    content, is_error, _ = registry.execute("list_figures", {})
    if is_error:
        return []
    import json as _json

    try:
        payload = _json.loads(content)
    except (_json.JSONDecodeError, TypeError):
        return []
    return payload.get("figures") or []


def total_figure_count(registry: ToolRegistry) -> int:
    figs = figure_list_from_registry(registry)
    return len(figs)


def build_force_pass_question(missing: list[int], registry: ToolRegistry) -> str:
    """Build the user message that forces discussion of missing figures."""
    figs = figure_list_from_registry(registry)
    by_num = {f.get("figure_number"): f for f in figs if f.get("figure_number")}

    bits = [
        "在进入最终的 assessment 阶段前，还有一些图表尚未在前序任何阶段"
        "被讨论过。请对以下每张图：调用 `get_figure` 获取其 caption 与 "
        "PNG，然后产出**一个 JSON 对象**（与常规阶段 JSON schema 相同，"
        "但 `phase` 字段设为 `figure_coverage`），其 `findings` 中为每"
        "张待分析图对应一个 key，value 为 2-3 句中文分析。",
        "",
        "待分析的图：",
    ]
    for n in missing:
        meta = by_num.get(n, {})
        short = (meta.get("caption_short") or "").strip()
        bits.append(f"  - Figure {n}: {short[:160]}")
    bits.append("")
    bits.append(
        "请把这些图号填入 `figures_discussed` 数组，并在 `citations` 中"
        "引用（如 `Fig 3`）。不要重复讨论前序阶段已经分析过的图。"
    )
    return "\n".join(bits)
