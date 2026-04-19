"""Shared runner utility used by teacher / student / autoloop.

Factors out the work of:
  1. building a :class:`ToolRegistry` pre-loaded with the paper
  2. creating an :class:`AgentLoop` around a provider
  3. running one question, capturing the structured JSON answer + CoT

Calling code supplies the provider, the system prompt, and an
:class:`AgentState` (which can carry prior autoloop context).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..agent.loop import AgentLoop, TurnResult
from ..agent.state import AgentState
from ..agent.tool_registry import ToolRegistry
from ..llm.base import Provider
from ..tools import (
    build_arxiv_search_tool,
    build_figure_tools,
    build_hepdata_tools,
    build_inspire_tools,
    build_paper_tools,
)
from ..tools.paper_tools import PaperContext


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
BARE_JSON_RE = re.compile(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", re.DOTALL)


@dataclass
class QuestionResult:
    """Per-question output from teacher/student runners."""

    question_id: str
    question: str
    category: str
    provider: str
    answer_raw: str = ""
    answer_parsed: dict[str, Any] | None = None
    thinking: str = ""  # joined convenience view
    thinking_records: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    usage_per_iteration: list[dict[str, int]] = field(default_factory=list)
    stop_reason: str = ""
    duration_s: float = 0.0
    autoloop_rounds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "provider": self.provider,
            "answer_raw": self.answer_raw,
            "answer_parsed": self.answer_parsed,
            "thinking": self.thinking,
            "thinking_records": self.thinking_records,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "usage_per_iteration": self.usage_per_iteration,
            "stop_reason": self.stop_reason,
            "duration_s": round(self.duration_s, 2),
            "autoloop_rounds": self.autoloop_rounds,
        }


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------


def build_hep_registry(
    paper_identifier: str,
    cache_dir: str,
    segmenter_provider: Any = None,
    skip_segmentation: bool = False,
) -> tuple[ToolRegistry, PaperContext, dict[str, Any] | None]:
    """Build the full MVP-1 tool registry and pre-load the paper.

    Returns ``(registry, paper_context, fetch_paper_result)``. The third
    element is the ``fetch_paper`` tool output (parsed dict) so callers
    can inject paper metadata into their system prompts.

    When ``segmenter_provider`` is provided and the paper has no cached
    semantic TOC, fetch_paper will trigger a one-time segmentation and
    cache it under ``<cache_dir>/<arxiv_id>_semantic_toc.json``.
    """
    ctx = PaperContext(
        cache_dir=cache_dir,
        segmenter_provider=segmenter_provider,
        skip_segmentation=skip_segmentation,
    )
    reg = ToolRegistry()

    for schema in build_paper_tools(ctx):
        reg.register_schema(schema)
    for schema in build_figure_tools(ctx):
        reg.register_schema(schema)
    for schema in build_inspire_tools(cache_dir):
        reg.register_schema(schema)
    for schema in build_hepdata_tools(cache_dir):
        reg.register_schema(schema)
    for schema in build_arxiv_search_tool(cache_dir):
        reg.register_schema(schema)

    # Pre-load the paper so the first question doesn't burn a tool call.
    content, is_error, _ = reg.execute(
        "fetch_paper", {"identifier": paper_identifier}
    )
    info: dict[str, Any] | None = None
    if not is_error:
        try:
            info = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            info = None
    return reg, ctx, info


def augment_system_prompt_with_paper(
    base_prompt: str, paper_info: dict[str, Any] | None
) -> str:
    """Append paper-loaded metadata to a teacher/student system prompt."""
    if not paper_info or not paper_info.get("arxiv_id"):
        return base_prompt

    title = (paper_info.get("title") or "").strip()
    arxiv_id = paper_info.get("arxiv_id", "")
    abstract = (paper_info.get("abstract") or "").strip()
    extra = (
        f"\n\n## Paper under analysis (pre-loaded)\n"
        f"- arxiv_id: {arxiv_id}\n"
        f"- title: {title}\n"
    )
    if abstract:
        short = abstract if len(abstract) <= 900 else abstract[:900] + " ..."
        extra += f"- abstract: {short}\n"
    extra += (
        "\nThe full LaTeX source is already cached. Do NOT call "
        "`fetch_paper`. Use the local tools directly.\n"
    )
    return base_prompt + extra


# ---------------------------------------------------------------------------
# Structured JSON extraction
# ---------------------------------------------------------------------------


def parse_structured_answer(text: str) -> dict[str, Any] | None:
    """Try hard to recover a JSON object from a model's free-text answer.

    Looks for (in order):
      1. Triple-backtick fenced ``json`` block
      2. The largest JSON-looking brace-matched substring

    Returns ``None`` if nothing parses.
    """

    text = text.strip()
    if not text:
        return None

    # Fenced blocks first.
    m = JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try the whole thing.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Greedy brace match — take the first top-level {...}.
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1
                    continue

    return None


# ---------------------------------------------------------------------------
# Single-question driver
# ---------------------------------------------------------------------------


def run_single_question(
    provider: Provider,
    registry: ToolRegistry,
    system_prompt: str,
    question: str,
    max_tool_iterations: int = 8,
    prior_state: AgentState | None = None,
) -> tuple[QuestionResult, AgentState]:
    """Run one question end-to-end. Returns the result + final AgentState."""

    state = prior_state or AgentState(system_prompt=system_prompt)
    # In case caller reused a state without the right prompt:
    if not state.system_prompt:
        state.system_prompt = system_prompt

    loop = AgentLoop(
        provider=provider,
        registry=registry,
        state=state,
        max_tool_iterations=max_tool_iterations,
    )

    t0 = time.time()
    turn: TurnResult = loop.run_round(question)
    duration = time.time() - t0

    answer_raw = turn.text
    parsed = parse_structured_answer(answer_raw)

    # Collect tool call bookkeeping (from this round only).
    turn_mark = state.turn_index - 1
    tool_calls = [
        {
            "name": tc.name,
            "arguments": tc.arguments,
            "iteration": tc.iteration_index,
            "is_error": tc.is_error,
            "result_preview": tc.result[:400],
            "duration_s": round(tc.duration_s, 3),
            "timestamp": tc.timestamp,
        }
        for tc in state.tool_calls_log
        if tc.turn_index == turn_mark
    ]

    # Per-iteration thinking records (preserves iteration boundaries + timestamps
    # so researchers can reconstruct the interleaved reasoning ↔ tool timeline).
    thinking_records = [
        {
            "iteration": t.iteration_index,
            "text": t.text,
            "provider": t.provider,
            "timestamp": t.timestamp,
            "char_count": len(t.text),
        }
        for t in state.thinking_log
        if t.turn_index == turn_mark
    ]

    # Convenience joined view.
    thinking = "\n\n".join(
        f"[iter {r['iteration']}] {r['text']}" for r in thinking_records
    )

    result = QuestionResult(
        question_id="",  # filled by caller
        question=question,
        category="",  # filled by caller
        provider=provider.name,
        answer_raw=answer_raw,
        answer_parsed=parsed,
        thinking=thinking,
        thinking_records=thinking_records,
        tool_calls=tool_calls,
        usage=dict(state.usage_totals),
        usage_per_iteration=list(state.usage_per_iteration),
        stop_reason=turn.stop_reason,
        duration_s=duration,
    )
    return result, state
