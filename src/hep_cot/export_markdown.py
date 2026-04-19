"""Export a CoT session JSON to a readable Markdown document.

Optionally translates hidden reasoning (English) to Chinese via GPT-5.4-mini.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s}s"


def _reasoning_to_md(steps: list[dict]) -> str:
    """Format hidden reasoning steps as a Markdown blockquote block."""
    if not steps:
        return ""
    parts: list[str] = []
    for step in steps:
        summary = step.get("summary", "").strip()
        if summary:
            parts.append(summary)
    if not parts:
        return ""
    combined = "\n\n".join(parts)
    lines = combined.split("\n")
    quoted = "\n".join(f"> {l}" for l in lines)
    return quoted


def _translate_reasoning_block(text: str, model: str = "gpt-5.4-mini") -> str:
    """Translate an English reasoning block to Chinese using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        return text + "\n\n*(openai package not installed, skipping translation)*"

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个高能物理领域的翻译助手。将以下英文推理过程翻译为中文。"
                    "物理专业术语保留英文（如 cross section, luminosity, branching ratio, "
                    "signal region, control region, nuisance parameter 等）。"
                    "保持原文的逻辑结构和推理链条，不要省略任何内容。"
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def session_to_markdown(
    session: dict,
    translate_reasoning: bool = False,
    translation_model: str = "gpt-5.4-mini",
) -> str:
    """Convert a session JSON dict to a Markdown string."""
    lines: list[str] = []

    # Header
    paper = session.get("paper", {})
    title = paper.get("title", session.get("session_id", "Unknown"))
    arxiv_id = paper.get("arxiv_id", "")
    lines.append(f"# CoT Session: {title}")
    if arxiv_id:
        lines.append(f"\n**arXiv**: [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})")
    lines.append(f"**Model**: {session.get('model', 'unknown')}")

    protocol = session.get("analysis_protocol", {})
    if protocol:
        completed = protocol.get("phases_completed", 0)
        total = protocol.get("total_phases", 0)
        lines.append(f"**Phases**: {completed}/{total}")

    total_turns = session.get("total_turns", 0)
    total_reasoning = session.get("total_reasoning_steps", 0)
    lines.append(f"**Turns**: {total_turns}  |  **Reasoning steps**: {total_reasoning}")
    lines.append("")

    # Figures
    figures = session.get("figures", [])
    if figures:
        lines.append("## Figures")
        for fig in figures:
            num = fig.get("number", "?")
            caption = fig.get("caption", "")[:120]
            lines.append(f"- **Figure {num}**: {caption}")
        lines.append("")

    # Phase names for labeling turns
    phase_names = protocol.get("phase_names", []) if protocol else []

    # Turns
    turns = session.get("turns", [])
    for i, turn in enumerate(turns):
        duration = turn.get("duration_seconds", 0)
        phase_label = phase_names[i] if i < len(phase_names) else f"Turn {i+1}"

        lines.append(f"---\n## {phase_label}")
        lines.append(f"*({_format_duration(duration)})*\n")

        # Hidden reasoning
        hidden = turn.get("hidden_reasoning", [])
        if hidden:
            reasoning_text = _reasoning_to_md(hidden)
            if reasoning_text:
                lines.append("### Hidden Reasoning (Model Internal)")
                if translate_reasoning:
                    raw = "\n\n".join(
                        s.get("summary", "") for s in hidden if s.get("summary")
                    )
                    if raw.strip():
                        translated = _translate_reasoning_block(
                            raw, model=translation_model,
                        )
                        lines.append(translated)
                    else:
                        lines.append(reasoning_text)
                else:
                    lines.append(reasoning_text)
                lines.append("")

        # Agent response (explicit reasoning)
        response = turn.get("agent_response", "")
        if response:
            lines.append("### Analysis")
            lines.append(response)
            lines.append("")

        # Tool calls (if any)
        tool_calls = turn.get("tool_calls", [])
        if tool_calls:
            lines.append("<details><summary>Tool calls</summary>\n")
            for tc in tool_calls:
                cmd = tc.get("command", "")
                status = tc.get("status", "")
                lines.append(f"- `{cmd}` ({status})")
            lines.append("\n</details>\n")

        # Usage stats
        usage = turn.get("usage", {})
        reasoning_tokens = usage.get("output_tokens_details", {}).get(
            "reasoning_tokens", 0
        )
        if reasoning_tokens:
            lines.append(
                f"*reasoning_tokens={reasoning_tokens}*\n"
            )

    # Footer
    lines.append("---")
    lines.append(
        f"*Generated by hep-cot | "
        f"{time.strftime('%Y-%m-%d %H:%M')}*"
    )

    return "\n".join(lines)


def export_session(
    json_path: str,
    output_path: str | None = None,
    translate_reasoning: bool = False,
    translation_model: str = "gpt-5.4-mini",
) -> str:
    """Load a session JSON file and export as Markdown."""
    with open(json_path, "r", encoding="utf-8") as f:
        session = json.load(f)

    md = session_to_markdown(
        session,
        translate_reasoning=translate_reasoning,
        translation_model=translation_model,
    )

    if output_path is None:
        output_path = str(Path(json_path).with_suffix(".md"))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    return output_path
