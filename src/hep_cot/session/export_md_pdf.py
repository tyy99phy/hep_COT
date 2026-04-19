"""ANALYSIS.md / .pdf exporter — refined layout.

Regenerates the per-phase analysis document for an existing study run
directory, without re-invoking any LLM. All data comes from the
already-persisted ``phase_NN_*.json`` files (written by
:class:`hep_cot.study.protocol.Phase6ProtocolRunner`).

Layout (per phase), in order:
  1. Phase header with TL;DR and meta row
  2. **Reasoning trace** (collapsible in MD viewers, always-open in PDF)
     — rendered as a distinct aside so it's visually separated from
     the main findings. Each iteration is its own nested block with
     its own token budget.
  3. Final findings: structured card list
  4. Key quantitative values (styled table)
  5. Citations / figures / tools (badge row)
  6. Actual tool calls (collapsible audit trail)
  7. Caveats + confidence pill
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _escape_md(s: Any) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _h(s: Any) -> str:
    """HTML-escape for inline values inside raw-HTML blocks."""
    return html.escape(str(s) if s is not None else "")


def _format_usage(usage: dict[str, Any] | None) -> str:
    if not usage:
        return "—"
    bits = []
    label_map = [
        ("reasoning_tokens", "reasoning"),
        ("completion_tokens", "completion"),
        ("output_tokens", "output"),
        ("prompt_tokens", "prompt"),
        ("input_tokens", "input"),
    ]
    for k, lbl in label_map:
        if k in usage and usage[k]:
            bits.append(f"<span class='chip'>{lbl} <b>{usage[k]:,}</b></span>")
    return "".join(bits) if bits else "—"


# ---------------------------------------------------------------------------
# Reasoning block (the "observer" trace from the API reasoning channel)
# ---------------------------------------------------------------------------


def _render_thinking_block(
    records: list[dict[str, Any]],
    usage_per_iteration: list[dict[str, Any]],
    provider_name: str,
) -> str:
    """Render per-iteration reasoning as nested collapsible blocks."""
    if not records:
        return (
            '<aside class="reasoning-empty">\n'
            '<em>（本阶段未捕获 reasoning tokens — 可能是 provider '
            "未返回 reasoning 通道）</em>\n"
            "</aside>\n"
        )

    total_chars = sum(r.get("char_count", len(r.get("text", ""))) for r in records)
    total_reasoning_tok = sum(
        (u.get("reasoning_tokens", 0) or 0)
        for u in usage_per_iteration
    )

    lines = [
        '<aside class="reasoning">',
        '  <details class="reasoning-outer" open>',
        "    <summary>",
        '      <span class="rs-badge">🧠 Reasoning trace</span>',
        f"      <span class='rs-meta'>{len(records)} iterations · "
        f"{total_chars:,} chars · {total_reasoning_tok:,} reasoning tokens · "
        f"<em>provider={_h(provider_name)}</em></span>",
        "    </summary>",
        '    <p class="rs-note">',
        "      由 <b>API reasoning 通道</b>捕获（非模型文本输出）。",
        "      每块对应 agent loop 的一次 provider 调用（iteration）。",
        "    </p>",
    ]

    for i, rec in enumerate(records):
        it = rec.get("iteration", i)
        text = (rec.get("text") or "").strip()
        chars = rec.get("char_count", len(text))
        upi = usage_per_iteration[i] if i < len(usage_per_iteration) else {}
        rt = upi.get("reasoning_tokens", 0) or 0
        ct = upi.get("completion_tokens", 0) or upi.get("output_tokens", 0) or 0

        lines += [
            f'    <details class="rs-iter" open>',
            "      <summary>",
            f"        <span class='iter-n'>iter {_h(it)}</span>",
            f"        <span class='iter-stats'>{chars:,} chars · "
            f"reasoning {rt:,} tok · completion {ct:,} tok</span>",
            "      </summary>",
            '      <pre class="rs-text">' + _h(text if text else "(空)") + "</pre>",
            "    </details>",
        ]

    lines += ["  </details>", "</aside>", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Final-output block (findings / key_values / citations / ... )
# ---------------------------------------------------------------------------


def _render_findings(findings: dict[str, Any]) -> str:
    if not findings:
        return ""
    items = []
    for k, v in findings.items():
        vs = _h(str(v).strip().replace("\n", " "))
        items.append(
            f'<div class="finding">'
            f'<div class="finding-key">{_h(k)}</div>'
            f'<div class="finding-val">{vs}</div>'
            f'</div>'
        )
    return (
        '<div class="findings-card">\n'
        '<h3 class="block-h">🔑 Findings</h3>\n'
        + "\n".join(items)
        + "\n</div>\n"
    )


def _render_key_values(kvs: list[dict[str, Any]]) -> str:
    if not kvs:
        return ""
    rows = []
    for kv in kvs:
        if not isinstance(kv, dict):
            continue
        q = _h(kv.get("quantity", "?"))
        v = _h(kv.get("value", "?"))
        s = _h(kv.get("source", "?"))
        rows.append(f"<tr><td>{q}</td><td><b>{v}</b></td><td><code>{s}</code></td></tr>")
    return (
        '<div class="kv-card">\n'
        '<h3 class="block-h">📊 Key quantitative values</h3>\n'
        '<table class="kv-table">\n'
        "<thead><tr><th>Quantity</th><th>Value</th><th>Source</th></tr></thead>\n"
        "<tbody>" + "\n".join(rows) + "</tbody>\n"
        "</table>\n"
        "</div>\n"
    )


def _render_meta_row(
    citations: list[str],
    figs: list[int],
    tools_cited: list[str],
) -> str:
    bits = []
    if citations:
        chips = "".join(f"<span class='cit-chip'>{_h(c)}</span>" for c in citations[:20])
        bits.append(f'<div class="meta-row"><b>Citations</b> {chips}</div>')
    if figs:
        chips = "".join(
            f"<span class='fig-chip'>Fig {_h(x)}</span>" for x in figs
        )
        bits.append(f'<div class="meta-row"><b>Figures</b> {chips}</div>')
    if tools_cited:
        chips = "".join(f"<span class='tool-chip'>{_h(t)}</span>" for t in tools_cited)
        bits.append(
            f'<div class="meta-row"><b>Tools (model-reported)</b> {chips}</div>'
        )
    if not bits:
        return ""
    return '<div class="meta-card">' + "".join(bits) + "</div>\n"


def _render_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    if not tool_calls:
        return ""
    rows = []
    for i, tc in enumerate(tool_calls, 1):
        name = tc.get("name", "?")
        iter_ = tc.get("iteration", "?")
        args_s = _h(str(tc.get("arguments", {}))[:140])
        err = "⚠️" if tc.get("is_error") else "✓"
        dur = tc.get("duration_s", 0)
        rows.append(
            f"<tr><td>{i}</td><td>{_h(iter_)}</td><td><code>{_h(name)}</code></td>"
            f"<td>{err}</td><td>{dur:.2f}s</td>"
            f"<td><code class='args'>{args_s}</code></td></tr>"
        )
    return (
        '<details class="tool-calls-card">\n'
        f"<summary>🔧 Actual tool calls ({len(tool_calls)}) — click to expand</summary>\n"
        '<table class="tc-table">\n'
        "<thead><tr><th>#</th><th>iter</th><th>tool</th><th>ok</th>"
        "<th>dur</th><th>args</th></tr></thead>\n"
        "<tbody>" + "\n".join(rows) + "</tbody>\n"
        "</table>\n"
        "</details>\n"
    )


def _render_caveats_and_conf(caveats: list[str], confidence: str) -> str:
    out = ""
    if caveats:
        items = "".join(f"<li>{_h(c)}</li>" for c in caveats)
        out += f"<div class='caveats-card'><h3 class='block-h'>⚠️ Caveats</h3><ul>{items}</ul></div>"
    if confidence:
        out += (
            f"<div class='conf-row'>Confidence: "
            f"<span class='conf-pill conf-{_h(confidence)}'>{_h(confidence).upper()}</span>"
            "</div>"
        )
    return out + "\n"


def _render_output_block(parsed: dict[str, Any]) -> str:
    """Render the full final-output section (findings + kv + meta + ...)."""
    findings = parsed.get("findings") or {}
    key_values = parsed.get("key_values") or []
    citations = parsed.get("citations") or []
    figs = parsed.get("figures_discussed") or []
    tools_cited = parsed.get("tools_used_this_phase") or []
    caveats = parsed.get("caveats") or []
    confidence = parsed.get("confidence", "")

    parts = ['<section class="output">']
    parts.append('<h3 class="output-h">📝 Final structured output</h3>')
    parts.append(
        '<p class="output-source">由 <b>submit_&lt;phase&gt;_answer</b> '
        "工具调用的 arguments 提取（API-level schema validation）。</p>"
    )
    parts.append(_render_findings(findings))
    parts.append(_render_key_values(key_values))
    parts.append(_render_meta_row(citations, figs, tools_cited))
    parts.append(_render_caveats_and_conf(caveats, confidence))
    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# One phase
# ---------------------------------------------------------------------------


def _render_phase(pdata: dict[str, Any], phase_index: int, provider_name: str) -> str:
    key = pdata.get("question_id") or pdata.get("category") or "?"
    parsed = pdata.get("answer_parsed") or {}
    short = parsed.get("short_answer", "")

    usage = pdata.get("usage") or {}
    thinking_records = pdata.get("thinking_records") or []
    usage_per_iter = pdata.get("usage_per_iteration") or []
    tool_calls = pdata.get("tool_calls") or []

    meta_chips = _format_usage(usage)
    dur = pdata.get("duration_s", "?")
    stop = pdata.get("stop_reason", "?")

    header = (
        f'<article class="phase">\n'
        f'<header class="phase-header">\n'
        f'<div class="phase-title">'
        f'<span class="phase-num">Phase {phase_index:02d}</span>'
        f'<span class="phase-key"><code>{_h(key)}</code></span>'
        f"</div>\n"
        f'<div class="phase-tldr"><b>TL;DR：</b>{_h(short)}</div>\n'
        f'<div class="phase-meta">'
        f"<span class='chip'>duration <b>{_h(dur)}s</b></span>"
        f"<span class='chip'>stop <b>{_h(stop)}</b></span>"
        f"{meta_chips}"
        f"</div>\n"
        f"</header>\n"
    )

    # Reasoning FIRST (per user preference)
    reasoning = _render_thinking_block(thinking_records, usage_per_iter, provider_name)

    # Then the final output
    output = _render_output_block(parsed)

    # Then tool-calls audit
    tool_audit = _render_tool_calls(tool_calls)

    return header + reasoning + output + tool_audit + "</article>\n"


# ---------------------------------------------------------------------------
# Full-document builder
# ---------------------------------------------------------------------------


def build_analysis_md(
    run_dir: Path,
    include_reasoning: bool = True,
) -> str:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "session_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}

    paper_info = manifest.get("paper_info") or {}
    label = manifest.get("label", run_dir.name)
    provider = manifest.get("provider", "?")
    model = manifest.get("model", "?")

    phase_files = sorted(run_dir.glob("phase_??_*.json"))
    phase_files = [p for p in phase_files if not re.search(r"_round\d+\.json$", p.name)]

    # Totals
    total_reasoning = 0
    total_completion = 0
    total_prompt = 0
    total_duration = 0.0
    phase_entries: list[dict[str, Any]] = []
    for pf in phase_files:
        try:
            d = json.loads(pf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        u = d.get("usage") or {}
        total_reasoning += u.get("reasoning_tokens", 0) or 0
        total_completion += (
            u.get("completion_tokens", 0) or u.get("output_tokens", 0) or 0
        )
        total_prompt += u.get("prompt_tokens", 0) or u.get("input_tokens", 0) or 0
        total_duration += d.get("duration_s", 0.0) or 0.0
        phase_entries.append(d)

    # Header / cover
    arxiv = paper_info.get("arxiv_id", "")
    title = paper_info.get("title", "")

    toc_items = "".join(
        f'<li><a href="#phase-{i}">Phase {i:02d} · '
        f"{_h(d.get('question_id') or '?')}"
        f"</a></li>"
        for i, d in enumerate(phase_entries, 1)
    )

    cover = (
        f"<header class='cover'>\n"
        f"<div class='cover-eyebrow'>hep-copilot · Protocol Analysis</div>\n"
        f"<h1 class='cover-title'>{_h(label)}</h1>\n"
        f"<div class='cover-paper'>\n"
        f"<div class='cover-paper-title'>{_h(title)}</div>\n"
        f"<div class='cover-paper-id'>arXiv:"
        f"<a href='https://arxiv.org/abs/{_h(arxiv)}'>{_h(arxiv)}</a></div>\n"
        f"</div>\n"
        f"<dl class='cover-meta'>\n"
        f"<dt>Provider</dt><dd><code>{_h(provider)}</code></dd>\n"
        f"<dt>Model</dt><dd><code>{_h(model)}</code></dd>\n"
        f"<dt>Phases</dt><dd>{len(phase_entries)}</dd>\n"
        f"<dt>Duration</dt><dd>{total_duration:.1f}s</dd>\n"
        f"<dt>Tokens</dt><dd>"
        f"reasoning <b>{total_reasoning:,}</b> · "
        f"completion <b>{total_completion:,}</b> · "
        f"prompt <b>{total_prompt:,}</b>"
        f"</dd>\n"
        f"</dl>\n"
        f"<nav class='cover-toc'>\n"
        f"<h3>Contents</h3>\n"
        f"<ol>{toc_items}</ol>\n"
        f"</nav>\n"
        f"</header>\n"
        f"<div class='page-break'></div>\n"
    )

    # Per-phase bodies
    phase_html: list[str] = []
    for i, d in enumerate(phase_entries, 1):
        if not include_reasoning:
            d = {**d, "thinking_records": [], "thinking": ""}
        phase_html.append(f'<a id="phase-{i}"></a>')
        phase_html.append(_render_phase(d, i, provider))
        phase_html.append("<div class='page-break'></div>")

    body_html = cover + "\n".join(phase_html)

    # We wrap the whole thing in a markdown file with a single raw-HTML
    # block. Markdown.py renders the HTML verbatim.
    return body_html


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------


_PDF_CSS = """
@page {
  size: A4;
  margin: 20mm 18mm 22mm 18mm;
  @bottom-right {
    content: counter(page) " / " counter(pages);
    font-size: 9pt;
    color: #888;
  }
  @bottom-left {
    content: "hep-copilot · ANALYSIS";
    font-size: 9pt;
    color: #b0b0b0;
    font-family: "Noto Sans", sans-serif;
  }
}
@page cover {
  margin: 28mm 22mm;
  @bottom-right { content: ""; }
  @bottom-left { content: ""; }
}

html {
  font-family: "Noto Sans CJK SC", "Noto Sans",
               "Source Han Sans SC", "AR PL UMing CN",
               "DejaVu Sans", sans-serif;
  font-size: 10.2pt;
  line-height: 1.55;
  color: #1f2328;
  -weasy-hyphens: auto;
}

code, pre, .mono {
  font-family: "Noto Sans Mono", "DejaVu Sans Mono",
               "Source Han Mono SC", "Courier New", monospace;
}
code {
  background: #f3f4f8;
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 88%;
  color: #3b3f44;
}

h1, h2, h3 {
  font-family: "Noto Sans CJK SC", "Noto Sans", sans-serif;
  color: #1f2f4b;
}

.page-break { page-break-after: always; height: 0; }

/* ---------- Cover ---------- */
header.cover {
  page: cover;
  padding-top: 20mm;
}
.cover-eyebrow {
  font-size: 10pt;
  letter-spacing: 4px;
  color: #556;
  text-transform: uppercase;
  margin-bottom: 16pt;
}
.cover-title {
  font-size: 28pt;
  font-weight: 300;
  margin: 0 0 20pt 0;
  border-bottom: 3px solid #2d4d7f;
  padding-bottom: 8pt;
  letter-spacing: 1px;
}
.cover-paper {
  background: #f4f7fc;
  border-left: 4px solid #2d4d7f;
  padding: 12pt 18pt;
  margin: 18pt 0;
}
.cover-paper-title {
  font-size: 13pt;
  line-height: 1.5;
  color: #233;
}
.cover-paper-id {
  margin-top: 4pt;
  font-size: 10pt;
  color: #556;
}
.cover-paper-id a { color: #2a5faf; text-decoration: none; }

.cover-meta {
  display: block;
  margin: 20pt 0;
  font-size: 10pt;
}
.cover-meta dt {
  display: inline-block;
  width: 82pt;
  color: #668;
  font-weight: 500;
  padding: 2pt 0;
}
.cover-meta dd {
  display: inline;
  margin: 0;
  color: #1f2328;
}
.cover-meta dd::after { content: "\\A"; white-space: pre; }
.cover-meta b { color: #2d4d7f; }
.cover-meta code { background: #eef; padding: 1px 6px; }

.cover-toc {
  margin-top: 24pt;
  background: #fafbfd;
  border: 1px solid #e3e6ee;
  border-radius: 6px;
  padding: 10pt 18pt;
}
.cover-toc h3 {
  margin: 0 0 8pt 0;
  font-size: 11pt;
  color: #2d4d7f;
}
.cover-toc ol {
  margin: 0;
  padding-left: 20pt;
  font-size: 10pt;
  line-height: 1.9;
}
.cover-toc a { color: #2a5faf; text-decoration: none; }

/* ---------- Phase ---------- */
article.phase {
  page-break-before: always;
  margin-top: 0;
}
header.phase-header {
  border-bottom: 2px solid #dde2ec;
  padding-bottom: 8pt;
  margin-bottom: 10pt;
}
.phase-title {
  display: flex;
  align-items: baseline;
  gap: 12pt;
  margin-bottom: 6pt;
}
.phase-num {
  font-size: 14pt;
  font-weight: 600;
  color: #2d4d7f;
  letter-spacing: 1px;
}
.phase-key code {
  font-size: 12pt;
  background: #2d4d7f;
  color: #fff;
  padding: 2pt 8pt;
  border-radius: 3px;
}
.phase-tldr {
  font-size: 11pt;
  line-height: 1.6;
  margin: 6pt 0 8pt 0;
  color: #223;
}
.phase-meta {
  display: block;
  font-size: 8.5pt;
  color: #666;
  margin-bottom: 4pt;
}

.chip {
  display: inline-block;
  background: #eef1f7;
  padding: 1px 7px;
  border-radius: 10px;
  margin-right: 4pt;
  font-size: 8.5pt;
  color: #455;
}
.chip b { color: #2d4d7f; }

/* ---------- Reasoning aside (FIRST in every phase) ---------- */
aside.reasoning, aside.reasoning-empty {
  margin: 10pt 0 14pt 0;
  background: #fffaf2;
  border: 1px solid #e8cfa6;
  border-left: 4px solid #d69d3f;
  border-radius: 5px;
  padding: 6pt 10pt;
}
aside.reasoning-empty {
  font-size: 9pt;
  color: #987;
}

details.reasoning-outer { padding: 0; }
details.reasoning-outer > summary {
  list-style: none;
  cursor: default;
  font-size: 10.5pt;
  padding: 2pt 0;
}
details.reasoning-outer > summary::-webkit-details-marker { display: none; }

.rs-badge {
  display: inline-block;
  background: #d69d3f;
  color: #fff;
  padding: 2pt 9pt;
  border-radius: 3px;
  font-size: 9pt;
  font-weight: 600;
  margin-right: 8pt;
  letter-spacing: 0.5px;
}
.rs-meta {
  font-size: 9pt;
  color: #855;
}
.rs-meta em { color: #987; }

.rs-note {
  font-size: 8.5pt;
  color: #865;
  margin: 2pt 0 6pt 0;
  font-style: italic;
}

details.rs-iter {
  background: #fff;
  border: 1px solid #ecd7b5;
  border-radius: 3px;
  margin: 5pt 0;
  padding: 4pt 8pt;
  page-break-inside: auto;
}
details.rs-iter > summary {
  list-style: none;
  cursor: default;
  font-size: 9pt;
  color: #754;
  padding: 0;
  display: flex;
  gap: 8pt;
  align-items: baseline;
}
details.rs-iter > summary::-webkit-details-marker { display: none; }
.iter-n {
  display: inline-block;
  background: #f4e7d2;
  color: #7a5312;
  padding: 1px 7px;
  border-radius: 3px;
  font-weight: 600;
  font-size: 8.5pt;
}
.iter-stats { color: #987; font-size: 8.5pt; }

pre.rs-text {
  margin: 4pt 0 2pt 0;
  padding: 6pt 9pt;
  background: #fffdf9;
  border: 1px dashed #e8cfa6;
  border-radius: 3px;
  font-size: 8.2pt;
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
  color: #3b2f1f;
}

/* ---------- Output section ---------- */
section.output {
  margin-top: 8pt;
}
.output-h {
  font-size: 12pt;
  margin: 0 0 4pt 0;
  padding-bottom: 3pt;
  border-bottom: 2px solid #2d4d7f;
  color: #2d4d7f;
  letter-spacing: 0.3px;
}
.output-source {
  font-size: 8.5pt;
  color: #667;
  margin: 0 0 8pt 0;
  font-style: italic;
}

.block-h {
  font-size: 10pt;
  margin: 10pt 0 5pt 0;
  color: #334a70;
}

.findings-card {
  margin: 6pt 0;
}
.finding {
  display: grid;
  grid-template-columns: 170pt 1fr;
  gap: 10pt;
  padding: 6pt 0;
  border-bottom: 1px dotted #dde2ec;
  font-size: 10pt;
  page-break-inside: avoid;
}
.finding:last-child { border-bottom: none; }
.finding-key {
  font-family: "Noto Sans Mono", monospace;
  font-size: 9pt;
  color: #2d4d7f;
  background: #f0f3f9;
  padding: 2pt 7pt;
  border-radius: 3px;
  align-self: start;
  font-weight: 500;
  word-break: break-all;
}
.finding-val {
  line-height: 1.55;
  color: #233;
}

.kv-card {
  margin: 10pt 0;
  page-break-inside: avoid;
}
table.kv-table, table.tc-table {
  border-collapse: collapse;
  width: 100%;
  font-size: 9pt;
  margin: 4pt 0;
}
table.kv-table th, table.kv-table td,
table.tc-table th, table.tc-table td {
  border: 1px solid #dde2ec;
  padding: 5pt 8pt;
  vertical-align: top;
  text-align: left;
}
table.kv-table th, table.tc-table th {
  background: #f4f7fc;
  color: #2d4d7f;
  font-weight: 600;
  font-size: 9pt;
}
table.kv-table td b { color: #0a3a74; }

/* Citation / figure / tool chips */
.meta-card {
  margin: 8pt 0;
  padding: 6pt 10pt;
  background: #fafbfd;
  border: 1px solid #e8ebf3;
  border-radius: 4px;
  page-break-inside: avoid;
}
.meta-row {
  font-size: 9pt;
  margin: 3pt 0;
  line-height: 1.7;
}
.meta-row b {
  display: inline-block;
  width: 140pt;
  color: #556;
  font-size: 8.5pt;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.cit-chip, .fig-chip, .tool-chip {
  display: inline-block;
  font-family: "Noto Sans Mono", monospace;
  font-size: 8.5pt;
  padding: 1px 7px;
  margin: 1pt 3pt 1pt 0;
  border-radius: 3px;
}
.cit-chip { background: #e7f1fb; color: #1a4571; border: 1px solid #bcd4ed; }
.fig-chip { background: #fce7eb; color: #7e2340; border: 1px solid #e9c2ca; }
.tool-chip { background: #edf5ed; color: #234d23; border: 1px solid #c6dac6; }

/* Caveats + confidence */
.caveats-card {
  margin: 8pt 0;
  padding: 6pt 12pt;
  background: #fff5f3;
  border-left: 3px solid #c54a2d;
  border-radius: 3px;
  page-break-inside: avoid;
}
.caveats-card ul { margin: 4pt 0; padding-left: 18pt; font-size: 9.5pt; }
.caveats-card li { margin: 2pt 0; color: #4a2114; }

.conf-row {
  margin: 8pt 0 4pt 0;
  font-size: 10pt;
  color: #445;
}
.conf-pill {
  display: inline-block;
  padding: 2pt 10pt;
  margin-left: 4pt;
  border-radius: 12pt;
  font-weight: 600;
  font-size: 9pt;
  letter-spacing: 1px;
}
.conf-high { background: #d5eed5; color: #1c4a1c; }
.conf-medium { background: #fff0c8; color: #6a4a10; }
.conf-low { background: #f8cfcf; color: #6b1818; }

/* Tool calls audit */
details.tool-calls-card {
  margin: 10pt 0;
  padding: 6pt 10pt;
  background: #f7f8fa;
  border: 1px dashed #c0c5d0;
  border-radius: 4px;
  font-size: 9pt;
}
details.tool-calls-card > summary {
  cursor: default;
  font-weight: 500;
  color: #556;
}
table.tc-table code.args {
  font-size: 8pt;
  background: #f0f3f9;
  padding: 1px 5px;
  color: #445;
}

/* Links */
a { color: #2a5faf; }
a:hover { color: #143d76; }

/* Block mono code blocks */
pre:not(.rs-text) {
  background: #f7f8fa;
  border: 1px solid #e2e4eb;
  padding: 8pt 10pt;
  border-radius: 4px;
  font-size: 9pt;
  line-height: 1.4;
  white-space: pre-wrap;
  word-break: break-word;
}
"""


def md_to_pdf(
    md_or_html: str, output_path: Path, title: str = "Protocol Analysis"
) -> Path:
    """Render (HTML-heavy) markdown text to PDF via weasyprint.

    The MD content we generate is mostly raw HTML with CSS classes.
    We still pass it through ``markdown`` so plain markdown fragments
    (like the TOC links in the cover) get rendered too.
    """

    try:
        import markdown as md  # type: ignore
    except ImportError as e:
        raise ImportError("markdown lib required for export") from e

    html_body = md.markdown(
        md_or_html,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
            "md_in_html",
        ],
    )

    # Force-open all <details> for PDF static rendering.
    open_html = re.sub(r"<details(?![^>]* open\b)", r"<details open", html_body)

    full_html = (
        "<!doctype html>\n"
        "<html><head><meta charset='utf-8'>\n"
        f"<title>{_h(title)}</title>\n"
        f"<style>{_PDF_CSS}</style>\n"
        "</head><body>\n"
        f"{open_html}\n"
        "</body></html>"
    )

    try:
        from weasyprint import HTML  # type: ignore
    except ImportError as e:
        raise ImportError("weasyprint is required for PDF export") from e

    output_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=full_html).write_pdf(str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Run-dir export (MD + PDF)
# ---------------------------------------------------------------------------


def export_run_dir(
    run_dir: Path,
    include_reasoning: bool = True,
    write_pdf: bool = True,
) -> dict[str, Path]:
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    body = build_analysis_md(run_dir, include_reasoning=include_reasoning)
    md_path = run_dir / "ANALYSIS.md"
    md_path.write_text(body, encoding="utf-8")

    results = {"md": md_path}
    if write_pdf:
        pdf_path = run_dir / "ANALYSIS.pdf"
        md_to_pdf(body, pdf_path, title=f"ANALYSIS — {run_dir.name}")
        results["pdf"] = pdf_path

    return results


def export_all_under(
    base_dir: Path,
    include_reasoning: bool = True,
    write_pdf: bool = True,
) -> list[dict[str, Path]]:
    base_dir = Path(base_dir)
    out: list[dict[str, Path]] = []
    for pat in ("teacher_*", "student_native_*", "student_aligned_*"):
        for d in sorted(base_dir.glob(pat)):
            if not d.is_dir():
                continue
            if not list(d.glob("phase_??_*.json")):
                continue
            try:
                r = export_run_dir(
                    d, include_reasoning=include_reasoning, write_pdf=write_pdf
                )
                out.append({"run_dir": d, **r})
                print(f"  exported: {d.name}  → md+pdf")
            except Exception as e:
                print(f"  FAILED: {d.name}  — {type(e).__name__}: {e}")
    return out
