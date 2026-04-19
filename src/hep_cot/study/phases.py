"""Six-phase HEP paper analysis protocol.

Each :class:`ProtocolPhase` carries the instruction template that asks
the model to analyse one aspect of the paper and to emit a structured
JSON block. The ``required_outputs`` keys are checked by
:mod:`hep_cot.study.phase_judge` for L4 completeness.

Derived from (but compacted from) :mod:`hep_cot.analysis_protocol`.
The original 7-phase sequence merged ``background_estimation`` into
``strategy`` per user request, and renamed ``statistics`` →
``stat_result`` to reflect its inclusion of the final results.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProtocolPhase:
    """One phase of the structured paper-analysis protocol."""

    key: str
    name: str
    description: str
    prompt_template: str
    required_outputs: list[str] = field(default_factory=list)
    hepdata_hint: bool = False
    figure_coverage_after: bool = False  # run coverage pass after this phase


# ---------------------------------------------------------------------------
# Per-phase prompt templates
#
# Each template is wrapped by the runner with:
#   * paper title + arxiv_id header
#   * prior_summaries (previous phases' conclusions, compact)
#   * HEPData reminder (only when hepdata_hint=True)
#   * the global JSON schema spec (injected separately via system prompt)
# ---------------------------------------------------------------------------


_OVERVIEW = """\
This is **Phase 1 (overview)** of a structured multi-phase analysis of a
HEP experimental paper. You are reading the paper for the first time.

**Tool order matters in this phase.** Before any `search_text` call,
FIRST call these three cheap discovery tools so you have a complete
picture:

  1. `get_paper_info` — confirm metadata
  2. `list_figures` — get the figure inventory
  3. `fetch_hepdata` — probe HEPData availability

Only after those three should you reach for `search_text` /
`get_paper_section` for specific facts.

Required `findings` keys (each must be non-empty):

1. `measurement_type` — one-sentence description: what measurement or
   search does this paper report? (e.g. "search for same-sign WWH
   production via VBF")
2. `final_state` — the detector-level final-state signature
   (e.g. "two same-sign leptons + ≥2 forward jets + ETmiss")
3. `dataset` — collision energy and integrated luminosity with units,
   plus year range if available (e.g. "13 TeV pp, 137 fb^-1, 2016-2018")
4. `key_result_headline` — the headline number: observed significance,
   signal strength, upper limit, or cross section measurement with
   uncertainty. Cite source.
5. `figure_inventory_ack` — total number of physics figures in the
   paper (from the `list_figures` call above), and a one-line
   categorisation (e.g. "9 figures: 2 validation, 3 BG-estimation, 2
   fit, 2 limit").
6. `hepdata_availability` — does a HEPData record exist for this paper?
   Give the outcome of the `fetch_hepdata` call above: HEPData record
   id and total number of tables, or "no record" if not found.

Do not yet comment on motivation, strategy or quality — that comes in
later phases.
"""

_MOTIVATION = """\
This is **Phase 2 (motivation)**. Prior phase summaries are above.

Analyse the **physics motivation** of this measurement/search.

Required `findings` keys:

1. `theory_motivation` — what theoretical prediction, BSM scenario, or
   SM precision question motivates this analysis? Cite the relevant
   introduction section or equations.
2. `prior_constraints` — what were the most stringent prior
   experimental constraints (or measurements) on this quantity? Cite
   the arXiv ids of prior ATLAS/CMS/LEP/Tevatron results if mentioned,
   or call `inspire_references` / `arxiv_search` if the paper's
   introduction is terse.
3. `BSM_or_SM_goal` — is this a BSM search (if so, what model and what
   parameter range) or a SM measurement (if so, what parameter and
   what precision target)?
4. `comparison_to_predecessors` — in one sentence, how does this
   analysis improve on its direct predecessor (more data / new channel
   / better technique)?

Assess from the perspective of an experienced HEP phenomenologist who
understands both theory and experimental feasibility.
"""

_STRATEGY = """\
This is **Phase 3 (strategy + background)**. Prior phase summaries are
above.

Describe the **analysis strategy**, including the background estimation.
Use `get_paper_section` to read the "Event selection", "Signal region",
and "Background estimation" (or equivalent) sections.

Required `findings` keys:

1. `trigger_selection` — triggers used (HLT paths or categories).
2. `object_definitions` — lepton, jet, photon, b-tag, MET definitions
   (pT cuts, isolation, working points). Be specific.
3. `signal_region_categorization` — how the SR is defined and whether
   it is binned, unbinned, or multiple sub-categories; describe the
   discriminating variable(s).
4. `background_processes` — list all significant backgrounds by name
   (prompt, non-prompt, charge-misID, rare SM, etc.).
5. `background_estimation_methods` — for each background, whether it
   is estimated from MC, data-driven (fake factor / ABCD / matrix
   method / sideband transfer), or hybrid. Describe each method
   precisely enough that someone could reproduce it.
6. `control_regions` — the control/validation regions and what each
   one constrains.

Be critical: if any method has obvious weak assumptions, flag them in
`caveats`.
"""

_SYST = """\
This is **Phase 4 (systematic uncertainties)**. Prior phase summaries
are above.

Review the **systematic uncertainty** treatment. A HEPData-hosted
breakdown table may be available — call `fetch_hepdata` and then
`get_hepdata_table` (looking for names like "Systematic breakdown" or
"Impact on signal strength") to pull structured numbers when possible.
Otherwise use `get_paper_section` on the "Systematic uncertainties"
section and any relevant table (`search_text` for "Table" near "syst").

Required `findings` keys:

1. `dominant_systematics` — ranked list (top 3–5) of the systematic
   uncertainties with largest impact on the final result, each with a
   numerical value and a source citation.
2. `experimental_systs` — how experimental systematics (JES, JER,
   b-tag SF, lepton scale/ID, pileup, luminosity) are evaluated. Note
   any analysis-specific complications.
3. `theoretical_systs` — how theoretical systematics (scale variation,
   PDF, parton shower, NLO/NNLO matching) are evaluated. Include the
   variation range.
4. `correlation_treatment` — are systematics correlated across
   categories, processes, data-taking years? Briefly describe the
   correlation model.
5. `impact_on_result` — the fractional impact of the dominant
   systematic(s) on the final measurement/limit.

Flag any systematic that looks under-estimated in `caveats`.
"""

_STAT_RESULT = """\
This is **Phase 5 (statistical analysis + results)**. Prior phase
summaries are above.

Analyse the **statistical methodology and the reported results**. If a
HEPData record exists, check whether a reusable likelihood / pyhf
workspace / cut-flow is provided — call `fetch_hepdata` and
`get_hepdata_table` for tables named like "Likelihood", "Covariance",
"Cut flow", etc.

Required `findings` keys:

1. `stat_framework` — frequentist or Bayesian; profile likelihood,
   CLs, hybrid, etc. Include test statistic type.
2. `fit_observables` — what is fit (binned template, unbinned ML, 2D
   etc.) and what are the observables/categories entering the
   likelihood.
3. `categories` — the category structure (number of bins, number of
   channels, how they are combined).
4. `measured_central_values` — central value(s) of the measurement(s)
   with total uncertainty. Quote sources.
5. `confidence_intervals_or_limits` — 68% / 95% CL intervals or upper
   limits (observed and expected). Quote exact numbers.
6. `sm_consistency` — whether the result is consistent with the SM
   expectation, and at what significance any excess/deficit is
   observed.

If the paper provides goodness-of-fit or look-elsewhere corrections,
note them in `caveats`.
"""

_ASSESSMENT = """\
This is **Phase 6 (critical assessment + lessons)**. All prior phase
summaries — including the figure coverage forced pass — are above.

Provide a **critical overall assessment**. Be constructive but honest.

Required `findings` keys:

1. `strengths` — what does this analysis do particularly well? Name
   specific techniques or design choices worth emulating.
2. `weaknesses` — main limitations. Which aspects could be improved
   with more data, better methods, or additional cross-checks?
3. `methodological_insights` — any technique used here that is novel
   or notably effective and could be adapted to other analyses.
4. `future_directions` — the natural next steps given these results
   (HL-LHC projection, channel combination, BSM reinterpretation,
   etc.).
5. `reproducibility_assessment` — based on the paper and its HEPData
   (if any), could another group reasonably reproduce the core result?
   Comment specifically on whether the likelihood, systematics model,
   and selection are described precisely enough.

Do NOT introduce new numerical claims here — you should only comment
on the physics/methods you have already surfaced in earlier phases.
"""


# ---------------------------------------------------------------------------
# Phase list (order matters; runner iterates in this order)
# ---------------------------------------------------------------------------


PROTOCOL_PHASES: list[ProtocolPhase] = [
    ProtocolPhase(
        key="overview",
        name="Paper Overview",
        description="Headline metadata, figure inventory, HEPData probe.",
        prompt_template=_OVERVIEW,
        required_outputs=[
            "measurement_type",
            "final_state",
            "dataset",
            "key_result_headline",
            "figure_inventory_ack",
            "hepdata_availability",
        ],
        hepdata_hint=True,
    ),
    ProtocolPhase(
        key="motivation",
        name="Physics Motivation",
        description="Theory motivation, prior constraints, goal framing.",
        prompt_template=_MOTIVATION,
        required_outputs=[
            "theory_motivation",
            "prior_constraints",
            "BSM_or_SM_goal",
            "comparison_to_predecessors",
        ],
    ),
    ProtocolPhase(
        key="strategy",
        name="Analysis Strategy + Background",
        description="Selection, categorisation, background estimation.",
        prompt_template=_STRATEGY,
        required_outputs=[
            "trigger_selection",
            "object_definitions",
            "signal_region_categorization",
            "background_processes",
            "background_estimation_methods",
            "control_regions",
        ],
    ),
    ProtocolPhase(
        key="syst",
        name="Systematic Uncertainties",
        description="Dominant systs, experimental vs theoretical breakdown, correlations.",
        prompt_template=_SYST,
        required_outputs=[
            "dominant_systematics",
            "experimental_systs",
            "theoretical_systs",
            "correlation_treatment",
            "impact_on_result",
        ],
        hepdata_hint=True,
    ),
    ProtocolPhase(
        key="stat_result",
        name="Statistical Analysis + Results",
        description="Fit framework, categories, central values, limits, SM consistency.",
        prompt_template=_STAT_RESULT,
        required_outputs=[
            "stat_framework",
            "fit_observables",
            "categories",
            "measured_central_values",
            "confidence_intervals_or_limits",
            "sm_consistency",
        ],
        hepdata_hint=True,
        figure_coverage_after=True,
    ),
    ProtocolPhase(
        key="assessment",
        name="Critical Assessment + Lessons",
        description="Strengths, weaknesses, methodological insights, future directions, reproducibility.",
        prompt_template=_ASSESSMENT,
        required_outputs=[
            "strengths",
            "weaknesses",
            "methodological_insights",
            "future_directions",
            "reproducibility_assessment",
        ],
    ),
]


def get_phase(key: str) -> ProtocolPhase | None:
    for p in PROTOCOL_PHASES:
        if p.key == key:
            return p
    return None


# ---------------------------------------------------------------------------
# Phase submission tool schema (Option B: submit via function-calling)
#
# Instead of asking the model to write a JSON object as its final text
# response (which is brittle — parse errors, format drift, "how do I
# close this bracket" polluting the reasoning channel), we give each
# phase a dedicated ``submit_<phase>_answer`` tool whose JSON Schema is
# exactly the expected phase output. The provider's function-calling
# layer enforces required fields at the API level.
# ---------------------------------------------------------------------------


def build_submit_tool_schema(phase: ProtocolPhase) -> dict:
    """Build the JSON Schema for the per-phase submit tool parameters.

    ``findings`` is dynamically schema'd per phase so each required
    output key is an explicitly required property — the provider will
    reject the tool call if any is missing.
    """
    findings_props: dict = {}
    for k in phase.required_outputs:
        findings_props[k] = {
            "type": "string",
            "description": (
                f"Value for `{k}` — 用中文自然语言（物理术语、数值单位、"
                f"引用标签保留英文）。若论文确实没有覆盖，填 `[uncertain]`。"
            ),
        }

    return {
        "type": "object",
        "properties": {
            "short_answer": {
                "type": "string",
                "description": "本阶段一句话核心结论（中文）。",
            },
            "findings": {
                "type": "object",
                "description": (
                    "本阶段要求的结构化发现；每个 required_outputs key 都"
                    "必须填充（必填字段见 required）。"
                ),
                "properties": findings_props,
                "required": list(phase.required_outputs),
                "additionalProperties": False,
            },
            "key_values": {
                "type": "array",
                "description": "本阶段关键量化数值（每条含 quantity/value/source）。",
                "items": {
                    "type": "object",
                    "properties": {
                        "quantity": {
                            "type": "string",
                            "description": "量的名字（中文，物理术语保留英文）。",
                        },
                        "value": {
                            "type": "string",
                            "description": "数值 + 单位（如 '3.2%'、'10.8 GeV'）。",
                        },
                        "source": {
                            "type": "string",
                            "description": (
                                "来源引用（§、Fig、Table、Eq、HEPData/xxx、arXiv:xxxx.xxxxx）。"
                            ),
                        },
                    },
                    "required": ["quantity", "value", "source"],
                    "additionalProperties": False,
                },
            },
            "citations": {
                "type": "array",
                "description": "所有引用列表（§、Fig、Table、arXiv:...、HEPData/...）。",
                "items": {"type": "string"},
            },
            "figures_discussed": {
                "type": "array",
                "description": "本阶段讨论过的图编号（整数）。",
                "items": {"type": "integer"},
            },
            "tools_used_this_phase": {
                "type": "array",
                "description": "本阶段用过的工具名字列表。",
                "items": {"type": "string"},
            },
            "caveats": {
                "type": "array",
                "description": "说明性警告或局限（中文）。",
                "items": {"type": "string"},
            },
            "confidence": {
                "type": "string",
                "description": "对本阶段回答的整体信心。",
                "enum": ["high", "medium", "low"],
            },
        },
        "required": [
            "short_answer",
            "findings",
            "citations",
            "confidence",
        ],
        "additionalProperties": False,
    }


def submit_tool_name(phase: ProtocolPhase) -> str:
    return f"submit_{phase.key}_answer"


def submit_tool_description(phase: ProtocolPhase) -> str:
    required = ", ".join(f"`{k}`" for k in phase.required_outputs)
    return (
        f"**[终止工具]** 本阶段 `{phase.key}` ({phase.name}) 的**最终结构化回答**。"
        f"所有信息收集完成后，调用本工具一次即结束本阶段。\n\n"
        f"调用时必须填齐 findings 下的所有键：{required}。\n"
        "不要在其他 tool 调用之前调用本工具；也不要在文本中再写 JSON——"
        "本工具的 arguments 就是 harness 唯一提取的结构化输出来源。"
    )


# ---------------------------------------------------------------------------
# Legacy JSON-schema hint (kept only for backwards compat / debugging;
# the live runner now uses submit_tool_schema instead)
# ---------------------------------------------------------------------------


PHASE_JSON_SCHEMA_HINT = """\
When you have gathered enough information, call
`submit_<phase_key>_answer` **exactly once** to finalise the phase.
The tool's arguments are the structured answer — the harness reads
them directly. Do not also write the JSON as text; the tool call IS
the output.

The schema is enforced at the API level; every required field must be
present. For required `findings` keys that the paper genuinely does
not cover, pass the string `"[uncertain]"` and list the reason in
`caveats`.

Respond in Chinese for all natural-language values; keep physics
terms, units, and citation labels (§, Fig, Table, arXiv:..., HEPData/...)
in their original English form.
"""
