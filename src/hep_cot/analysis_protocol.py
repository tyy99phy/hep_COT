"""Paper Analysis Protocol -- structured multi-phase deep reading of HEP papers.

Designed from a CMS convener-like perspective: assessing both the physics
motivation and the soundness of experimental methodology, while extracting
transferable insights.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .paper_fetcher import PaperInfo

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

@dataclass
class AnalysisPhase:
    """One phase of the structured paper analysis."""
    name: str
    key: str
    prompt_template: str
    follow_up_prompts: list[str] = field(default_factory=list)
    description: str = ""


ANALYSIS_PHASES: list[AnalysisPhase] = [
    # ------------------------------------------------------------------
    # Phase 0: Paper ingestion
    # ------------------------------------------------------------------
    AnalysisPhase(
        name="Paper Overview",
        key="overview",
        description="Read the paper and produce a structured overview.",
        prompt_template="""\
I want you to carefully read the following HEP experimental paper and provide
a structured overview. This is the start of a deep analysis session -- we will
go through the paper systematically in later steps.

Paper: {title}
arXiv: {arxiv_id}
Abstract: {abstract}

{text_snippet}

Please provide:
1. A one-paragraph summary of what measurement or search this paper reports.
2. The final-state signature and the dataset used (collision energy, luminosity).
3. The key physics result(s) -- central values, uncertainties, and significance
   if applicable.
4. A list of the major analysis techniques used (bullet points).
5. A brief inventory of all figures and tables: for each one, state what it
   shows and where in the analysis narrative it belongs (motivation, strategy,
   background estimation, results, etc.).

Be precise and quantitative where the paper provides numbers.""",
    ),

    # ------------------------------------------------------------------
    # Phase 1: Physics motivation
    # ------------------------------------------------------------------
    AnalysisPhase(
        name="Physics Motivation",
        key="motivation",
        description="Analyze the physics motivation and context.",
        prompt_template="""\
Now let's examine the physics motivation of this analysis in depth.

1. What is the theoretical motivation for this measurement/search?
   Why is it interesting or important for the field?
2. What were the previous experimental constraints or measurements
   of this quantity before this paper? How does this paper improve
   on them (more data, better techniques, new channel)?
3. Is the choice of final state and kinematic regime well-justified?
   Are there alternative channels that could be competitive?
4. Does the paper clearly state what it aims to test or constrain
   in terms of BSM physics or Standard Model parameters?

Assess this from the perspective of an experienced HEP experimentalist
who understands both the theoretical landscape and practical feasibility.""",
        follow_up_prompts=[
            "Are there any implicit assumptions in the physics motivation "
            "that deserve more scrutiny?",
            "How does this analysis fit into the broader program of measurements "
            "at the LHC? What complementary measurements exist?",
        ],
    ),

    # ------------------------------------------------------------------
    # Phase 2: Analysis strategy
    # ------------------------------------------------------------------
    AnalysisPhase(
        name="Analysis Strategy",
        key="strategy",
        description="Evaluate the overall analysis strategy and event selection.",
        prompt_template="""\
Let's analyze the overall analysis strategy.

1. Describe the event selection step by step: trigger, object definitions,
   kinematic cuts, and any categorization of events.
2. For each major cut or selection requirement, what is the physics
   rationale? Is the choice reasonable given the signal topology and
   the dominant backgrounds?
3. How is the signal region defined? Are the control and validation
   regions well-designed to constrain the major backgrounds?
4. Is the analysis binned, unbinned, or using a multivariate approach?
   Is this choice well-motivated for the given signal-to-background
   ratio and available statistics?
5. Are there any aspects of the strategy that seem overly aggressive
   (risk of bias) or overly conservative (leaving sensitivity on the
   table)?

Think like a convener reviewing this analysis for approval.""",
        follow_up_prompts=[
            "Walk me through the signal/background discrimination: "
            "how effective is the selection, and what is the expected "
            "signal-to-background ratio after all cuts?",
            "Are the control regions kinematically close enough to "
            "the signal region to reliably extrapolate background predictions?",
        ],
    ),

    # ------------------------------------------------------------------
    # Phase 3: Background estimation
    # ------------------------------------------------------------------
    AnalysisPhase(
        name="Background Estimation",
        key="background",
        description="Scrutinize the background estimation methods.",
        prompt_template="""\
Now focus on how backgrounds are estimated.

1. List all significant background processes. Which are estimated
   from simulation and which from data-driven methods?
2. For each data-driven method, explain the technique in detail:
   what assumptions does it rely on, and how are those validated?
3. Are there closure tests or cross-checks that demonstrate the
   method works? What level of non-closure is observed?
4. For MC-based backgrounds, what generators and PDF sets are used?
   Are higher-order corrections (NLO, NNLO) applied where appropriate?
5. Is there any background that seems under-estimated or whose
   uncertainty might be underestimated?

Be critical and specific. Identify the weakest link in the background
estimation chain.""",
        follow_up_prompts=[
            "If you had to identify the single biggest risk in the "
            "background estimation, what would it be and why?",
            "How sensitive is the final result to the background "
            "normalization? What happens if backgrounds shift by 1 sigma?",
        ],
    ),

    # ------------------------------------------------------------------
    # Phase 4: Systematic uncertainties
    # ------------------------------------------------------------------
    AnalysisPhase(
        name="Systematic Uncertainties",
        key="systematics",
        description="Evaluate the treatment of systematic uncertainties.",
        prompt_template="""\
Let's examine the systematic uncertainty treatment.

1. What are the dominant sources of systematic uncertainty in this
   analysis? Rank them by impact on the final result.
2. For experimental systematics (JES, lepton scale, luminosity, etc.),
   are they evaluated using standard methods? Are there any
   analysis-specific complications?
3. For theoretical systematics (scale variations, PDF, parton shower),
   are they handled appropriately? Is the choice of variation range
   conservative or aggressive?
4. How are systematics correlated across categories, processes, or
   data-taking years? Are these correlations well-motivated?
5. Is there any systematic that might be missing or underestimated?
   Any sign of over-constraining nuisance parameters in the fit?

Focus on whether the error budget is honest and complete.""",
        follow_up_prompts=[
            "Are there any nuisance parameters that get pulled or "
            "constrained significantly in the fit? What does that tell us?",
            "How would the result change if the dominant systematic "
            "were doubled?",
        ],
    ),

    # ------------------------------------------------------------------
    # Phase 5: Statistical analysis and results
    # ------------------------------------------------------------------
    AnalysisPhase(
        name="Statistical Analysis & Results",
        key="statistics",
        description="Analyze the statistical methodology and physics results.",
        prompt_template="""\
Now evaluate the statistical analysis and the results.

1. What statistical framework is used (frequentist/Bayesian, profile
   likelihood, CLs, etc.)? Is it the appropriate choice?
2. How is the signal model parameterized in the fit? What are the
   observables and categories entering the likelihood?
3. Describe the main results: measured values, confidence intervals,
   upper limits, or significances. Quote the numbers.
4. Are the results consistent with the Standard Model expectation?
   If there is any tension or excess, how significant is it, and
   how robust is it to analysis choices?
5. Are goodness-of-fit checks reported? Do the post-fit distributions
   look reasonable?

Assess whether the statistical treatment is rigorous and whether the
conclusions are supported by the data.""",
        follow_up_prompts=[
            "Are the expected and observed limits consistent? If not, "
            "what does the discrepancy suggest?",
            "Were any look-elsewhere effects or trial factors accounted for?",
        ],
    ),

    # ------------------------------------------------------------------
    # Phase 6: Critical assessment and lessons
    # ------------------------------------------------------------------
    AnalysisPhase(
        name="Critical Assessment & Lessons",
        key="assessment",
        description="Provide overall assessment, lessons learned, and improvement ideas.",
        prompt_template="""\
Finally, step back and provide a critical overall assessment.

1. STRENGTHS: What does this analysis do particularly well? What
   techniques or ideas are worth learning from and applying elsewhere?
2. WEAKNESSES: What are the main limitations? Which aspects could
   be improved with more data, better methods, or additional studies?
3. METHODOLOGICAL INSIGHTS: Are there any analysis techniques used
   here that are novel or particularly effective? Could they be
   adapted for other analyses?
4. FUTURE DIRECTIONS: Given the results, what are the natural next
   steps? More data (HL-LHC projections)? Combination with other
   channels? New analysis strategies?
5. REPRODUCIBILITY: Based on the paper, could another group
   reproduce this analysis? Is the methodology described with
   sufficient detail?

Be constructive but honest. This assessment should be useful for
someone designing a similar analysis.""",
        follow_up_prompts=[
            "If you were starting a similar analysis from scratch today, "
            "what would you do differently based on what you learned here?",
            "What is the single most important lesson from this paper "
            "for the field?",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Protocol engine
# ---------------------------------------------------------------------------

def build_initial_context(paper: PaperInfo, max_text_chars: int = 60000) -> str:
    """Build the initial context string for a paper analysis session."""
    text_snippet = ""
    if paper.text_content:
        text_snippet = paper.text_content[:max_text_chars]
        if len(paper.text_content) > max_text_chars:
            text_snippet += "\n\n[... text truncated for context window ...]"

    return text_snippet


REASONING_EXTERNALIZATION_INSTRUCTION = """\

## 输出格式要求（必须严格遵守）

你的回答必须采用**编号步骤**格式，每一步对应一个独立的推理单元：

```
### Step 1: [这一步在做什么]
[具体分析内容]

### Step 2: [这一步在做什么]
[具体分析内容]

...

### Step N: 小结
[本阶段的关键结论，不超过 5 句]
```

每一步中：
- 引用论文中的具体证据（数字、表格编号、图编号）。
- 如果存在替代解读，简要说明并解释为何排除。
- 基于领域经验而非论文直接陈述的判断，用 **[经验判断]** 标注。
- 不确定之处用 **[不确定]** 标注并说明来源。

## 禁止事项

- **禁止重复前序阶段已分析的内容。** 如果前面阶段已经讨论过某个结论或数字，
  直接引用"如 Phase N 已述"即可，不要复述。
- **禁止添加任何前言、寒暄、免责声明或过渡句。** 直接从 Step 1 开始。
- **禁止输出"我无法提供 chain-of-thought"之类的声明。**

## 语言

请用中文回复。物理专业术语保留英文原文（如 cross section, luminosity,
branching ratio, signal region, control region 等），但分析、讨论和推理过程
请全部使用中文。"""


def format_phase_prompt(
    phase: AnalysisPhase,
    paper: PaperInfo,
    text_snippet: str = "",
    figure_inventory: str = "",
    previous_phases_summary: str = "",
) -> str:
    """Format a phase prompt template with paper information.

    Args:
        previous_phases_summary: Brief summary of conclusions from earlier phases.
            Injected to prevent the model from repeating prior analysis.
    """
    prompt = phase.prompt_template.format_map(defaultdict(str, {
        "title": paper.title or "Unknown",
        "arxiv_id": paper.arxiv_id or "Unknown",
        "abstract": paper.abstract or "Not available",
        "text_snippet": text_snippet,
        "doi": paper.doi or "",
        "authors": ", ".join(paper.authors[:5]) + ("..." if len(paper.authors) > 5 else ""),
    }))

    if previous_phases_summary:
        prompt += (
            f"\n\n---\n**前序阶段已完成的分析（不要重复这些内容，可直接引用）：**\n"
            f"{previous_phases_summary}\n---"
        )

    if figure_inventory and phase.key != "overview":
        prompt += (
            f"\n\n{figure_inventory}\n"
            f"If any of the figures are relevant to this phase of the analysis, "
            f"include them naturally in your discussion -- describe what you see "
            f"in the figure and how it relates to your assessment."
        )

    prompt += f"\n\n{REASONING_EXTERNALIZATION_INSTRUCTION}"

    return prompt


def get_phase_by_key(key: str) -> AnalysisPhase | None:
    """Look up a phase by its key."""
    for phase in ANALYSIS_PHASES:
        if phase.key == key:
            return phase
    return None


def extract_phase_summary(agent_response: str, phase_name: str, max_chars: int = 600) -> str:
    """Extract a concise summary from a completed phase's agent response.

    Looks for the final '小结' step; if not found, takes the last paragraph.
    """
    # Try to find a "小结" or "Summary" step
    for marker in ("### Step", "小结", "Summary", "关键结论"):
        parts = agent_response.rsplit(marker, maxsplit=1)
        if len(parts) == 2 and len(parts[1].strip()) > 50:
            summary = parts[1].strip()[:max_chars]
            return f"- **{phase_name}**: {summary}"

    # Fallback: last non-empty paragraph
    paragraphs = [p.strip() for p in agent_response.split("\n\n") if p.strip()]
    if paragraphs:
        return f"- **{phase_name}**: {paragraphs[-1][:max_chars]}"
    return f"- **{phase_name}**: (completed, no summary extracted)"


# ---------------------------------------------------------------------------
# HEP metadata schema
# ---------------------------------------------------------------------------

@dataclass
class HepExpMetadata:
    """Metadata for a paper analysis CoT session."""
    arxiv_id: str = ""
    title: str = ""
    physics_channel: str = ""
    collision_energy: str = ""
    luminosity: str = ""
    experiment: str = ""

    analysis_methods: list[str] = field(default_factory=list)
    detector_concepts: list[str] = field(default_factory=list)
    statistical_methods: list[str] = field(default_factory=list)
    background_methods: list[str] = field(default_factory=list)

    reasoning_scenarios: list[str] = field(default_factory=list)
    # e.g. strategy_design, method_evaluation, result_interpretation,
    #      anomaly_diagnosis, sensitivity_estimation, cross_check

    key_results: list[str] = field(default_factory=list)
    assessment_strengths: list[str] = field(default_factory=list)
    assessment_weaknesses: list[str] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}
