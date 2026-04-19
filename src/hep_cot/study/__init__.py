"""Comparative reasoning study: protocol-driven 6-phase analysis."""

from .autoloop import run_protocol_autoloop
from .figure_coverage import (
    FigureCoverageReport,
    compute_coverage,
    figure_list_from_registry,
    total_figure_count,
)
from .judge import JudgeReport, judge_answer
from .metrics import aggregate_protocol_metrics, save_metrics
from .phase_judge import PhaseJudgeReport, judge_phase
from .phases import PHASE_JSON_SCHEMA_HINT, PROTOCOL_PHASES, ProtocolPhase, get_phase
from .protocol import Phase6ProtocolRunner, extract_phase_summary, make_runner
from .runner import (
    QuestionResult,
    augment_system_prompt_with_paper,
    build_hep_registry,
    parse_structured_answer,
    run_single_question,
)
from .student import run_student_protocol_native
from .teacher import run_teacher_protocol

__all__ = [
    "FigureCoverageReport",
    "JudgeReport",
    "PHASE_JSON_SCHEMA_HINT",
    "PROTOCOL_PHASES",
    "Phase6ProtocolRunner",
    "PhaseJudgeReport",
    "ProtocolPhase",
    "QuestionResult",
    "aggregate_protocol_metrics",
    "augment_system_prompt_with_paper",
    "build_hep_registry",
    "compute_coverage",
    "extract_phase_summary",
    "figure_list_from_registry",
    "get_phase",
    "judge_answer",
    "judge_phase",
    "make_runner",
    "parse_structured_answer",
    "run_protocol_autoloop",
    "run_single_question",
    "run_student_protocol_native",
    "run_teacher_protocol",
    "save_metrics",
    "total_figure_count",
]
