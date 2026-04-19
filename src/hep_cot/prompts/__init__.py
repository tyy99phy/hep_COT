"""System prompts for hep-copilot."""

from .system_prompts import (
    COPILOT_SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    STUDENT_SYSTEM_PROMPT,
    build_copilot_prompt,
    build_feedback_prompt,
)

__all__ = [
    "COPILOT_SYSTEM_PROMPT",
    "TEACHER_SYSTEM_PROMPT",
    "STUDENT_SYSTEM_PROMPT",
    "build_copilot_prompt",
    "build_feedback_prompt",
]
