"""LLM provider abstraction for hep-copilot.

Unified streaming event protocol over multiple backends:
- OpenAI Responses API (GPT-5.4, reasoning summary)
- DeepSeek-reasoner (V3.2 thinking, raw CoT via reasoning_content)
"""

from .base import (
    Event,
    Message,
    Provider,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    ToolSpec,
    TurnEnd,
    build_provider,
)

__all__ = [
    "Event",
    "Message",
    "Provider",
    "TextDelta",
    "ThinkingDelta",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "TurnEnd",
    "build_provider",
]
