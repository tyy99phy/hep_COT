"""hep-copilot: HEP paper literature reasoning + comparative CoT study."""

__version__ = "0.2.0"

# Load secrets (~/.codex/hep-copilot.env) into os.environ at package import
# so downstream provider construction picks up API keys without requiring
# callers to source a file first. Idempotent.
from .llm.deepseek_reasoner import _load_env_file_once as _load_secrets  # noqa: E402

_load_secrets()

from .agent.loop import AgentLoop
from .agent.state import AgentState
from .agent.tool_registry import ToolRegistry, ToolSchema
from .llm.base import (
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
    "AgentLoop",
    "AgentState",
    "Event",
    "Message",
    "Provider",
    "TextDelta",
    "ThinkingDelta",
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
    "ToolSchema",
    "ToolSpec",
    "TurnEnd",
    "build_provider",
]


__all__ = [
    "AgentLoop",
    "AgentState",
    "Event",
    "Message",
    "Provider",
    "TextDelta",
    "ThinkingDelta",
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
    "ToolSchema",
    "ToolSpec",
    "TurnEnd",
    "build_provider",
]
