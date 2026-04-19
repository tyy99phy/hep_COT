"""Provider abstraction + unified streaming event protocol.

All LLM backends implement :class:`Provider` and emit the same sequence
of :class:`Event` objects during a turn, so :mod:`hep_cot.agent.loop`
can drive any of them identically.

Event sequence in a single turn::

    (ThinkingDelta*)  (TextDelta*)  (ToolCall*)  TurnEnd

After tool execution, :class:`ToolResult` is appended to the message
history by the caller before the next turn.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """Function-style tool specification (OpenAI-compatible).

    Both OpenAI Responses and DeepSeek chat.completions consume this
    unchanged; Anthropic-style adaptation is no longer needed after
    Claude was dropped from the provider set.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Message:
    """A single turn message in the conversation history.

    Follows the OpenAI chat message shape but with explicit thinking
    preserved for audit (never replayed to the model).
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None  # for role=tool
    name: str | None = None  # for role=tool (function name)
    thinking: str = ""  # not sent back to model; kept for audit


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass
class ThinkingDelta:
    """Incremental reasoning text.

    OpenAI provides natural-language summary; DeepSeek provides raw
    reasoning_content. The consumer should treat the text as opaque and
    store it verbatim.
    """

    text: str


@dataclass
class TextDelta:
    """Incremental final-answer text."""

    text: str


@dataclass
class ToolCall:
    """A complete function-call request from the model.

    Emitted once per tool call after its arguments have fully streamed
    in. The agent loop executes the tool, then feeds the result back
    via :class:`ToolResult` on the next turn.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result from executing a tool (bookkeeping, not emitted by providers)."""

    id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class TurnEnd:
    """Marks the end of a streamed turn."""

    stop_reason: str  # "stop" | "tool_calls" | "length" | "error"
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: Any = None  # provider-specific, for debugging


Event = ThinkingDelta | TextDelta | ToolCall | TurnEnd


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class Provider(ABC):
    """Abstract base for LLM backends."""

    name: str = "base"

    @abstractmethod
    def stream_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[Event]:
        """Stream one model turn as events.

        The provider is responsible for:
          * serializing ``messages`` and ``tools`` into its native wire format
          * streaming the response
          * emitting ``ThinkingDelta`` / ``TextDelta`` / ``ToolCall`` / ``TurnEnd``

        ``kwargs`` can carry provider-specific knobs (``reasoning_effort``,
        ``temperature``, ``max_tokens``, ...). Unknown kwargs should be
        silently ignored to keep the interface stable.
        """

    # ------------------------------------------------------------------
    # Optional helpers — subclasses may override
    # ------------------------------------------------------------------

    def supports_tools(self) -> bool:
        return True

    def supports_raw_thinking(self) -> bool:
        """Whether this provider exposes raw (not summarised) reasoning."""
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_provider(name: str, **kwargs: Any) -> Provider:
    """Instantiate a provider by short name.

    Supported names:
      * ``openai`` / ``gpt`` — :class:`hep_cot.llm.openai_responses.OpenAIResponsesProvider`
      * ``deepseek`` / ``deepseek-reasoner`` — :class:`hep_cot.llm.deepseek_reasoner.DeepSeekReasonerProvider`
    """

    key = name.lower().strip()

    if key in ("openai", "gpt", "gpt-5", "gpt-5.4"):
        from .openai_responses import OpenAIResponsesProvider

        return OpenAIResponsesProvider(**kwargs)
    if key in ("deepseek", "deepseek-reasoner", "ds-r"):
        from .deepseek_reasoner import DeepSeekReasonerProvider

        return DeepSeekReasonerProvider(**kwargs)

    raise ValueError(
        f"Unknown provider '{name}'. "
        "Supported: 'openai' (GPT-5.4), 'deepseek' (DeepSeek-reasoner V3.2)."
    )


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v is not None and v.strip() else default
