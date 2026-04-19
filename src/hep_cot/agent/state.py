"""Agent conversation state.

Keeps the message history in a shape suitable for replay to a
:class:`~hep_cot.llm.base.Provider`, plus parallel audit structures
(thinking, tool calls, usage) used by the session store.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import Message


@dataclass
class ThinkingRecord:
    """One contiguous block of reasoning text for a single provider call.

    For DeepSeek this is the raw ``reasoning_content`` chunk; for
    OpenAI Responses it is the concatenated ``reasoning_summary_text``.
    One record per provider invocation; within a single agent-loop
    round there can be multiple records (one per tool-use iteration).
    """

    turn_index: int
    iteration_index: int = 0  # 0-based; distinct per provider call within a round
    text: str = ""
    provider: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolCallRecord:
    """Bookkeeping for one executed tool call."""

    turn_index: int
    iteration_index: int = 0  # which provider iteration triggered this call
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    is_error: bool = False
    duration_s: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentState:
    """Mutable state for one agent conversation."""

    system_prompt: str = ""
    messages: list[Message] = field(default_factory=list)
    thinking_log: list[ThinkingRecord] = field(default_factory=list)
    tool_calls_log: list[ToolCallRecord] = field(default_factory=list)
    usage_totals: dict[str, int] = field(default_factory=dict)
    usage_per_iteration: list[dict[str, int]] = field(default_factory=list)
    turn_index: int = 0
    iteration_index: int = 0  # incremented each time a provider turn completes

    # ------------------------------------------------------------------
    # Message mutation
    # ------------------------------------------------------------------

    def add_user(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_assistant(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        thinking: str = "",
    ) -> None:
        self.messages.append(
            Message(
                role="assistant",
                content=content,
                tool_calls=tool_calls or [],
                thinking=thinking,
            )
        )
        if thinking:
            self.thinking_log.append(
                ThinkingRecord(
                    turn_index=self.turn_index,
                    iteration_index=self.iteration_index,
                    text=thinking,
                    timestamp=time.time(),
                )
            )

    def add_tool_result(
        self, tool_call_id: str, name: str, content: str, is_error: bool = False
    ) -> None:
        self.messages.append(
            Message(
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
                name=name,
            )
        )

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def add_usage(self, usage: dict[str, int]) -> None:
        for k, v in usage.items():
            if not isinstance(v, int):
                continue
            self.usage_totals[k] = self.usage_totals.get(k, 0) + v
        # Also keep the per-iteration snapshot.
        if usage:
            self.usage_per_iteration.append(
                {k: v for k, v in usage.items() if isinstance(v, int)}
            )

    def next_iteration(self) -> None:
        self.iteration_index += 1

    def finish_turn(self) -> None:
        self.turn_index += 1
        self.iteration_index = 0

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "turn_count": self.turn_index,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "tool_calls": m.tool_calls,
                    "tool_call_id": m.tool_call_id,
                    "name": m.name,
                    "thinking": m.thinking,
                }
                for m in self.messages
            ],
            "thinking_log": [
                {
                    "turn": t.turn_index,
                    "iteration": t.iteration_index,
                    "text": t.text,
                    "provider": t.provider,
                    "timestamp": t.timestamp,
                }
                for t in self.thinking_log
            ],
            "tool_calls_log": [
                {
                    "turn": tc.turn_index,
                    "iteration": tc.iteration_index,
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "result": tc.result[:2000],  # truncate in summary
                    "is_error": tc.is_error,
                    "duration_s": tc.duration_s,
                    "timestamp": tc.timestamp,
                }
                for tc in self.tool_calls_log
            ],
            "usage_totals": self.usage_totals,
            "usage_per_iteration": self.usage_per_iteration,
        }
