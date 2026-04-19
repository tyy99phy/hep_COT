"""Routes App Server events and extracts CoT-relevant data."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReasoningStep:
    """A single reasoning summary emitted during one inference call."""
    item_id: str
    summary_parts: list[str] = field(default_factory=list)
    raw_content: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolCall:
    """A tool/command executed by the agent."""
    item_id: str
    tool_type: str  # commandExecution, mcpToolCall, fileChange, webSearch
    command: str = ""
    cwd: str = ""
    output: str = ""
    status: str = ""
    exit_code: int | None = None
    duration_ms: int | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class TurnRecord:
    """All data collected from a single turn."""
    turn_id: str
    human_input: str
    reasoning_steps: list[ReasoningStep] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    agent_response: str = ""
    agent_response_phase: str = ""
    plan_text: str = ""
    phase_key: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    status: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0


class EventRouter:
    """Processes App Server notifications and builds structured turn records."""

    def __init__(self):
        self._current_turn: TurnRecord | None = None
        self._current_reasoning: ReasoningStep | None = None
        self._current_reasoning_summary_idx: int = -1
        self._completed_turns: list[TurnRecord] = []
        self._agent_message_buffer: str = ""

        # Callbacks
        self.on_reasoning_delta: list[Any] = []
        self.on_agent_message_delta: list[Any] = []
        self.on_tool_started: list[Any] = []
        self.on_tool_completed: list[Any] = []
        self.on_turn_completed: list[Any] = []
        self.on_turn_started: list[Any] = []
        self.on_error: list[Any] = []

    @property
    def current_turn(self) -> TurnRecord | None:
        return self._current_turn

    @property
    def completed_turns(self) -> list[TurnRecord]:
        return self._completed_turns

    def handle_event(self, msg: dict) -> None:
        method = msg.get("method", "")
        params = msg.get("params", {})

        if method == "turn/started":
            self._handle_turn_started(params)
        elif method == "turn/completed":
            self._handle_turn_completed(params)
        elif method == "error":
            self._handle_error(params)
        elif method == "item/started":
            self._handle_item_started(params)
        elif method == "item/completed":
            self._handle_item_completed(params)
        elif method == "item/reasoning/summaryTextDelta":
            self._handle_reasoning_summary_delta(params)
        elif method == "item/reasoning/summaryPartAdded":
            self._handle_reasoning_summary_part_added(params)
        elif method == "item/reasoning/textDelta":
            self._handle_reasoning_text_delta(params)
        elif method == "item/agentMessage/delta":
            self._handle_agent_message_delta(params)
        elif method == "item/commandExecution/outputDelta":
            self._handle_command_output_delta(params)
        elif method == "item/plan/delta":
            self._handle_plan_delta(params)

    def _handle_turn_started(self, params: dict) -> None:
        turn = params.get("turn", {})
        turn_id = turn.get("id", f"turn_{time.time()}")
        self._current_turn = TurnRecord(
            turn_id=turn_id,
            human_input="",
            start_time=time.time(),
        )
        self._agent_message_buffer = ""
        for cb in self.on_turn_started:
            cb(self._current_turn)

    def _handle_turn_completed(self, params: dict) -> None:
        if not self._current_turn:
            return
        self._finalize_current_reasoning()
        self._current_turn.agent_response = self._agent_message_buffer

        turn_data = params.get("turn", {})
        self._current_turn.status = (
            turn_data.get("status")
            or params.get("status")
            or "completed"
        )
        self._current_turn.usage = params.get("usage", {})
        self._current_turn.end_time = time.time()
        self._completed_turns.append(self._current_turn)
        for cb in self.on_turn_completed:
            cb(self._current_turn)
        self._current_turn = None

    def _handle_error(self, params: dict) -> None:
        error = params.get("error", {})
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        will_retry = params.get("willRetry", True)
        for cb in self.on_error:
            cb(message, will_retry)

    def _handle_item_started(self, params: dict) -> None:
        item = params.get("item", {})
        item_type = item.get("type", "")

        if item_type == "reasoning":
            self._finalize_current_reasoning()
            self._current_reasoning = ReasoningStep(
                item_id=item.get("id", ""),
                timestamp=time.time(),
            )
            self._current_reasoning_summary_idx = -1

        elif item_type == "userMessage":
            if self._current_turn:
                content = item.get("content", [])
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                self._current_turn.human_input = "\n".join(texts)

        elif item_type == "commandExecution":
            if self._current_turn:
                tc = ToolCall(
                    item_id=item.get("id", ""),
                    tool_type="commandExecution",
                    command=item.get("command", ""),
                    cwd=item.get("cwd", ""),
                    status="in_progress",
                    timestamp=time.time(),
                )
                self._current_turn.tool_calls.append(tc)
                for cb in self.on_tool_started:
                    cb(tc)

        elif item_type == "mcpToolCall":
            if self._current_turn:
                tc = ToolCall(
                    item_id=item.get("id", ""),
                    tool_type="mcpToolCall",
                    command=f"{item.get('server', '')}::{item.get('tool', '')}",
                    status="in_progress",
                    timestamp=time.time(),
                )
                self._current_turn.tool_calls.append(tc)
                for cb in self.on_tool_started:
                    cb(tc)

        elif item_type == "fileChange":
            if self._current_turn:
                changes = item.get("changes", [])
                desc = "; ".join(
                    f"{c.get('kind', '?')} {c.get('path', '?')}" for c in changes
                )
                tc = ToolCall(
                    item_id=item.get("id", ""),
                    tool_type="fileChange",
                    command=desc,
                    status="in_progress",
                    timestamp=time.time(),
                )
                self._current_turn.tool_calls.append(tc)
                for cb in self.on_tool_started:
                    cb(tc)

        elif item_type == "webSearch":
            if self._current_turn:
                tc = ToolCall(
                    item_id=item.get("id", ""),
                    tool_type="webSearch",
                    command=item.get("query", ""),
                    status="in_progress",
                    timestamp=time.time(),
                )
                self._current_turn.tool_calls.append(tc)
                for cb in self.on_tool_started:
                    cb(tc)

    def _handle_item_completed(self, params: dict) -> None:
        item = params.get("item", {})
        item_type = item.get("type", "")
        item_id = item.get("id", "")

        if item_type == "reasoning":
            self._finalize_current_reasoning()
            # Update with final data from completed event
            if self._current_turn and self._current_turn.reasoning_steps:
                last = self._current_turn.reasoning_steps[-1]
                if last.item_id == item_id:
                    summary = item.get("summary", [])
                    if summary:
                        # Overwrite with the authoritative completed summary
                        final_parts = [
                            s.get("text", "") for s in summary
                            if s.get("type") == "summary_text" and s.get("text", "").strip()
                        ]
                        if final_parts:
                            last.summary_parts = final_parts
                    # Capture raw text if available in completed event
                    raw_text = item.get("text", "")
                    if raw_text and not last.raw_content:
                        last.raw_content = raw_text
            else:
                # No streaming delta was received; create step from completed data
                summary = item.get("summary", [])
                summary_parts = [
                    s.get("text", "") for s in summary
                    if s.get("type") == "summary_text" and s.get("text", "").strip()
                ]
                raw_text = item.get("text", "")
                if (summary_parts or raw_text) and self._current_turn:
                    step = ReasoningStep(
                        item_id=item_id,
                        summary_parts=summary_parts,
                        raw_content=raw_text,
                        timestamp=time.time(),
                    )
                    self._current_turn.reasoning_steps.append(step)

        elif item_type == "agentMessage":
            text = item.get("text", "")
            if text:
                self._agent_message_buffer = text
            if self._current_turn:
                self._current_turn.agent_response_phase = item.get("phase", "")

        elif item_type in ("commandExecution", "mcpToolCall", "fileChange", "webSearch"):
            if self._current_turn:
                for tc in self._current_turn.tool_calls:
                    if tc.item_id == item_id:
                        tc.status = item.get("status", "completed")
                        tc.exit_code = item.get("exitCode")
                        tc.duration_ms = item.get("durationMs")
                        if item_type == "commandExecution":
                            tc.output = item.get("aggregatedOutput", tc.output)
                        elif item_type == "mcpToolCall":
                            tc.output = str(item.get("result", ""))
                        for cb in self.on_tool_completed:
                            cb(tc)
                        break

    def _handle_reasoning_summary_delta(self, params: dict) -> None:
        delta = params.get("delta", "")
        summary_idx = params.get("summaryIndex", 0)

        if not self._current_reasoning:
            self._current_reasoning = ReasoningStep(
                item_id=params.get("itemId", ""),
                timestamp=time.time(),
            )

        if summary_idx != self._current_reasoning_summary_idx:
            if self._current_reasoning_summary_idx >= 0:
                pass  # new section
            self._current_reasoning_summary_idx = summary_idx
            self._current_reasoning.summary_parts.append("")

        if self._current_reasoning.summary_parts:
            self._current_reasoning.summary_parts[-1] += delta
        else:
            self._current_reasoning.summary_parts.append(delta)

        for cb in self.on_reasoning_delta:
            cb(delta, summary_idx)

    def _handle_reasoning_summary_part_added(self, params: dict) -> None:
        pass  # boundary marker, handled by summaryIndex in delta

    def _handle_reasoning_text_delta(self, params: dict) -> None:
        delta = params.get("delta", "")
        if not self._current_reasoning:
            self._current_reasoning = ReasoningStep(
                item_id=params.get("itemId", ""),
                timestamp=time.time(),
            )
        self._current_reasoning.raw_content += delta
        # If no summary deltas are being received, also feed the reasoning
        # callbacks so the terminal shows something.
        if not self._current_reasoning.summary_parts:
            for cb in self.on_reasoning_delta:
                cb(delta, 0)

    def _handle_agent_message_delta(self, params: dict) -> None:
        delta = params.get("delta", "")
        self._agent_message_buffer += delta
        for cb in self.on_agent_message_delta:
            cb(delta)

    def _handle_command_output_delta(self, params: dict) -> None:
        delta = params.get("delta", "")
        item_id = params.get("itemId", "")
        if self._current_turn:
            for tc in self._current_turn.tool_calls:
                if tc.item_id == item_id:
                    tc.output += delta
                    break

    def _handle_plan_delta(self, params: dict) -> None:
        delta = params.get("delta", "")
        if self._current_turn:
            self._current_turn.plan_text += delta

    def _finalize_current_reasoning(self) -> None:
        if self._current_reasoning and self._current_turn:
            self._current_turn.reasoning_steps.append(self._current_reasoning)
            self._current_reasoning = None
            self._current_reasoning_summary_idx = -1
