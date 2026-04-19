"""Agent main loop: drive a provider through model ↔ tool iterations."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import (
    Event,
    Provider,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    TurnEnd,
)
from .state import AgentState, ToolCallRecord
from .tool_registry import ToolRegistry


@dataclass
class TurnResult:
    """Summary of one model turn (possibly with tool calls)."""

    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)


EventCallback = Callable[[Event], None]


class AgentLoop:
    """Multi-turn agent loop.

    Given a provider, a tool registry, and a conversation state, runs
    one "round" on user input: the model is called in a loop until it
    stops requesting tools.
    """

    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        state: AgentState,
        max_tool_iterations: int = 8,
        on_event: EventCallback | None = None,
    ):
        self.provider = provider
        self.registry = registry
        self.state = state
        self.max_tool_iterations = max_tool_iterations
        self.on_event = on_event

    # ------------------------------------------------------------------
    # Event emission helper
    # ------------------------------------------------------------------

    def _emit(self, event: Event) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                # UI callbacks must not break the loop.
                pass

    # ------------------------------------------------------------------
    # One-turn helper
    # ------------------------------------------------------------------

    def _run_single_turn(
        self,
        tools_filter: list[str] | None = None,
        **provider_kwargs: Any,
    ) -> TurnResult:
        """Call the provider once, collect events, return a TurnResult.

        ``tools_filter`` (optional): if given, only tools whose name is
        in this list are passed to the provider. Used by the
        force-answer logic to constrain the model to terminal tools
        (e.g. ``submit_*``) after the main tool budget is exhausted.
        """

        result = TurnResult()
        thinking_buf: list[str] = []
        text_buf: list[str] = []

        specs = self.registry.specs() if len(self.registry) else None
        if specs is not None and tools_filter is not None:
            specs = [s for s in specs if s.name in tools_filter]
            if not specs:
                specs = None

        stream: Iterator[Event] = self.provider.stream_turn(
            messages=self.state.messages,
            tools=specs,
            system=self.state.system_prompt or None,
            **provider_kwargs,
        )

        for ev in stream:
            self._emit(ev)

            if isinstance(ev, ThinkingDelta):
                thinking_buf.append(ev.text)
            elif isinstance(ev, TextDelta):
                text_buf.append(ev.text)
            elif isinstance(ev, ToolCall):
                result.tool_calls.append(ev)
            elif isinstance(ev, TurnEnd):
                result.stop_reason = ev.stop_reason
                result.usage = ev.usage

        result.text = "".join(text_buf)
        result.thinking = "".join(thinking_buf)
        return result

    # ------------------------------------------------------------------
    # Public: run a full user round
    # ------------------------------------------------------------------

    def run_round(self, user_input: str, **provider_kwargs: Any) -> TurnResult:
        """Process one user input until the model is done calling tools.

        Returns the *final* TurnResult (with the final assistant text).
        """

        self.state.add_user(user_input)

        last: TurnResult | None = None

        for _ in range(self.max_tool_iterations):
            turn = self._run_single_turn(**provider_kwargs)

            # Serialise tool_calls for Message history.
            tc_payload = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "arguments_json": json.dumps(tc.arguments, ensure_ascii=False),
                }
                for tc in turn.tool_calls
            ]

            self.state.add_assistant(
                content=turn.text,
                tool_calls=tc_payload,
                thinking=turn.thinking,
            )
            if turn.thinking and self.state.thinking_log:
                self.state.thinking_log[-1].provider = self.provider.name
            self.state.add_usage(turn.usage)

            if not turn.tool_calls:
                # Model is done calling tools. But if text is ALSO empty,
                # try a force-answer fallback before returning (avoids the
                # pathological "stop with empty output" from GPT-5.4 xhigh
                # after long tool chains).
                if not turn.text.strip():
                    self.state.next_iteration()
                    break  # fall through to force-answer below
                self.state.finish_turn()
                return turn

            # Execute each requested tool and append results to messages.
            for tc in turn.tool_calls:
                content, is_error, duration = self.registry.execute(
                    tc.name, tc.arguments
                )
                self.state.add_tool_result(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=content,
                    is_error=is_error,
                )
                self.state.tool_calls_log.append(
                    ToolCallRecord(
                        turn_index=self.state.turn_index,
                        iteration_index=self.state.iteration_index,
                        id=tc.id,
                        name=tc.name,
                        arguments=tc.arguments,
                        result=content,
                        is_error=is_error,
                        duration_s=duration,
                    )
                )

            last = turn
            self.state.next_iteration()
            # Continue loop: call the model again with tool results.

        # Ran out of iterations without the model stopping. First try
        # to force the model to submit its answer via the dedicated
        # ``submit_*`` tool — by restricting the tool set to just those
        # names, the model has no other option than to terminate. This
        # is the Option-B-native end-of-round path.
        submit_names = [n for n in self.registry.names() if n.startswith("submit_")]
        final: TurnResult | None = None

        if submit_names:
            self.state.add_user(
                "**工具调用预算已用完。请立即调用 "
                f"`{submit_names[0]}` 工具**提交结构化答案即可结束本阶段，"
                "不要再请求其他工具，也不要输出文本。submit 工具的 "
                "arguments 就是 harness 唯一读取的结构化结果；即便某些"
                " findings 键没有完全确定的答案，也把 `[uncertain]` 填入"
                "该键并在 caveats 里解释——宁可填 [uncertain] 也不要空着。"
            )
            # Run a full single-turn loop (with tool execution) but only
            # allow the submit tool. If the model complies, its args
            # get captured via the registry's _submit callback.
            try:
                submit_turn = self._run_single_turn(
                    tools_filter=submit_names, **provider_kwargs
                )
            except Exception:
                submit_turn = None

            if submit_turn is not None:
                # Serialise and execute any tool_calls (should be just submit).
                tc_payload = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "arguments_json": json.dumps(
                            tc.arguments, ensure_ascii=False
                        ),
                    }
                    for tc in submit_turn.tool_calls
                ]
                self.state.add_assistant(
                    content=submit_turn.text,
                    tool_calls=tc_payload,
                    thinking=submit_turn.thinking,
                )
                if submit_turn.thinking and self.state.thinking_log:
                    self.state.thinking_log[-1].provider = self.provider.name
                self.state.add_usage(submit_turn.usage)

                for tc in submit_turn.tool_calls:
                    content, is_error, duration = self.registry.execute(
                        tc.name, tc.arguments
                    )
                    self.state.add_tool_result(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content=content,
                        is_error=is_error,
                    )
                    self.state.tool_calls_log.append(
                        ToolCallRecord(
                            turn_index=self.state.turn_index,
                            iteration_index=self.state.iteration_index,
                            id=tc.id,
                            name=tc.name,
                            arguments=tc.arguments,
                            result=content,
                            is_error=is_error,
                            duration_s=duration,
                        )
                    )
                final = submit_turn

        # Last-ditch safety net: some models will still refuse to call
        # the submit tool and insist on emitting text. Give them one
        # more call with NO tools so at least we get a textual JSON we
        # can parse as a fallback.
        if final is None or (not final.tool_calls and not final.text.strip()):
            self.state.add_user(
                "如果你无法调用 submit 工具，就请直接在文本里输出完整的"
                " JSON 对象（符合 schema），不要再拒绝应答。"
            )
            fallback_kwargs = dict(provider_kwargs)
            fallback_kwargs["reasoning_effort"] = "low"
            fallback_kwargs["summary"] = "concise"
            try:
                final = self._run_single_turn_no_tools(**fallback_kwargs)
            except Exception:
                pass

            if final is not None:
                self.state.add_assistant(
                    content=final.text,
                    tool_calls=[],
                    thinking=final.thinking,
                )
                if final.thinking and self.state.thinking_log:
                    self.state.thinking_log[-1].provider = self.provider.name
                self.state.add_usage(final.usage)

        self.state.finish_turn()
        return final or last or TurnResult(stop_reason="max_iterations")

    # ------------------------------------------------------------------
    # Helper: call provider with no tools (used when tool budget is exhausted)
    # ------------------------------------------------------------------

    def _run_single_turn_no_tools(self, **provider_kwargs: Any) -> TurnResult:
        """Call provider without any tools. Emits events through ``on_event``."""

        result = TurnResult()
        thinking_buf: list[str] = []
        text_buf: list[str] = []

        stream: Iterator[Event] = self.provider.stream_turn(
            messages=self.state.messages,
            tools=None,
            system=self.state.system_prompt or None,
            **provider_kwargs,
        )

        for ev in stream:
            self._emit(ev)
            if isinstance(ev, ThinkingDelta):
                thinking_buf.append(ev.text)
            elif isinstance(ev, TextDelta):
                text_buf.append(ev.text)
            elif isinstance(ev, ToolCall):
                # Ignore — we told the provider not to call tools, but be defensive.
                pass
            elif isinstance(ev, TurnEnd):
                result.stop_reason = ev.stop_reason
                result.usage = ev.usage

        result.text = "".join(text_buf)
        result.thinking = "".join(thinking_buf)
        return result
