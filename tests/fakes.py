"""Fake provider + fake tools for offline testing of the agent loop."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from hep_cot.llm.base import (
    Event,
    Message,
    Provider,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolSpec,
    TurnEnd,
)


class ScriptedProvider(Provider):
    """Replays a canned sequence of events over multiple turns.

    ``script`` is a list of lists; each inner list is the events for
    one provider call.
    """

    name = "scripted"

    def __init__(self, script: list[list[Event]]):
        self._script = script
        self._call = 0

    def stream_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[Event]:
        if self._call >= len(self._script):
            yield TurnEnd(stop_reason="stop", usage={})
            return
        events = self._script[self._call]
        self._call += 1
        yield from events


def build_scripted(
    *rounds: list[Event],
) -> ScriptedProvider:
    return ScriptedProvider(list(rounds))


def make_tool_call(name: str, args: dict[str, Any], cid: str = "c1") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=args)


def make_turn_end(
    stop: str = "stop", usage: dict[str, int] | None = None
) -> TurnEnd:
    return TurnEnd(stop_reason=stop, usage=usage or {})
