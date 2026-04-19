"""Smoke tests for the agent loop with a scripted provider."""

from __future__ import annotations

import json

from hep_cot.agent.loop import AgentLoop
from hep_cot.agent.state import AgentState
from hep_cot.agent.tool_registry import ToolRegistry
from hep_cot.llm.base import TextDelta, ThinkingDelta
from tests.fakes import ScriptedProvider, make_tool_call, make_turn_end


def _echo_registry() -> ToolRegistry:
    reg = ToolRegistry()

    def echo(text: str) -> dict[str, str]:
        return {"echoed": text}

    reg.register(
        name="echo",
        description="echo the input",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        func=echo,
    )
    return reg


def test_simple_text_turn():
    provider = ScriptedProvider(
        [
            [
                ThinkingDelta(text="thinking..."),
                TextDelta(text="Hello, "),
                TextDelta(text="world."),
                make_turn_end(stop="stop", usage={"output_tokens": 5}),
            ]
        ]
    )
    state = AgentState(system_prompt="SYS")
    loop = AgentLoop(provider=provider, registry=ToolRegistry(), state=state)

    result = loop.run_round("hi")

    assert result.text == "Hello, world."
    assert result.thinking == "thinking..."
    assert state.messages[0].role == "user"
    assert state.messages[1].role == "assistant"
    assert state.messages[1].content == "Hello, world."
    assert state.messages[1].thinking == "thinking..."
    assert state.usage_totals.get("output_tokens") == 5


def test_tool_call_round_trip():
    """Model emits a tool call, executes, then final answer."""

    provider = ScriptedProvider(
        [
            [
                ThinkingDelta(text="need to call echo"),
                make_tool_call("echo", {"text": "hi"}, cid="call_1"),
                make_turn_end(stop="tool_calls"),
            ],
            [
                TextDelta(text="Done: result received."),
                make_turn_end(stop="stop"),
            ],
        ]
    )
    state = AgentState(system_prompt="SYS")
    loop = AgentLoop(provider=provider, registry=_echo_registry(), state=state)

    result = loop.run_round("please echo")

    assert result.text == "Done: result received."
    # Expect: user, assistant(w/ tool_call), tool, assistant(final)
    roles = [m.role for m in state.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    # Tool result message content is a JSON-encoded dict.
    tool_msg = state.messages[2]
    payload = json.loads(tool_msg.content)
    assert payload == {"echoed": "hi"}
    assert tool_msg.tool_call_id == "call_1"
    # Tool call bookkeeping
    assert len(state.tool_calls_log) == 1
    tc_rec = state.tool_calls_log[0]
    assert tc_rec.name == "echo"
    assert tc_rec.is_error is False


def test_unknown_tool_returns_error():
    provider = ScriptedProvider(
        [
            [
                make_tool_call("nonexistent", {}, cid="call_x"),
                make_turn_end(stop="tool_calls"),
            ],
            [
                TextDelta(text="Could not call tool."),
                make_turn_end(stop="stop"),
            ],
        ]
    )
    state = AgentState()
    loop = AgentLoop(
        provider=provider, registry=_echo_registry(), state=state
    )

    loop.run_round("do the thing")

    # Tool result should carry an error.
    tool_msg = [m for m in state.messages if m.role == "tool"][0]
    payload = json.loads(tool_msg.content)
    assert "error" in payload


def test_max_tool_iterations_cap():
    """Provider asks for a tool forever; loop should stop at iteration cap."""

    infinite_script: list[list] = []
    for _ in range(20):
        infinite_script.append(
            [
                make_tool_call("echo", {"text": "again"}, cid=f"call_{_}"),
                make_turn_end(stop="tool_calls"),
            ]
        )
    provider = ScriptedProvider(infinite_script)
    state = AgentState()
    loop = AgentLoop(
        provider=provider,
        registry=_echo_registry(),
        state=state,
        max_tool_iterations=3,
    )

    loop.run_round("loop forever")

    # At most 3 tool executions — even though the provider wanted 20.
    assert len(state.tool_calls_log) == 3
