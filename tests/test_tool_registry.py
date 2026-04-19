"""Tests for tool registry dispatch + error handling."""

from __future__ import annotations

import json

from hep_cot.agent.tool_registry import ToolRegistry


def test_register_and_execute_happy_path():
    reg = ToolRegistry()
    reg.register(
        name="adder",
        description="add two ints",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        func=lambda a, b: {"sum": a + b},
    )

    content, is_error, _ = reg.execute("adder", {"a": 2, "b": 3})
    assert is_error is False
    assert json.loads(content) == {"sum": 5}


def test_execute_unknown_tool():
    reg = ToolRegistry()
    content, is_error, _ = reg.execute("nope", {})
    assert is_error is True
    assert "unknown tool" in content.lower()


def test_execute_bad_arguments():
    reg = ToolRegistry()
    reg.register(
        name="need_a",
        description="requires kwarg a",
        parameters={"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
        func=lambda a: {"got": a},
    )

    content, is_error, _ = reg.execute("need_a", {})
    assert is_error is True
    assert "bad arguments" in content.lower()


def test_specs_for_provider():
    reg = ToolRegistry()
    reg.register(
        name="t1",
        description="d1",
        parameters={"type": "object", "properties": {}},
        func=lambda: None,
    )
    specs = reg.specs()
    assert len(specs) == 1
    openai_form = specs[0].to_openai()
    assert openai_form["type"] == "function"
    assert openai_form["function"]["name"] == "t1"
