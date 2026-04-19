"""OpenAI Responses API provider (GPT-5.4 and friends).

Exposes reasoning *summaries* (not raw CoT) via the Responses API, plus
function-calling. DeepSeek users should prefer :mod:`hep_cot.llm.deepseek_reasoner`
which returns raw ``reasoning_content``.

Config resolution order (first win):
  1. Explicit ``api_key`` / ``base_url`` constructor arguments
  2. Environment: ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``
  3. ``YTYFREE_BASE_URL`` environment variable (local codex proxy)
  4. Fallback: ``~/.codex/auth.json`` ``OPENAI_API_KEY``

The ytyfree setup runs a local OpenAI-compatible proxy at
``http://127.0.0.1:8317/v1`` that fronts GPT-5.4 with ``xhigh``
reasoning effort. When a codex install is present we mirror its
provider so the copilot uses the same backend Codex already talks to.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .base import (
    Event,
    Message,
    Provider,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolSpec,
    TurnEnd,
)


_CODEX_AUTH = Path.home() / ".codex" / "auth.json"


def _fallback_api_key() -> str | None:
    """Recover an API key from ``~/.codex/auth.json`` if env is empty."""
    if not _CODEX_AUTH.is_file():
        return None
    try:
        with open(_CODEX_AUTH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    key = data.get("OPENAI_API_KEY")
    if isinstance(key, str) and key.strip():
        return key.strip()
    return None


def _resolve_api_key(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    return _fallback_api_key()


def _resolve_base_url(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for name in ("OPENAI_BASE_URL", "YTYFREE_BASE_URL"):
        v = os.environ.get(name)
        if v and v.strip():
            return v.strip()
    return None


class OpenAIResponsesProvider(Provider):
    """Streams turns from OpenAI Responses API.

    Ytyfree / codex users: leave ``base_url`` unset — the provider will
    auto-pick up ``YTYFREE_BASE_URL`` and the key from ``~/.codex/auth.json``.
    """

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-5.4",
        reasoning_effort: str = "xhigh",
        summary: str = "detailed",
        api_key: str | None = None,
        base_url: str | None = None,
        max_output_tokens: int = 16384,
        **kwargs: Any,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "openai>=1.50 is required for OpenAIResponsesProvider. "
                "Install with: pip install 'openai>=1.50'"
            ) from e

        self._OpenAI = OpenAI
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.summary = summary
        self.max_output_tokens = max_output_tokens

        self._client = OpenAI(
            api_key=_resolve_api_key(api_key),
            base_url=_resolve_base_url(base_url),
        )

    def supports_tools(self) -> bool:
        return True

    def supports_raw_thinking(self) -> bool:
        # Responses API emits *summaries*, not raw CoT.
        return False

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_input(
        messages: list[Message], system: str | None
    ) -> list[dict[str, Any]]:
        """Serialise our Message list to the Responses API ``input`` format.

        The Responses API accepts a structured input list with role+content
        items, plus ``function_call`` / ``function_call_output`` items.
        """

        out: list[dict[str, Any]] = []

        if system:
            out.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system}],
                }
            )

        for msg in messages:
            if msg.role == "system":
                out.append(
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": msg.content}],
                    }
                )
            elif msg.role == "user":
                out.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": msg.content}],
                    }
                )
            elif msg.role == "assistant":
                # Emit prior text then any function calls this turn requested.
                if msg.content:
                    out.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": msg.content}],
                        }
                    )
                for tc in msg.tool_calls:
                    out.append(
                        {
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": tc.get("name", ""),
                            "arguments": tc.get("arguments_json", "{}"),
                        }
                    )
            elif msg.role == "tool":
                out.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id or "",
                        "output": msg.content,
                    }
                )

        return out

    @staticmethod
    def _tools_to_api(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        # Responses API uses a flattened function tool shape.
        return [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in tools
        ]

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[Event]:
        """Stream a turn with automatic retry on transient network errors."""

        import time as _time

        max_retries = kwargs.pop("max_retries", 3)
        backoff_base = kwargs.pop("retry_backoff", 3.0)

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                events: list[Event] = list(
                    self._stream_turn_once(messages, tools, system, **kwargs)
                )
            except Exception as e:
                last_exc = e
                ename = type(e).__name__
                transient = any(
                    k in ename
                    for k in (
                        "RemoteProtocol",
                        "ReadTimeout",
                        "ConnectTimeout",
                        "ReadError",
                        "ConnectError",
                        "APIConnection",
                        "InternalServerError",
                        "ServiceUnavailable",
                        "APITimeoutError",
                    )
                )
                if transient and attempt < max_retries:
                    wait = backoff_base * (attempt + 1)
                    print(
                        f"[openai] {ename} on attempt {attempt+1}/{max_retries+1}: "
                        f"{e}. retrying in {wait:.1f}s..."
                    )
                    _time.sleep(wait)
                    continue
                raise
            else:
                yield from events
                return

        if last_exc is not None:
            raise last_exc

    def _stream_turn_once(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        system: str | None,
        **kwargs: Any,
    ) -> Iterator[Event]:
        input_items = self._messages_to_input(messages, system)
        api_tools = self._tools_to_api(tools)

        effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        summary = kwargs.get("summary", self.summary)
        max_out = kwargs.get("max_output_tokens", self.max_output_tokens)

        req_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "reasoning": {"effort": effort, "summary": summary},
            "max_output_tokens": max_out,
            "stream": True,
        }
        if api_tools:
            req_kwargs["tools"] = api_tools

        # Partial state for assembling function calls across deltas.
        fn_call_buf: dict[str, dict[str, Any]] = {}

        stream = self._client.responses.create(**req_kwargs)

        final_usage: dict[str, int] = {}
        stop_reason: str = "stop"
        final_response: Any = None

        for ev in stream:
            etype = getattr(ev, "type", "") or ""

            # Reasoning summary streaming
            if etype == "response.reasoning_summary_text.delta":
                delta = getattr(ev, "delta", "") or ""
                if delta:
                    yield ThinkingDelta(text=delta)
                continue
            if etype == "response.reasoning_summary_part.added":
                # Boundary between summary parts — inject a separator so that
                # downstream persistence can keep them distinct.
                yield ThinkingDelta(text="\n")
                continue

            # Final answer streaming
            if etype == "response.output_text.delta":
                delta = getattr(ev, "delta", "") or ""
                if delta:
                    yield TextDelta(text=delta)
                continue

            # Function call argument streaming — assemble, emit on done.
            if etype == "response.output_item.added":
                item = getattr(ev, "item", None)
                if item is not None and getattr(item, "type", "") == "function_call":
                    fn_call_buf[getattr(item, "id", "")] = {
                        "call_id": getattr(item, "call_id", "")
                        or getattr(item, "id", ""),
                        "name": getattr(item, "name", ""),
                        "args": "",
                    }
                continue
            if etype == "response.function_call_arguments.delta":
                item_id = getattr(ev, "item_id", "")
                delta = getattr(ev, "delta", "") or ""
                if item_id in fn_call_buf:
                    fn_call_buf[item_id]["args"] += delta
                continue
            if etype == "response.function_call_arguments.done":
                item_id = getattr(ev, "item_id", "")
                buf = fn_call_buf.get(item_id)
                if buf:
                    try:
                        parsed_args = json.loads(buf["args"] or "{}")
                    except json.JSONDecodeError:
                        parsed_args = {"_raw": buf["args"]}
                    yield ToolCall(
                        id=buf["call_id"],
                        name=buf["name"],
                        arguments=parsed_args,
                    )
                continue

            # Completion
            if etype == "response.completed":
                resp = getattr(ev, "response", None)
                final_response = resp
                if resp is not None:
                    usage_obj = getattr(resp, "usage", None)
                    if usage_obj is not None:
                        final_usage = {
                            "input_tokens": getattr(usage_obj, "input_tokens", 0),
                            "output_tokens": getattr(usage_obj, "output_tokens", 0),
                            "reasoning_tokens": getattr(
                                getattr(usage_obj, "output_tokens_details", None),
                                "reasoning_tokens",
                                0,
                            ),
                        }
                # stop_reason inferred from whether tool calls were emitted.
                stop_reason = "tool_calls" if fn_call_buf else "stop"
                continue

            if etype == "response.incomplete":
                stop_reason = "length"
                continue
            if etype.startswith("response.error"):
                stop_reason = "error"
                continue

        yield TurnEnd(
            stop_reason=stop_reason, usage=final_usage, raw_response=final_response
        )
