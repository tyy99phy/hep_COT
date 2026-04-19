"""DeepSeek-reasoner provider (V3.2 thinking mode).

Uses the OpenAI-compatible ``chat.completions`` endpoint at
``https://api.deepseek.com``. The key differentiator from OpenAI is
that DeepSeek streams **raw** reasoning via ``delta.reasoning_content``
instead of a summary — the full CoT is recoverable.

Multi-turn contract (per DeepSeek docs): subsequent turns **must not**
echo ``reasoning_content`` back; only ``content`` + ``tool_calls`` are
replayed. Our :mod:`hep_cot.agent.state` honours this by storing
thinking in :attr:`Message.thinking` (audit-only) instead of
:attr:`Message.content`.
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


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-reasoner"

_SECRETS_FILE = Path.home() / ".codex" / "hep-copilot.env"


def _load_env_file_once() -> None:
    """Load KEY=VAL lines from ``~/.codex/hep-copilot.env`` into os.environ.

    We do NOT overwrite already-set env vars — explicit export wins.
    """
    if not _SECRETS_FILE.is_file():
        return
    try:
        with open(_SECRETS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except OSError:
        pass


_load_env_file_once()


class DeepSeekReasonerProvider(Provider):
    """Streams turns from DeepSeek's thinking-mode chat completions."""

    name = "deepseek"

    def __init__(
        self,
        model: str = DEFAULT_DEEPSEEK_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 16384,
        temperature: float | None = None,
        **kwargs: Any,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "openai>=1.50 is required for DeepSeekReasonerProvider "
                "(DeepSeek uses the OpenAI-compatible SDK). "
                "Install with: pip install 'openai>=1.50'"
            ) from e

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self._client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or DEFAULT_DEEPSEEK_BASE_URL,
        )

    def supports_tools(self) -> bool:
        return True

    def supports_raw_thinking(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_api(
        messages: list[Message], system: str | None
    ) -> list[dict[str, Any]]:
        """Serialise our Message list to OpenAI chat-message format.

        Thinking-mode has two opposite replay rules depending on context:

        * **Plain multi-turn chat (no tool calls in that turn)**:
          ``reasoning_content`` must be **stripped** from prior assistant
          messages or the API returns 400.
        * **Tool-calling flow (assistant turn carries tool_calls)**:
          ``reasoning_content`` must be **echoed back** in that same
          assistant message, again or the API returns 400.

        We therefore attach ``reasoning_content`` only on assistant turns
        that have ``tool_calls``.
        """

        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "system":
                out.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                out.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                api_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or None,
                }
                if msg.tool_calls:
                    api_msg["tool_calls"] = [
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": tc.get("arguments_json", "{}"),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                    # Thinking-mode tool-calling REQUIRES reasoning_content
                    # to be echoed on the assistant turn that triggered the
                    # call; omitting it returns a 400 from the API.
                    if msg.thinking:
                        api_msg["reasoning_content"] = msg.thinking
                out.append(api_msg)
            elif msg.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "",
                        "content": msg.content,
                    }
                )

        return out

    @staticmethod
    def _tools_to_api(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [t.to_openai() for t in tools]

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

        # Buffer events so a mid-stream failure can be retried cleanly.
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                events: list[Event] = list(
                    self._stream_turn_once(messages, tools, system, **kwargs)
                )
            except Exception as e:
                last_exc = e
                ename = type(e).__name__
                # Retry on known transient errors from httpx/openai SDK.
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
                        f"[deepseek] {ename} on attempt {attempt+1}/{max_retries+1}: "
                        f"{e}. retrying in {wait:.1f}s..."
                    )
                    _time.sleep(wait)
                    continue
                raise
            else:
                yield from events
                return

        # Should not reach here; raise is in the except branch.
        if last_exc is not None:
            raise last_exc

    def _stream_turn_once(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        system: str | None,
        **kwargs: Any,
    ) -> Iterator[Event]:
        api_messages = self._messages_to_api(messages, system)
        api_tools = self._tools_to_api(tools)

        max_tok = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        req_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tok,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if api_tools:
            req_kwargs["tools"] = api_tools
            req_kwargs["tool_choice"] = "auto"
        if temperature is not None:
            req_kwargs["temperature"] = temperature

        # Per-call function-call accumulator keyed by choice index.
        # DeepSeek streams tool_calls with indices and partial arguments.
        tc_buf: dict[int, dict[str, Any]] = {}
        final_usage: dict[str, int] = {}
        stop_reason: str = "stop"

        stream = self._client.chat.completions.create(**req_kwargs)

        for chunk in stream:
            # Usage appears only on the final chunk when include_usage=True.
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                final_usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                    "reasoning_tokens": getattr(
                        getattr(usage, "completion_tokens_details", None),
                        "reasoning_tokens",
                        0,
                    ),
                }

            choices = getattr(chunk, "choices", None) or []
            for choice in choices:
                delta = getattr(choice, "delta", None)
                finish = getattr(choice, "finish_reason", None)

                if delta is not None:
                    # Raw reasoning (the DeepSeek value-add).
                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        yield ThinkingDelta(text=rc)

                    # Final answer text.
                    content = getattr(delta, "content", None)
                    if content:
                        yield TextDelta(text=content)

                    # Tool call deltas.
                    tcs = getattr(delta, "tool_calls", None) or []
                    for tc in tcs:
                        idx = getattr(tc, "index", 0) or 0
                        slot = tc_buf.setdefault(
                            idx, {"id": "", "name": "", "args": ""}
                        )
                        tc_id = getattr(tc, "id", None)
                        if tc_id:
                            slot["id"] = tc_id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            name = getattr(fn, "name", None)
                            if name:
                                slot["name"] = name
                            args = getattr(fn, "arguments", None)
                            if args:
                                slot["args"] += args

                if finish:
                    stop_reason = finish
                    if finish == "tool_calls":
                        for slot in tc_buf.values():
                            try:
                                parsed_args = json.loads(slot["args"] or "{}")
                            except json.JSONDecodeError:
                                parsed_args = {"_raw": slot["args"]}
                            yield ToolCall(
                                id=slot["id"],
                                name=slot["name"],
                                arguments=parsed_args,
                            )

        yield TurnEnd(stop_reason=stop_reason, usage=final_usage, raw_response=None)
