"""Interactive REPL for hep-copilot.

Loads a paper, wires up tools, drives an :class:`AgentLoop`, streams
events to the terminal, and persists session data to
``./cot_sessions/`` via the existing :class:`CotStore`.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from ..agent.loop import AgentLoop
from ..agent.state import AgentState
from ..agent.tool_registry import ToolRegistry
from ..cot_store import CotStore
from ..event_router import TurnRecord
from ..llm.base import (
    Event,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    TurnEnd,
    build_provider,
)
from ..prompts import build_copilot_prompt
from ..tools import (
    build_arxiv_search_tool,
    build_figure_tools,
    build_hepdata_tools,
    build_inspire_tools,
    build_paper_tools,
)
from ..tools.paper_tools import PaperContext
from ..ui import blue, bold, cyan, dim, green, magenta, red, yellow


BANNER = """\
==================================================
  hep-copilot — HEP paper reasoning assistant
==================================================
"""


class CopilotRepl:
    """REPL main class."""

    def __init__(
        self,
        provider_name: str = "openai",
        model: str | None = None,
        paper_identifier: str | None = None,
        cache_dir: str = "./paper_cache",
        output_dir: str = "./cot_sessions",
        max_tool_iterations: int = 8,
        reasoning_effort: str = "high",
        show_thinking: bool = True,
    ):
        self.provider_name = provider_name
        self.paper_identifier = paper_identifier
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.show_thinking = show_thinking
        self.reasoning_effort = reasoning_effort

        provider_kwargs: dict[str, Any] = {}
        if model:
            provider_kwargs["model"] = model
        # OpenAI uses this; DeepSeek silently ignores unknown kwargs.
        provider_kwargs["reasoning_effort"] = reasoning_effort
        self.provider = build_provider(provider_name, **provider_kwargs)

        # Paper-shared context for paper + figure tools.
        self.paper_ctx = PaperContext(cache_dir=cache_dir)

        # Tool registry.
        self.registry = ToolRegistry()
        for schema in build_paper_tools(self.paper_ctx):
            self.registry.register_schema(schema)
        for schema in build_figure_tools(self.paper_ctx):
            self.registry.register_schema(schema)
        for schema in build_inspire_tools(cache_dir):
            self.registry.register_schema(schema)
        for schema in build_hepdata_tools(cache_dir):
            self.registry.register_schema(schema)
        for schema in build_arxiv_search_tool(cache_dir):
            self.registry.register_schema(schema)

        # Agent state + loop.
        self.state = AgentState(
            system_prompt=build_copilot_prompt(paper_identifier)
        )
        self.loop = AgentLoop(
            provider=self.provider,
            registry=self.registry,
            state=self.state,
            max_tool_iterations=max_tool_iterations,
            on_event=self._on_event,
        )

        # Streaming print state.
        self._in_thinking = False
        self._in_text = False

        # Persistence.
        session_id = self._make_session_id()
        self.store = CotStore(output_dir=output_dir, session_id=session_id)
        self.store.set_meta(
            model=getattr(self.provider, "model", provider_name),
            thread_id=session_id,
        )
        self.store.update_meta("provider", provider_name)
        self.store.update_meta("system_prompt_preview", self.state.system_prompt[:500])

    # ------------------------------------------------------------------
    # Session id
    # ------------------------------------------------------------------

    def _make_session_id(self) -> str:
        suffix = int(time.time())
        tag = "copilot"
        if self.paper_identifier:
            tag += f"_{self.paper_identifier.replace('/', '_').replace(':', '_')}"
        return f"{tag}_{self.provider_name}_{suffix}"

    # ------------------------------------------------------------------
    # Event streaming to terminal
    # ------------------------------------------------------------------

    def _flush_inline(self) -> None:
        if self._in_thinking or self._in_text:
            print()
            self._in_thinking = False
            self._in_text = False

    def _on_event(self, ev: Event) -> None:
        if isinstance(ev, ThinkingDelta):
            if not self.show_thinking:
                return
            if not self._in_thinking:
                self._flush_inline()
                print(f"{magenta('[thinking]')} ", end="", flush=True)
                self._in_thinking = True
            sys.stdout.write(dim(ev.text))
            sys.stdout.flush()
        elif isinstance(ev, TextDelta):
            if not self._in_text:
                self._flush_inline()
                print(f"{green('[answer]')} ", end="", flush=True)
                self._in_text = True
            sys.stdout.write(ev.text)
            sys.stdout.flush()
        elif isinstance(ev, ToolCall):
            self._flush_inline()
            args_preview = str(ev.arguments)[:140]
            print(
                f"{cyan('[tool]')} {ev.name}({args_preview})",
                flush=True,
            )
        elif isinstance(ev, TurnEnd):
            self._flush_inline()
            usage_bits = []
            if ev.usage.get("reasoning_tokens"):
                usage_bits.append(f"think={ev.usage['reasoning_tokens']}")
            if ev.usage.get("output_tokens"):
                usage_bits.append(f"out={ev.usage['output_tokens']}")
            elif ev.usage.get("completion_tokens"):
                usage_bits.append(f"out={ev.usage['completion_tokens']}")
            if ev.usage.get("input_tokens"):
                usage_bits.append(f"in={ev.usage['input_tokens']}")
            elif ev.usage.get("prompt_tokens"):
                usage_bits.append(f"in={ev.usage['prompt_tokens']}")
            usage_str = " ".join(usage_bits)
            print(dim(f"  (stop={ev.stop_reason} {usage_str})"))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(bold(BANNER))
        print(f"Provider: {cyan(self.provider_name)}")
        print(f"Model:    {cyan(getattr(self.provider, 'model', 'n/a'))}")
        print(
            f"Tools:    {dim(', '.join(self.registry.names()))}  "
            f"({len(self.registry)} total)"
        )
        print(f"Session:  {dim(self.store.session_id)}")
        if self.paper_identifier:
            print(f"Paper:    {cyan(self.paper_identifier)} {dim('(pre-load)')}")
            # Pre-load the paper via tool so the state is warm.
            prime = self.registry.execute(
                "fetch_paper", {"identifier": self.paper_identifier}
            )
            content, is_error, _ = prime
            if is_error:
                print(red(f"[warn] failed to pre-load paper: {content}"))
            else:
                print(dim(f"[pre-load] {content[:240]}..."))
                if self.paper_ctx.is_loaded:
                    self.store.set_context(self.paper_ctx.paper.text_content)  # type: ignore[union-attr]
                    # Refresh system prompt with paper metadata so the model
                    # knows the paper is already cached.
                    import json as _json

                    try:
                        info = _json.loads(content)
                    except (_json.JSONDecodeError, TypeError):
                        info = None
                    if info:
                        self.state.system_prompt = build_copilot_prompt(
                            self.paper_identifier, paper_info=info
                        )
        print(dim("Commands: /quit  /tools  /provider <name>  /paper <id>  /save"))
        print()

        while True:
            try:
                line = input(f"{bold(cyan('You > '))}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                continue

            if line.lower() in ("/quit", "/exit", "/q"):
                break
            if line.lower() == "/tools":
                self._print_tools()
                continue
            if line.lower() == "/save":
                path = self._save_session()
                print(f"{green('saved:')} {path}")
                continue
            if line.lower().startswith("/provider "):
                new_provider = line.split(None, 1)[1].strip()
                self._switch_provider(new_provider)
                continue
            if line.lower().startswith("/paper "):
                ident = line.split(None, 1)[1].strip()
                self._load_paper(ident)
                continue

            # Normal user question.
            try:
                self.loop.run_round(line)
            except KeyboardInterrupt:
                self._flush_inline()
                print(yellow("\n[interrupted]"))
            except Exception as e:
                self._flush_inline()
                print(red(f"[error] {type(e).__name__}: {e}"))
            finally:
                # Persist after each round.
                self._log_last_turn_to_store()

        self._save_session()
        print(green("\nBye."))

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _print_tools(self) -> None:
        for spec in self.registry.specs():
            print(f"  {bold(spec.name)} — {dim(spec.description[:120])}")

    def _switch_provider(self, name: str) -> None:
        try:
            self.provider = build_provider(name)
        except ValueError as e:
            print(red(str(e)))
            return
        self.loop.provider = self.provider
        self.provider_name = name
        self.store.update_meta("provider", name)
        print(f"{green('provider switched to')} {cyan(name)}")

    def _load_paper(self, identifier: str) -> None:
        self.paper_identifier = identifier
        self.state.system_prompt = build_copilot_prompt(identifier)
        content, is_error, _ = self.registry.execute(
            "fetch_paper", {"identifier": identifier}
        )
        if is_error:
            print(red(f"[error] {content}"))
        else:
            print(f"{green('loaded:')} {content[:240]}")
            if self.paper_ctx.is_loaded:
                self.store.set_context(
                    self.paper_ctx.paper.text_content  # type: ignore[union-attr]
                )

    # ------------------------------------------------------------------
    # Store bridging — we convert AgentState into CotStore TurnRecord
    # ------------------------------------------------------------------

    def _log_last_turn_to_store(self) -> None:
        """Emit one synthetic TurnRecord for the last completed round.

        The underlying CotStore was designed around Codex event schema, but
        its public ``log_turn`` method accepts our :class:`TurnRecord` too.
        We build a minimal record that preserves human_input / reasoning /
        tool_calls / agent_response.
        """

        msgs = self.state.messages
        if not msgs:
            return

        # Find the most recent user message + everything after it.
        user_idx = None
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].role == "user":
                user_idx = i
                break
        if user_idx is None:
            return

        tail = msgs[user_idx:]

        human_input = tail[0].content
        assistant_text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_records: list[Any] = []

        from ..event_router import ReasoningStep, ToolCall as RouterToolCall

        for m in tail[1:]:
            if m.role == "assistant":
                if m.thinking:
                    thinking_parts.append(m.thinking)
                if m.content:
                    assistant_text_parts.append(m.content)
            if m.role == "tool":
                # tool result row — pair with corresponding call in state.tool_calls_log
                pass

        # Attach tool calls (from the log) newer than what's already persisted.
        already = len(self.store._session_meta.get("turns", []))  # noqa: SLF001
        new_calls = [
            tc
            for tc in self.state.tool_calls_log
            if tc.turn_index >= already
        ]
        for tc in new_calls:
            rtc = RouterToolCall(
                item_id=tc.id,
                tool_type="function",
                command=f"{tc.name}({str(tc.arguments)[:120]})",
                cwd="",
                output=tc.result[:2000],
                status="failed" if tc.is_error else "completed",
                exit_code=1 if tc.is_error else 0,
                duration_ms=int(tc.duration_s * 1000),
                timestamp=time.time(),
            )
            tool_records.append(rtc)

        reasoning_steps: list[Any] = []
        for i, t in enumerate(thinking_parts):
            reasoning_steps.append(
                ReasoningStep(
                    item_id=f"think_{self.state.turn_index}_{i}",
                    summary_parts=[t],
                    raw_content=t if self.provider.supports_raw_thinking() else "",
                    timestamp=time.time(),
                )
            )

        record = TurnRecord(
            turn_id=f"turn_{self.state.turn_index}",
            human_input=human_input,
            reasoning_steps=reasoning_steps,
            tool_calls=tool_records,
            agent_response="\n".join(assistant_text_parts),
            phase_key="free",
            usage=self.state.usage_totals,
            status="completed",
            start_time=time.time(),
            end_time=time.time(),
        )
        self.store.log_turn(record)

    def _save_session(self) -> str:
        self.store.update_meta("agent_state", self.state.to_dict())
        self.store.update_meta(
            "paper",
            {
                "arxiv_id": getattr(self.paper_ctx.paper, "arxiv_id", ""),
                "title": getattr(self.paper_ctx.paper, "title", ""),
            }
            if self.paper_ctx.paper
            else {},
        )
        return self.store.finalize()
