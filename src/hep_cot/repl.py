"""Interactive REPL for guiding Codex and extracting CoT."""

from __future__ import annotations

import threading

from .app_server_client import CodexSession
from .cot_store import CotStore
from .event_router import EventRouter, TurnRecord
from .ui import (
    TerminalPrinter,
    bold, cyan, dim, green, red, yellow,
)


class CotRepl:
    """Interactive terminal REPL that drives Codex while capturing CoT."""

    def __init__(
        self,
        codex_binary: str = "codex",
        model: str = "gpt-5.4",
        cwd: str | None = None,
        reasoning_effort: str = "high",
        output_dir: str = "./cot_sessions",
        sandbox: str = "workspace-write",
        approval_policy: str = "onFailure",
        show_reasoning: bool = True,
        show_tools: bool = True,
    ):
        self.session = CodexSession(
            codex_binary=codex_binary,
            model=model,
            cwd=cwd,
            reasoning_effort=reasoning_effort,
            sandbox=sandbox,
            approval_policy=approval_policy,
        )
        self.router = EventRouter()
        self.store = CotStore(output_dir=output_dir)
        self.show_reasoning = show_reasoning
        self.show_tools = show_tools
        self._turn_done = threading.Event()
        self._current_turn_id: str | None = None
        self._printer = TerminalPrinter()
        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        self.session.client.on_notification(self._on_notification)
        self.session.client.on_server_request(self._on_server_request)

        if self.show_reasoning:
            self.router.on_reasoning_delta.append(self._printer.print_reasoning_delta)

        self.router.on_agent_message_delta.append(self._printer.print_agent_delta)

        if self.show_tools:
            self.router.on_tool_started.append(self._printer.print_tool_started)
            self.router.on_tool_completed.append(self._printer.print_tool_completed)

        self.router.on_turn_started.append(self._on_turn_started)
        self.router.on_turn_completed.append(self._on_turn_completed)

    def _on_notification(self, msg: dict) -> None:
        self.store.log_raw_event(msg)
        self.router.handle_event(msg)

    def _on_server_request(self, msg: dict) -> dict | None:
        self.store.log_raw_event(msg)
        method = msg.get("method", "")
        params = msg.get("params", {})

        if "requestApproval" in method:
            return self._handle_approval(method, params)
        return None

    def _handle_approval(self, method: str, params: dict) -> dict:
        cmd = params.get("command", params.get("reason", "unknown action"))
        print(f"\n{yellow('[APPROVAL]')} Codex wants to: {bold(str(cmd))}")
        print(f"  {dim('(a)ccept / (d)ecline / (s)ession-accept / (c)ancel')}")

        try:
            choice = input(f"  {yellow('>')} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "a"

        decision_map = {
            "a": "accept",
            "d": "decline",
            "s": "acceptForSession",
            "c": "cancel",
        }
        decision = decision_map.get(choice, "accept")
        print(f"  {dim(f'-> {decision}')}")
        return {"decision": decision}

    def _on_turn_started(self, turn: TurnRecord) -> None:
        self._current_turn_id = turn.turn_id
        self._turn_done.clear()
        self._printer.reset()

    def _on_turn_completed(self, turn: TurnRecord) -> None:
        self._printer.end_blocks()
        self.store.log_turn(turn)

        # Print turn stats
        n_reasoning = len(turn.reasoning_steps)
        n_tools = len(turn.tool_calls)
        usage = turn.usage
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        reasoning_tokens = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
        duration = round(turn.end_time - turn.start_time, 1)

        print(f"\n{dim('---')}")
        print(
            dim(
                f"  [Turn {turn.status}] "
                f"reasoning_steps={n_reasoning} tools={n_tools} "
                f"duration={duration}s "
                f"tokens(in={input_tokens} out={output_tokens} reasoning={reasoning_tokens})"
            )
        )
        print(dim("---"))
        self._turn_done.set()

    def run(self) -> None:
        print(bold("=== HEP CoT Extractor ==="))
        print(f"Model: {cyan(self.session.model)}")
        print(f"Reasoning effort: {cyan(self.session.reasoning_effort)}")
        if self.session.cwd:
            print(f"Working dir: {cyan(self.session.cwd)}")
        print()

        try:
            print(dim("Connecting to Codex App Server..."), flush=True)
            self.session.connect()
            print(dim("Starting thread..."), flush=True)
            thread_id = self.session.start_thread()
            self.store.set_meta(self.session.model, thread_id)
            print(f"Thread: {cyan(thread_id)}")
            print()
            print(dim("Commands: /quit /status /export"))
            print(dim("Type your instructions to guide Codex."))
            print()

            self._repl_loop()

        except KeyboardInterrupt:
            print(f"\n{yellow('Interrupted.')}")
        except Exception as e:
            print(f"\n{red(f'Error: {e}')}")
        finally:
            summary_path = self.store.finalize()
            print(f"\n{green('Session saved:')} {summary_path}")
            print(
                f"  Turns: {len(self.router.completed_turns)}, "
                f"Raw events: {self.store.raw_file}"
            )
            self.session.disconnect()

    def _repl_loop(self) -> None:
        while True:
            try:
                user_input = input(f"\n{bold(cyan('You > '))}").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    break
                continue

            self.session.send_turn(user_input)
            self._turn_done.wait(timeout=600)

    def _handle_command(self, cmd: str) -> bool:
        """Returns True if REPL should exit."""
        cmd = cmd.lower().strip()

        if cmd in ("/quit", "/exit", "/q"):
            return True

        elif cmd == "/status":
            turns = self.router.completed_turns
            total_reasoning = sum(len(t.reasoning_steps) for t in turns)
            total_tools = sum(len(t.tool_calls) for t in turns)
            print(f"\n{bold('Session Status:')}")
            print(f"  Thread: {self.session.thread_id}")
            print(f"  Turns completed: {len(turns)}")
            print(f"  Total reasoning steps: {total_reasoning}")
            print(f"  Total tool calls: {total_tools}")
            print(f"  Raw events file: {self.store.raw_file}")

        elif cmd == "/export":
            path = self.store.finalize()
            print(f"\n{green('Exported:')} {path}")

        elif cmd.startswith("/effort "):
            new_effort = cmd.split(maxsplit=1)[1]
            if new_effort in ("none", "low", "medium", "high", "xhigh"):
                self.session.reasoning_effort = new_effort
                print(f"  Reasoning effort set to: {cyan(new_effort)}")
            else:
                print(f"  {red('Invalid effort.')} Use: none/low/medium/high/xhigh")

        else:
            print(dim(f"  Unknown command: {cmd}"))
            print(dim("  Available: /quit /status /export /effort <level>"))

        return False
