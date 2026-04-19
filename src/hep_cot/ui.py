"""Shared terminal UI helpers -- ANSI colors and streaming print callbacks."""

from __future__ import annotations

import sys
from typing import Any


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

def _c(code: int, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def dim(text: str) -> str:
    return _c(2, text)

def cyan(text: str) -> str:
    return _c(36, text)

def green(text: str) -> str:
    return _c(32, text)

def yellow(text: str) -> str:
    return _c(33, text)

def red(text: str) -> str:
    return _c(31, text)

def bold(text: str) -> str:
    return _c(1, text)

def magenta(text: str) -> str:
    return _c(35, text)

def blue(text: str) -> str:
    return _c(34, text)


# ---------------------------------------------------------------------------
# TerminalPrinter -- streaming print callbacks with block state tracking
# ---------------------------------------------------------------------------

class TerminalPrinter:
    """Manages streaming print state for reasoning / agent / tool blocks.

    Tracks whether we are currently inside a reasoning or agent-message
    block so that transitions between block types produce clean output.
    """

    def __init__(self) -> None:
        self._in_reasoning = False
        self._in_agent_msg = False

    def reset(self) -> None:
        self._in_reasoning = False
        self._in_agent_msg = False

    def end_blocks(self) -> None:
        """Close any open streaming block with a newline."""
        if self._in_reasoning:
            print()
            self._in_reasoning = False
        if self._in_agent_msg:
            print()
            self._in_agent_msg = False

    # -- callbacks ---------------------------------------------------------

    def print_reasoning_delta(self, delta: str, _summary_idx: int) -> None:
        if not self._in_reasoning:
            print(f"\n{magenta('[Reasoning]')}", end=" ", flush=True)
            self._in_reasoning = True
        sys.stdout.write(dim(delta))
        sys.stdout.flush()

    def print_agent_delta(self, delta: str) -> None:
        if self._in_reasoning:
            print()
            self._in_reasoning = False
        if not self._in_agent_msg:
            print(f"\n{green('[Agent]')}", end=" ", flush=True)
            self._in_agent_msg = True
        sys.stdout.write(delta)
        sys.stdout.flush()

    def print_tool_started(self, tc: Any) -> None:
        self.end_blocks()
        label = tc.tool_type
        print(f"\n{cyan(f'[{label}]')} {dim(tc.command)}", flush=True)

    def print_tool_completed(self, tc: Any) -> None:
        status_color = green if tc.status == "completed" else red
        extra = ""
        if tc.exit_code is not None:
            extra += f" exit={tc.exit_code}"
        if tc.duration_ms is not None:
            extra += f" {tc.duration_ms}ms"
        print(f"  {status_color(f'-> {tc.status}')}{dim(extra)}", flush=True)
