"""Structured storage for Chain-of-Thought extraction data.

Output JSON follows a training-corpus-friendly structure:

  session level:
    context          -- paper full text (stored once)
    system_prompt    -- externalization + language instructions (stored once)
    paper            -- metadata
    turns[]          -- the conversation

  per turn:
    role: "user"     -- human_input (the phase prompt, no repeated boilerplate)
    role: "reasoning"-- hidden_reasoning (model internal summary)
    role: "assistant"-- agent_response (explicit reasoning output)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .event_router import TurnRecord


class CotStore:
    """Persists raw events and structured CoT data to disk."""

    def __init__(self, output_dir: str = "./cot_sessions", session_id: str | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"session_{int(time.time())}"
        self._raw_file = self.output_dir / f"{self.session_id}.jsonl"
        self._summary_file = self.output_dir / f"{self.session_id}.json"
        self._session_meta: dict[str, Any] = {
            "session_id": self.session_id,
            "start_time": time.time(),
            "model": "",
            "thread_id": "",
            "context": "",
            "system_prompt": "",
            "turns": [],
        }

    @property
    def raw_file(self) -> Path:
        """Path to the raw JSONL events file."""
        return self._raw_file

    def update_meta(self, key: str, value: Any) -> None:
        """Set an arbitrary key in the session metadata."""
        self._session_meta[key] = value

    def set_meta(self, model: str, thread_id: str) -> None:
        self._session_meta["model"] = model
        self._session_meta["thread_id"] = thread_id

    def set_context(self, paper_text: str) -> None:
        """Store the full paper text once at session level."""
        self._session_meta["context"] = paper_text

    def set_system_prompt(self, prompt: str) -> None:
        """Store the system-level instructions once."""
        self._session_meta["system_prompt"] = prompt

    def log_raw_event(self, event: dict) -> None:
        with open(self._raw_file, "a", encoding="utf-8") as f:
            record = {
                "timestamp": time.time(),
                "event": event,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_turn(self, turn: TurnRecord) -> None:
        turn_data = self._turn_to_dict(turn)
        self._session_meta["turns"].append(turn_data)
        self._write_summary()

    def finalize(self) -> str:
        self._session_meta["end_time"] = time.time()
        self._session_meta["total_turns"] = len(self._session_meta["turns"])

        total_reasoning = 0
        total_tools = 0
        for t in self._session_meta["turns"]:
            total_reasoning += len(t.get("hidden_reasoning", []))
            total_tools += len(t.get("tool_calls", []))
        self._session_meta["total_reasoning_steps"] = total_reasoning
        self._session_meta["total_tool_calls"] = total_tools

        self._write_summary()
        return str(self._summary_file)

    def _write_summary(self) -> None:
        with open(self._summary_file, "w", encoding="utf-8") as f:
            json.dump(self._session_meta, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _turn_to_dict(turn: TurnRecord) -> dict:
        reasoning_chain = []
        for i, step in enumerate(turn.reasoning_steps):
            summary = "\n---\n".join(step.summary_parts) if step.summary_parts else ""
            # Fallback: if no summary was streamed, use raw_content as summary
            if not summary.strip() and step.raw_content:
                summary = step.raw_content
            reasoning_chain.append({
                "step": i + 1,
                "item_id": step.item_id,
                "summary": summary,
                "raw_content": step.raw_content if step.raw_content else None,
                "timestamp": step.timestamp,
            })

        tool_calls = []
        for tc in turn.tool_calls:
            tool_calls.append({
                "item_id": tc.item_id,
                "type": tc.tool_type,
                "command": tc.command,
                "cwd": tc.cwd,
                "output_preview": tc.output[:500] if tc.output else "",
                "status": tc.status,
                "exit_code": tc.exit_code,
                "duration_ms": tc.duration_ms,
                "timestamp": tc.timestamp,
            })

        return {
            "turn_id": turn.turn_id,
            "phase_key": turn.phase_key,
            "human_input": turn.human_input,
            "hidden_reasoning": reasoning_chain,
            "agent_response": turn.agent_response,
            "tool_calls": tool_calls,
            "plan_text": turn.plan_text if turn.plan_text else None,
            "usage": turn.usage,
            "status": turn.status,
            "start_time": turn.start_time,
            "end_time": turn.end_time,
            "duration_seconds": round(turn.end_time - turn.start_time, 2),
        }
