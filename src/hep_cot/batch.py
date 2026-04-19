"""Batch mode: run a list of tasks through Codex and extract all CoT non-interactively."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .app_server_client import CodexSession
from .cot_store import CotStore
from .event_router import EventRouter


class BatchRunner:
    """Run a sequence of prompts through Codex, auto-approving all actions,
    and capture the full CoT for each turn."""

    def __init__(
        self,
        tasks: list[str],
        codex_binary: str = "codex",
        model: str = "gpt-5.4",
        cwd: str | None = None,
        reasoning_effort: str = "high",
        output_dir: str = "./cot_sessions",
        sandbox: str = "workspace-write",
        auto_approve: bool = True,
    ):
        self.tasks = tasks
        self.session = CodexSession(
            codex_binary=codex_binary,
            model=model,
            cwd=cwd,
            reasoning_effort=reasoning_effort,
            sandbox=sandbox,
            approval_policy="never" if auto_approve else "onFailure",
        )
        self.router = EventRouter()
        self.store = CotStore(output_dir=output_dir)
        self.auto_approve = auto_approve
        self._turn_done_event = threading.Event()

    def run(self) -> str:
        self.session.client.on_notification(self._on_notification)
        self.session.client.on_server_request(self._on_server_request)
        self.router.on_turn_completed.append(self._on_turn_completed)

        self.session.connect()
        thread_id = self.session.start_thread()
        self.store.set_meta(self.session.model, thread_id)

        for i, task in enumerate(self.tasks):
            print(f"[{i+1}/{len(self.tasks)}] {task[:80]}...")
            self._turn_done_event.clear()
            self.session.send_turn(task)
            self._turn_done_event.wait(timeout=600)

        summary_path = self.store.finalize()
        self.session.disconnect()
        return summary_path

    def _on_notification(self, msg: dict) -> None:
        self.store.log_raw_event(msg)
        self.router.handle_event(msg)

    def _on_server_request(self, msg: dict) -> dict | None:
        self.store.log_raw_event(msg)
        if self.auto_approve:
            return {"decision": "accept"}
        return None

    def _on_turn_completed(self, turn: Any) -> None:
        self.store.log_turn(turn)
        self._turn_done_event.set()


def run_batch_from_file(
    task_file: str,
    codex_binary: str = "codex",
    model: str = "gpt-5.4",
    cwd: str | None = None,
    reasoning_effort: str = "high",
    output_dir: str = "./cot_sessions",
) -> str:
    """Load tasks from a JSON file (list of strings) and run batch extraction."""
    tasks = json.loads(Path(task_file).read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("Task file must contain a JSON array of strings")
    runner = BatchRunner(
        tasks=tasks,
        codex_binary=codex_binary,
        model=model,
        cwd=cwd,
        reasoning_effort=reasoning_effort,
        output_dir=output_dir,
    )
    return runner.run()
