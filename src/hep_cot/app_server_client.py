"""Codex App Server JSON-RPC client over stdio."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any


class AppServerClient:
    """Manages a Codex App Server child process and provides JSON-RPC communication."""

    def __init__(self, codex_binary: str = "codex"):
        self._codex_binary = codex_binary
        self._proc: subprocess.Popen | None = None
        self._next_id = 0
        self._pending: dict[int, threading.Event] = {}
        self._results: dict[int, Any] = {}
        self._notification_handlers: list[Callable[[dict], None]] = []
        self._server_request_handlers: list[Callable[[dict], dict | None]] = []
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        self._proc = subprocess.Popen(
            [self._codex_binary, "app-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._running = True
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._proc:
            self._proc.stdin.close()
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def on_notification(self, handler: Callable[[dict], None]) -> None:
        self._notification_handlers.append(handler)

    def on_server_request(self, handler: Callable[[dict], dict | None]) -> None:
        self._server_request_handlers.append(handler)

    def send_request(self, method: str, params: dict | None = None, timeout: float = 300) -> Any:
        msg_id = self._alloc_id()
        msg: dict[str, Any] = {"method": method, "id": msg_id}
        if params is not None:
            msg["params"] = params

        event = threading.Event()
        with self._lock:
            self._pending[msg_id] = event

        self._write(msg)

        if not event.wait(timeout=timeout):
            with self._lock:
                self._pending.pop(msg_id, None)
            raise TimeoutError(f"Request {method} (id={msg_id}) timed out after {timeout}s")

        with self._lock:
            result = self._results.pop(msg_id, None)

        if isinstance(result, Exception):
            raise result
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"Server error: {result['error']}")
        return result

    def send_notification(self, method: str, params: dict | None = None) -> None:
        msg: dict[str, Any] = {"method": method}
        if params is not None:
            msg["params"] = params
        self._write(msg)

    def _alloc_id(self) -> int:
        with self._lock:
            self._next_id += 1
            return self._next_id

    def _write(self, msg: dict) -> None:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("App server process not running")
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

    def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            while self._running:
                line = self._proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._dispatch(msg)
        finally:
            # Wake up any pending requests so callers don't hang forever
            with self._lock:
                for msg_id, event in self._pending.items():
                    self._results[msg_id] = ConnectionError(
                        "Reader thread exited; app-server connection lost"
                    )
                    event.set()
                self._pending.clear()

    def _dispatch(self, msg: dict) -> None:
        if "id" in msg and "method" in msg:
            # Server-initiated request (e.g. approval)
            response = None
            for handler in self._server_request_handlers:
                response = handler(msg)
                if response is not None:
                    break
            if response is not None:
                self._write({"id": msg["id"], "result": response})
            else:
                self._write({"id": msg["id"], "result": {"decision": "accept"}})
        elif "id" in msg and ("result" in msg or "error" in msg):
            # Response to our request
            msg_id = msg["id"]
            with self._lock:
                event = self._pending.pop(msg_id, None)
                if event:
                    self._results[msg_id] = msg.get("result", msg.get("error"))
                    event.set()
        elif "method" in msg and "id" not in msg:
            # Notification
            for handler in self._notification_handlers:
                handler(msg)


class CodexSession:
    """High-level wrapper managing a single Codex thread with CoT extraction."""

    def __init__(
        self,
        codex_binary: str = "codex",
        model: str = "gpt-5.4",
        cwd: str | None = None,
        reasoning_effort: str = "high",
        sandbox: str = "workspace-write",
        approval_policy: str = "onFailure",
    ):
        self.client = AppServerClient(codex_binary=codex_binary)
        self.model = model
        self.cwd = cwd
        self.reasoning_effort = reasoning_effort
        self.sandbox = sandbox
        self.approval_policy = approval_policy
        self.thread_id: str | None = None
        self._initialized = False
        self._thread_id_from_notification: str | None = None

    def _capture_thread_id_notification(self, msg: dict) -> None:
        method = msg.get("method", "")
        if method == "thread/started":
            params = msg.get("params", {})
            thread = params.get("thread", {})
            tid = thread.get("id", "")
            if tid:
                self._thread_id_from_notification = tid

    def connect(self) -> None:
        self.client.start()
        self.client.on_notification(self._capture_thread_id_notification)
        time.sleep(0.5)  # give process time to start

        self.client.send_request("initialize", {
            "clientInfo": {
                "name": "hep_cot",
                "title": "HEP CoT Extractor",
                "version": "0.1.0",
            },
            "capabilities": {
                "experimentalApi": True,
            },
        })
        self.client.send_notification("initialized", {})
        self._initialized = True

    def start_thread(self) -> str:
        if not self._initialized:
            raise RuntimeError("Must call connect() first")

        params: dict[str, Any] = {
            "model": self.model,
            "sandbox": self.sandbox,
            "approvalPolicy": self.approval_policy,
        }
        if self.cwd:
            params["cwd"] = self.cwd

        self._thread_id_from_notification = None
        result = self.client.send_request("thread/start", params)

        # Handle varying response shapes from different Codex versions
        tid = None
        if isinstance(result, dict):
            if "thread" in result and isinstance(result["thread"], dict):
                tid = result["thread"].get("id")
            elif "threadId" in result:
                tid = result["threadId"]
            elif "id" in result:
                tid = result["id"]

        # Fallback: thread/started notification may arrive before/after result
        if not tid:
            time.sleep(0.3)
            tid = self._thread_id_from_notification

        if not tid:
            raise RuntimeError(
                f"thread/start: could not extract thread id. "
                f"result={result!r}, notification={self._thread_id_from_notification!r}"
            )

        self.thread_id = tid
        return self.thread_id

    def send_turn(self, text: str, summary: str = "detailed") -> None:
        if not self.thread_id:
            raise RuntimeError("No active thread")

        self.client.send_request("turn/start", {
            "threadId": self.thread_id,
            "input": [{"type": "text", "text": text}],
            "effort": self.reasoning_effort,
            "summary": summary,
        })

    def interrupt(self, turn_id: str) -> None:
        if not self.thread_id:
            return
        self.client.send_request("turn/interrupt", {
            "threadId": self.thread_id,
            "turnId": turn_id,
        })

    def disconnect(self) -> None:
        self.client.stop()
