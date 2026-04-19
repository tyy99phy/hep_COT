"""Tool registry.

Tools are registered with a callable and a JSON-Schema parameter spec,
then exported as :class:`~hep_cot.llm.base.ToolSpec` for the provider
and dispatched by name when the model calls them.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..llm.base import ToolSpec


@dataclass
class ToolSchema:
    """Internal tool record."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ToolExecutionError(Exception):
    """Raised when a tool is not registered or its call signature is wrong."""


class ToolRegistry:
    """Holds a set of callable tools keyed by name."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSchema] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Any],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")
        self._tools[name] = ToolSchema(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )

    def register_schema(self, schema: ToolSchema) -> None:
        if schema.name in self._tools:
            raise ValueError(f"Tool '{schema.name}' already registered")
        self._tools[schema.name] = schema

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def specs(self) -> list[ToolSpec]:
        return [schema.to_spec() for schema in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def execute(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[str, bool, float]:
        """Run a tool by name.

        Returns ``(content, is_error, duration_s)``. ``content`` is always
        a string so it can be sent back to the model directly.
        """

        if name not in self._tools:
            return (
                json.dumps({"error": f"unknown tool '{name}'"}),
                True,
                0.0,
            )

        schema = self._tools[name]
        t0 = time.time()
        try:
            result = schema.func(**arguments)
        except TypeError as e:
            return (
                json.dumps({"error": f"bad arguments: {e}"}),
                True,
                time.time() - t0,
            )
        except Exception as e:  # pragma: no cover - defensive
            return (
                json.dumps({"error": f"{type(e).__name__}: {e}"}),
                True,
                time.time() - t0,
            )

        duration = time.time() - t0

        # Normalise to a JSON string — tools may return str, dict, or list.
        if isinstance(result, str):
            return result, False, duration
        try:
            return (
                json.dumps(result, ensure_ascii=False, default=_json_default),
                False,
                duration,
            )
        except (TypeError, ValueError):
            return str(result), False, duration


def _json_default(obj: Any) -> Any:
    # Dataclasses / pathlib paths / any object with __dict__
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)
