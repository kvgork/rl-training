"""Structured lab errors: a closed set of codes, mapped 1:1 onto the MCP error shape.

``{"ok": false, "error": {"code", "message", "details", "hint", "retryable"}}``
"""

from __future__ import annotations

from typing import Any

CODES = frozenset(
    {
        "INVALID_ARGUMENT",
        "NOT_FOUND",
        "CONFIG_INVALID",
        "UNKNOWN_TRAINER",
        "UNKNOWN_TASK",
        "BUDGET_EXHAUSTED",
        "HALTED",
        "STATE_CONFLICT",
        "GPU_BUSY",
        "IDEMPOTENCY_MISMATCH",
        "TRAINER_UNAVAILABLE",
        "GATE_MISSING",
        "LAUNCH_FAILED",
        "INTERNAL",
    }
)


class LabError(Exception):
    """A refusal or failure with a stable code. ``message`` and ``hint`` are one line each."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        hint: str | None = None,
        retryable: bool = False,
    ) -> None:
        if code not in CODES:
            raise ValueError(f"unknown lab error code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details or {}
        self.hint = hint
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "hint": self.hint,
                "retryable": self.retryable,
            },
        }
