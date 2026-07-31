"""Structured observability shared by compiler stages and backend runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class TraceCode(IntEnum):
    CLOSURE_PLANNED = 1
    LOOP_ENTER = 2
    REGION_EXECUTE = 3
    STATE_COMMIT = 4
    OUTPUT_PUBLISH = 5
    ERROR = 255


@dataclass(frozen=True)
class TraceEvent:
    code: TraceCode
    closure: str
    subject_id: int
    fields: tuple[tuple[str, Any], ...] = ()


@dataclass
class CompilerTrace:
    events: list[TraceEvent] = field(default_factory=list)

    def emit(self, code, closure, subject_id, **fields) -> None:
        self.events.append(TraceEvent(
            TraceCode(code),
            str(closure),
            int(subject_id),
            tuple(sorted(fields.items())),
        ))

    def render_ascii(self) -> str:
        return "\n".join(
            f"{index:04d} {event.code.name:<16} "
            f"{event.closure} #{event.subject_id} "
            + " ".join(
                f"{name}={value!r}" for name, value in event.fields
            )
            for index, event in enumerate(self.events)
        )


__all__ = ["CompilerTrace", "TraceCode", "TraceEvent"]
