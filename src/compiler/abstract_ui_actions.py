"""Action-edge tables updated by the AbstractUI system timer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence


ABSTRACT_UI_ACTION_VERSION = "abstract-ui-action-edges-v0"


@dataclass(frozen=True, slots=True)
class IssuedAction:
    identity: str
    actor: str
    type: str
    destination: str
    edge: str
    issued_at: float


@dataclass(frozen=True, slots=True)
class ActionEdgeRow:
    identity: str
    source: str
    type: str
    destination: str
    issue_count: int = 0
    last_issued_at: float | None = None
    recent: bool = False


@dataclass(frozen=True, slots=True)
class ActionEdgeTable:
    """Immutable reference table wired to one system-timer identity."""

    identity: str
    timer: str
    rows: tuple[ActionEdgeRow, ...] = ()
    time: float = 0.0
    recent_window: float = 0.8

    def register(
        self,
        *,
        identity: str,
        source: str,
        type: str,
        destination: str,
    ) -> "ActionEdgeTable":
        if any(row.identity == identity for row in self.rows):
            return self
        return replace(self, rows=(*self.rows, ActionEdgeRow(
            identity, source, type, destination,
        )))

    def update(
        self,
        actions: Sequence[IssuedAction],
        *,
        time: float | None = None,
    ) -> "ActionEdgeTable":
        """Apply one timer delivery and recompute recently active rows."""

        now = self.time if time is None else float(time)
        issued_by_edge: dict[str, list[IssuedAction]] = {}
        for action in actions:
            issued_by_edge.setdefault(action.edge, []).append(action)
        known = {row.identity for row in self.rows}
        missing = set(issued_by_edge).difference(known)
        if missing:
            raise KeyError(f"actions reference unregistered edges: {sorted(missing)!r}")
        rows = []
        for row in self.rows:
            issued = issued_by_edge.get(row.identity, ())
            last = row.last_issued_at
            if issued:
                last = max(action.issued_at for action in issued)
            rows.append(replace(
                row,
                issue_count=row.issue_count + len(issued),
                last_issued_at=last,
                recent=last is not None and now - last <= self.recent_window,
            ))
        return replace(self, rows=tuple(rows), time=now)


@dataclass(frozen=True, slots=True)
class SystemTimer:
    """The neutral root clock connection; backends own actual scheduling."""

    identity: str
    sequence: int = 0
    time: float = 0.0
    connections: tuple[str, ...] = ()

    def connect(self, destination: str) -> "SystemTimer":
        if destination in self.connections:
            return self
        return replace(self, connections=(*self.connections, destination))

    def tick(
        self,
        time: float,
        *,
        actions: Sequence[IssuedAction],
        action_edges: ActionEdgeTable,
    ) -> tuple["SystemTimer", ActionEdgeTable]:
        if action_edges.identity not in self.connections:
            raise ValueError("system timer is not connected to the action-edge table")
        next_timer = replace(self, sequence=self.sequence + 1, time=float(time))
        return next_timer, action_edges.update(actions, time=float(time))


def system_action_mezzanine_model(system_root: str) -> dict[str, object]:
    timer = SystemTimer(f"{system_root}/timer")
    table = ActionEdgeTable(
        f"{system_root}/entities/action-edges", timer.identity,
    )
    timer = timer.connect(table.identity)
    return {
        "schema": ABSTRACT_UI_ACTION_VERSION,
        "timer": {
            "identity": timer.identity,
            "sequence": timer.sequence,
            "time": timer.time,
            "connections": list(timer.connections),
        },
        "action_edges": {
            "identity": table.identity,
            "timer": table.timer,
            "recent_window": table.recent_window,
            "operation": "update(actions)",
            "rows": [],
        },
        "edges": [{
            "source": timer.identity,
            "target": table.identity,
            "relationship": "update(actions)",
        }],
    }


__all__ = [
    "ABSTRACT_UI_ACTION_VERSION",
    "ActionEdgeRow",
    "ActionEdgeTable",
    "IssuedAction",
    "SystemTimer",
    "system_action_mezzanine_model",
]
