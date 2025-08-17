# -*- coding: utf-8 -*-
"""Shared state table for dt-graph engines.

This module provides a lightweight, global state table so that:
- All engines derive their working state from the table at the start of a step.
- All engines publish their updated state back into the table after a step.

It uses duck-typed adapters to avoid tight coupling with specific engines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional


Key = Tuple[str, str, str]


@dataclass
class StateTable:
    store: Dict[Key, Any] = field(default_factory=dict)

    def get(self, scope: str, name: str, field_name: str) -> Optional[Any]:
        return self.store.get((str(scope), str(name), str(field_name)))

    def set(self, scope: str, name: str, field_name: str, value: Any) -> None:
        self.store[(str(scope), str(name), str(field_name))] = value

    def clear_scope(self, scope: str, name: Optional[str] = None) -> None:
        if name is None:
            keys = [k for k in self.store.keys() if k[0] == scope]
        else:
            keys = [k for k in self.store.keys() if k[0] == scope and k[1] == name]
        for k in keys:
            self.store.pop(k, None)


# Global default table; can be replaced/injected by runners when needed
GLOBAL_STATE_TABLE = StateTable()


# --------------------- engine sync/publish helpers -------------------------

def _sync_demo_state(engine: Any, name: str, table: StateTable) -> bool:
    """Sync classic mechanics DemoState if present."""
    s = getattr(engine, "s", None)
    if s is None:
        return False
    touched = False
    for field_name in ("pos", "vel", "acc", "mass"):
        data = table.get("engine", name, field_name)
        if data is not None and hasattr(s, field_name):
            try:
                setattr(s, field_name, data)
                touched = True
            except Exception:
                pass
    return touched


def _publish_demo_state(engine: Any, name: str, table: StateTable) -> bool:
    s = getattr(engine, "s", None)
    if s is None:
        return False
    touched = False
    for field_name in ("pos", "vel", "acc", "mass"):
        if hasattr(s, field_name):
            try:
                table.set("engine", name, field_name, getattr(s, field_name))
                touched = True
            except Exception:
                pass
    return touched


def _sync_fluid_state(engine: Any, name: str, table: StateTable) -> bool:
    sim = getattr(engine, "sim", None)
    if sim is None:
        return False
    # pull typical arrays
    touched = False
    for field_name in ("x", "v", "m", "rho", "T", "S"):
        data = table.get("engine", name, field_name)
        if data is not None and hasattr(sim, field_name):
            try:
                setattr(sim, field_name, data)
                touched = True
            except Exception:
                pass
    return touched


def _publish_fluid_state(engine: Any, name: str, table: StateTable) -> bool:
    sim = getattr(engine, "sim", None)
    if sim is None:
        return False
    touched = False
    for field_name in ("x", "v", "m", "rho", "T", "S"):
        if hasattr(sim, field_name):
            try:
                table.set("engine", name, field_name, getattr(sim, field_name))
                touched = True
            except Exception:
                pass
    return touched


def sync_engine_from_table(engine: Any, reg_name: str, table: StateTable | None = None) -> None:
    """Load engine state from the table if present.

    Order: engine-provided hook -> known adapters -> no-op.
    """
    table = table or GLOBAL_STATE_TABLE
    # Engine-supplied hook takes precedence
    hook = getattr(engine, "sync_from_state", None)
    if callable(hook):
        try:
            hook(table)  # type: ignore[misc]
            return
        except Exception:
            pass
    # Known adapters
    if _sync_demo_state(engine, reg_name, table):
        return
    if _sync_fluid_state(engine, reg_name, table):
        return


def publish_engine_to_table(engine: Any, reg_name: str, table: StateTable | None = None) -> None:
    """Publish engine state into table.

    Order: engine-provided hook -> known adapters -> no-op.
    """
    table = table or GLOBAL_STATE_TABLE
    hook = getattr(engine, "publish_to_state", None)
    if callable(hook):
        try:
            hook(table)  # type: ignore[misc]
            return
        except Exception:
            pass
    if _publish_demo_state(engine, reg_name, table):
        return
    if _publish_fluid_state(engine, reg_name, table):
        return


__all__ = [
    "StateTable",
    "GLOBAL_STATE_TABLE",
    "sync_engine_from_table",
    "publish_engine_to_table",
]
