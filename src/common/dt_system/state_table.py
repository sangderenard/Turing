# -*- coding: utf-8 -*-
"""Shared state table for dt-graph engines.

This module provides a lightweight, global state table so that:
- All engines derive their working state from the table at the start of a step.
- All engines publish their updated state back into the table after a step.

It uses duck-typed adapters to avoid tight coupling with specific engines.
"""
from __future__ import annotations


import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional


Key = Tuple[str, str, str]


@dataclass



class StateTable:
    store: Dict[Key, Any] = field(default_factory=dict)

    # --- Enhanced group/graph structure ---
    group_to_vertices: Dict[str, set] = field(default_factory=dict)
    group_to_edges: Dict[str, dict] = field(default_factory=dict)  # group_label -> {color: set(edge_ids)}
    group_to_faces: Dict[str, set] = field(default_factory=dict)
    vertex_to_groups: Dict[Any, set] = field(default_factory=dict)
    edge_to_groups: Dict[Any, set] = field(default_factory=dict)
    face_to_groups: Dict[Any, set] = field(default_factory=dict)

    # --- Identity registry ---
    # Each identity is a UUID mapping to a dict with at least 'pos' and 'mass'
    identity_registry: Dict[str, dict] = field(default_factory=dict)


    def register_identity(self, pos: Any, mass: Any = 0.0, uuid_str: str = None, dedup: bool = False) -> str:
        """Register or update an identity object with position and mass. Returns the UUID string.
        If dedup is True, return the UUID of an existing identity with the same pos (and mass), else create new."""
        if dedup:
            # Try to find an existing identity with the same pos and mass
            for existing_uuid, identity in self.identity_registry.items():
                if identity.get('pos') == pos and identity.get('mass') == mass:
                    return existing_uuid
        if uuid_str is None:
            uuid_str = str(uuid.uuid4())
        self.identity_registry[uuid_str] = {'pos': pos, 'mass': mass}
        return uuid_str

    def get_identity(self, uuid_str: str) -> dict:
        """Get the identity object by UUID string."""
        return self.identity_registry.get(uuid_str, None)

    def update_identity(self, uuid_str: str, pos: Any = None, mass: Any = None):
        """Update position and/or mass for an identity object."""
        if uuid_str in self.identity_registry:
            if pos is not None:
                self.identity_registry[uuid_str]['pos'] = pos
            if mass is not None:
                self.identity_registry[uuid_str]['mass'] = mass

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

    # --- Group/graph management ---
    def register_group(self, group_label: str, vertices: set, edges: dict = None, faces: set = None):
        """Register or update a group with its members. Edges can be colored: {color: set(edge_ids)}."""
        self.group_to_vertices[group_label] = set(vertices)
        if edges is not None:
            self.group_to_edges[group_label] = {color: set(eids) for color, eids in edges.items()}
        else:
            self.group_to_edges[group_label] = {}
        if faces is not None:
            self.group_to_faces[group_label] = set(faces)
        else:
            self.group_to_faces[group_label] = set()
        # Update reverse lookups
        for v in vertices:
            self.vertex_to_groups.setdefault(v, set()).add(group_label)
        for color, eids in (edges or {}).items():
            for e in eids:
                self.edge_to_groups.setdefault(e, set()).add(group_label)
        for f in (faces or set()):
            self.face_to_groups.setdefault(f, set()).add(group_label)

    def get_group_vertices(self, group_label: str) -> set:
        return self.group_to_vertices.get(group_label, set())

    def get_group_edges(self, group_label: str, color: str = None) -> set:
        if color is None:
            # Return all edges for this group
            all_edges = set()
            for s in self.group_to_edges.get(group_label, {}).values():
                all_edges.update(s)
            return all_edges
        return self.group_to_edges.get(group_label, {}).get(color, set())

    def get_group_faces(self, group_label: str) -> set:
        return self.group_to_faces.get(group_label, set())

    def get_vertex_groups(self, vertex_id: Any) -> set:
        return self.vertex_to_groups.get(vertex_id, set())

    def get_edge_groups(self, edge_id: Any) -> set:
        return self.edge_to_groups.get(edge_id, set())

    def get_face_groups(self, face_id: Any) -> set:
        return self.face_to_groups.get(face_id, set())

    def clear_groups(self):
        self.group_to_vertices.clear()
        self.group_to_edges.clear()
        self.group_to_faces.clear()
        self.vertex_to_groups.clear()
        self.edge_to_groups.clear()
        self.face_to_groups.clear()


# Global default table; can be replaced/injected by runners when needed


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


def sync_engine_from_table(engine: Any, reg_name: str, table: StateTable) -> None:
    """Load engine state from the table if present. (table is now required)

    Order: engine-provided hook -> known adapters -> no-op.
    """
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


def publish_engine_to_table(engine: Any, reg_name: str, table: StateTable) -> None:
    """Publish engine state into table. (table is now required)

    Order: engine-provided hook -> known adapters -> no-op.
    """
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
    "sync_engine_from_table",
    "publish_engine_to_table",
]
