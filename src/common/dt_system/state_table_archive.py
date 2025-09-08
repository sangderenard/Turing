# --- Builder for CausalEvolutionMap ---
class CausalEvolutionMapBuilder:
    """
    Builds a CausalEvolutionMap from a state table and an assembly.
    For each item in the assembly, computes the causal region view and adds it to the map.
    Optionally, can operate on an archive for higher-order analysis.
    """
    def __init__(self, state_table, assembly, archive=None, spatial_key=None, t_key=None, limits=None):
        self.state_table = state_table
        self.assembly = assembly
        self.archive = archive
        self.spatial_key = spatial_key or (lambda st: st.get('pos', (0.0, 0.0)))
        self.t_key = t_key or (lambda st: st.get('t', 0.0))
        self.limits = limits or {}

    def build(self):
        causal_map = CausalEvolutionMap()
        # If archive is provided, use it; else, treat state_table as a single frame
        archive = self.archive or StateTableArchive()
        if not self.archive:
            # Insert the current state_table at t=0 (or from t_key)
            t = self.t_key(self.state_table)
            archive.insert(t, self.state_table)
        view = CausalRegionView(archive, self.spatial_key, self.t_key, self.limits)
        for item in self.assembly:
            # Assume item is a key or index into the state_table
            st = self.state_table[item] if isinstance(self.state_table, dict) else item
            query_spatial = self.spatial_key(st)
            query_t = self.t_key(st)
            influence_set = view.interference_hash(query_spatial, query_t)
            dt_limit = self.limits.get('dt_limit', 1.0)  # Placeholder; should query engine API
            causal_map.add(item, influence_set, dt_limit)
        return causal_map

# --- Next-evolution data structure: Causal Evolution Map ---
class CausalEvolutionMap:
    """
    Maps each participant (region/cell/entity) to:
      - its influence set (who it can causally affect)
      - its dt_limit (how fast it can evolve)
      - optional metadata (risk, percentile, etc)
    This is the backbone for nonlinear, causally-aware, and optimally scheduled simulation.
    """
    def __init__(self):
        self.map = {}  # key: participant_id, value: {'influence_set': set, 'dt_limit': float, ...}

    def add(self, participant_id, influence_set, dt_limit, **metadata):
        self.map[participant_id] = {
            'influence_set': set(influence_set),
            'dt_limit': dt_limit,
            **metadata
        }

    def get(self, participant_id):
        return self.map.get(participant_id, None)

    def participants(self):
        return list(self.map.keys())

    def items(self):
        return self.map.items()

    # Placeholder for future: percentile ordering, risk analysis, region grouping, etc.
import bisect
from typing import Callable, Optional, Any

class CausalRegionView:
    def state_table_mask(self, query_spatial: tuple, query_t: float) -> dict:
        """
        Return a boolean mask (dict) indicating which (t, spatial) keys in the archive are in the causal set for (query_spatial, query_t).
        Useful for locking a frame for evolution and preventing coevolution.
        """
        keys = set((t, spatial) for t, spatial, _ in self._index)
        causal_keys = self.interference_hash(query_spatial, query_t)
        mask = {k: (k in causal_keys) for k in keys}
        return mask
    """
    A view on a StateTableArchive that organizes data into n^2-trees (quad/octrees) by a selected spatial dimension.
    Delivers a filtered state table representing the local causal region, using programmable limits (e.g., Lorentz, max force, etc).
    """
    def __init__(self, archive: 'StateTableArchive',
                 spatial_key: Callable[[Any], tuple],
                 t_key: Callable[[Any], float],
                 limits: Optional[dict] = None):
        """
        archive: the StateTableArchive to view
        spatial_key: function mapping a state_table to its spatial coordinates (tuple)
        t_key: function mapping a state_table to its inertial time (float)
        limits: dict of causal limits (e.g., {'max_speed': c, 'lorentz': True, ...})
        """
        self.archive = archive
        self.spatial_key = spatial_key
        self.t_key = t_key
        self.limits = limits or {}
        # For now, just keep a list of (t, spatial, state_table)
        self._index = []
        for t in archive.t_vectors:
            st = archive.select(t)
            self._index.append((t, spatial_key(st), st))

    def _lorentz_factor(self, v: float, c: float) -> float:
        """Return Lorentz gamma factor for velocity v and speed of light c."""
        return 1.0 / (1.0 - (v / c) ** 2) ** 0.5 if abs(v) < c else float('inf')

    def interference_hash(self, query_spatial: tuple, query_t: float) -> set:
        """
        Return a set of t,spatial keys that are within the causal region of (query_spatial, query_t)
        according to the programmed limits (e.g., max_speed, lorentz, etc).
        """
        result = set()
        c = self.limits.get('max_speed', None)
        lorentz = self.limits.get('lorentz', False)
        for t, spatial, st in self._index:
            dt = abs(query_t - t)
            try:
                qs = AbstractTensor.tensor(query_spatial)
                sp = AbstractTensor.tensor(spatial)
                diff = qs - sp
                dx = (diff * diff).sum().sqrt().item()
            except Exception:
                dx = np.linalg.norm(np.array(query_spatial) - np.array(spatial))
            if c is not None:
                if lorentz:
                    # Use Lorentz interval: s^2 = c^2 dt^2 - dx^2 >= 0
                    s2 = (c * dt) ** 2 - dx ** 2
                    if s2 >= 0:
                        result.add((t, spatial))
                else:
                    # Use max speed: dx <= c * dt
                    if dx <= c * dt:
                        result.add((t, spatial))
            else:
                # No speed limit: all within dt threshold
                if dt < self.limits.get('max_dt', float('inf')):
                    result.add((t, spatial))
        return result

    def filtered_state_table(self, query_spatial: tuple, query_t: float) -> list:
        """
        Return a list of state tables in the local causal region of (query_spatial, query_t).
        """
        keys = self.interference_hash(query_spatial, query_t)
        return [self.archive.select(t) for t, spatial in keys]
# -*- coding: utf-8 -*-
"""
StateTableArchive: Efficient, t-vector-indexed archive for state tables in dt_system.

This archive stores state tables along a (potentially multi-dimensional) time vector,
allowing for time drift and non-uniform time axes, as can occur in advanced dt_systems.

This design is intended to support both efficient compression and future extensions for
relativistic or nonstandard time models.
"""

from typing import Any, Dict, Tuple, List
import numpy as np
from ..tensors.abstraction import AbstractTensor

class StateTableArchive:
    def __init__(self):
        # Archive is a dict mapping t (as a tuple or vector) to state tables
        self.archive: Dict[Tuple[float, ...], Any] = {}
        self.t_vectors: List[Tuple[float, ...]] = []

    # --- DB-style methods ---
    def insert(self, t_vector: Tuple[float, ...], state_table: Any):
        """Insert a state table at the given t_vector."""
        self.archive[t_vector] = state_table
        self.t_vectors.append(t_vector)

    def select(self, t_vector: Tuple[float, ...]) -> Any:
        """Select (retrieve) the state table for the given t_vector."""
        return self.archive.get(t_vector, None)

    def update(self, t_vector: Tuple[float, ...], state_table: Any):
        """Update the state table at the given t_vector, if it exists."""
        if t_vector in self.archive:
            self.archive[t_vector] = state_table

    def delete(self, t_vector: Tuple[float, ...]):
        """Delete the state table at the given t_vector, if it exists."""
        if t_vector in self.archive:
            del self.archive[t_vector]
            self.t_vectors.remove(t_vector)

    # --- Legacy/compatibility methods ---
    def archive_state(self, t_vector: Tuple[float, ...], state_table: Any):
        self.insert(t_vector, state_table)

    def get_state(self, t_vector: Tuple[float, ...]) -> Any:
        return self.select(t_vector)

    def available_t_vectors(self) -> List[Tuple[float, ...]]:
        return self.t_vectors[:]
