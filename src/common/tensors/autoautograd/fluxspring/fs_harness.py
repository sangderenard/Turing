# -*- coding: utf-8 -*-
"""Auxiliary harness managing ring buffers for FluxSpring graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from ...abstraction import AbstractTensor as AT


@dataclass
class LineageLedger:
    """Ledger mapping tick numbers to lineage identifiers.

    The ledger tracks which lineage was active at a given tick.  This allows
    callers to later retrieve lineage‑aligned histories from the
    :class:`RingHarness` rather than relying on the wall‑clock ordering of
    pushes.  Lineage identifiers are kept abstract and are typically simple
    integers.
    """

    tick_to_lineage: Dict[int, int] = field(default_factory=dict)

    def record(self, tick: int, lineage_id: int) -> None:
        """Associate ``tick`` with ``lineage_id``."""

        self.tick_to_lineage[tick] = lineage_id

    def lineages(self) -> Tuple[int, ...]:
        """Return the unique lineage identifiers seen so far."""

        return tuple(dict.fromkeys(self.tick_to_lineage.values()))


@dataclass
class RingBuffer:
    """Simple differentiable ring buffer."""

    buf: AT
    idx: int = 0

    def push(self, val: AT) -> AT:
        i = self.idx % int(len(self.buf))
        self.buf = AT.scatter_row(self.buf.clone(), i, val, dim=0)
        self.idx = i + 1
        return self.buf


@dataclass
class RingHarness:
    """Own per-node and per-edge ring buffers keyed by lineage."""

    default_size: Optional[int] = None
    node_rings: Dict[Tuple[int, ...], RingBuffer] = field(default_factory=dict)
    edge_rings: Dict[Tuple[int, ...], RingBuffer] = field(default_factory=dict)

    def _key(self, obj_id: int, lineage: Tuple[int, ...] | None) -> Tuple[int, ...]:
        return (obj_id,) if lineage is None else (obj_id, *lineage)

    def _ensure_node_ring(
        self, key: Tuple[int, ...], D: int, size: Optional[int]
    ) -> Optional[RingBuffer]:
        size = size or self.default_size
        if size is None:
            return None
        if key not in self.node_rings:
            buf = AT.zeros((size, D), dtype=float)
            self.node_rings[key] = RingBuffer(buf)
        return self.node_rings[key]

    def _ensure_edge_ring(
        self, key: Tuple[int, ...], size: Optional[int]
    ) -> Optional[RingBuffer]:
        size = size or self.default_size
        if size is None:
            return None
        if key not in self.edge_rings:
            buf = AT.zeros(size, dtype=float)
            self.edge_rings[key] = RingBuffer(buf)
        return self.edge_rings[key]

    def push_node(
        self,
        node_id: int,
        val: AT,
        *,
        lineage: Tuple[int, ...] | None = None,
        size: Optional[int] = None,
    ) -> AT | None:
        key = self._key(node_id, lineage)
        t = AT.get_tensor(val)
        D = int(t.shape[0]) if getattr(t, "ndim", 0) > 0 else 1
        rb = self._ensure_node_ring(key, D, size)
        if rb is None:
            return None
        return rb.push(val)

    def push_edge(
        self,
        edge_idx: int,
        val: AT,
        *,
        lineage: Tuple[int, ...] | None = None,
        size: Optional[int] = None,
    ) -> AT | None:
        key = self._key(edge_idx, lineage)
        rb = self._ensure_edge_ring(key, size)
        if rb is None:
            return None
        return rb.push(val)

    def get_node_ring(
        self, node_id: int, *, lineage: Tuple[int, ...] | None = None
    ) -> Optional[RingBuffer]:
        return self.node_rings.get(self._key(node_id, lineage))

    def get_edge_ring(
        self, edge_idx: int, *, lineage: Tuple[int, ...] | None = None
    ) -> Optional[RingBuffer]:
        return self.edge_rings.get(self._key(edge_idx, lineage))

