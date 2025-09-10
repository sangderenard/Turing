# -*- coding: utf-8 -*-
"""Auxiliary harness managing ring buffers for FluxSpring graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from ...abstraction import AbstractTensor as AT


@dataclass
class LineageLedger:
    """Track lineage identifiers and their arrival order.

    The ledger assigns monotonically increasing *lineage identifiers* (LIDs)
    whenever an input sample is ingested.  Each ingestion is associated with
    the current ``tick`` counter which increases in lock‑step.  Two look‑up
    dictionaries allow callers to translate between ticks and LIDs so that
    ring‑buffer histories can be aligned with the originating input.
    """

    tick: int = 0
    next_lid: int = 0
    tick_of_lid: Dict[int, int] = field(default_factory=dict)
    lid_by_tick: Dict[int, int] = field(default_factory=dict)

    def ingest(self) -> int:
        """Register a fresh input arrival and return its lineage ID.

        Each call assigns the next available lineage identifier and records the
        bidirectional mappings between ``tick`` and ``lineage``.  The ``tick``
        counter is then advanced so that subsequent ingestions map to later
        ticks.
        """

        lid = self.next_lid
        t = self.tick
        self.next_lid += 1
        self.tick += 1
        self.tick_of_lid[lid] = t
        self.lid_by_tick[t] = lid
        return lid

    def record(self, tick: int, lineage_id: int) -> None:
        """Associate ``tick`` with ``lineage_id``.

        This helper mirrors the previous public API so legacy callers and unit
        tests remain valid.  The internal counters are updated to stay
        consistent with any manually recorded events.
        """

        self.tick_of_lid[lineage_id] = tick
        self.lid_by_tick[tick] = lineage_id
        if tick >= self.tick:
            self.tick = tick + 1
        if lineage_id >= self.next_lid:
            self.next_lid = lineage_id + 1

    def lineages(self) -> Tuple[int, ...]:
        """Return the known lineage identifiers ordered by tick."""

        return tuple(self.lid_by_tick[t] for t in sorted(self.lid_by_tick))


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

