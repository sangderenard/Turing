# -*- coding: utf-8 -*-
"""Auxiliary harness managing ring buffers for FluxSpring graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from ...abstraction import AbstractTensor as AT
from .fs_types import FluxSpringSpec


def _tape():
    """Return the global autograd tape."""
    try:
        return AT.autograd.tape
    except Exception:
        from ...autograd import autograd as _ag
        return _ag.tape


_LABEL_STAGE_DEPTH: Dict[str, int] = {}


def label_stage_depth(label: str) -> int:
    """Return the pipeline stage depth for ``label``."""

    for prefix, depth in _LABEL_STAGE_DEPTH.items():
        if label.startswith(prefix):
            return depth
    return 0


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
class ParamRing:
    """Ring buffer storing parameter snapshots and their ticks."""

    buf: AT
    ticks: List[int]
    idx: int = 0

    def push(self, tick: int, val: AT) -> AT:
        i = self.idx % int(len(self.buf))
        self.buf = AT.scatter_row(self.buf.clone(), i, val, dim=0)
        self.ticks[i] = tick
        self.idx = i + 1
        return self.buf

    def value_at(self, tick: int) -> AT:
        for i, t in enumerate(self.ticks):
            if t == tick:
                return self.buf[i]
        raise KeyError(f"tick {tick} not found")


@dataclass
class RingHarness:
    """Own per-node and per-edge ring buffers keyed by lineage."""

    default_size: Optional[int] = None
    node_rings: Dict[Tuple[int, ...], RingBuffer] = field(default_factory=dict)
    edge_rings: Dict[Tuple[int, ...], RingBuffer] = field(default_factory=dict)
    param_rings: Dict[str, ParamRing] = field(default_factory=dict)
    param_labels: List[str] = field(default_factory=list)
    tick: int = 0

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

    def _ensure_param_ring(
        self, label: str, D: int, size: Optional[int]
    ) -> Optional[ParamRing]:
        size = size or self.default_size
        if size is None:
            return None
        if label not in self.param_rings:
            buf = AT.zeros((size, D), dtype=float)
            ticks = [0] * size
            self.param_rings[label] = ParamRing(buf, ticks)
        return self.param_rings[label]

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

    # ------------------------------------------------------------------
    # Parameter versioning
    # ------------------------------------------------------------------
    def snapshot_learnables(
        self, spec: FluxSpringSpec, *, size: Optional[int] = None
    ) -> None:
        """Record current learnable parameters with tick index."""

        tape = _tape()

        def _maybe(label: str | None, p: AT | None) -> Iterable[Tuple[str, AT]]:
            if p is None or not getattr(p, "requires_grad", False):
                return []
            if label is None:
                label = tape.graph.nodes.get(id(p), {}).get("annotations", {}).get("label")
            if label is None:
                return []
            return [(label, p)]

        # Nodes
        for n in spec.nodes:
            for attr in ("alpha", "w", "b"):
                p = getattr(n.ctrl, attr)
                lbl = tape.graph.nodes.get(id(p), {}).get("annotations", {}).get("label") if p is not None else None
                for label, param in _maybe(lbl, p):
                    val = AT.get_tensor(param).reshape(-1)
                    D = int(val.shape[0])
                    ring = self._ensure_param_ring(label, D, size)
                    if ring is not None:
                        ring.push(self.tick, val)
                        if label not in self.param_labels:
                            self.param_labels.append(label)

        # Edges
        for e in spec.edges:
            for attr in ("alpha", "w", "b"):
                p = getattr(e.ctrl, attr)
                lbl = tape.graph.nodes.get(id(p), {}).get("annotations", {}).get("label") if p is not None else None
                for label, param in _maybe(lbl, p):
                    val = AT.get_tensor(param).reshape(-1)
                    D = int(val.shape[0])
                    ring = self._ensure_param_ring(label, D, size)
                    if ring is not None:
                        ring.push(self.tick, val)
                        if label not in self.param_labels:
                            self.param_labels.append(label)
            for attr in ("kappa", "k", "l0", "lambda_s", "x"):
                p = getattr(e.transport, attr)
                lbl = tape.graph.nodes.get(id(p), {}).get("annotations", {}).get("label") if p is not None else None
                for label, param in _maybe(lbl, p):
                    val = AT.get_tensor(param).reshape(-1)
                    D = int(val.shape[0])
                    ring = self._ensure_param_ring(label, D, size)
                    if ring is not None:
                        ring.push(self.tick, val)
                        if label not in self.param_labels:
                            self.param_labels.append(label)

        # Faces
        for f in spec.faces:
            for attr in ("alpha", "c"):
                p = getattr(f, attr, None)
                lbl = tape.graph.nodes.get(id(p), {}).get("annotations", {}).get("label") if p is not None else None
                for label, param in _maybe(lbl, p):
                    val = AT.get_tensor(param).reshape(-1)
                    D = int(val.shape[0])
                    ring = self._ensure_param_ring(label, D, size)
                    if ring is not None:
                        ring.push(self.tick, val)
                        if label not in self.param_labels:
                            self.param_labels.append(label)

        self.tick += 1

    def get_params_for_lineages(
        self, lineage_ids: Iterable[int], ledger: LineageLedger
    ) -> AT:
        """Return concatenated parameter snapshots for ``lineage_ids``."""

        rows = []
        for lid in lineage_ids:
            base_tick = ledger.tick_of_lid[lid]
            parts = []
            for label in self.param_labels:
                ring = self.param_rings[label]
                t = base_tick + label_stage_depth(label)
                val = ring.value_at(t)
                parts.append(val)
            rows.append(AT.concat(parts, dim=0))
        return AT.stack(rows)

