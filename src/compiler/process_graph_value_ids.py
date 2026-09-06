"""Monotonic ProcessGraph value-id allocation.

A ProcessGraph value id is an identity, not an array slot.  Allocating
``max(nodes) + 1`` after a pass has removed the highest-numbered nodes hands a
freed id to an unrelated new node; every ledger that still names the old id
(aggregate member lists, loop port correlations, identity histories) then
silently points at the wrong value.  Allocation therefore advances a
per-graph watermark that never moves backwards.
"""
from __future__ import annotations

from typing import Any

WATERMARK_KEY = "value_id_watermark"


def next_process_value_id(graph_or_digraph: Any) -> int:
    """Return a fresh value id for ``graph`` and advance its watermark."""

    digraph = getattr(graph_or_digraph, "G", graph_or_digraph)
    metadata = digraph.graph
    watermark = int(metadata.get(WATERMARK_KEY, -1))
    highest = max((int(node_id) for node_id in digraph.nodes), default=-1)
    allocated = max(watermark, highest) + 1
    metadata[WATERMARK_KEY] = allocated
    return allocated


__all__ = ["WATERMARK_KEY", "next_process_value_id"]
