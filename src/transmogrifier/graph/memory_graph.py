"""
Backward-compatible shim for transmogrifier.graph.memory_graph

This module re-exports the split package implementation now located under
transmogrifier.graph.memory_graph.* so legacy imports keep working:

    from transmogrifier.graph.memory_graph import BitTensorMemoryGraph, NodeEntry, ...
"""

# Primary graph API
from .memory_graph import BitTensorMemoryGraph  # noqa: F401

# Helper re-exports to preserve the original public surface
from .memory_graph.helpers import (  # noqa: F401
    BitTensorMemoryDAGHelper,
    StructView,
    BitTensorMemory,
    NodeEntry,
    NodeRegion,
    SetMicrograinEntry,
    EdgeEntry,
    MetaGraphEdge,
    META_GRAPH_TRANSFER_BUFFER_SIZE,
    BTGraphHeader,
    NetworkxEmulation,
    GraphSearch,
    Deque3D,
    MaskConsolidation,
    BitTensorMemoryUnits,
)

__all__ = [
    # core
    "BitTensorMemoryGraph",
    # helpers
    "BitTensorMemoryDAGHelper",
    "StructView",
    "BitTensorMemory",
    "NodeEntry",
    "NodeRegion",
    "SetMicrograinEntry",
    "EdgeEntry",
    "MetaGraphEdge",
    "META_GRAPH_TRANSFER_BUFFER_SIZE",
    "BTGraphHeader",
    "NetworkxEmulation",
    "GraphSearch",
    "Deque3D",
    "MaskConsolidation",
    "BitTensorMemoryUnits",
]