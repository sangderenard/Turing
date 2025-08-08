import collections
import ctypes
from importlib import abc
import itertools
import math
import random
import re
import sys
import threading
from uuid import uuid4

from ....cells.simulator import Simulator
from ....cells.simulator_methods.salinepressure import SalineHydraulicSystem
from ....cells.cell_consts import Cell

import json

# Mirror LinearCells flag used by older region manager
IMMUTABLE = 1 << 5
from .node_entry import NodeEntry
from .edge_entry import EdgeEntry

class NetworkxEmulation:
    """
    A class to emulate NetworkX-like graph operations on bit tensors.
    This is a placeholder for future implementation.
    """
    class NodesEmulator:
        def __init__(self, bit_tensor_memory_graph):
            self.bit_tensor_memory_graph = bit_tensor_memory_graph
        def __getitem__(self, node_id):
            return self.bit_tensor_memory_graph.get_node(node_id)
        def __setitem__(self, node_id, attr):
            self.bit_tensor_memory_graph.set_node(node_id, **attr)
        def append(self, node_id, attr):
            self.bit_tensor_memory_graph.add_node(node_id, **attr)

    class EdgesEmulator:
        def __init__(self, bit_tensor_memory_graph):
            self.bit_tensor_memory_graph = bit_tensor_memory_graph
        def __getitem__(self, edge_id):
            return self.bit_tensor_memory_graph.get_edge(edge_id)
        def __setitem__(self, edge_id, attr):
            self.bit_tensor_memory_graph.add_edge(*edge_id, **attr)
        def append(self, edge):
            self.bit_tensor_memory_graph.add_edge(edge[0], edge[1], **edge[2] if len(edge) > 2 else {})

    def __init__(self, bit_tensor_memory_graph):
        self.bit_tensor_memory_graph = bit_tensor_memory_graph
        self.nodes = NetworkxEmulation.NodesEmulator(bit_tensor_memory_graph)
        self.edges = NetworkxEmulation.EdgesEmulator(bit_tensor_memory_graph)
        self.node_count = self.bit_tensor_memory_graph.node_count
        self.edge_count = self.bit_tensor_memory_graph.edge_count
    def add_node(self, node_id, **attr):
        return self.nodes.append(node_id, attr)
    
    def add_edge(self, src, dst, **attr):
        self.edges.append((src, dst, attr))
    

    def get_node(self, node_id):
        return self.nodes[node_id]

    def get_edges(self):
        return self.edges
    
    def to_edges(self, target):
        """
        all edges in the graph to a target bit tensor
        """
        return self.bit_tensor_memory_graph.find_edges(source=None, target=target)
    def from_edges(self, source):
        """
        all edges from a source bit tensor
        """
        return self.bit_tensor_memory_graph.find_edges(source=source, target=None)
    # ───────────────────────────────────────────────────────────
    # ①  core: node enumeration straight from BitTensorMemoryGraph
    # ───────────────────────────────────────────────────────────
    def _iter_node_ids(self):
        """
        Generator over all node_id values currently present
        in the backing BitTensorMemoryGraph.
        """
        bt = self.bit_tensor_memory_graph
        offs = bt.find_in_span((bt.n_start, bt.e_start), ctypes.sizeof(NodeEntry))
        if offs == getattr(bt, "NOTHING_TO_FLY", -1):
            return
        for off in offs:
            raw = bt.hard_memory.read(off, ctypes.sizeof(NodeEntry))
            entry = NodeEntry.from_buffer_copy(raw)
            yield entry.node_id

    # ───────────────────────────────────────────────────────────
    # ②  NetworkX-style protocol methods
    # ───────────────────────────────────────────────────────────
    def __iter__(self):
        """`for n in G:` → iterate over node IDs"""
        return self._iter_node_ids()

    def __contains__(self, nid):
        """`nid in G` and `if nid not in G:`"""
        for x in self._iter_node_ids():
            if x == nid:
                return True
        return False

    def __len__(self):
        """`len(G)`"""
        return sum(1 for _ in self._iter_node_ids())

    # convenience accessors (optional but handy)
    def nodes(self):
        return list(self._iter_node_ids())

    def edges(self):
        # simple edge-list; expand as needed
        bt = self.bit_tensor_memory_graph
        offs = bt.find_in_span((bt.e_start, bt.p_start), ctypes.sizeof(EdgeEntry))
        if offs == getattr(bt, "NOTHING_TO_FLY", -1):
            return []
        out = []
        for off in offs:
            e = EdgeEntry.from_buffer_copy(bt.hard_memory.read(off, ctypes.sizeof(EdgeEntry)))
            out.append((e.src_ptr, e.dst_ptr))
        return out
# in python we can't create something permanent inside a function
# so we need a container for all active meta nodes
import ctypes, zlib
import hashlib
meta_nodes = {}
root_meta_nodes = set()
master_graph = None
