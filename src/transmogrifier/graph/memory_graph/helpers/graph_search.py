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

from src.cells.simulator import Simulator
from src.cells.cellsim.api.saline import SalinePressureAPI as SalineHydraulicSystem
from src.cells.cell_consts import Cell
import zlib
from .bt_graph_header import BTGraphHeader
from .node_entry import NodeEntry
from .edge_entry import EdgeEntry
from .meta_graph_edge import MetaGraphEdge
from .bit_tensor_memory import BitTensorMemory

import json

# Mirror LinearCells flag used by older region manager
IMMUTABLE = 1 << 5

meta_nodes = {}
root_meta_nodes = set()
master_graph = None

class GraphSearch:
    """ A class to handle searching and building meta nodes in a graph."""
    def __init__(self, meta_nodes_override=None, root_meta_nodes_override=None, master_graph_override=None):
        global meta_nodes, root_meta_nodes, master_graph
        self.meta_nodes = meta_nodes_override if meta_nodes_override is not None else meta_nodes
        self.root_meta_nodes = root_meta_nodes_override if root_meta_nodes_override is not None else root_meta_nodes
        self.master_graph = master_graph_override if master_graph_override is not None else master_graph

        if self.master_graph is None:
            self.build_master_graph(force=True)

        if self.meta_nodes is None:
            self.meta_nodes = {}

        if self.root_meta_nodes is None:
            self.root_meta_nodes = set()

    @staticmethod
    def _build_struct_bytes(inst):
        """
        Return *bytes* for `inst` with a fresh CRC-32.

        • Any field whose name contains '_pad' (case-insensitive) is zeroed.
        • Any field whose name starts with 'checksum' is treated as checksum
        storage: it is blanked before hashing, then overwritten with the
        CRC in little-endian order, chunked to the field's ctype size.
        • Works with mixed-size checksum fields (e.g. one c_uint32, or a
        run of c_uint16 words).
        """
        T    = type(inst)
        size = ctypes.sizeof(T)
        raw  = bytearray(ctypes.string_at(ctypes.addressof(inst), size))

        pad_rx = re.compile(r'_?pad\d*$',      re.I)      # …_pad0, pad_1, _pad, etc.
        sum_rx = re.compile(r'^checksum',      re.I)      # checksum, checksuma, …

        checksum_slots = []         # (name, offset, nbytes, ctype)

        # ── Pass-1 : mask pads & checksum fields, remember checksum slots
        for fld_name, fld_ctype in T._fields_:
            off   = getattr(T, fld_name).offset
            nbyte = ctypes.sizeof(fld_ctype)

            if pad_rx.search(fld_name):
                raw[off:off+nbyte] = b'\x00' * nbyte
            elif sum_rx.search(fld_name):
                raw[off:off+nbyte] = b'\x00' * nbyte
                checksum_slots.append((fld_name, off, nbyte, fld_ctype))

        # ── Pass-2 : compute CRC-32 on the masked blob
        crc32  = zlib.crc32(raw) & 0xFFFFFFFF
        crc_le = crc32.to_bytes(4, "little")   # 4-byte little-endian buffer
        cursor = 0

        # ── Pass-3 : write CRC back into struct & raw array
        for fld_name, off, nbyte, fld_ctype in checksum_slots:
            slice_ = crc_le[cursor:cursor+nbyte]
            slice_ += b'\x00' * (nbyte - len(slice_))     # pad if fewer than nbyte
            raw[off:off+nbyte] = slice_

            # keep the live instance coherent (works for scalar ctypes)
            if issubclass(fld_ctype, ctypes._SimpleCData):
                setattr(inst, fld_name, fld_ctype(int.from_bytes(slice_, "little")))

            cursor += nbyte
            if cursor >= 4:            # CRC fully written: extra checksum*
                break                  # fields remain zero

        return bytes(raw)



    def heuristic_memory_build(self, memory):
        new_memory = BitTensorMemory(memory.size, self.master_graph)
        sanity_patterns = [
            BTGraphHeader,
            NodeEntry,
            EdgeEntry,
            MetaGraphEdge,
        ]
        captures = []
        captures = [pattern() for pattern in sanity_patterns if hasattr(pattern, 'checksuma')]
        if not captures:
            raise ValueError("No valid sanity patterns found in memory")
        
        size = ctypes.sizeof(memory.data)
        for i in range(size):
            for j, pattern in enumerate(sanity_patterns):
                # Attempt to match struct by validating checksum and pad fields via _build_struct_bytes
                raw_bytes = ctypes.string_at(ctypes.addressof(memory.data) + i, ctypes.sizeof(pattern))
                # Create an instance from the raw bytes
                inst = pattern.from_buffer_copy(raw_bytes)
                # Recompute canonical bytes (pads zeroed, checksum fields masked and CRC32 applied)
                recomputed = self._build_struct_bytes(inst)
                # If recomputed bytes match the raw memory, we have a valid capture
                if recomputed == raw_bytes:
                    captures[j].append(inst)
                    break
        header_offset = ctypes.sizeof(BTGraphHeader)
        #leave the header empty but provide courtesy space for optimum packing
        header_found = captures[0] if captures else None
        if header_found:
            self.master_graph.hard_memory.write(0, ctypes.string_at(ctypes.addressof(header_found), ctypes.sizeof(BTGraphHeader)))
        node_count = len(captures[1]) if len(captures) > 1 else 0
        edge_count = len(captures[2]) if len(captures) > 2 else 0
        associations = len(captures[3]) if len(captures) > 3 else 0

        def offsetter(offset):
            return (offset + self.master_graph.hard_memory.granular_size - 1) // self.master_graph.hard_memory.granular_size

        node_offset = offsetter(header_offset)
        edge_offset = node_offset + ([offsetter(node_count * ctypes.sizeof(NodeEntry))for _ in range(node_count)])
        parent_offset = edge_offset + ([offsetter(edge_count * ctypes.sizeof(EdgeEntry)) for _ in range(edge_count)])
        child_offset = parent_offset + ([offsetter(associations * ctypes.sizeof(MetaGraphEdge)) for _ in range(associations)])

        self.master_graph.start_n = node_offset
        self.master_graph.start_e = edge_offset
        self.master_graph.start_p = parent_offset
        self.master_graph.start_c = child_offset

        self.master_graph.node_count = node_count
        self.master_graph.edge_count = edge_count
        self.master_graph.parent_count = associations
        self.master_graph.child_count = 0 #become fully subordinate as a regenerated graph

        # Build contiguous byte blocks and write in one go
        nodespan = [n for n in captures[1] if isinstance(n, NodeEntry)]
        edgespan = [e for e in captures[2] if isinstance(e, EdgeEntry)]
        parentspan = [p for p in captures[3] if isinstance(p, MetaGraphEdge)]

        # Serialize spans to bytes
        node_bytes = b''.join(ctypes.string_at(ctypes.addressof(n), ctypes.sizeof(NodeEntry)) for n in nodespan)
        edge_bytes = b''.join(ctypes.string_at(ctypes.addressof(e), ctypes.sizeof(EdgeEntry)) for e in edgespan)
        parent_bytes = b''.join(ctypes.string_at(ctypes.addressof(p), ctypes.sizeof(MetaGraphEdge)) for p in parentspan)

        self.master_graph.hard_memory.write(node_offset, node_bytes)
        self.master_graph.hard_memory.write(edge_offset, edge_bytes)
        self.master_graph.hard_memory.write(parent_offset, parent_bytes)

        return self.master_graph

    def parent_child_traversal_build(self, graph, meta_nodes, root_meta_nodes):
        """
        Traverse the parent-child relationships in the meta nodes
        and build the master graph accordingly.
        """
        new_nodes_present = 1
        visited = set()
        while new_nodes_present > 0:
            for i in range(2):
                original_length = len(visited)
                new_nodes_present = len(visited)
                upflow = i % 2 == 0
                if upflow:
                    for node_id, node in meta_nodes.items():
                        
                        parents = node.get_parents()
                        parents = [parent for parent in parents if parent not in visited]
                        if not parents:
                            root_meta_nodes.add(node_id)
                        else:
                            parent_nodes = [meta_nodes.get(p) for p in parents if p in meta_nodes]
                            
                            for parent_node in parent_nodes:
                                if parent_node:
                                    graph.add_edge(parent_node.node_id, node.node_id, **node.attributes)
                        # Add node to the graph
                        graph.add_node(node.node_id, **node.attributes)
                        visited.add(node.node_id)
                else:
                    for node_id, node in meta_nodes.items():
                        children = node.get_children()
                        children = [child for child in children if child not in visited]
                        
                        child_nodes = [meta_nodes.get(c) for c in children if c in meta_nodes]
                        for child_node in child_nodes:
                            if child_node:
                                graph.add_edge(node.node_id, child_node.node_id, **child_node.attributes)
                        # Add node to the graph
                        graph.add_node(node.node_id, **node.attributes)
                        visited.add(node.node_id)

            new_nodes_present = original_length - new_nodes_present
        
        self.meta_nodes = meta_nodes
        self.root_meta_nodes = root_meta_nodes

        return graph
    
    def straight_build(self, nodes, edges):
        """
        Build a straight graph from the given nodes and edges.
        This is a placeholder for future implementation.
        """
        # Placeholder for straight build logic
        from ..memory_graph import BitTensorMemoryGraph
        return BitTensorMemoryGraph(1024 * 1024 * len(nodes))
        
    def find_permanent_storage_spores(self, folder, capsid_ids=None):
        """
        Find and return permanent storage spores from the given folder.
        This is a placeholder for future implementation.
        """
        # Placeholder for finding spores logic
        return []

    def rehydrate_spore(self, compressed_data, header):
        """
        Rehydrate a spore from compressed data and header.
        This is a placeholder for future implementation.
        """
        # Placeholder for rehydration logic
        from ..memory_graph import BitTensorMemoryGraph
        return BitTensorMemoryGraph(1024 * 1024 * len(compressed_data))

    def push_to_global(self):
        """
        Push the current state of meta nodes, root meta nodes, and master graph
        to the global variables.
        """
        global meta_nodes, root_meta_nodes, master_graph
        if self.meta_nodes is not None:
            meta_nodes = self.meta_nodes
        if self.root_meta_nodes is not None:
            root_meta_nodes = self.root_meta_nodes
        if self.master_graph is not None:
            master_graph = self.master_graph

    def build_master_graph(self, force=False, whatif=False):
        global meta_nodes, root_meta_nodes, master_graph

        if master_graph is not None and not force:
            return master_graph
        
        from ..memory_graph import BitTensorMemoryGraph
        master_graph = BitTensorMemoryGraph(1024 * 1024 * len(meta_nodes))  # 1MB default size
        master_graph.capsid_id = 0  # root capsid ID
        master_graph.encapsidate_capsid()  # encapsulate the capsid

        master_graph = self.parent_child_traversal_build(master_graph, meta_nodes, root_meta_nodes)

        if whatif:
            # If whatif is True, we don't push to global variables
            return master_graph
        
        # Push the built graph to global variables
        self.push_to_global()
        return master_graph
