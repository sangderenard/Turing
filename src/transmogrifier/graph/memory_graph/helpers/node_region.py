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
class NodeRegion(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("pad_0", ctypes.c_byte * 16),          # prediction leader metadata
        ("active_region_0", ctypes.c_byte * 32), # prediction window
        ("pad_1", ctypes.c_byte * 48),
        ("active_region_1", ctypes.c_byte * 32), # delta trail or PID components
        ("active_region_2", ctypes.c_byte * 128),# meta / schema / exponent delta
        ("pad_2", ctypes.c_byte * 256)           # overflow, persistent tail, or hooks
    ]
    def inverted_mask_view(self) -> memoryview:
        """
        Return a memoryview of the full 512-byte structure,
        bitwise-inverted, non-destructively.
        """
        raw = bytes(ctypes.string_at(ctypes.addressof(self), ctypes.sizeof(self)))
        inverted = bytes(b ^ 0xFF for b in raw)
        return memoryview(inverted)

    def register_set(self, node_ids, memory_graph):
        """
        Register a set of nodes from a memory graph
        as being members of this region instance
        and subject to all its handler
        """

        if not hasattr(self, 'node_ids'):
            self.node_ids = set()
        if not hasattr(self, 'memory_graphs'):
            self.memory_graphs = {}

        not_yet_registered = set(node_ids) - set(self.node_ids)
        if not_yet_registered:
            # Trigger any hooks or handlers for the newly registered nodes
            for node_id in not_yet_registered:
                self.hooks.get('register', lambda n, g: None)(node_id, memory_graph)
            self.node_ids.update(not_yet_registered)
        if not hasattr(self, 'memory_graphs'):
            self.memory_graphs = {}
        if id(memory_graph) not in self.memory_graphs:
            # Register the memory graph if not already done
            self.hooks.get('register_graph', lambda g: None)(memory_graph)
            self.memory_graphs[id(memory_graph)] = memory_graph

    def zero_all(self):
        ctypes.memset(ctypes.addressof(self), 0, ctypes.sizeof(self))

    def get_full_region(self) -> memoryview:
        return memoryview(ctypes.string_at(ctypes.addressof(self), ctypes.sizeof(self)))

    def get_active_regions(self) -> dict:
        return {
            "32_0": bytes(self.active_region_0),
            "32_1": bytes(self.active_region_1),
            "128_0":       bytes(self.active_region_2)
        }

    def get_contiguous_active_regions(self) -> memoryview:
        """
        Returns a memoryview of the active regions concatenated.
        """
        return memoryview(self.active_region_0) + memoryview(self.active_region_1) + memoryview(self.active_region_2)

    def set_contiguous_active_regions(self, data: bytes):

        if len(data) != 192:
            raise ValueError("Data must be exactly 192 bytes long")

        ctypes.memmove(self.active_region_0, data[:32], 32)
        ctypes.memmove(self.active_region_1, data[32:64], 32)
        ctypes.memmove(self.active_region_2, data[64:], 128)

    def set_active_region(self, name: str, data: bytes):
        if name == "32_0":
            ctypes.memmove(self.active_region_0, data, min(len(data), 32))
        elif name == "32_1":
            ctypes.memmove(self.active_region_1, data, min(len(data), 32))
        elif name == "128_0":
            ctypes.memmove(self.active_region_2, data, min(len(data), 128))

    def install_schema_handler(self, hooks, schema_handler_fns, graphs=None, stack=False):
        """Attach schema-specific init hook or tracking logic."""
        if graphs is None:
            graphs = list(self.memory_graphs) if hasattr(self, 'memory_graphs') else []
            if not graphs:
                return -1
        if not isinstance(hooks, list):
            hooks = [hooks]
        if not isinstance(schema_handler_fns, list):
            schema_handler_fns = [schema_handler_fns]
        if not hasattr(self, 'hooks'):
            self.hooks = {}
        for graph in graphs:
            if graph not in self.hooks:
                self.hooks[graph] = {}
            for schema_handler_fn, hook in zip(schema_handler_fns, hooks):
                if callable(schema_handler_fn):
                    if stack:
                        if hook not in self.hooks[graph]:
                            self.hooks[graph][hook] = []
                        self.hooks[graph][hook].append(schema_handler_fn)
                    else:
                        if hook not in self.hooks[graph]:
                            self.hooks[graph][hook] = [schema_handler_fn]
                        else:
                            self.hooks[graph][hook] = [schema_handler_fn]
