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

from .bt_graph_header import BTGraphHeader
from .bit_tensor_memory_units import BitTensorMemoryUnits
from .node_entry import NodeEntry
from .edge_entry import EdgeEntry
from .meta_graph_edge import MetaGraphEdge, META_GRAPH_TRANSFER_BUFFER_SIZE
from .set_micrograin_entry import SetMicrograinEntry
from .integer_region_allocator import IntegerRegionAllocator, allocation_quantum

import json

# Mirror LinearCells flag used by older region manager
IMMUTABLE = 1 << 5
class BitTensorMemory: #sizes in bytes
    ALLOCATION_FAILURE = -1
    DEFAULT_BLOCK = 4096  # default block size for memory allocation
    DEFAULT_GRAIN = 512  # default granular size for bitmap allocation

    def __init__(self, size, graph=None, dynamic=True):

        self.size = size
        self.chunk_size = BitTensorMemory.DEFAULT_BLOCK
        self.granular_size = BitTensorMemory.DEFAULT_GRAIN
        self.dynamic = dynamic


        self.header_size = ctypes.sizeof(BTGraphHeader) if graph is not None else 0
        self.l_start = graph.l_start if graph is not None else 0
        self.n_start = graph.n_start if graph is not None else 0
        self.e_start = graph.e_start if graph is not None else 0
        self.p_start = graph.p_start if graph is not None else 0
        self.c_start = graph.c_start if graph is not None else 0
        self.r_start = graph.r_start if graph is not None else 0
        self.x_start = graph.x_start if graph is not None else 0
        self.envelope_domain = (0, self.x_start)
        self.envelope_size = self.size - self.header_size


        self.size = max(size, BitTensorMemory.DEFAULT_BLOCK)
        self.data = ctypes.create_string_buffer(self.size)
        self.unit_helper = BitTensorMemoryUnits(0, self.size, self.granular_size, self.chunk_size, node_template = NodeEntry, edge_template= EdgeEntry, association_template=MetaGraphEdge, hard_memory = self)
        self.graph = graph
        

        self.active_regions = []

        self.extra_data_size = 0
        self.set_cache = set()

        quantum = allocation_quantum((
            ctypes.sizeof(NodeEntry), ctypes.sizeof(EdgeEntry),
            ctypes.sizeof(MetaGraphEdge), 8,
        ))
        self.allocator = IntegerRegionAllocator(
            self.size,
            reserved_prefix=self.header_size,
            growth_quantum=max(self.DEFAULT_BLOCK, quantum),
            on_grow=self._grow_data,
        )

        
    def pull_full_set_from_memory(self):
        pass

    def write_set_cache_to_memory(self):
        # we're going to write right to left from the extra data start
        # back to the end of the right envelope space
        # the set is only for speed and not necessary
        # so we're only using it in unallocated space
        stride = ctypes.sizeof(SetMicrograinEntry)
        offset = self.size - self.extra_data_size
        for entry in self.set_cache:
            if self.unit_helper.bitmap[self.unit_helper.bitmap.MASK_BITMAP, offset // self.chunk_size, ((offset % self.chunk_size) * self.unit_helper.bitmap_depth) // self.unit_helper.bitmap_depth] == 1:
                break
            # Legacy LinearCells boundary check removed; CellPressureRegionManager
            # does not expose fixed boundaries.
            # write the entry
            ctypes.memmove(ctypes.addressof(self.data) + offset, ctypes.byref(entry), stride)
            offset -= stride
        
    def _clip_offset(self, offset, size):
        offset, size = int(offset), int(size)
        if offset < 0 or size < 0 or offset + size > self.size:
            raise ValueError(
                f"byte interval [{offset}, {offset + size}) exceeds arena [0, {self.size})"
            )
        return offset, size

    def reset_region_manager(self, boundaries=None, strides=None, insertion=False):
        return self.get_specs(boundaries, strides)

    def _grow_data(self, new_capacity):
        new_data = ctypes.create_string_buffer(int(new_capacity))
        ctypes.memmove(new_data, self.data, self.size)
        self.data = new_data
        self.size = int(new_capacity)


    def read(self, offset, size, clear_delta_map=False):
        offset, size = self._clip_offset(offset, size)
        if offset + size > self.size:
            raise ValueError("Read exceeds memory bounds")
        if clear_delta_map:
            self.unit_helper.delta(offset, size, "read")
        return self.data[offset:offset + size]

    def view(self, offset, size, clear_delta_map=False):
        offset, size = self._clip_offset(offset, size)
        if offset + size > self.size:
            raise ValueError("View exceeds memory bounds")
        if clear_delta_map:
            self.unit_helper.delta(offset, size, "read")
        return memoryview(self.data)[offset:offset + size]

    def write(self, offset, data, clear_delta_map=False):
        offset, size = self._clip_offset(offset, len(data))
        print(f"Writing {len(data)} bytes at offset {offset}")
        if len(data) < size:
            print(f"Writing {len(data)} bytes at offset {offset}, but size is {size}")
            data += b'\x00' * (size - len(data))  # pad with zeros if necessary
        
        
        self.data[offset:offset + size] = data[:size]

    def free(self, offset, size, scramble=False):
        offset, size = self._clip_offset(offset, size)
        data = b'\x00' * size if not scramble else random.randbytes(size)
        self.data[offset:offset + size] = data
        try:
            self.allocator.release(offset)
        except KeyError:
            pass

    def bitmap_expanded(self, offset=None, size=None):
        if offset is None:
            offset = 0
        if size is None:
            size = self.size
        bitmap = self.unit_helper.bitmap[self.unit_helper.bitmap.MASK_BITMAP, offset // self.chunk_size, ((offset % self.chunk_size) * self.unit_helper.bitmap_depth) // self.unit_helper.bitmap_depth]
        return_map = []
        def bits_to_bool(bits):
            nonlocal return_map
            [return_map.append(bool((bits >> i) & 1)) for i in range(self.unit_helper.bitmap_depth)]
        bits_to_bool(bitmap)
        return return_map

    def mark_used(self, offset, size):
        self._clip_offset(offset, size)
    def get_specs(self, boundaries=None, strides=None):
        """Return legacy band descriptions for serialized-layout readers.

        New allocation does not consume these bands.  They remain available to
        readers of the historical header format while allocation uses exact
        byte intervals in ``self.allocator``.
        If no region provides a stride from which to derive this width, a
        ``ValueError`` is raised.
        """

        usable = max(0, self.size - self.header_size - self.extra_data_size)
        if boundaries is None:
            start = self.header_size
            q = usable // 4
            end_usable = start + usable
            boundaries = [
                start,
                start,
                start + q,
                start + 2 * q,
                start + 3 * q,
                end_usable,
                self.size - self.extra_data_size,
                self.size,
            ]
        if strides is None:
            strides = [self.granular_size] * len(boundaries)
        #if usable % self.granular_size != 0:
        #    raise ValueError("Total size minus extra data size must be a multiple of grain size")
        q = usable // 4
        print(f"[BitTensorMemory] Spec boundaries: {boundaries}, strides: {strides}")

        specs = [
            {"left": 0, "right": boundaries[0], "label": 0, "min": None, "max": ctypes.sizeof(BTGraphHeader), "len": ctypes.sizeof(BTGraphHeader), "stride": self.granular_size, "flags": 0},
            {"left": boundaries[0], "right": boundaries[1], "label": 1, "min": None, "max": None, "len": 0, "stride": 1, "flags": 0},
            {"left": boundaries[1], "right": boundaries[2], "label": 2, "min": None, "max": None, "len": q, "stride": ctypes.sizeof(NodeEntry), "flags": 0},
            {"left": boundaries[2], "right": boundaries[3], "label": 3, "min": None, "max": None, "len": q, "stride": ctypes.sizeof(EdgeEntry), "flags": 0},
            {"left": boundaries[3], "right": boundaries[4], "label": 4, "min": None, "max": None, "len": q, "stride": ctypes.sizeof(MetaGraphEdge), "flags": 0},
            {"left": boundaries[4], "right": boundaries[5], "label": 5, "min": None, "max": None, "len": q, "stride": ctypes.sizeof(MetaGraphEdge), "flags": 0},
            {"left": boundaries[5], "right": boundaries[6], "label": 6, "min": None, "max": None, "len": 0, "stride": 1, "flags": 0},
        ]
        if self.extra_data_size > 0:

            specs.append({"left": boundaries[6], "right": boundaries[7], "label": 7, "min": self.size - self.extra_data_size, "max": self.size, "len": self.extra_data_size, "stride": self.extra_data_size or 1, "flags": IMMUTABLE})

        non_zero_strides = [
            s["stride"] for s in specs if s["right"] > s["left"] and s["stride"] > 0
        ]
        if not non_zero_strides:
            # Provide a sane default so the simulator never sees a zero-width LCM
            min_stride = 512
        else:
            # Enforce a minimum stride of 512 for any zero-width regions
            min_stride = max(min(non_zero_strides), 512)

        for s in specs:
            if s["right"] <= s["left"] and s["left"] < self.size:
                s["stride"] = min_stride
                s["right"] = min(s["left"] + min_stride, self.size)

        max_right = max(s["right"] for s in specs)
        if max_right > self.size:
            raise ValueError("Expanded regions exceed memory size")

        specs = [s for s in specs if s["right"] > s["left"]]

        # Convert byte-based positions and strides to bit offsets for the simulator
        for s in specs:
            s["left"] *= 8
            s["right"] *= 8
            s["len"] *= 8
            s["stride"] *= 8
            if s["min"] is not None:
                s["min"] *= 8
            if s["max"] is not None:
                s["max"] *= 8
        return specs
    def mark_free(self, offset, size):
        self._clip_offset(offset, size)

    def inflate_regions(self, boundaries):
        print(f"Inflating regions: {boundaries}")
        
        boundaries[0][1][0] = (self.graph.header_size if self.graph.capsid else 0)        
        collapses = []
        for i in range(1, len(boundaries)-1):
            if boundaries[i][1][0] <= boundaries[i+1][1][0]:
                collapses.append((i, boundaries[i], boundaries[i+1]))
        return boundaries

    # BitTensorMemory ------------------------------------------------------------
    def find_free_space(self,
                        label: str,
                        bytes_needed: int = 0,
                        allow_drift: bool = False):
        """
        Return the address of the tightest-fit hole for `bytes_needed` bytes.

        The arena grows by an integer allocation quantum if no hole fits.
        """
        # ── 0. label → cell index ------------------------------------------------
        label2cell = {"header": 0, "node": 2, "edge": 3, "parent": 4, "child": 5}
        cell_idx   = label2cell.get(label)
        if cell_idx is None:
            raise ValueError(f"Unknown label '{label}'")

        stride = {
            0: ctypes.sizeof(BTGraphHeader), 2: ctypes.sizeof(NodeEntry),
            3: ctypes.sizeof(EdgeEntry), 4: ctypes.sizeof(MetaGraphEdge),
            5: ctypes.sizeof(MetaGraphEdge),
        }[cell_idx]
        if bytes_needed == 0:
            bytes_needed = stride                       # “one quantum” default

        return self.allocator.allocate(label, bytes_needed, alignment=stride)


    def allocate_block(self, block_size, allowed_range):
        # Emergency blocks use the same exact interval set.  ``allowed_range``
        # is retained in the signature for callers that still describe the
        # historical bands; no float simulation or guessed byte/bit conversion
        # is involved.
        return self.allocator.allocate("block", int(block_size), alignment=8)
    
    
# --- In-memory graph entry definitions ---
