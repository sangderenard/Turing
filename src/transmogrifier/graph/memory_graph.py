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

from ..cells.simulator import Simulator
from ..cells.cell_pressure_region_manager import CellPressureRegionManager
from ..cells.cell_consts import Cell

import json

# Mirror LinearCells flag used by older region manager
IMMUTABLE = 1 << 5
class BitTensorMemoryDAGHelper:
    def __init__(self, bit_tensor_memory, chunk_size=8, bit_width=32):
        self.bit_tensor_memory = bit_tensor_memory
        self.chunk_size = chunk_size
        self.bit_width = bit_width
        self.hard_memory_size = sys.getsizeof(bit_tensor_memory.data)
        self.hard_memory = BitTensorMemory(self.hard_memory_size)
        self.lock_manager = None  # placeholder for lock manager
        self.envelope_domain = (0, self.hard_memory_size // self.chunk_size)
        self.envelope_size = self.hard_memory_size // self.chunk_size

    def merge(self, one, theother):
        """
        Merge two procedural and memory concurrency dags for no-lock
        memory moving.
        """
        
        return self

class StructView:

    def make_view(self, raw_ptr: memoryview, Template: type,
                  *, data_key_guess=("node_data","edge_data")):
        """
        Return a live Mapping over *raw_ptr* cast as *Template*.
        Works for NodeEntry, EdgeEntry, MetaGraphEdge, or any future template.

            >>> off = graph.n_start
            >>> raw = graph.hard_memory.view(off, ctypes.sizeof(NodeEntry))
            >>> n   = units.make_view(raw, NodeEntry)
            >>> n['bit_width'] = 16          # writes straight into the buffer
        """
        field_names = [n for n,_ in Template._fields_]
        data_field  = next((k for k in data_key_guess if k in field_names), None)
        T_size      = ctypes.sizeof(Template)

        # cast the slice as a live ctypes object (zero-copy)
        entry = Template.from_buffer(raw_ptr)

        # ---------- dynamic proxy class (one per call, cheap) ----------
        # Using a closure keeps a reference to *entry* and *raw_ptr*
        class _Proxy(abc.MutableMapping):
            __slots__ = ("_e","_kv","_blob_mv")
            def __init__(self):                      # bind outer-scope vars
                self._e   = entry
                self._kv  = None                     # lazy-parse dict
                if data_field:
                    start = getattr(Template, data_field).offset
                    length= getattr(Template, data_field).size
                    self._blob_mv = raw_ptr[start:start+length]
                else:
                    self._blob_mv = None

            # ---- helper: parse / flush --------------------------------
            def _ensure_cache(self):
                if self._kv is not None or self._blob_mv is None:
                    return
                raw = bytes(self._blob_mv).rstrip(b"\0")
                if not raw:
                    self._kv = {}
                else:
                    try: self._kv = json.loads(raw.decode())
                    except Exception:
                        kv={}
                        for tok in raw.decode().split(";"):
                            if "=" in tok:
                                k,v=tok.split("=",1)
                                kv[k]=v
                        self._kv=kv

            def _flush(self):
                if self._kv is None or self._blob_mv is None:
                    return
                blob = json.dumps(self._kv, separators=(",",":")).encode()
                blob = blob[:len(self._blob_mv)].ljust(len(self._blob_mv), b"\0")
                self._blob_mv[:] = blob     # in-place write

            # ---- mapping interface -----------------------------------
            def __getitem__(self, k):
                if hasattr(self._e, k):        return getattr(self._e, k)
                self._ensure_cache();          return self._kv[k]

            def __setitem__(self, k, v):
                if hasattr(self._e, k):        setattr(self._e, k, v)
                else:                          self._ensure_cache(); self._kv[k]=v; self._flush()

            def __delitem__(self, k):
                if hasattr(self._e, k):        raise TypeError("cannot delete fixed field")
                self._ensure_cache();          del self._kv[k]; self._flush()

            def __iter__(self):
                yield from field_names
                if data_field:
                    self._ensure_cache();  yield from self._kv

            def __len__(self):
                self._ensure_cache();      return len(field_names)+len(self._kv or {})

            # optional convenience
            def __repr__(self):
                d={k:self[k] for k in self};   return f"<{Template.__name__}View {d}>"

        return _Proxy()


class BitTensorMemory: #sizes in bytes
    ALLOCATION_FAILURE = -1
    DEFAULT_BLOCK = 4096  # default block size for memory allocation
    DEFAULT_GRAIN = 512  # default granular size for bitmap allocation

    def __init__(self, size, graph=None, dynamic=True):

        self.size = size
        self.chunk_size = BitTensorMemory.DEFAULT_BLOCK
        self.granular_size = BitTensorMemory.DEFAULT_GRAIN
        self.dynamic = dynamic


        self.header_size = ctypes.sizeof(BTGraphHeader) if graph else 0
        self.l_start = graph.l_start if graph else 0
        self.n_start = graph.n_start if graph else 0
        self.e_start = graph.e_start if graph else 0
        self.p_start = graph.p_start if graph else 0
        self.c_start = graph.c_start if graph else 0
        self.r_start = graph.r_start if graph else 0
        self.x_start = graph.x_start if graph else 0
        self.envelope_domain = (0, self.x_start)
        self.envelope_size = self.size - self.header_size


        self.data = ctypes.create_string_buffer(size if size > BitTensorMemory.DEFAULT_BLOCK else BitTensorMemory.DEFAULT_BLOCK)
        self.unit_helper = BitTensorMemoryUnits(0, self.size, self.granular_size, self.chunk_size, node_template = NodeEntry, edge_template= EdgeEntry, association_template=MetaGraphEdge, hard_memory = self)
        self.graph = graph
        

        self.active_regions = []

        self.extra_data_size = 0
        self.set_cache = set()

        specs = self.get_specs()
        cells = [
            Cell(stride=s["stride"], left=s["left"], right=s["right"],
                 len=s["len"], label=s["label"])
            for s in specs
        ]
        self.region_simulator = Simulator(cells)
        self.region_manager = CellPressureRegionManager(
            self.region_simulator.bitbuffer, cells
        )
        # Initialise the hydraulic model upfront so equilibrium fractions and
        # buffer expansion are resolved before any writes occur.
        self.region_simulator.run_saline_sim()

        
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
        offest = self.unit_helper.bitmap._clip(offset + size) - size
        size = self.unit_helper.bitmap._clip(size)
        return offset, size

    def reset_region_manager(self, boundaries=None, strides=None, insertion=False):
        # Recreate region manager with new specs
        specs = self.get_specs(boundaries, strides)
        cells = [
            Cell(stride=s["stride"], left=s["left"], right=s["right"],
                 len=s["len"], label=s["label"])
            for s in specs
        ]
        self.region_simulator = Simulator(cells)
        self.region_manager = CellPressureRegionManager(
            self.region_simulator.bitbuffer, cells
        )
        # Ensure hydraulic model is primed on reset as well
        self.region_simulator.run_saline_sim()
        return self.region_manager.cells


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
        
        self.unit_helper.delta(offset, size, "write", string=data[:size])
        if clear_delta_map:
            self.unit_helper.delta(offset, size, "read", string=data[:size])

    def free(self, offset, size, scramble=False):
        offset, size = self._clip_offset(offset, size)
        self.mark_free(offset, size)

        data = b'\x00' * size if not scramble else random.randbytes(size)
        self.unit_helper.delta(offset, size, "free", string=data)

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
        # mark used bits and increment delta
        offset, size = self._clip_offset(offset, size)
        self.unit_helper.delta(offset, size, "alloc")
    def get_specs(self, boundaries=None, strides=None):
        """Return region specifications for the underlying simulator.

        Zero-width regions are expanded to a minimal non-zero stride so that
        the pressure simulator always receives valid ``left < right`` domains.
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
        return specs
    def mark_free(self, offset, size):
        # mark free bits and reset delta
        offset, size = self._clip_offset(offset, size)
        self.unit_helper.delta(offset, size, "free")

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

        Falls back to enqueueing a zeroed quantum via the pressure simulator
        until the hole exists.
        Raises RuntimeError if nothing can be made to fit (immutable cells, etc.).
        """
        # ── 0. label → cell index ------------------------------------------------
        label2cell = {"header": 0, "node": 2, "edge": 3, "parent": 4, "child": 5}
        cell_idx   = label2cell.get(label)
        if cell_idx is None:
            raise ValueError(f"Unknown label '{label}'")

        # convenience
        stride = self.region_manager.cells[cell_idx].stride
        if bytes_needed == 0:
            bytes_needed = stride                       # “one quantum” default

        while True:
            # ── 1. current layout snapshot --------------------------------------
            snap = CellPressureRegionManager.dump_cells(self.region_manager)
            free  = snap["free_spaces"]                 # [(label, addr, size), …]

            # collect holes that fit
            candidates = [(addr, size) for lbl, addr, size in free
                        if size >= bytes_needed and
                            (lbl == cell_idx or allow_drift)]

            # ── 2. choose the best hole, if any ---------------------------------
            if candidates:
                addr, size = min(candidates, key=lambda p: (p[1], p[0]))  # smallest first
                # NOTE: nothing is written yet; write()/mark_used() will record it
                return addr

            # ── 3. no hole ⇒ enqueue one empty stride to force expansion -----
            cell = self.region_manager.cells[cell_idx]
            stride_bits = cell.stride * self.region_simulator.bitbuffer.bitsforbits
            payload = b"\0" * ((stride_bits + 7) // 8)
            self.region_simulator.write_data(cell.label, payload)
            self.region_simulator.run_saline_sim()
            # discard the placeholder write used to trigger expansion
            self.region_simulator.input_queues.pop(cell.label, None)
            cell.injection_queue = 0
            cell.salinity = 0
            # loop retries with the now-larger cell


    def allocate_block(self, block_size, allowed_range):
        # allowed_range = (min_offset, max_offset)
        start_chunk = allowed_range[0] // self.chunk_size
        end_chunk = allowed_range[1] // self.chunk_size

        # Sort candidate chunks by density descending
        candidates = sorted(range(start_chunk, end_chunk),
                            key=lambda i: self.density[i],
                            reverse=True)

        candidates = [self.unit_helper.bytes_for_chunks(c) for c in candidates]
        for candidate in candidates:

            offset = self.find_free_space(candidate, block_size)
            if offset is not None:
                self.mark_used(offset, block_size)
                return offset



        # Fallback: expand or trigger bifurcation
        return BitTensorMemory.ALLOCATION_FAILURE
    
    
# --- In-memory graph entry definitions ---

class NodeEntry(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        # ─── Line 0 (offset 0) ───
        ('node_id', ctypes.c_uint64),          # 0x00
        ('node_type', ctypes.c_uint8),         # 0x08
        ('node_flags', ctypes.c_uint16),       # 0x09
        ('node_depth', ctypes.c_uint8),        # 0x0B
        ('encoding', ctypes.c_uint8),          # 0x0C
        ('bit_width', ctypes.c_uint8),         # 0x0D
        ('checksuma', ctypes.c_uint16),        # 0x0E (alignment pad)
        ('pad_0', ctypes.c_byte * (32)),


        # ─── Line 64 (offset 0x80) ───
        ('handler_ref', ctypes.c_uint64),      # 0x80
        ('input_schema', ctypes.c_uint16),     # 0x88
        ('output_schema', ctypes.c_uint16),    # 0x8A
        ('flags', ctypes.c_uint16),            # 0x8C
        ('checksumb', ctypes.c_uint16),        # 0x8E
        ('params_ref', ctypes.c_uint64),       # 0x90
        ('return_ref', ctypes.c_uint64),       # 0x98
        ('caller_ref', ctypes.c_uint64),       # 0xA0
        ('resume_ref', ctypes.c_uint64),       # 0xA8
        ('pad_1', ctypes.c_byte * (32)),

        ('pad_2', ctypes.c_byte * (128)),  # align to 128 bytes
        # ─── Line 256 (offset 0x100) ───
        
        ('node_data', ctypes.c_char * 256),    # 0x140

        
    ]

    def __init__(self, node_id=0, node_data=None, **kwargs):
        """Create a new ``NodeEntry``.

        Parameters
        ----------
        node_id:
            Numeric identifier for the node.  The previous implementation
            unconditionally reset this field to ``0`` after construction,
            which meant any ID supplied by callers was lost.  That in turn
            caused lookups to fail because every node appeared to have an ID
            of zero.
        node_data:
            Optional blob to copy into the ``node_data`` field.  Only the
            first 256 bytes are retained.
        **kwargs:
            Additional structure field overrides.
        """
        # Initialise the base structure with any recognised fields from
        # ``kwargs``.  Unknown keys are ignored to preserve ``ctypes``
        # behaviour, but we still set ``node_id`` explicitly afterwards so it
        # is never overwritten.
        super().__init__(**{k: v for k, v in kwargs.items() if k in
                            {f[0] for f in self._fields_}})

        self.node_id = node_id

        if node_data is not None:
            if isinstance(node_data, str):
                data = node_data.encode("utf-8")
            elif isinstance(node_data, (bytes, bytearray)):
                data = bytes(node_data)
            else:
                data = str(node_data).encode("utf-8")

            # Copy at most 256 bytes into the fixed-size ``node_data`` field.
            data = data[:256]
            dest = ctypes.addressof(self) + self.__class__.node_data.offset
            ctypes.memset(dest, 0, 256)
            ctypes.memmove(dest, data, len(data))


    def __getattr__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        node_data = self.node_data
        if isinstance(node_data, ctypes.Array):
            # the data as a dict would've been added
            # by using str() so we need to parse that
            # in reverse from char_8 data
            node_data = node_data.decode('utf-8')
            try:
                node_data = json.loads(node_data)
            except json.JSONDecodeError:
                pass

        
            
        elif hasattr(node_data, key):
            return node_data[key]
        else:
            raise AttributeError(f"Attribute {key} not found in NodeEntry")
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):    
        if hasattr(self, key):
            setattr(self, key, value)

        node_data = self.node_data.decode('utf-8')
        try:
            node_data = json.loads(node_data)
        except json.JSONDecodeError:
            pass

        if hasattr(self.node_data, key):
            self.node_data[key] = value
        else:
            raise KeyError(f"Key {key} not found in NodeEntry")
    
#    def __delitem__(self, key):
#        del self._data[key]

#    def __iter__(self):
#        return iter(self._data)

#    def __len__(self):
#        return len(self._data)
    
    def __contains__(self, key):
        try:
            success = self.__getattr__(key)
        except AttributeError:
            return False
        return True

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
class SetMicrograinEntry(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('object_id', ctypes.c_uint32),  # size of the grain in bytes
        ('object_address', ctypes.c_uint64),  # address of the grain in memory
    ]


class EdgeEntry(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('src_ptr', ctypes.c_uint64),      # memory offset or absolute ptr
        ('dst_ptr', ctypes.c_uint64),      # idem
        ('src_graph_id', ctypes.c_uint64), # pointer or UUID
        ('dst_graph_id', ctypes.c_uint64), # idem
        ('data_type', ctypes.c_uint16),    # semantic type of transmission
        ('edge_flags', ctypes.c_uint16),   # async, inline, compressed
        ('timestamp', ctypes.c_uint64),    # for causal graphs / DAG sorting
        ('alignment', ctypes.c_uint16),    # alignment mask or slot
        ('checksuma', ctypes.c_uint16),
        ('checksumb', ctypes.c_uint16),    # checksum for integrity
        ('_pad', ctypes.c_byte * (128 - 50))  # align to 128
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.src_ptr = 0
        self.dst_ptr = 0
        self.src_graph_id = args[0] if args else 0
        self.dst_graph_id = args[1] if len(args) > 1 else 0
        self.data_type = 0
        self.edge_flags = 0
        self.timestamp = 0
        self.alignment = 0
        self.checksuma = 0
        self.checksumb = 0
    
META_GRAPH_TRANSFER_BUFFER_SIZE = 60  # 60 uint64 slots for transfer buffer
class MetaGraphEdge(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        # ─── Core capsule linkage ───
        ('local_capsid_ref', ctypes.c_uint64),     # local graph pointer
        ('linked_capsid_ref', ctypes.c_uint64),    # linked (parent or child)
        ('capsid_id', ctypes.c_uint32),            # unique identifier
        # ─── Routing and flow metadata ───
        ('permeability_weight', ctypes.c_uint32),   # ease of transfer
        ('pressure', ctypes.c_uint16),              # flow pressure
        ('flags', ctypes.c_uint16),             # routing flags
        ('checksuma', ctypes.c_uint16),          # checksum for integrity
        ('checksumb', ctypes.c_uint16),          # additional checksum

        

        # ─── Preallocated transfer buffer ───
        ('transfer_buffer', ctypes.c_uint64 * META_GRAPH_TRANSFER_BUFFER_SIZE),   # actual data being passed
    ]
# Avoid noisy output during import; retain statement for manual debugging only.
# print(ctypes.sizeof(NodeEntry), ctypes.sizeof(EdgeEntry), ctypes.sizeof(MetaGraphEdge))
assert ctypes.sizeof(MetaGraphEdge) == 512, "MetaGraphEdge must be exactly 512 bytes"
class BTGraphHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        
        ("capsid_id",      ctypes.c_uint32),
        ("chunk_size",     ctypes.c_uint32),
        ("bit_width",      ctypes.c_uint32),
        ("header_size",    ctypes.c_uint8),
        ("encoding",       ctypes.c_uint8),
        ("capsid",         ctypes.c_uint8),  # 0 = no capsid, 1 = encapsulated
        ("dynamic",        ctypes.c_uint8),  # 0 = static, 1 = dynamic
        
        ("p_rational",     ctypes.c_uint8),
        ("c_rational",     ctypes.c_uint8),
        ("p_start",        ctypes.c_uint64),
        ("c_start",        ctypes.c_uint64),
        
        ("n_rational",     ctypes.c_uint8),
        ("e_rational",     ctypes.c_uint8),
        ("n_start",        ctypes.c_uint64),
        ("e_start",        ctypes.c_uint64),
        
        ("node_count",     ctypes.c_uint16),
        ("edge_count",     ctypes.c_uint16),
        ("parent_count",   ctypes.c_uint16),
        ("child_count",    ctypes.c_uint16),
        
        ("meta_graph_root",ctypes.c_uint64),
        ("generative_parent", ctypes.c_uint64),
        ("emergency_reference", ctypes.c_uint64),  # fallback reference for emergency allocation
        ("8_4_pad",       ctypes.c_uint8 * 4),  # reserved for future use
        ("32_4_pad",     ctypes.c_uint32 * 4),
        ("64_2_pad",     ctypes.c_uint64 * 3)
    ]
# print("BTGraphHeader size:", ctypes.sizeof(BTGraphHeader))
assert ctypes.sizeof(BTGraphHeader) == 128, "BTGraphHeader must be exactly 128 bytes"


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
        bt   = self.bit_tensor_memory_graph
        offs = bt.find_in_span((bt.n_start, bt.e_start),
                               ctypes.sizeof(NodeEntry))
        if offs == BitTensorMemoryGraph.NOTHING_TO_FLY:
            return
        for off in offs:
            raw   = bt.hard_memory.read(off, ctypes.sizeof(NodeEntry))
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
        bt   = self.bit_tensor_memory_graph
        offs = bt.find_in_span((bt.e_start, bt.p_start),
                               ctypes.sizeof(EdgeEntry))
        if offs == BitTensorMemoryGraph.NOTHING_TO_FLY:
            return []
        out = []
        for off in offs:
            e = EdgeEntry.from_buffer_copy(
                    bt.hard_memory.read(off, ctypes.sizeof(EdgeEntry)))
            out.append((e.src_ptr, e.dst_ptr))
        return out
# in python we can't create something permanent inside a function
# so we need a container for all active meta nodes
import ctypes, zlib
import hashlib
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

class BitTensorMemoryGraph:
    LINE_L = 5
    LINE_R = 6
    LINE_N = 1
    LINE_E = 2
    LINE_P = 3
    LINE_C = 4
    NOTHING_TO_FLY = -1
    OPEN_ALLOCATION = -2

    # Helper proxy for chained access: Graph[nid1][nid2] = ...
    class _NodeOrEdgeProxy:
        def __init__(self, graph, node_id):
            self._graph = graph
            self._node_id = node_id
        def __getitem__(self, other):
            # Graph[nid1][nid2] returns edge (nid1, nid2)
            return self._graph[(self._node_id, other)]
        def __setitem__(self, other, value):
            # Graph[nid1][nid2] = value sets edge (nid1, nid2)
            self._graph[(self._node_id, other)] = value
        def __call__(self):
            # Graph[nid1]() returns the node itself
            node_offsets = self._graph.find_in_span((self._graph.n_start, self._graph.c_start), ctypes.sizeof(NodeEntry))
            if node_offsets == BitTensorMemoryGraph.NOTHING_TO_FLY:
                return None
            for off in node_offsets:
                raw = self._graph.hard_memory.read(off, ctypes.sizeof(NodeEntry))
                node = NodeEntry.from_buffer_copy(raw)
                if node.node_id == self._node_id:
                    return node
            return None


    """
    A class representing a metamemory graph for bit tensors.
    """
    def __init__(self, size=0, bit_width=32, encoding="gray", meta_graph_root=0, generative_parent=0):
        self.capsid_id = ( 1 + id(generative_parent) + uuid4().int) % 2**32
        self.chunk_size = 8
        self.hard_memory_size = ctypes.sizeof(BTGraphHeader)
        self.header_size = self.hard_memory_size

        # Provide a minimal payload when no explicit size is requested so
        # downstream region specifications always have space to expand.
        if size <= 0:
            size = 512

        # Envelope spans the bytes after the header.  Ensure it always
        # represents a valid non-negative range even when ``size`` is 0.
        self.hard_memory_size += size
        min_payload = (
            ctypes.sizeof(NodeEntry)
            + ctypes.sizeof(EdgeEntry)
            + ctypes.sizeof(MetaGraphEdge)
        )
        if self.hard_memory_size < self.header_size + min_payload:
            self.hard_memory_size = self.header_size + min_payload

        if self.hard_memory_size % self.chunk_size != 0:
            self.hard_memory_size += self.chunk_size - (
                self.hard_memory_size % self.chunk_size
            )

        self.envelope_domain = (self.header_size, self.hard_memory_size)
        self.envelope_size = self.envelope_domain[1] - self.envelope_domain[0]
        self.envelope_config = {"type": "greedy"}
        self.l_start = self.header_size
        self.r_start = self.envelope_domain[1]
        self.x_start = self.envelope_domain[1]
        self.n_rational = 1
        self.e_rational = 1
        self.p_rational = 1
        self.c_rational = 1
        total_ratio_sum = self.n_rational + self.e_rational + self.p_rational + self.c_rational
        quantum = self.envelope_size // total_ratio_sum if total_ratio_sum else 0
        self.n_start = self.envelope_domain[0]
        self.e_start = self.n_start + quantum * self.n_rational
        self.p_start = self.e_start + quantum * self.e_rational
        self.c_start = self.p_start + quantum * self.p_rational

        self.hard_memory = BitTensorMemory(self.hard_memory_size, self)  # default memory size
        self.hard_memory.region_manager.register_object_maps()
        self.meta_graph_root = meta_graph_root  # root of the meta graph
        self.generative_parent = generative_parent
        self.lock_manager = None  # placeholder for lock manager

        
        self.dynamic = False
        self.emergency_reference = BitTensorMemory.ALLOCATION_FAILURE
        


        self.node_count = 0
        self.edge_count = 0
        self.parent_count = 0
        self.child_count = 0

        self.bit_width = bit_width
        self.encoding = encoding
        self.capsid = True
        
        
        self.G = NetworkxEmulation(self)
        self.concurrency_dag = None

        self.struct_viewer = StructView()

        self.region_layout = self.compute_region_boundaries()   
        self.encapsidate_capsid()
    # -- Nodes ----------------------------------------------------------
    def _node_offset(self, node_id) -> int|None:
        """Return byte-offset of the NodeEntry whose node_id matches, else None."""
        offs = self.find_in_span((self.n_start, self.c_start), ctypes.sizeof(NodeEntry))
        if offs == self.NOTHING_TO_FLY:
            return None
        for off in offs:
            raw = self.hard_memory.read(off, ctypes.sizeof(NodeEntry))
            if NodeEntry.from_buffer_copy(raw).node_id == node_id:
                return off
        return None

    def _node_view(self, off):
        raw = self.hard_memory.view(off, ctypes.sizeof(NodeEntry))
        return self.struct_view.make_view(raw, NodeEntry)

    # -- Edges (normal + meta) -----------------------------------------
    def _edge_view(self, off):
        raw = self.hard_memory.view(off, ctypes.sizeof(EdgeEntry))
        return self.struct_view.make_view(raw, EdgeEntry)

    def _meta_edge_view(self, off):
        raw = self.hard_memory.view(off, ctypes.sizeof(MetaGraphEdge))
        return self.struct_view.make_view(raw, MetaGraphEdge)


    def __setitem__(self, key, value):
        """
        Graph[key] = value
        - If key is a node id, update or insert node data.
        - If key is a tuple (nid1, nid2), update or insert edge data between nid1 and nid2.
        """
        def set_struct_vals(dict_obj, struct_obj):
            """
            Set values from dict_obj to struct_obj.
            """
            for k, v in dict_obj.items():
                if hasattr(struct_obj, k):
                    setattr(struct_obj, k, v)
                else:
                    raise KeyError(f"Key {k} not found in {struct_obj.__class__.__name__}")
            return struct_obj

        inner_assignment = False
        if not isinstance(value, (NodeEntry, EdgeEntry, MetaGraphEdge)):
            inner_assignment = True

        if isinstance(key, tuple) and len(key) == 2:
            src, dst = key
            # Find if edge exists
            edge_offs = self.find_in_span((self.e_start, self.p_start), ctypes.sizeof(EdgeEntry))
            if edge_offs != BitTensorMemoryGraph.NOTHING_TO_FLY:
                for off in edge_offs:
                    raw = self.hard_memory.read(off, ctypes.sizeof(EdgeEntry))
                    edge = EdgeEntry.from_buffer_copy(raw)
                    if edge.src_ptr == src and edge.dst_ptr == dst:
                        if inner_assignment:
                            edge = set_struct_vals(value, edge)
                        else:
                            edge = value
                        patched = GraphSearch._build_struct_bytes(edge)
                        self.hard_memory.write(off, patched)
                        return
            # Edge not found, add new
            self.add_edge(src, dst, **(value if isinstance(value, dict) else {}))
            return
        # Otherwise, treat as node
        NODE_SIZE = ctypes.sizeof(NodeEntry)
        def _encode_node_data(obj) -> bytes:
            if isinstance(obj, NodeEntry):
                buf = bytes(obj.node_data)
            elif isinstance(obj, bytes):
                buf = obj
            elif isinstance(obj, str):
                buf = obj.encode("utf-8")
            else:
                buf = json.dumps(obj).encode("utf-8")
            return (buf[:256]).ljust(256, b"\x00")
        node_offsets = self.find_in_span((self.n_start, self.c_start), NODE_SIZE)
        if node_offsets == BitTensorMemoryGraph.NOTHING_TO_FLY:
            self.add_node(node_id=key, node_data=value)
            return
        for off in node_offsets:
            raw = self.hard_memory.read(off, NODE_SIZE)
            node = NodeEntry.from_buffer_copy(raw)
            if node.node_id == key:
                if inner_assignment:
                    node = set_struct_vals(value, node)
                else:
                    node.node_data = _encode_node_data(value)
                patched = GraphSearch._build_struct_bytes(node)
                self.hard_memory.write(off, patched)
                return
        self.add_node(node_id=key, node_data=value)

    def __repr__(self):
        self.encapsidate_capsid()
        byte_output = self.hard_memory.read(0, self.hard_memory_size)
        return f"Graph({byte_output})"

    def __getitem__(self, key):
        print(f"Getting item: {key}")
        # ---- (a) slice of nodes  -------------------------------------
        if isinstance(key, slice):
            start, stop, step = key.indices(self.node_count)
            views = []
            for idx in range(start, stop, step):
                off = self.n_start + idx * ctypes.sizeof(NodeEntry)
                views.append(self._node_view(off))
            return views

        # ---- (b) single node by ID  ----------------------------------
        if isinstance(key, int):
            off = self._node_offset(key)
            if off is None:
                raise KeyError(f"node_id {key} not found")
            return self._node_view(off)                 # <- live mapping proxy

        # ---- (c) (src, dst) edge lookup ------------------------------
        if isinstance(key, tuple) and len(key) == 2:
            src, dst = key
            edge_offs = self.find_in_span((self.e_start, self.p_start), ctypes.sizeof(EdgeEntry))
            if edge_offs != self.NOTHING_TO_FLY:
                for off in edge_offs:
                    raw = self.hard_memory.read(off, ctypes.sizeof(EdgeEntry))
                    e   = EdgeEntry.from_buffer_copy(raw)
                    if e.src_ptr == src and e.dst_ptr == dst:
                        return self._edge_view(off)
            raise KeyError(f"edge {src}->{dst} not found")

        # ---- (d) meta-edge (True, srcCapsid, dstCapsid) --------------
        if isinstance(key, tuple) and len(key) == 3 and key[0] is True:
            _, src, dst = key
            offs = self.find_in_span((self.p_start, self.envelope_domain[1]),
                                    ctypes.sizeof(MetaGraphEdge))
            if offs != self.NOTHING_TO_FLY:
                for off in offs:
                    raw = self.hard_memory.read(off, ctypes.sizeof(MetaGraphEdge))
                    m   = MetaGraphEdge.from_buffer_copy(raw)
                    if m.local_capsid_ref == src and m.linked_capsid_ref == dst:
                        return self._meta_edge_view(off)
            raise KeyError(f"meta edge {src}->{dst} not found")

        # ---- (e) chained syntax:  G[nid1][nid2]  ----------------------
        return self._NodeOrEdgeProxy(self, key)




    def add_edge(self, src, dst, **kwargs):
        edge = EdgeEntry(src, dst, **kwargs)
        free_space = self.hard_memory.find_free_space("edge", ctypes.sizeof(EdgeEntry))

        if free_space is not None:
            self.hard_memory.write(free_space, ctypes.string_at(ctypes.addressof(edge), ctypes.sizeof(edge)))
        self.edge_count += 1

        return (src, dst)

    def add_meta_parent(self, parent_id):
        # THESE ARE NOT THE SAME AS THE NODES AND EDGES IN THE GRAPH
        # THESE ARE META GRAPH CONNECTIONS FOR THE GRAPH OF GRAPH
        # THEY ONLY EXIST AS NO-SIBLING TABLE RELATIONSHIPS
        """
        Add a meta parent node to the graph.
        """
        space = self.hard_memory.find_free_space("parent")
        
        if space is None:
            self.emergency_reference = self.hard_memory.allocate_block(ctypes.sizeof(MetaGraphEdge), (self.LINE_P, self.LINE_C))
            if self.emergency_reference == BitTensorMemory.ALLOCATION_FAILURE:
                raise MemoryError("Failed to allocate emergency reference for parent node")
        edge = MetaGraphEdge()
        edge.local_capsid_ref = self.capsid_id
        edge.remote_capsid_ref = parent_id
        
        # THESE ARE NOT THE SAME AS THE NODES AND EDGES IN THE GRAPH
        # THESE ARE META GRAPH CONNECTIONS FOR THE GRAPH OF GRAPH
        # THEY ONLY EXIST AS NO-SIBLING TABLE RELATIONSHIPS


        self.parent_count += 1
        self.hard_memory.write(space, ctypes.string_at(ctypes.addressof(edge), ctypes.sizeof(edge)))

        # THESE ARE NOT THE SAME AS THE NODES AND EDGES IN THE GRAPH
        # THESE ARE META GRAPH CONNECTIONS FOR THE GRAPH OF GRAPH
        # THEY ONLY EXIST AS NO-SIBLING TABLE RELATIONSHIPS
        return

    def serialize_header(self):
        """
        Serialize the header data into a byte string.
        """
        header = BTGraphHeader()
        # core layout
        header.chunk_size = self.chunk_size
        header.header_size = self.header_size
        header.capsid_id = self.capsid
        # rationals and starts
        header.n_rational = self.n_rational
        header.e_rational = self.e_rational
        header.p_rational = self.p_rational
        header.c_rational = self.c_rational
        header.n_start = self.n_start
        header.e_start = self.e_start
        header.p_start = self.p_start
        header.c_start = self.c_start
        # counts
        header.node_count = self.node_count
        header.edge_count = self.edge_count
        header.parent_count = self.parent_count
        header.child_count = self.child_count
        # bit and encoding
        header.bit_width = self.bit_width
        header.encoding = 0 if self.encoding == 'gray' else 1
        # flags
        header.dynamic = int(self.dynamic)
        header.capsid = int(self.capsid)

        # emergency reference
        header.emergency_reference = self.emergency_reference
        header.meta_graph_root = self.meta_graph_root
        header.generative_parent = self.generative_parent
        # return packed header
        return ctypes.string_at(ctypes.addressof(header), ctypes.sizeof(header))
    
    def deserialize_header(self):
        """
        Read header bytes back from hard memory and populate fields.
        """
        size = ctypes.sizeof(BTGraphHeader)
        data = self.hard_memory.read(0, size)
        header = BTGraphHeader.from_buffer_copy(data)
        # core
        self.chunk_size = header.chunk_size
        self.header_size = header.header_size
        
        # rationals and starts
        self.n_rational = header.n_rational
        self.e_rational = header.e_rational
        self.p_rational = header.p_rational
        self.c_rational = header.c_rational
        self.n_start = header.n_start
        self.e_start = header.e_start
        self.p_start = header.p_start
        self.c_start = header.c_start
        # counts
        self.node_count = header.node_count
        self.edge_count = header.edge_count
        self.parent_count = header.parent_count
        self.child_count = header.child_count
        # bit and encoding
        self.bit_width = header.bit_width
        self.encoding = 'gray' if header.encoding == 0 else 'binary'
        # flags
        self.dynamic = bool(header.dynamic)
        self.capsid = bool(header.capsid)

        return header


    def decondensate_capsid(self, burn=False):
        """
        Decondensate the capsid of the header data.
        This is a placeholder for future implementation.
        """

        if not self.capsid:
            return
        if burn:
            # if burn is True, we remove the capsid reference
            self.hard_memory.free(0, self.header_size)
            self.capsid = False
            return

        header = self.hard_memory.read(0, self.header_size)

        header_node = self.add_node(NodeEntry.from_buffer_copy(header))

        return header_node

    def encapsidate_capsid(self):
        """
        Encapsidate the capsid of the header data.
        This is a placeholder for future implementation.
        """
        #this method exists to put the object's own header data
        # the instance of the object itself inside the first bytes
        # of the object, as it is when instantiated, in preparation
        # for sporulation or bifurcation.

        print(f"Debugging: Encapsidating capsid with size {self.hard_memory_size} and header size {self.header_size}.")

        dirty = False
        if self.n_start >= self.header_size:     
            print(f"Debugging: n_start {self.n_start} is greater than header size {self.header_size}.")      
            self.hard_memory.write(0, self.serialize_header())
        else:
            print(f"Debugging: n_start {self.n_start} is less than header size {self.header_size}. Checking for dirty memory.")
            bitmask = self.hard_memory.bitmap_expanded()
            for i in range((self.header_size + self.hard_memory.granular_size - 1) // self.hard_memory.granular_size):
                if i in bitmask:
                    dirty = True
                    break
            if not dirty:
                print(f"Debugging: No dirty memory found, writing header at start.")
                self.hard_memory.write(0, self.serialize_header())
                self.start_n = ((self.header_size + self.hard_memory.granular_size - 1) // self.hard_memory.granular_size) * self.hard_memory.granular_size
            else:
                print(f"Debugging: Dirty memory found, reallocating hard memory.")
                free_space = self.hard_memory.find_free_space("header")
                print(f"Debugging: Free space for header serialization: {free_space}")
                if free_space is None:
                    assert False, "currently unhandled lack of memory for header"
                print(f"Debugging: Found free space at {free_space} for header serialization.")
                self.hard_memory.write(free_space, self.serialize_header())
                self.n_start = free_space + self.header_size

                # add this later, sweep the space before for entries, use heuristic
                # building in graph search for an easy way out, in fact, do do that.
                dummy_graph = BitTensorMemoryGraph(self.hard_memory.size - self.header_size, bit_width=self.bit_width, encoding=self.encoding)
                graph_search_helper = GraphSearch(self, self, dummy_graph)
                graph_search_helper.heuristic_memory_build(self.hard_memory, embed_capcid=self.serialize_header())

                self.hard_memory.free(0, self.hard_memory_size)
                self.hard_memory = dummy_graph.hard_memory



            #this recursion here needs to be controlled
            
            self.hard_memory.write(0, self.serialize_header())
        pass

    def confirm_layout(self, layout):
        self.region_layout = layout
        self.envelope_domain = (layout[0][1][0], layout[-1][1][0])
        print(f"Debugging: Confirmed layout with envelope domain {self.envelope_domain} and size {self.envelope_size}.")
        self.envelope_size = self.envelope_domain[-1] - self.envelope_domain[0]
        self.n_start = layout[1]
        self.e_start = layout[2]
        self.p_start = layout[3]
        self.c_start = layout[4]
        
        self.n_rational = self.n_start // self.envelope_size
        self.e_rational = self.e_start // self.envelope_size
        self.p_rational = self.p_start // self.envelope_size
        self.c_rational = self.c_start // self.envelope_size

    def sporulate(self):
        """
        Sporulate the memory graph, creating a new instance
        with the same properties but an "inactive"
        compressed ctype payload requiring decompression.
        """

        # This method is for defragmanting the memory graph
        # and compressing the data into a long term storage
        # or transmission format, after ensuring the capsid
        # has been encapsidated.
        # This represents a minimum footprint of the memory graph
        # without actually inhibiting its functionality.

        
        header = self.encapsidate_capsid()
        import zlib
        compressed_body = zlib.compress(self.hard_memory.data.raw[ctypes.sizeof(BTGraphHeader):self.hard_memory_size])
        self.hard_memory.free(0, self.hard_memory_size)
        self.hard_memory = BitTensorMemory(self.header_size + len(compressed_body), self)
        self.hard_memory.write(0, ctypes.string_at(ctypes.addressof(header), ctypes.sizeof(BTGraphHeader)))
        self.hard_memory.write(self.header_size, compressed_body)
        return compressed_body, header

    def add_child(self, free_space=None, child_id=None, meta_graph_node=None, byref=False):
        global meta_nodes
        if free_space is None:
            free_space = self.hard_memory.find_free_space("child")
        if not byref:
            new_child_entry = MetaGraphEdge()
        else:
            new_child_entry = free_space
        new_child = None    
        new_child_entry.local_capsid_ref = self.capsid_id
        if child_id is not None:
            new_child_entry.remote_capsid_ref = child_id
        if meta_graph_node is not None:
            new_child = meta_graph_node
        else:
            new_child = BitTensorMemoryGraph(self.hard_memory.size - self.header_size, bit_width=self.bit_width, encoding=self.encoding)
        new_child_entry.linked_capsid_ref = new_child.capsid_id
        new_child_entry.permeability_weight = 255
        new_child_entry.pressure = 255
        new_child_entry.queue_space = 0
        new_child_entry.flags = 0
        

        new_child.generative_parent = self.capsid_id
        if self.parent_count == 0:
            new_child.meta_graph_root = self.capsid_id

        if not isinstance(free_space, MetaGraphEdge):
            status = self.hard_memory.write(free_space, ctypes.string_at(ctypes.addressof(new_child_entry), ctypes.sizeof(MetaGraphEdge)))
            if status == BitTensorMemory.ALLOCATION_FAILURE:
                raise MemoryError("Failed to create new child: no free space available")
        
        meta_nodes[new_child.capsid_id] = new_child
        self.child_count += 1
        print(f"Debugging: Created new child with capsid ID {new_child.capsid_id} and linked to parent {self.capsid_id}")
        print(f"meta_nodes now has {len(meta_nodes)} entries.")
        print(f"meta_nodes[{new_child.capsid_id}] = {new_child}")
        print("New child created with properties:")
        print(f"  - Capsid ID: {new_child.capsid_id}")
        print(f"  - Generative Parent: {new_child.generative_parent}")
        print(f"  - Meta Graph Root: {new_child.meta_graph_root}")

        return new_child.capsid_id

    def get_ids(self, entries):
        """
        Extracts and returns a list of node IDs from the given entries.
        """
        ids = []
        for entry in entries:
            if isinstance(entry, NodeEntry):
                ids.append(entry.node_id)
            elif isinstance(entry, MetaGraphEdge):
                ids.extend(list(entry.transfer_buffer))
        return ids

    def push_exodus_to_children(self, node_ids):
        queues = self.find_in_span((self.c_start, self.envelope_domain[1]), ctypes.sizeof(MetaGraphEdge))
        if queues == BitTensorMemoryGraph.NOTHING_TO_FLY:
            return BitTensorMemoryGraph.NOTHING_TO_FLY
        total_capacity = 0
        capacities = []
        for queue in queues:
            capacity = queue.capacity
            total_capacity += capacity
            capacities.append(capacity)

        transfer_buffer_offset = ctypes.sizeof(MetaGraphEdge) - ctypes.sizeof(ctypes.c_uint64) * META_GRAPH_TRANSFER_BUFFER_SIZE

        if total_capacity < len(node_ids):
            for queue in queues:
                queue = MetaGraphEdge.from_buffer_copy(queue)
                for item in queue.transfer_buffer:
                    if item == 0:
                        item = node_ids.pop(0)
                    self.hard_memory.write(queue + transfer_buffer_offset, ctypes.c_uint64(item))

    def check_queue_spaces(self):
        span = (self.p_start, self.envelope_domain[1])
        queues = self.find_in_span(span, ctypes.sizeof(MetaGraphEdge))
        nodes = []
        capsids = []
        if queues == BitTensorMemoryGraph.NOTHING_TO_FLY:
            return BitTensorMemoryGraph.NOTHING_TO_FLY
        for queue in queues:
            queue = MetaGraphEdge.from_buffer_copy(queue)
            queue = list(queue.transfer_buffer)
            
            for node_id in queue:
                if node_id == 0:
                    continue
                if node_id not in nodes:
                    nodes.append(node_id)
            
            capsids.append(queue.linked_capsid_ref)

        my_nodes = self.get_ids(self.find_in_span((self.n_start, self.c_start), NodeEntry))
        
        their_nodes = [node for node in nodes if node not in my_nodes]

        my_nodes = [node for node in nodes if node in my_nodes]

        # pressure is active bits over capacity in the hard memory bitmap
        densities = self.hard_memory.density
        total_density = sum(densities)//len(densities)
        their_densities = [meta_nodes[capsid].hard_memory.density for capsid in capsids]
        their_total_densities = [sum(density)//len(density) for density in their_densities]

        deltas = [(total_density - density) for density in their_total_densities]

        weighted_deltas = [queue.permeability_weight * delta for queue, delta in zip(queues, deltas)]

        # what is my total experienced delta for the meta-graph
        total_delta = sum(weighted_deltas)

        if total_delta < 0:
            # if total delta is negative, we have more pressure than capacity
            # we need to exchange with the meta graph
            for capsid in capsids:
                if capsid in meta_nodes:
                    meta_nodes[capsid].evaluate_memory_pressure_and_exchange_with_metagraph()

        elif total_delta > 0:
            
            i_will_take = len(nodes) * total_delta

            proportions = [delta / total_delta for delta in weighted_deltas]

            quotas = [(capsid, queue, i_will_take * proportion) for capsid, queue, proportion in zip(capsids, queues, proportions)]

            pull_in = [quota for quota in quotas if quota > 0]

            keep = random.choices(pull_in, k=i_will_take)

            original_quantity = self.node_count

            for capsid, queue, amount in keep:
                
                queue = random.choices(queue.transfer_buffer, k=amount)
                
                for node in queue:
                    if node in my_nodes:
                        continue
                    self.hard_memory.transfer(capsid, node, byref=True)
                    self.node_count += 1

            i_will_give = self.node_count - original_quantity + i_will_take

            push_out = self.any_isolated_nodes(i_will_give)

            give = [(self.capsid_id, node_id) for node_id in push_out]

            should_have_been_push_out = [quota for quota in quotas if quota < 0]

            for quota in should_have_been_push_out:
                capsid, queue, amount = quota
                if capsid in meta_nodes:
                    for _ in amount:
                        meta_nodes[capsid].hard_memory.transfer(self.capsid_id, give[-1], byref=True)
                        self.node_count -= 1
                        give.pop()

            

        return nodes

        
    def evaluate_memory_pressure_and_exchange_with_metagraph(self):
        self.check_queue_spaces()
    def reconfigure_hard_memory_from_header(self):
        self.__init__(self.hard_memory.size - self.header_size, bit_width=self.bit_width, encoding=self.encoding, meta_graph_root=self.meta_graph_root, generative_parent=self.generative_parent)
        self.hard_memory.unit_helper.bitmap.rebuild_density()
    def bifurcate(self):
        """
        Bifurcate the memory graph, creating a new instance
        with the same properties but a different memory layout.
        This is a placeholder for future implementation.
        """

        # This is a method for splitting storage burdens across
        # the meta graph by creating chidren, negotiating to make siblings
        # negotiating on parent relationship splitting if handling too many
        # sources or direct throughput, this should be achieved with
        # permeability of parent child relationships and memory pressure
        # relative to capacity and density of the memory graph.
        # 
        print(self.hard_memory.size, self.header_size, self.hard_memory.size - self.header_size)
        free_space = self.hard_memory.find_free_space("child")
        new_child = None
        if free_space == BitTensorMemory.ALLOCATION_FAILURE or free_space is None:
            if self.emergency_reference == BitTensorMemory.ALLOCATION_FAILURE:
                self.emergency_reference = MetaGraphEdge()
            new_child = self.add_child(self.emergency_reference, byref=True)
        else:
            new_child = self.add_child(free_space)
        if new_child is None:
            raise MemoryError("Failed to bifurcate: no free space available")

        #new_child.capsid_id = (id(self.capsid_id) + uuid4().int) % 2**32

        #there's something I'm forgetting...
        # new_child.hard_memory.write(0, new_child.serialize_header())?
        # actually, it's the fact that we have not instantiated the
        # node we have just referenced by creating a meta edge
        # which originally was a pause in the process because
        # python does not use malloc so I can't permanently
        # instantiate departed from other processes the memory instance
        # so I'm not sure where to pass it, or wasn't, though now 
        # we know it should go through graph_search's meta meta graph
        # or else the global set of meta nodes

        new_child_object = meta_nodes.get(new_child.capsid_id, None)
        if new_child_object is None:
            new_child_object = BitTensorMemoryGraph((self.hard_memory.size - self.header_size) * sum(self.hard_memory.density), bit_width=self.bit_width, encoding=self.encoding)
            meta_nodes[new_child.capsid_id] = new_child_object

        new_child_object.hard_memory.write(0, new_child.serialize_header())
        new_child_object.capsid = True
        new_child_object.deserialize_header()
        new_child_object.reconfigure_hard_memory_from_header()



        
        self.evaluate_memory_pressure_and_exchange_with_metagraph()
        


    def inverse_set(self, offsets_slice):
        hard_memory_bitmask = self.hard_memory.bitmap_expanded()

        
        inverse_set = []
        for i in [i for i in range(len(hard_memory_bitmask)) if hard_memory_bitmask[i] == 1 and i % ctypes.sizeof(NodeEntry) == 0]:
            if offsets_slice is None or offsets_slice == [] or offsets_slice == BitTensorMemoryGraph.NOTHING_TO_FLY:
                inverse_set.append(i)
            elif i not in offsets_slice:
                inverse_set.append(i)

            
        return inverse_set
    def meta_find_node(self, target=None, meta_nodes=None, tracepath=None):
        
        
        if tracepath is None:
            tracepath = set()
        else:
            tracepath = tracepath.copy()
        if meta_nodes is None:
            meta_nodes = {self.capsid_id: self}
        
        search_field = [ metanode for metanode in meta_nodes.values() if metanode not in tracepath ]

        if target is None:
            return GraphSearch(meta_nodes, search_field=search_field).find_all_nodes()
        return GraphSearch(meta_nodes, search_field=search_field).find_node(target)

#when you get back fix bitmask, the logic is inconsistent

    def empty_set(self, offsets_slice):
        """
        Returns a set of empty offsets in the hard memory
        that are not occupied by the given offsets_slice.
        """
        hard_memory_bitmask = self.hard_memory.unit_helper.bitmap[MaskConsolidation.MASK_BITMAP]
        empty_set = []
        for i in range(len(hard_memory_bitmask)):
            if hard_memory_bitmask[i] == 0:
                if offsets_slice is None or offsets_slice == [] or offsets_slice == BitTensorMemoryGraph.NOTHING_TO_FLY:
                    empty_set.append(i)
                elif i not in offsets_slice:
                    empty_set.append(i)
        return empty_set
        
    def get_header_view(self, tracepath=set(), depth=0, depth_limit=10):
        def finalize(return_val):
            tracepath.add(self.capsid_id)
            return return_val, tracepath
        if depth > depth_limit:
            return finalize(BitTensorMemoryGraph.NOTHING_TO_FLY)
        if self.capsid_id in tracepath:
            return finalize(BitTensorMemoryGraph.NOTHING_TO_FLY)
        def build_header_view():
            """
            Build a memoryview of the header data.
            """
            header_data = self.serialize_header()
            return memoryview(header_data)
        if self.capsid:
            # If the capsid is encapsulated, we need to read the header from hard memory
            header_data = self.hard_memory.view(0, self.header_size)
            return finalize(header_data)
        else:
            header_node = self.capsid_node_ref
            if header_node is None:
                my_nodes = self.find_in_span((self.n_start, self.c_start), ctypes.sizeof(NodeEntry))
                if my_nodes == BitTensorMemoryGraph.NOTHING_TO_FLY:
                    meta_associates = self.find_in_span((self.p_start, self.envelope_domain[1]), ctypes.sizeof(MetaGraphEdge))
                    if meta_associates == BitTensorMemoryGraph.NOTHING_TO_FLY:
                        new_capsid_node = build_header_view()
                    else:
                        for associate in meta_associates:
                            associate = MetaGraphEdge.from_buffer_copy(associate)

                            associate_location = associate.linked_capsid_ref
                            
                            associate_header, new_tracepath = associate.get_header_view(tracepath, depth+1, depth_limit)
                            
                            tracepath.update(new_tracepath)
                            
                            if associate_location in meta_nodes:
                                new_capsid_node = meta_nodes[associate_location].find_in_span((associate_header.n_start, associate_header.c_start), ctypes.sizeof(NodeEntry))
                                return finalize(new_capsid_node)

                else:
                    for node in my_nodes:
                        node = NodeEntry.from_buffer_copy(node)
                        if node.node_id == self.capsid_id:
                            new_capsid_node = node.get_full_region()
                            return finalize(new_capsid_node)
            if tracepath is None:
                tracepath = [self.capsid_id]
                success = self.meta_find_node(meta_nodes, tracepath)
                if success:
                    finalize(success)

            new_capsid_node = build_header_view()
            self.add_node(new_capsid_node, node_id=self.capsid_id, byref=True)
            self.capsid_node_ref = new_capsid_node

        return finalize(new_capsid_node)

    def compute_region_boundaries(self):
        """Initialise the Region-Manager with the canonical layout.

        The classic ``memory_graph`` implementation expected ``region_layout``
        to be a simple list of 4-tuples ``(label, start, end, stride)``.  Recent
        experiments wrapped this information in a dictionary to expose additional
        bookkeeping data, but much of the surrounding code still assumes the
        original list structure.  To keep the harmoniser simple and avoid
        brittle type‑checks in hot paths, we compute the extra maps but store the
        tuple list on the instance and return it directly.
        """

        # ── 1. declarative layout ─────────────────────────────────────────────
        boundaries = [
            self.header_size,           # end of header   (Cell-0)
            self.envelope_domain[0],    # start of arena  (Cell-1)
            self.n_start,               # nodes           (Cell-2)
            self.e_start,               # edges           (Cell-3)
            self.p_start,               # parents         (Cell-4)
            self.c_start,               # children        (Cell-5)
            self.envelope_domain[1],    # spare / scratch (Cell-6)
            self.hard_memory_size,      # immutable tail  (Cell-7)
        ]

        strides = [
            ctypes.sizeof(BTGraphHeader),   # header grain
            8,                              # envelope filler
            ctypes.sizeof(NodeEntry),       # node stride
            ctypes.sizeof(EdgeEntry),       # edge stride
            ctypes.sizeof(MetaGraphEdge),   # parent stride
            ctypes.sizeof(MetaGraphEdge),   # child  stride
            8,                              # scratch stride
            self.hard_memory.extra_data_size or 1,  # tail (≥1 to avoid div-by-0)
        ]

        # ── 2. push layout into Region-Manager ────────────────────────────────
        raw_regions = self.hard_memory.reset_region_manager(boundaries, strides)

        # ── 3. harvest per-cell maps ──────────────────────────────────────────
        (active_regions,
        free_by_cell,
        occupied_by_cell) = self.process_active_regions(raw_regions)

        # keep instance in sync
        self.active_regions = active_regions
        self.free_spaces_by_cell = free_by_cell
        self.occupied_spaces_by_cell = occupied_by_cell

        # pre-compute “best fit” per cell for later allocation helpers
        self.best_free_space = {}
        for cell_idx, holes in free_by_cell.items():
            self.best_free_space[cell_idx] = (
                min(holes, key=lambda t: (t[1], t[0])) if holes else None
            )

        return active_regions

    def initialize_regions(self):
        """
        Initialize the regions of the memory graph.
        This is a placeholder for future implementation.
        """
        print("Debugging: Initializing regions.")
        # right now it looks like the buck is stopping here for the initial
        # boundary definitions, while other definitions also exist for that purpose
        # it will be vital to tie these definitions together somewhere
        # but for now we'll reiterate something valid
        
        active_regions = []
        for i in range(8):
            start = i * self.hard_memory.granular_size
            end = start + self.hard_memory.granular_size
            stride = self.hard_memory.granular_size
            label = f"Region {i}"
            active_regions.append((label, start, end, stride))

        self.region_layout = active_regions
        self.envelope_domain = (active_regions[0][1], active_regions[-1][2])
        self.envelope_size = self.envelope_domain[1] - self.envelope_domain[0]
        self.n_start = active_regions[1][1]
        self.e_start = active_regions[2][1]
        self.p_start = active_regions[3][1]
        self.c_start = active_regions[4][1]
        self.n_rational = self.n_start // self.envelope_size
        self.e_rational = self.e_start // self.envelope_size
        self.p_rational = self.p_start // self.envelope_size
        self.c_rational = self.c_start // self.envelope_size
        print(f"Debugging: Initialized regions with envelope domain {self.envelope_domain} and size {self.envelope_size}.")
        print(f"Debugging: Region layout: {self.region_layout}")
        self.l_start = active_regions[0][1]
        self.r_start = active_regions[-2][2]
        self.x_start = active_regions[-1][2]
        return self.region_layout

    def process_active_regions(self, active_regions):
        # ── 0.  Book-keeping & old debug — keep whatever you still need
        print(f"Debugging: Processing active regions: {active_regions}")
        if not active_regions:
            active_regions = self.initialize_regions()
        print(f"Debugging: Active regions found: {len(active_regions)}")

        # Convert ``Cell`` objects (from the region manager) into 4-tuples so the
        # rest of the code can treat them uniformly.
        region_tuples = []
        for region in active_regions:
            if region is None:
                continue
            region_tuples.append(
                (
                    getattr(region, "label", ""),
                    getattr(region, "left", 0),
                    getattr(region, "right", 0),
                    getattr(region, "stride", 0),
                )
            )

        # ── 1.  Fresh quanta-level dump from pressure-based manager
        cell_dump = CellPressureRegionManager.dump_cells(self.hard_memory.region_manager)

        def group_by_cell(ranges):
            buckets = collections.defaultdict(list)
            for label, addr, size in ranges:
                buckets[label].append((addr, size))
            for vals in buckets.values():
                vals.sort(key=lambda t: t[1])  # sort each cell’s list by size
            return buckets

        free_spaces_by_cell = group_by_cell(cell_dump["free_spaces"])
        occupied_spaces_by_cell = group_by_cell(cell_dump["occupied_spaces"])

        return region_tuples, free_spaces_by_cell, occupied_spaces_by_cell


    def find_in_span(self, delta_band, entry_size, return_objects=False):
        delta_band = (int(delta_band[0]), int(delta_band[1]))
        # Ensure harmonization is always up-to-date before each operation:
        print(f"Debugging: Finding in span {delta_band} with entry size {entry_size}")
        if not hasattr(self, "region_layout") or not self.region_layout:
            # Cache the full layout information from the region manager.
            self.region_layout = self.compute_region_boundaries()

        active_regions = self.region_layout
        print(f"Debugging: Active regions after harmonization: {active_regions}")
        # Find the region in the layout that matches this entry_size and delta_band
        for label, start, end, stride in active_regions:
            print(f"Debugging: Checking region {label}, {start}-{end} with stride {stride}")
            if stride == entry_size and start <= delta_band[0] < end:
                region_start = max(start, delta_band[0])
                region_end = min(end, delta_band[1])
                break
        else:
            # Fallback to original (but only if you want undefined behavior!)
            region_start, region_end, stride = delta_band[0], delta_band[1], entry_size

        flight_zone = []
        for i in range(region_start, region_end, stride):
            flight_zone.append(i)

        hard_memory_bitmask = self.hard_memory.unit_helper.bitmap[MaskConsolidation.MASK_BITMAP]
        if hard_memory_bitmask == BitTensorMemoryGraph.NOTHING_TO_FLY:
            return BitTensorMemoryGraph.NOTHING_TO_FLY
        # Now check occupancy:
        flight_zone = [i for i in flight_zone if hard_memory_bitmask[self.hard_memory.unit_helper.grains_for_bytes(i)] == 1]
        if not flight_zone:
            return BitTensorMemoryGraph.NOTHING_TO_FLY

        if return_objects:
            return [self.hard_memory.read(i, stride) for i in flight_zone]
        return flight_zone
    def snap_to_alignment(self, offset, entry_size, direction="down"):
        """
        Snap a byte offset to the nearest legal entry alignment.
        - offset: arbitrary byte offset
        - entry_size: struct size (e.g., ctypes.sizeof(NodeEntry))
        - direction: "down" (default) for floor, "up" for ceil
        Returns the snapped offset (aligned to entry boundary).
        """
        if direction == "down":
            return int((offset // entry_size) * entry_size)
        elif direction == "up":
            return int(((offset + entry_size - 1) // entry_size) * entry_size)
        else:
            raise ValueError("direction must be 'down' or 'up'")

    def any_isolated_nodes(self, count):
        """
        Returns a list of isolated nodes in the memory graph.
        An isolated node is one that has no edges connected to it.
        """

        nodes = self.find_in_span((self.n_start, self.c_start), ctypes.sizeof(NodeEntry), return_objects=True)
        edges = self.find_in_span((self.e_start, self.p_start), ctypes.sizeof(EdgeEntry))

        if nodes == BitTensorMemoryGraph.NOTHING_TO_FLY:
            return BitTensorMemoryGraph.NOTHING_TO_FLY
        if edges == BitTensorMemoryGraph.NOTHING_TO_FLY:
            return nodes
        flat_edge_membership = [edge.src_ptr for edge in edges] + [edge.dst_ptr for edge in edges]
        connectivity = {node: 0 for node in nodes}
        for member in flat_edge_membership:
            if member in connectivity:
                connectivity[member] += 1
        isolated_nodes = [node for node, count in connectivity.items() if count == 0]
        if len(isolated_nodes) < count:
            def sort_connectivity(node_connectivity_dictionary):
                return sorted(node_connectivity_dictionary.items(), key=lambda item: item[1], reverse=True)
            sorted_isolated = sort_connectivity(connectivity)
            if len(sorted_isolated) < count:
                return self.NOTHING_TO_FLY
            return [node for node, _ in sorted_isolated[:count]]

        return isolated_nodes[:count]

    # ───────────────────────────────────────────────────────────────
    # Edge scrubber: replace every occurrence of one node-ref
    #                 with another and refresh the edge checksum.
    # ───────────────────────────────────────────────────────────────
    def scrub_edges(self, old_ptr: int, new_ptr: int, *, touch_graph_ids=False) -> int:
        """
        Replace **all** occurrences of `old_ptr` in every EdgeEntry’s
        src_ptr/dst_ptr  (and optionally src_graph_id/dst_graph_id)
        with `new_ptr`, then recompute checksums in-place.

        Returns the number of EdgeEntry records modified.
        """
        # 1. Locate every edge struct in memory
        edge_offs = self.find_in_span((self.e_start, self.p_start),
                                    ctypes.sizeof(EdgeEntry))
        if edge_offs == BitTensorMemoryGraph.NOTHING_TO_FLY:
            return 0

        patched = 0
        for off in edge_offs:
            # read -> struct
            raw  = self.hard_memory.read(off, ctypes.sizeof(EdgeEntry))
            edge = EdgeEntry.from_buffer_copy(raw)

            changed = False
            # 2. swap pointers
            if edge.src_ptr == old_ptr:
                edge.src_ptr = new_ptr
                changed = True
            if edge.dst_ptr == old_ptr:
                edge.dst_ptr = new_ptr
                changed = True
            # 3. (optional) swap graph-IDs too
            if isinstance(touch_graph_ids, tuple):
                if edge.src_graph_id == touch_graph_ids[0]:
                    edge.src_graph_id = touch_graph_ids[1]
                    changed = True
                if edge.dst_graph_id == touch_graph_ids[3]:
                    edge.dst_graph_id = touch_graph_ids[4]
                    changed = True

            if not changed:
                continue

            # 4. refresh checksum & write back
            patched_bytes = GraphSearch._build_struct_bytes(edge)
            self.hard_memory.write(off, patched_bytes)
            patched += 1

        return patched
    def find_in_edges(self, node_ptr):
        """
        Find all edges in the graph that are connected to a given node.
        """
        edge_offs = self.find_in_span((self.e_start, self.p_start),
                                    ctypes.sizeof(EdgeEntry))
        if edge_offs == BitTensorMemoryGraph.NOTHING_TO_FLY:
            return []

        edges = []
        for off in edge_offs:
            raw  = self.hard_memory.read(off, ctypes.sizeof(EdgeEntry))
            edge = EdgeEntry.from_buffer_copy(raw)
            if edge.src_ptr == node_ptr or edge.dst_ptr == node_ptr:
                edges.append(edge)

        return edges

    # ---- 1.  planner ----------------------------------------------
    def plan_edge_relocations(self, sources, destinations):
        """
        Returns two things:
            * best_assignment  – list of (src_idx, dst_idx) pairs
            * total_delta      – signed sum of distance changes (<0 is improvement)
        """
        calc = lambda src, dst: abs(src - dst)  # distance function
        # pre-cache edge lists for every source
        edge_map = {
            s : self.find_in_edges(s) or []
            for s in sources
        }
        def _edge_distance(edge, calc):
            return calc(edge.src_ptr, edge.dst_ptr)
        # --- build cost matrix -------------------------------------
        cost = [[0]*len(destinations) for _ in sources]
        
        
        for i, s in enumerate(sources):
            for j, d in enumerate(destinations):
                delta = 0
                for off in edge_map[s]:
                    e   = EdgeEntry.from_buffer_copy(
                            self.hard_memory.read(off, ctypes.sizeof(EdgeEntry)))
                    old = _edge_distance(e, calc)
                    new_src = d if e.src_ptr == s else e.src_ptr
                    new_dst = d if e.dst_ptr == s else e.dst_ptr
                    new = calc(new_src, new_dst)
                    delta += (new - old)
                cost[i][j] = delta
        
        
        # --- Hungarian assignment (O(n³) but n is small here) -------
        try:
            from scipy.optimize import linear_sum_assignment as hungarian
            rows, cols = hungarian(cost)          # scipy returns numpy arrays
            assign     = list(zip(rows.tolist(), cols.tolist()))
        except ImportError:                       # tiny pure-python fallback
            best, best_val = None, math.inf
            for perm in itertools.permutations(range(len(destinations)),
                                            len(sources)):
                val = sum(cost[i][p] for i, p in enumerate(perm))
                if val < best_val:
                    best_val, best = val, perm
            assign = list(enumerate(best))

        total_delta = sum(cost[i][j] for i, j in assign)
        return assign, total_delta

    # ---- 2.  mover ------------------------------------------------
    def relocate_edges(self, sources, destinations, assignment):
        """
        `assignment` is the list returned by plan_edge_relocations.
        """
        for src_idx, dst_idx in assignment:
            old = sources[src_idx]
            new = destinations[dst_idx]

            # lock once per source-move
            stale = self.find_in_edges(old)
            if not stale or stale == self.NOTHING_TO_FLY:
                continue
            if self.lock_manager:
                self.lock_manager.lock(stale)

            self.scrub_edges(old, new)   # graph-ID fields remain unchanged   

    def relocate_hard_memory_sites(self, sources, destinations, distance_matrix):
        #this is for moving collections with indifference to ordering
        #which involves a preliminary search for what the most
        #relaxed network state could be for the ordering

        relevant_edges = self.find_in_span((self.e_start, self.p_start), ctypes.sizeof(EdgeEntry))
        if relevant_edges == BitTensorMemoryGraph.NOTHING_TO_FLY:
            # the nodes are not relevant to the graph and can be moved arbitrarily
            for i, (src, dest) in enumerate(zip(sources, destinations)):
                self.hard_memory.move(src, dest)
            return



        for i, (src, dest) in enumerate(zip(sources, destinations)):
            distance = distance_matrix[i]
            # for now we'll take the best scores first for single move relaxation of edge distances
            # by summing the differences in distances
    def get_node(self, node_id):
        print(f"Retrieving node with ID: {node_id} from memory graph.")
        """
        Retrieve a NodeEntry by its node_id.
        Returns None if the node is not found.
        """
        nodes = self.find_in_span((self.n_start, self.c_start), ctypes.sizeof(NodeEntry), return_objects=True)
        print(f"Found {len(nodes)} nodes in the memory graph.")
        
        if nodes == BitTensorMemoryGraph.NOTHING_TO_FLY:
            print(f"Node with ID: {node_id} not found.")
            return None
        
        print(f"Found {len(nodes)} nodes in the memory graph.")
        
        for node in nodes:
            node_entry = NodeEntry.from_buffer_copy(node)
            print(f"node: {node}, node_entry: {node_entry}")
            if node_entry.node_id == node_id:
                print(f"Node with ID: {node_id} found in memory graph.")
                return node_entry
        print(f"Node with ID: {node_id} not found in the memory graph.")
        return None
    
    def add_node(self, node_entry=None, node_id=None, *args, **kwargs):
        print(f"Adding node with args: {args}, kwargs: {kwargs}, node_entry: {node_entry}, node_id: {node_id}")

        # Prepare a byte string from any positional/keyword arguments to store
        # in ``node_data`` when a struct isn't explicitly provided.
        bytes_args = b"".join(bytes(arg) for arg in args)
        bytes_kwargs = b"".join(f"{k}={v}".encode("utf-8") for k, v in kwargs.items())

        if node_entry is None:
            node_entry = bytes_args + bytes_kwargs

        def _to_bytes(obj):
            if isinstance(obj, bytes):
                return obj
            if isinstance(obj, str):
                return obj.encode("utf-8")
            if isinstance(obj, ctypes.Array):
                return bytes(obj)
            if isinstance(obj, list):
                return bytes(obj)
            return str(obj).encode("utf-8")

        if isinstance(node_entry, NodeEntry):
            # ``node_entry`` may already carry an ID, but treat ``0`` as
            # "unset" so we always emit a usable identifier.  Callers can
            # still override via the explicit ``node_id`` parameter.
            if node_id in (None, 0):
                node_id = node_entry.node_id
            if node_id in (None, 0):
                node_id = uuid4().int % 2**32
            node_entry.node_id = node_id
            node_bytes = ctypes.string_at(ctypes.addressof(node_entry), ctypes.sizeof(NodeEntry))
        else:
            if node_id in (None, 0):
                node_id = uuid4().int % 2**32
            node_data = _to_bytes(node_entry)
            new_node_entry = NodeEntry(node_id=node_id, node_data=node_data)
            node_bytes = ctypes.string_at(ctypes.addressof(new_node_entry), ctypes.sizeof(NodeEntry))

        new_node_slot = self.hard_memory.find_free_space("node", ctypes.sizeof(NodeEntry))
        if new_node_slot == BitTensorMemory.ALLOCATION_FAILURE:
            raise MemoryError("Failed to add node: no free space available")

        if new_node_slot is None:
            raise MemoryError("Allocation returned None")

        status = self.hard_memory.write(new_node_slot, node_bytes)
        if status == BitTensorMemory.ALLOCATION_FAILURE:
            raise MemoryError("Failed to write node entry to hard memory")

        self.node_count += 1

        return node_id
        

class Deque3D(ctypes.Structure):
    # 3 contiguous 64-bit values (could be int, float, etc.)
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("z", ctypes.c_double)
    ]

    def __init__(self, x=0, y=0, z=0):
        super().__init__(x, y, z)

    def __repr__(self):
        return f"<Deque3D x={self.x}, y={self.y}, z={self.z}>"

    def as_tuple(self):
        return (self.x, self.y, self.z)

    def __getitem__(self, idx):
        return (self.x, self.y, self.z)[idx]

    def __setitem__(self, idx, value):
        if idx == 0:
            self.x = value
        elif idx == 1:
            self.y = value
        elif idx == 2:
            self.z = value
        else:
            raise IndexError("Deque3D index out of range")

    def __len__(self):
        return 3


class MaskConsolidation(ctypes.Structure):
    """
    A structure to hold a mask for consolidating memory regions.
    This is used to track which regions are occupied and which are free.
    """
    MASK_DELTA = 0
    MASK_DENSITY = 1
    MASK_BITMAP = 2


    _fields_ = [
        ("delta_style", ctypes.c_int),
        ("density_style", ctypes.c_int),
        ("bitmap_size", ctypes.c_int),
        ("density_size", ctypes.c_int),
        ("delta_size", ctypes.c_int),
    ]

    def __init__(self, memory_units, total_chunks, total_grains, bitmap_depth=8, density_depth=8, delta_depth=8, deep_delta=False):
        super().__init__()
        self.memory_units = memory_units
        self.bitmap_depth = bitmap_depth
        self.total_chunks = self.bitmap_size = total_chunks
        self.bitmap_size //= self.bitmap_depth
        self.density_depth = bitmap_depth | density_depth
        self.delta_depth = bitmap_depth | delta_depth
        self.total_grains = total_grains


        self.bitmap = ctypes.create_string_buffer(self.bitmap_size)
        
        self.density = ctypes.create_string_buffer(self.total_chunks * self.density_depth // 8)
        self.delta = ctypes.create_string_buffer(self.total_chunks * self.delta_depth // 8)
        self.delta_style = 0  # Default delta style
        self.density_style = 0  # Default density style
        self.delta_size = self.total_chunks * self.delta_depth
        self.density_size = self.total_chunks * self.density_depth
        self.bitmap_style = 0  # Default bitmap style

        if deep_delta:
            self.delta = ctypes.create_string_buffer(self.byte_size * self.delta_depth)
    
    def _clip(self, value, mode="byte"):
        """
        Clip the value to the valid range for the given mode.
        """
        if mode == "byte":
            return max(0, min(value, self.bitmap_size))
        elif mode == "grain":
            return max(0, min(value, self.total_grains))
        elif mode == "chunk":
            return max(0, min(value, self.total_chunks))
        else:
            raise ValueError("Invalid mode for clipping value")

        
    def bool_array(self):
        """
        Returns a boolean array representation of the bitmap.
        Each byte in the bitmap is treated as a set of bits.
        """
        ptr = ctypes.cast(self.bitmap, ctypes.POINTER(ctypes.c_uint8))
        return [bool((ptr[i] >> j) & 1) for i in range(self.total_grains) for j in range(8)]

    def obtain_map_as_byte_string(self, dataset, offset, size):
        """
        Returns the bitmap as a byte string.
        This is useful for serialization or saving to disk.
        """
        if dataset == "bitmap":
            return ctypes.addressof(self.bitmap) + offset * self.bitmap_depth

        elif dataset == "density":
            return ctypes.addressof(self.density) + offset * self.density_depth
        elif dataset == "delta":
            return ctypes.addressof(self.delta) + offset * self.delta_depth

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        primary_index = idx[0]
        secondary_index = idx[1] if len(idx) > 1 else None
        tertiary_index = idx[2] if len(idx) > 2 else None

        if primary_index == 0:  # delta
            if secondary_index is None:
                return self.delta
            ptr = ctypes.cast(self.delta, ctypes.POINTER(ctypes.c_uint8))
            return ptr[secondary_index]
        elif primary_index == 1:  # density
            if secondary_index is None:
                return self.density
            ptr = ctypes.cast(self.density, ctypes.POINTER(ctypes.c_uint8))
            return ptr[secondary_index]
        elif primary_index == 2:  # bitmap
            if secondary_index is None:
                return self.bool_array()
            ptr = ctypes.cast(self.bitmap, ctypes.POINTER(ctypes.c_uint8))
            byte_val = ptr[secondary_index]
            if tertiary_index is not None:
                return (byte_val >> tertiary_index) & 1
            return byte_val
        else:
            raise IndexError("MaskConsolidation index out of range")

    def rebuild_density(self):
        for chunk_index in range(self.total_chunks):
            self.calculate_density(chunk_index)

    def calculate_density(self, chunk_index):
        """
        Calculate the density for a given chunk index.
        This updates the density array based on the bitmap.
        """
        bits = [self[self.MASK_BITMAP, chunk_index, bit] for bit in range(self.bitmap_depth)]
        density_value = sum(bits) / self.bitmap_depth
        if self.density_depth not in (8, 16, 32, 64):
            self.density_depth = self.bitmap_depth
            self.density = ctypes.create_string_buffer(self.total_chunks * self.density_depth)
        if self.density_depth == 8:
            self[self.MASK_DENSITY, chunk_index] = int(density_value * 255)
        elif self.density_depth == 16:
            self[self.MASK_DENSITY, chunk_index] = int(density_value * 65535)
        elif self.density_depth == 32:
            self[self.MASK_DENSITY, chunk_index] = int(density_value * 4294967295)
        elif self.density_depth == 64:
            self[self.MASK_DENSITY, chunk_index] = int(density_value * 18446744073709551615)
        self[self.MASK_DENSITY, chunk_index] = int(density_value * 255)  # Scale to 0-255

    def __setitem__(self, idx, value):
        if not isinstance(idx, tuple):
            raise IndexError("Index must be a tuple (type, index, [bit])")

        primary_index = idx[0]
        secondary_index = idx[1] if len(idx) > 1 else None
        tertiary_index = idx[2] if len(idx) > 2 else None

        if primary_index == 0:  # delta
            if secondary_index is None:
                raise IndexError("Delta index required")
            ptr = ctypes.cast(self.delta, ctypes.POINTER(ctypes.c_uint8))
            ptr[secondary_index] = value
        elif primary_index == 1:  # density
            if secondary_index is None:
                raise IndexError("Density index required")
            ptr = ctypes.cast(self.density, ctypes.POINTER(ctypes.c_uint8))
            ptr[secondary_index] = value
        elif primary_index == 2:  # bitmap
            if secondary_index is None or tertiary_index is None:
                raise IndexError("Bitmap index and bit required")
            ptr = ctypes.cast(self.bitmap, ctypes.POINTER(ctypes.c_uint8))
            if value:
                ptr[secondary_index] |= (1 << tertiary_index)
            else:
                ptr[secondary_index] &= ~(1 << tertiary_index)

            self.calculate_density(secondary_index)
        else:
            raise IndexError("MaskConsolidation index out of range")
    def __iter__(self):
        ptr = ctypes.cast(self.bitmap, ctypes.POINTER(ctypes.c_uint8))
        byte_len = self.total_grains * self.bitmap_size
        for byte_index in range(byte_len):
            byte_val = ptr[byte_index]
            for bit in range(8):
                yield bool((byte_val >> bit) & 1)
    def __repr__(self):
        """
        ASCII density bar with region markers
        ---------------------------------------------------------------
        Requires:  the BitTensorMemory object that owns this helper
                   must have done:
                       self.unit_helper.graph = <BitTensorMemoryGraph>
        (That single line can be added right after `self.unit_helper`
         is created in BitTensorMemory.__init__.)
        """
        # -------------- 1. build the raw density glyphs ---------------
        ramp = " .:-=+*%@#"           # 10-step density ramp
        ramp_max = len(ramp) - 1
        step     = max(1, 256 // ramp_max)

        glyphs = []
        for i in range(self.total_chunks):
            d_val = int(self[self.MASK_DENSITY, i])      # 0-255
            glyphs.append(ramp[min(ramp_max, d_val // step)])

        # -------------- 2. overlay graph layout markers ---------------
        g = getattr(self.memory_units, "graph", None)

        if g is not None:                              # only if available
            sz   = self.memory_units.chunk_size
            marks = {
                "L": g.envelope_domain[0] // sz,
                "R": (g.envelope_domain[1]-1) // sz,
                "N": g.n_start // sz,
                "E": g.e_start // sz,
                "P": g.p_start // sz,
                "C": g.c_start // sz,
            }
            # overwrite glyphs (later keys win on collision)
            for sym, idx in marks.items():
                if 0 <= idx < len(glyphs):
                    glyphs[idx] = sym

        density_line = "".join(glyphs)

        # -------------- 3. summary line -------------------------------
        summary = (
            f"<MaskConsolidation bitmap_size={self.bitmap_size}, "
            f"density_size={self.density_size}, delta_size={self.delta_size}, "
            f"bitmap_depth={self.bitmap_depth}, "
            f"density_depth={self.density_depth}, delta_depth={self.delta_depth}>"
        )
        return "\n" + density_line + "\n" + summary

class BitTensorMemoryUnits:
    
    class Chunk:
        def __init__(self, start, end, size):
            self.start, self.end, self.size = start, end, size
        def __repr__(self): return f"<Chunk {self.start:#x}-{self.end:#x} size={self.size}>"
        def __getitem__(self, idx): return (self.start, self.end, self.size)[idx]

    class Byte:
        def __init__(self, start, end): self.start, self.end = start, end
        def __repr__(self): return f"<Byte {self.start:#x}-{self.end:#x}>"
        def __getitem__(self, idx): return (self.start, self.end)[idx]

    class Grain:
        def __init__(self, start, end, size): self.start, self.end, self.size = start, end, size
        def __repr__(self): return f"<Grain {self.start:#x}-{self.end:#x} size={self.size}>"
        def __getitem__(self, idx): return (self.start, self.end, self.size)[idx]

    class Node:
        def __init__(self, start, end, template_type):
            self.start, self.end, self.type = start, end, template_type
            self.size = ctypes.sizeof(template_type) if template_type else (end - start)
        def __repr__(self):
            return f"<Node {self.start:#x}-{self.end:#x} type={getattr(self.type, '__name__', str(self.type))} size={self.size}>"
        def __getitem__(self, idx): return (self.start, self.end, self.size)[idx]

    def __init__(self, offset=None, size=None, grain_size=None, chunk_size=None, left=None, right=None, node_template=NodeEntry, edge_template=EdgeEntry, association_template=MetaGraphEdge, node_data_template=None, edge_data_template=None, snap_on_input=None, snap_direction=None, bitmap_depth=ctypes.sizeof(ctypes.c_uint8), hard_memory=None):
        self.hard_memory = hard_memory
        if grain_size is None:
            grain_size = BitTensorMemory.DEFAULT_GRAIN
        if chunk_size is None:
            chunk_size = BitTensorMemory.DEFAULT_CHUNK

        if left is not None and right is not None:
            size = right - left
        elif offset is not None and size is not None:
            left = offset
            right = offset + size
                
        if snap_on_input is not None:
            if snap_direction == 'left':
                if 'grain' in snap_on_input:
                    offset = (offset + grain_size - 1) // grain_size * grain_size
                if 'chunk' in snap_on_input:
                    offset = (offset + chunk_size - 1) // chunk_size * chunk_size
            elif snap_direction == 'right':
                if 'grain' in snap_on_input:
                    offset = offset // grain_size * grain_size
                if 'chunk' in snap_on_input:
                    offset = offset // chunk_size * chunk_size
        self.size = size
        self.grain_size = grain_size
        self.chunk_size = chunk_size
        self.node_template = node_template
        self.node_dictionary = self.extract_field_metadata(node_template)
        self.edge_template = edge_template
        self.edge_dictionary = self.extract_field_metadata(edge_template)
        self.association_template = association_template
        self.association_dictionary = self.extract_field_metadata(association_template)
        self.left = left
        self.right = right
        self.offset = offset
        self.bitmap_depth = bitmap_depth
        def build_a_bitmap(self):
            return MaskConsolidation(
                    memory_units=self, total_chunks=chunk_size, total_grains=grain_size,
                    bitmap_depth=self.bitmap_depth, density_depth=self.bitmap_depth, delta_depth=self.bitmap_depth,
            )
        self.bitmap = build_a_bitmap(self)
        self.bitmap_backend = build_a_bitmap(self)

        self.deltamap = self.bitmap.delta
        self.density = self.bitmap.density

        self.boundaries = [
            ('header', 0, self.hard_memory.header_size, ctypes.sizeof(BTGraphHeader)),
            ('envelope', self.hard_memory.envelope_domain[0], self.hard_memory.envelope_domain[1], 8),
            ('nodes', self.hard_memory.n_start, self.hard_memory.e_start, ctypes.sizeof(NodeEntry)),
            ('edges', self.hard_memory.e_start, self.hard_memory.p_start, ctypes.sizeof(EdgeEntry)),
            ('meta_edges', self.hard_memory.p_start, self.hard_memory.c_start, ctypes.sizeof(MetaGraphEdge)),
            ('capsid', self.hard_memory.c_start, self.hard_memory.envelope_domain[1], ctypes.sizeof(MetaGraphEdge)),
            ('extra_data', self.hard_memory.envelope_domain[1], self.hard_memory.size, 8),
        ]
        self.boundaries_backend = [
            ('header', 0, self.hard_memory.header_size, ctypes.sizeof(BTGraphHeader)),
            ('envelope', self.hard_memory.envelope_domain[0], self.hard_memory.envelope_domain[1], 8),
            ('nodes', self.hard_memory.n_start, self.hard_memory.e_start, ctypes.sizeof(NodeEntry)),
            ('edges', self.hard_memory.e_start, self.hard_memory.p_start, ctypes.sizeof(EdgeEntry)),
            ('meta_edges', self.hard_memory.p_start, self.hard_memory.c_start, ctypes.sizeof(MetaGraphEdge)),
            ('capsid', self.hard_memory.c_start, self.hard_memory.envelope_domain[1], ctypes.sizeof(MetaGraphEdge)),
            ('extra_data', self.hard_memory.envelope_domain[1], self.hard_memory.size, 8),
        ]

    def relocation_trinary_map(self):
        trinary_map = []
        for i in range(self.size // self.grain_size):
            bitmap_index = self.grain_to_bitmap(i, self.bitmap_depth)
            bitmap_value = self.bitmap[self.bitmap.MASK_BITMAP, bitmap_index[0], bitmap_index[1]]
            backend_bitmap_value = self.bitmap_backend[self.bitmap.MASK_BITMAP, bitmap_index[0], bitmap_index[1]]
            if bitmap_value == backend_bitmap_value:
                trinary_map.append(0)
            elif bitmap_value == 1:
                trinary_map.append(-1)
            elif backend_bitmap_value == 1:
                trinary_map.append(1)

        return trinary_map
    
    def check_in_with_remap(self, left=True):
        changemap = self.relocation_trinary_map()
        copy_cache = {}
        copy_destinations_lacking_objects = {}
        copy_objects_lacking_destinations = {}
        boundaries = self.boundaries_backend

        for i, (label, start, end, stride) in enumerate(boundaries):
            if i % stride != 0:
                continue
            elif changemap[i] == 0:
                continue
            elif changemap[i] == -1:
                copy_destinations_lacking_objects[(start, end)] = ctypes.cast(self.bitmap_backend, ctypes.POINTER(ctypes.c_uint8)) + start
            elif changemap[i] == 1:
                copy_objects_lacking_destinations[(start, end)] = ctypes.cast(self.bitmap_backend, ctypes.POINTER(ctypes.c_uint8)) + start
        
            

        for start_origin, end_origin in copy_objects_lacking_destinations.keys():
            for start_destination, end_destination in copy_destinations_lacking_objects.keys():
                actual_data_in_graph = ctypes.cast(self.hard_memory.data, ctypes.POINTER(ctypes.c_uint8)) + start_origin
                ctypes.memmove(
                    ctypes.cast(self.hard_memory.data, ctypes.POINTER(ctypes.c_uint8)) + start_destination,
                    actual_data_in_graph,
                    end_origin - start_origin
                )
        
        self.swap_bitmap()

        return

    def check_out_for_remap(self, boundaries, left=True):

        return_maps = {}
        new_boundaries = []
        for label, start, end, stride in boundaries:
            full_pointer = ctypes.cast(self.bitmap_backend, ctypes.POINTER(ctypes.c_uint8))
            changed_start = start % stride
            changed_end = end % stride
            changed = changed_start != 0 or changed_end != 0
            if changed and left:
                start = start // stride * stride
                end = (end + stride - 1) // stride * stride
            elif changed:
                start = (start + stride - 1) // stride * stride
                end = end // stride * stride

            if end - start <= 0 or start < 0 or end > self.size:
                if end-start == 0 and end < self.size:
                    return_maps[label] = (start, end+1, ctypes.cast(full_pointer + start, ctypes.POINTER(ctypes.c_uint8)))
                raise ValueError(f"Invalid range for remap: {start}-{end} with stride {stride}")
                
            else:    
                return_maps[label] = (start, end, ctypes.cast(full_pointer + start, ctypes.POINTER(ctypes.c_uint8)))

            new_boundaries.append((label, start, end, stride))

        return return_maps, new_boundaries

    def swap_bitmap(self):
        """
        Swap the bitmap with the bitmap backend.
        This is used to switch between different bitmap representations.
        """
        self.bitmap, self.bitmap_backend = self.bitmap_backend, self.bitmap
        self.deltamap = self.bitmap.delta
        self.density = self.bitmap.density
        self.boundaries, self.boundaries_backend = self.boundaries_backend, self.boundaries

    @staticmethod
    def extract_field_metadata(template_cls: ctypes.Structure) -> dict:
        return {
            field[0]: {
                'offset': getattr(template_cls, field[0]).offset,
                'size': ctypes.sizeof(field[1])
            }
            for field in template_cls._fields_
        }

    def grain_to_bitmap(self,start, chunk_size, bitmap_depth):
        """
        Convert a grain start position to a bitmap position.
        This is used to map grain positions to bitmap positions.
        """
        main_index = start // chunk_size
        sub_index = (start % chunk_size) * bitmap_depth // chunk_size
        return (main_index, sub_index)

    def install_node_data_template(self, node_data_template):
        """
        Install a custom node data template for the NodeEntry.
        This allows for custom data structures to be used in nodes.
        """
        if not issubclass(node_data_template, ctypes.Structure):
            raise TypeError("node_data_template must be a subclass of ctypes.Structure")
        self.node_template.node_data = node_data_template
        self.node_dictionary['node_data'] = {
            'offset': getattr(self.node_template, 'node_data').offset,
            'size': ctypes.sizeof(node_data_template)
        }

    def deque_runner(self, dequeue_id, side_a_template, side_b_template, deque_template):
        """
        Attach a deque template to the memory graph.
        This allows for custom deque structures to be used in nodes.
        """
        if not issubclass(deque_template, ctypes.Structure):
            raise TypeError("deque_template must be a subclass of ctypes.Structure")
        def get_side(side_a_template):
            if isinstance(side_a_template, (NodeEntry, EdgeEntry, MetaGraphEdge)):
                side_a_template = ctypes.pointer(side_a_template)
                if hasattr(side_a_template, 'node_data'):
                    side_a_data_offset = getattr(side_a_template, 'node_data').offset
                elif hasattr(side_a_template, 'edge_data'):
                    side_a_data_offset = getattr(side_a_template, 'edge_data').offset
                else:
                    side_a_data_offset = getattr(side_a_template, 'buffer').offset
                    
                side_a_data_size = ctypes.sizeof(side_a_template.node_data)
                side_a_type = deque_template.side_a_type
                side_a_length = side_a_data_size // ctypes.sizeof(side_a_type)
                return {
                    'offset': side_a_data_offset,
                    'size': side_a_data_size,
                    'type': side_a_type,
                    'length': side_a_length,
                    'unit': BitTensorMemoryUnits.Node(side_a_data_offset, side_a_data_offset + side_a_data_size, side_a_template),

                }
        a_side_details = get_side(side_a_template)
        b_side_details = get_side(side_b_template)

        while True:
            # get the current node data
            # sync with the deque object backend
            # repeat for both sides with dynamic delay
            pass
        


    def attach_deque_to_node_template(self, deque_template):
        
        dequeue_token = uuid4().hex
        if not hasattr(self, 'deque_runners'):
            self.deque_runners = {}
        deque_runner_thread = threading.Thread(target=self.deque_runner, args=(deque_template,dequeue_token))
        
        if not issubclass(deque_template, ctypes.Structure):
            raise TypeError("deque_template must be a subclass of ctypes.Structure")
        
        self.deque_runners[dequeue_token] = deque_runner_thread
        deque_runner_thread.start()
        return dequeue_token

    def apply_bitmask(self, offset, size, value):
        """
        Apply a bitmask to the memory graph.
        This is used to mark regions as occupied or free.
        """
        if offset < 0 or size <= 0 or offset + size > self.size:
            raise ValueError("Offset and size must be within the bounds of the memory graph")
        
        start_grain = offset // self.grain_size
        end_grain = (offset + size + self.grain_size - 1) // self.grain_size
        for grain in range(start_grain, end_grain):
            byte_index, bit_index = self.grain_to_bitmap(grain, self.chunk_size, self.bitmap_depth)
            self.bitmap[self.bitmap.MASK_BITMAP, byte_index, bit_index] = value

    TOUCH_SUM = 0
    XOR_LAST = 1

    

    def apply_delta(self, offset, size, string=None, mode="write"):
        start_chunk = offset // self.chunk_size
        end_chunk = (offset + size + self.chunk_size - 1) // self.chunk_size
        for chunk in range(start_chunk, end_chunk):
            self.bitmap[self.bitmap.MASK_DELTA, chunk] = 1
            if mode == "read":
                self.bitmap[self.bitmap.MASK_DELTA, chunk] = 0
                return
            
            if self.bitmap.delta_style == self.TOUCH_SUM:
                size_inside_this_chunk = min(size, self.chunk_size - (offset % self.chunk_size))
                self.bitmap[self.bitmap.MASK_DELTA, chunk] += size_inside_this_chunk
            elif self.bitmap.delta_style == self.XOR_LAST:
                if string is not None:
                    self.bitmap[self.bitmap.MASK_DELTA, chunk] ^= hash(string) & 0xFF
                else:
                    self.bitmap[self.bitmap.MASK_DELTA, chunk] ^= 1

    def delta(self, offset, size, mode="write", string=None, destination=None):
        """
        Calculate the delta for a given offset and size.
        The delta is the difference between the offset and the size.
        """
        if mode == "write":
            self.apply_bitmask(offset, size, True)
            self.apply_delta(offset, size, string, mode)
        elif mode == "read":
            self.apply_delta(offset, size, mode)
        elif mode == "free":
            self.apply_bitmask(offset, size, False)
        elif mode == "alloc":
            self.apply_bitmask(offset, size, True)
        elif mode == "move":
            self.relocate_delta(offset, size, destination)
            self.relocate_bitmask(offset, size, destination)
        else:
            raise ValueError("Invalid mode for delta calculation")

    def install_edge_data_template(self, edge_data_template):
        """
        Install a custom edge data template for the EdgeEntry.
        This allows for custom data structures to be used in edges.
        """
        if not issubclass(edge_data_template, ctypes.Structure):
            raise TypeError("edge_data_template must be a subclass of ctypes.Structure")
        self.edge_template.edge_data = edge_data_template
        self.edge_dictionary['edge_data'] = {
            'offset': getattr(self.edge_template, 'edge_data').offset,
            'size': ctypes.sizeof(edge_data_template)
        }

    def get_unit_type(self, unit, desired_format, left=True):
        def leftright(val, unit_size, left):
            if left:
                return (val + unit_size - 1) // unit_size * unit_size
            else:
                return val // unit_size * unit_size
        if isinstance(unit, BitTensorMemoryUnits.Chunk):
            if desired_format == "bytes":
                return leftright(unit.start, self.chunk_size, left)
            elif desired_format == "grains":
                return leftright(unit.start * self.chunk_size // self.grain_size, self.grain_size, left)
            elif desired_format == "chunks":
                return leftright(unit.start // self.chunk_size, self.chunk_size, left)
        elif isinstance(unit, BitTensorMemoryUnits.Byte):
            if desired_format == "bytes":
                return leftright(unit.start, 1, left)
            elif desired_format == "grains":
                return leftright(unit.start // self.grain_size, self.grain_size, left)
            elif desired_format == "chunks":
                return leftright(unit.start // self.chunk_size, self.chunk_size, left)
        elif isinstance(unit, BitTensorMemoryUnits.Grain):
            if desired_format == "bytes":
                return leftright(unit.start, 1, left)
            elif desired_format == "grains":
                return leftright(unit.start, self.grain_size, left)
            elif desired_format == "chunks":
                return leftright(unit.start // self.chunk_size, self.chunk_size, left)

    def get_node_size(self):
        return ctypes.sizeof(self.node_template)
    
    def get_edge_size(self):
        return ctypes.sizeof(self.edge_template)
    
    def get_association_size(self):
        return ctypes.sizeof(self.association_template)
        
    def grains_per_chunk(self):
        return self.chunk_size // self.grain_size

    def chunks_in_size(self):
        return self.size // self.chunk_size

    def grains_in_size(self):
        return self.size // self.grain_size

    def bytes_for_grains(self, n):
        return n * self.grain_size

    def grains_for_bytes(self, n):
        return n // self.grain_size

    def chunks_for_bytes(self, n):
        return n // self.chunk_size

    def bytes_for_chunks(self, n):
        return n * self.chunk_size


def main():
    # Create a memory graph
    graph = BitTensorMemoryGraph()

    # Allocate some memory
    node_a = graph.add_node()
    node_b = graph.add_node()
    edge = graph.add_edge(node_a, node_b)
    child = graph.add_child()

    print(f"Node A ID: {node_a}, Node B ID: {node_b}, Edge ID: {edge}, Child ID: {child}")  

    # Write some data
    graph[node_a] = "Node A data"
    graph[node_b] = "Node B data"
    graph[edge] = "Edge data"
    graph[child] = "Child data"

    print(graph[edge])
    print(graph[child])
    print(graph[node_a])
    print(graph[node_b])

    assert graph[edge] == graph[node_a][node_b] == "Edge data"
    assert graph[child] == "Child data"
    assert graph[node_a] == "Node A data"
    assert graph[node_b] == "Node B data"

    # Read the data back
    print(node_a)
    print(node_b)
    print(edge)
    print(child)

    graph.del_node(node_a)
    graph.del_node(node_b)
    graph.del_edge(edge)
    graph.del_child(child)

    print("Memory graph operations completed successfully.")

if __name__ == "__main__":
    main()