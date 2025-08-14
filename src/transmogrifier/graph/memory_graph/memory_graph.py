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
from .helpers.meta_graph_edge import META_GRAPH_TRANSFER_BUFFER_SIZE

import json

# Mirror LinearCells flag used by older region manager
IMMUTABLE = 1 << 5

from .helpers import *  # expose helper classes into this module's namespace

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
    def __init__(
        self,
        size=0,
        bit_width=32,
        encoding="gray",
        meta_graph_root=0,
        generative_parent=0,
        *,
        dynamic=False,
    ):
        self.capsid_id = (1 + id(generative_parent) + uuid4().int) % 2**32
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

        # Initial dynamic state must align between the graph and its backing
        # memory.  Set ``_dynamic`` prior to memory construction and propagate
        # the flag into the BitTensorMemory instance.
        self._dynamic = bool(dynamic)
        self.hard_memory = BitTensorMemory(
            self.hard_memory_size, self, dynamic=self._dynamic
        )  # default memory size
        #self.hard_memory.region_manager.register_object_maps()
        self.meta_graph_root = meta_graph_root  # root of the meta graph
        self.generative_parent = generative_parent
        self.lock_manager = None  # placeholder for lock manager

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

    # -- Dynamic flag synchronisation ---------------------------------
    @property
    def dynamic(self):
        """Whether this graph may grow/shrink at runtime."""
        return self._dynamic

    @dynamic.setter
    def dynamic(self, value):
        self._dynamic = bool(value)
        if getattr(self, "hard_memory", None) is not None:
            self.hard_memory.dynamic = self._dynamic
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
        # System adaptation: set salinity and balance before allocation
        edge_size = ctypes.sizeof(EdgeEntry)
        self.hard_memory.region_manager.cells[3].salinity += edge_size  # 3 = 'edge'
        SalineHydraulicSystem.run_saline_sim(self.hard_memory.region_manager)
        free_space = self.hard_memory.find_free_space("edge", edge_size)

        if free_space is not None:
            self.hard_memory.write(free_space, ctypes.string_at(ctypes.addressof(edge), edge_size))
        else:
            raise MemoryError("Failed to allocate space for edge after balancing.")
        self.edge_count += 1

        return (src, dst)

    def add_meta_parent(self, parent_id):
        # THESE ARE NOT THE SAME AS THE NODES AND EDGES IN THE GRAPH
        # THESE ARE META GRAPH CONNECTIONS FOR THE GRAPH OF GRAPH
        # THEY ONLY EXIST AS NO-SIBLING TABLE RELATIONSHIPS
        """
        Add a meta parent node to the graph.
        """
        # System adaptation: set salinity and balance before allocation
        parent_size = ctypes.sizeof(MetaGraphEdge)
        self.hard_memory.region_manager.cells[4].salinity += parent_size  # 4 = 'parent'
        SalineHydraulicSystem.run_saline_sim(self.hard_memory.region_manager)
        space = self.hard_memory.find_free_space("parent", parent_size)
        if space is None:
            self.emergency_reference = self.hard_memory.allocate_block(parent_size, (self.LINE_P, self.LINE_C))
            if self.emergency_reference == BitTensorMemory.ALLOCATION_FAILURE:
                raise MemoryError("Failed to allocate emergency reference for parent node")
            space = self.emergency_reference
        edge = MetaGraphEdge()
        edge.local_capsid_ref = self.capsid_id
        edge.remote_capsid_ref = parent_id

        self.parent_count += 1
        self.hard_memory.write(space, ctypes.string_at(ctypes.addressof(edge), parent_size))
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
                # System adaptation: set salinity and balance before allocation
                header_size = self.header_size
                self.hard_memory.region_manager.cells[0].salinity += header_size  # 0 = 'header'
                SalineHydraulicSystem.run_saline_sim(self.hard_memory.region_manager)
                free_space = self.hard_memory.find_free_space("header", header_size)
                print(f"Debugging: Free space for header serialization: {free_space}")
                if free_space is None:
                    raise MemoryError("currently unhandled lack of memory for header after balancing")
                print(f"Debugging: Found free space at {free_space} for header serialization.")
                self.hard_memory.write(free_space, self.serialize_header())
                self.n_start = free_space + header_size

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
            # System adaptation: set salinity and balance before allocation
            child_size = ctypes.sizeof(MetaGraphEdge)
            self.hard_memory.region_manager.cells[5].salinity += child_size  # 5 = 'child'
            SalineHydraulicSystem.run_saline_sim(self.hard_memory.region_manager)
            free_space = self.hard_memory.find_free_space("child", child_size)
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
        # System adaptation: set salinity and balance before allocation
        child_size = ctypes.sizeof(MetaGraphEdge)
        self.hard_memory.region_manager.cells[5].salinity += child_size  # 5 = 'child'
        SalineHydraulicSystem.run_saline_sim(self.hard_memory.region_manager)
        free_space = self.hard_memory.find_free_space("child", child_size)
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
        cell_dump = self.hard_memory.region_manager.dump_cells()

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
            from scipy.optimize import linear_sum_assignment as hungarian # type: ignore
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
        # System adaptation: set salinity and balance before allocation
        node_size = ctypes.sizeof(NodeEntry)
        self.hard_memory.region_manager.cells[2].salinity += node_size  # 2 = 'node'
        SalineHydraulicSystem.run_saline_sim(self.hard_memory.region_manager)
        new_node_slot = self.hard_memory.find_free_space("node", node_size)
        if new_node_slot == BitTensorMemory.ALLOCATION_FAILURE:
            raise MemoryError("Failed to add node: no free space available after balancing")

        if new_node_slot is None:
            raise MemoryError("Allocation returned None after balancing")

        status = self.hard_memory.write(new_node_slot, node_bytes)
        if status == BitTensorMemory.ALLOCATION_FAILURE:
            raise MemoryError("Failed to write node entry to hard memory")

        self.node_count += 1

        return node_id
        
