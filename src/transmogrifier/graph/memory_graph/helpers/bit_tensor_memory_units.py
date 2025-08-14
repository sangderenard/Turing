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
from .mask_consolidation import MaskConsolidation
from .bt_graph_header import BTGraphHeader
from .node_entry import NodeEntry
from .edge_entry import EdgeEntry
from .meta_graph_edge import MetaGraphEdge
from .set_micrograin_entry import SetMicrograinEntry

import json

# Mirror LinearCells flag used by older region manager
IMMUTABLE = 1 << 5
class BitTensorMemoryUnits:
    # Local defaults to avoid circular import on BitTensorMemory
    DEFAULT_GRAIN = 512
    DEFAULT_CHUNK = 4096
    
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
            grain_size = BitTensorMemoryUnits.DEFAULT_GRAIN
        if chunk_size is None:
            chunk_size = BitTensorMemoryUnits.DEFAULT_CHUNK

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
    from ..memory_graph import BitTensorMemoryGraph
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
