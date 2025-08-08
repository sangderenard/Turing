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
