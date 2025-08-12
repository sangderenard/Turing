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
from ....cells.cellsim.api.saline import SalinePressureAPI as SalineHydraulicSystem
from ....cells.cell_consts import Cell

import json

# Mirror LinearCells flag used by older region manager
IMMUTABLE = 1 << 5

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
