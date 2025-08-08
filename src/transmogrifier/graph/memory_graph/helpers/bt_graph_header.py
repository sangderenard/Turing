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
