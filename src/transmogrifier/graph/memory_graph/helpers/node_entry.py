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

import json

# Mirror LinearCells flag used by older region manager
IMMUTABLE = 1 << 5
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
