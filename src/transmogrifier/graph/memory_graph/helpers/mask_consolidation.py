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
from .bt_graph_header import BTGraphHeader

import json

# Mirror LinearCells flag used by older region manager
IMMUTABLE = 1 << 5
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
