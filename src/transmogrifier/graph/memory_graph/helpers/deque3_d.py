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
