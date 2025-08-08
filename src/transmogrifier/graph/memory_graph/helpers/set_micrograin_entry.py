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
class SetMicrograinEntry(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('object_id', ctypes.c_uint32),  # size of the grain in bytes
        ('object_address', ctypes.c_uint64),  # address of the grain in memory
    ]
