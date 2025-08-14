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
# Avoid importing via the parent ``memory_graph`` package to prevent
# circular initialisation when ``memory_graph`` itself imports helpers.
# ``BitTensorMemory`` lives alongside this module, so we can import it
# directly from the local helper package.
from .bit_tensor_memory import BitTensorMemory

import json

# Mirror LinearCells flag used by older region manager
IMMUTABLE = 1 << 5
class BitTensorMemoryDAGHelper:
    def __init__(self, bit_tensor_memory, chunk_size=8, bit_width=32):
        self.bit_tensor_memory = bit_tensor_memory
        self.chunk_size = chunk_size
        self.bit_width = bit_width
        self.hard_memory_size = sys.getsizeof(bit_tensor_memory.data)
        # Use the provided BitTensorMemory instance to avoid redundant
        # allocation and circular init concerns.
        self.hard_memory = bit_tensor_memory
        self.lock_manager = None  # placeholder for lock manager
        self.envelope_domain = (0, self.hard_memory_size // self.chunk_size)
        self.envelope_size = self.hard_memory_size // self.chunk_size

    def merge(self, one, theother):
        """
        Merge two procedural and memory concurrency dags for no-lock
        memory moving.
        """
        
        return self
