
from enum import IntFlag, auto
from .cellsim.constants import (
    DEFAULT_ELASTIC_K,
    DEFAULT_LP0,
    SALINITY_PER_DATA_UNIT,
)



# Left‐wall flags
class LeftWallFlags(IntFlag):
    LOCK       = auto()
    ELASTIC    = auto()
    PERMEABLE  = auto()
    REFLECTIVE = auto()

# Right‐wall flags
class RightWallFlags(IntFlag):
    LOCK       = auto()
    ELASTIC    = auto()
    PERMEABLE  = auto()
    REFLECTIVE = auto()

# Whole‐cell flags
class CellFlags(IntFlag):
    ZERO_SUM = auto()
    INERT    = auto()
    SOURCE   = auto()
    SINK     = auto()
    ADHESION = auto()

# System‐wide policies
class SystemFlags(IntFlag):
    AUTO_EXPAND    = auto()
    AUTO_SHRINK    = auto()
    PRESERVE_LOCKS = auto()
    LOG_EVENTS     = auto()
    TRACK_FRAG     = auto()
import ctypes
# New Cell class definition for testing
class Cell:
    def __init__(self, stride, left, right, len=None, profile='default', leftmost=None, rightmost=None, label=None):
        self.len = len if len is not None else right - left
        self.label = f"cell_{id(self)}" if label is None else label
        self.salinity = 0
        self.temperature = 0
        self.leftmost = leftmost
        self.rightmost = rightmost
        buf = ctypes.create_string_buffer((self.len + 7)// 8)
        self.obj_map = ctypes.addressof(buf)
        self.left = left
        self.right = right
        self.compressible = 1
        flags = DEFAULT_FLAG_PROFILES.get(profile, DEFAULT_FLAG_PROFILES['default'])
        self.l_wall_flags = flags['left_wall']
        self.r_wall_flags = flags['right_wall']
        self.c_flags      = flags['cell']
        self.system_flags = flags['system']
        self.l_solvent_permiability = DEFAULT_LP0
        self.r_solvent_permiability = DEFAULT_LP0
        self.injection_queue = 0
        self.resize_queue = 0
        self.stride = stride

        # --- Biophysical properties --------------------------------------
        # Treat ``salinity`` as the total solute quantity S_i.
        # ``volume`` tracks the current cell volume V_i and defaults to the
        # initial length of the region.  ``reference_volume`` (V0) is used for
        # turgor pressure calculations.  ``elastic_coeff`` (k) models the
        # Hookean response of the cell wall.  ``base_pressure`` (P0) represents
        # any baseline hydrostatic pressure in the system.
        self.volume = float(self.len)
        self.reference_volume = float(self.len)
        self.elastic_coeff = DEFAULT_ELASTIC_K
        self.salinity_per_data_unit = SALINITY_PER_DATA_UNIT
        self.base_pressure = 0.0
        self.pressure = 0.0
        self.concentration = 0.0

        # Retain reference to avoid garbage collection
        self._buf = None#buf
    def bitview(self, buffer):
        return bytes(buffer[self.left:self.right])

    def apply_proposal(self, proposal):
        self.left = proposal.left
        self.right = proposal.right
        self.leftmost = proposal.leftmost
        self.rightmost = proposal.rightmost

# ─── Default flag‑profiles ─────────────────────────────────────────────────────
DEFAULT_FLAG_PROFILES = {
    'default': {
        'left_wall':   LeftWallFlags.ELASTIC,
        'right_wall':  RightWallFlags.ELASTIC,
        'cell':        CellFlags.ZERO_SUM,
        'system':      SystemFlags.AUTO_EXPAND | SystemFlags.PRESERVE_LOCKS,
    },
    'rigid_partition': {
        'left_wall':   LeftWallFlags.LOCK,
        'right_wall':  RightWallFlags.LOCK,
        'cell':        CellFlags.ZERO_SUM | CellFlags.INERT,
        'system':      SystemFlags.PRESERVE_LOCKS,
    },
    'open_pipe': {
        'left_wall':   LeftWallFlags.PERMEABLE,
        'right_wall':  RightWallFlags.PERMEABLE,
        'cell':        CellFlags.ZERO_SUM,
        'system':      SystemFlags.AUTO_SHRINK,
    },
    'source_driven': {
        'left_wall':   LeftWallFlags.ELASTIC,
        'right_wall':  RightWallFlags.ELASTIC,
        'cell':        CellFlags.SOURCE,
        'system':      SystemFlags.AUTO_EXPAND,
    },
    'sink_driven': {
        'left_wall':   LeftWallFlags.ELASTIC,
        'right_wall':  RightWallFlags.ELASTIC,
        'cell':        CellFlags.SINK,
        'system':      SystemFlags.AUTO_SHRINK,
    },
    'adhesive_net': {
        'left_wall':   LeftWallFlags.ELASTIC,
        'right_wall':  RightWallFlags.ELASTIC,
        'cell':        CellFlags.ZERO_SUM | CellFlags.ADHESION,
        'system':      SystemFlags.TRACK_FRAG,
    },
}

STRIDE = 12
CELL_COUNT = 1
MASK_BITS_TO_DATA_BITS = 16
TEST_SIZE_STRIDE_TIMES_UNITS = (STRIDE ** 2 * 8 * STRIDE * CELL_COUNT) // (8 * STRIDE * CELL_COUNT)
assert TEST_SIZE_STRIDE_TIMES_UNITS % STRIDE == 0
