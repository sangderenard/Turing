
from enum import IntFlag, auto



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

# New Cell class definition for testing
class Cell:
    def __init__(self, stride, left, right, len, profile='default', leftmost=None, rightmost=None, label=None):
        self.len = len
        self.label = f"cell_{id(self)}" if label is None else label
        self.salinity = 0
        self.temperature = 0
        self.leftmost = leftmost
        self.rightmost = rightmost
        #buf = ctypes.create_string_buffer((self.len + 7)// 8)
        #self.obj_map = ctypes.addressof(buf)
        self.left = left
        self.right = right
        self.compressible = 1
        flags = DEFAULT_FLAG_PROFILES.get(profile, DEFAULT_FLAG_PROFILES['default'])
        self.l_wall_flags = flags['left_wall']
        self.r_wall_flags = flags['right_wall']
        self.c_flags      = flags['cell']
        self.system_flags = flags['system']
        self.l_solvent_permiability = 1
        self.r_solvent_permiability = 1
        self.injection_queue = 0
        self.resize_queue = 0
        self.stride = stride


        
        self.pressure = 0
        # Retain reference to avoid garbage collection
        self._buf = None#buf
    def bitview(self, buffer):
        return bytes(buffer[self.left:self.right])



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

LEFT_WALL = Cell(
    stride=1, left=0, right=0, len=0, profile='rigid_partition', rightmost=0, leftmost=0
)
RIGHT_WALL = Cell(
    stride=1, left=0, right=0, len=0, profile='rigid_partition'
)

STRIDE = 12
CELL_COUNT = 1
MASK_BITS_TO_DATA_BITS = 16
TEST_SIZE_STRIDE_TIMES_UNITS = (STRIDE ** 2 * 8 * STRIDE * CELL_COUNT) // ( 8 * STRIDE * CELL_COUNT)
assert TEST_SIZE_STRIDE_TIMES_UNITS % STRIDE == 0
# Simulator class to coordinate simulation steps