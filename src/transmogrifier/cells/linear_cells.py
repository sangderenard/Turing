# ────────────────────────────────────────────────────────────────
# linear_cells_redux.py  – fixed‑domain, stride‑aware 1‑D solver
#   v6  —  targeted injection that **steals solvent** before inflating
# ---------------------------------------------------------------------------
# CHANGES v6
#   • `inject_item` now tries to pull a solvent quantum from neighbouring
#     cells (edge‑to‑edge transfers) so the domain grows **only when no solvent
#     exists anywhere**.
#   • The helper `steal_solvent(cells, src, dst)` propagates one solvent
#     quantum from cell *src* to *dst* by iteratively shifting boundaries.
#   • All previous physics (micro_tick, bitmap, slip, etc.) is intact.
# ---------------------------------------------------------------------------
from __future__ import annotations
import ctypes, math, shutil, time, os, random, itertools

from typing import ByteString

import shutil


def shift_right(buf: bytearray, stride: int) -> None:
    """
    In-place bucket-brigade: push every quantum ONE slot to the right.
      • buf length must be k*stride.
      • leftmost stride is overwritten with zeros (solvent).
    """
    if not buf:            # empty cell
        return
    mv = memoryview(buf)
    last = mv[-stride:].tobytes()     # stash rightmost quantum
    for off in range(len(buf) - stride, 0, -stride):
        print(f"Shifting right: {off} {off-stride} {off+stride}")
        mv[off:off+stride] = mv[off-stride:off]
    mv[0:stride] = b'\x00' * stride   # new solvent
    # return dropped quantum (caller decides what to do with it)
    return last
def first_solvent(buf: bytearray, stride: int) -> int | None:
    """Return index of first solvent quantum or None."""
    for k in range(0, len(buf), stride):
        if all(b == 0 for b in buf[k:k+stride]):
            return k // stride
    return None
def last_solvent(buf: bytearray, stride: int) -> int | None:
    for k in range(len(buf)-stride, -1, -stride):
        if all(b == 0 for b in buf[k:k+stride]):
            return k // stride
    return None

def set_quantum(buf: bytearray, k: int, stride: int, value: int):
    fill = b'\x01' * stride if value else b'\x00' * stride
    buf[k*stride:(k+1)*stride] = fill
def shift_left(buf: bytearray, stride: int) -> None:
    """Mirror of shift_right."""
    if not buf:
        return
    mv = memoryview(buf)
    first = mv[0:stride].tobytes()
    for off in range(0, len(buf)-stride, stride):
        mv[off:off+stride] = mv[off+stride:off+2*stride]
    mv[-stride:] = b'\x00' * stride
    return first

def solvent_leftmost(buf: ByteString) -> bool:
    """True ⇢ first quantum all-zeros (= solvent)."""
    return all(b == 0 for b in buf[:1])
def load_bitmap_window(cell, unit_helper):
    """Populate cell.obj_map from the canonical grain bitmap once."""
    stride = cell.stride
    mv     = memoryview(cell.obj_map)
    for k in range(cell.len // stride):
        grain = unit_helper.grains_from_bytes(cell.left + k*stride)
        b, bit = unit_helper.grain_to_bitmap(grain)
        if unit_helper.bitmap[b] & (1 << bit):
            mv[k*stride:(k+1)*stride] = b'\x01'  # mark as object
        else:
            mv[k*stride:(k+1)*stride] = b'\x00'

# Utility: Least Common Multiple
def lcm(a, b):
    """Return the least common multiple of a and b."""
    return abs(a * b) // math.gcd(a, b) if a and b else 0

# 1. low‑level types & flags ────────────────────────────────────
LEN  = ctypes.c_uint32
STR  = ctypes.c_char_p
MAPPTR = ctypes.c_void_p    
FLAG = ctypes.c_uint16
PRES = ctypes.c_int32
FORC = ctypes.c_int32
FLT = ctypes.c_float
class LinearCells:

    def __init__(self, cell_specs, memory_graph=None, maxsize=2**31-1):
        self.memory_graph = memory_graph
        for spec in cell_specs:
            # if min is None, allow shrinking down to zero
            spec['min'] = 0 if spec.get('min') is None else spec['min']
            # if max is None, allow to grow arbitrarily (within Python’s int range)
            spec['max'] = maxsize if spec.get('max') is None else spec['max']
        print(f"[LinearCells] Initializing with {len(cell_specs)} cells")
        print(f"[LinearCells] Specs: {cell_specs}")
        self.labels = [spec['label'] for spec in cell_specs]
        self._initial_cells = [self.Cell( left=spec['left'], right=spec['right'],
            label=spec['label'],
            min=spec['min'], max=spec['max'], len=spec['len'], stride=spec['stride'],
            l_flags=spec['flags'], l_solvent_permiability=spec.get('l_solvent_permiability',0),
            r_flags=spec.get('r_flags',0), r_solvent_permiability=spec.get('r_solvent_permiability',0),
            obj_flags=spec.get('obj_flags',0), cell_pressure_fn=spec.get('cell_pressure_fn', None),
            cell_force_fn=spec.get('cell_force_fn', None), obj_force_fn=spec.get('obj_force_fn', None),
            obj_template=spec.get('obj_template', None), bitmap_fn=spec.get('bitmap_fn', None),
            density_fn=spec.get('density_fn', None), relocate_hook=spec.get('relocate_hook', None)
        ) for spec in cell_specs]
        self.cells = (self.Cell * len(self._initial_cells))(*[self._copy_cell(c) for c in self._initial_cells])
        self._tick = 0
        print('[LinearCells] Initialized with cells:')
        for cell in self.cells:
            print(f'  {cell.label}: {cell.left}..{cell.right}, len={cell.len}, stride={cell.stride}, flags={cell.flags}')
    def register_object_maps(self):
        for cell in self.cells:
            if cell.obj_map is None:
                # lazy init of obj_map
                cell.obj_map = self.memory_graph.hard_memory.unit_helper.bitmap.obtain_map_as_byte_string("bitmap", cell.left, cell.right - cell.left)

    def _copy_cell(self, c):
        print(f"[LinearCells] Copying cell {c.label.decode() if isinstance(c.label, bytes) else c.label}")
        nc = self.Cell()
        for f, _ in self.Cell._fields_:
            print(f"[LinearCells] Copying field {f} from {c} to {nc}")
            setattr(nc, f, getattr(c, f))
        return nc

    def relax(self, max_iter=1024):
        pre_relax_maps = [(cell.label, cell.obj_map if cell.obj_map else None) for cell in self.cells]

        for _ in range(max_iter):
            self._tick += 1
            if not self.micro_tick(self.cells):
                break

        post_relax_maps = [(cell.label, cell.obj_map) if cell.obj_map else None for cell in self.cells]


        return self.cells

    def apply_cell_pressure(self):
        """Invoke the pressure simulator to rebalance cell widths."""
        from .cell_pressure import Simulator
        sim = Simulator(self.cells)
        sim.evolution_tick(self.cells)
        return self.cells

    def manifest(self):
        manifest = []
        for i, label in enumerate(self.labels):
            old = {f: getattr(self._initial_cells[i], f) for f, _ in self.Cell._fields_}
            new = {f: getattr(self.cells[i], f) for f, _ in self.Cell._fields_}
            changed = any(old[k] != new[k] for k in old)
            manifest.append({'label': label, 'old': old, 'new': new, 'changed': changed})
        return manifest
    
    @staticmethod    
    def _decode_bitmap(ptr, length, stride):
        """
        Return a list of booleans, one per quantum in the cell.
        True  -> object present
        False -> solvent (empty)
        """
        if not ptr or length == 0:
            return []
        raw = ctypes.string_at(ptr, length)      # grab bytes
        quanta = []
        for k in range(0, length, stride):
            chunk = raw[k:k+stride]
            quanta.append(any(chunk))            # 1-bit presence test
        return quanta

    # ──────────────────────────────────────────────────────────────
    #  Convenience: dump_cells()  →  {cells, free_spaces, occupied_spaces}
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def dump_cells(lc: "LinearCells"):
        """
        Return a full snapshot that higher-level code can consume without
        worrying about internals.
        """
        cells = lc.quanta_map()               # detailed per-cell view

        free_spaces = [(c["label"], addr, size)
                       for c in cells for addr, size in c["free"]]
        occupied    = [(c["label"], addr, size)
                       for c in cells for addr, size in c["used"]]

        return {"cells": cells,
                "free_spaces": free_spaces,
                "occupied_spaces": occupied}

    # ──────────────────────────────────────────────────────────────
    #  NEW helper:  quanta_map()
    #      returns a list of per-cell dicts
    # ──────────────────────────────────────────────────────────────
    def quanta_map(self, *, coalesce_free=True):
        """
        Build a full address-space view of every quantum.

        Returns
        -------
        List[Dict] – one dict per cell, in the original order:
            {
              "label" : <cell.label>,
              "stride": <cell.stride>,            # quantum size
              "used"  : [(addr, size)],           # contiguous object blocks
              "free"  : [(addr, size)],           # contiguous solvent blocks
            }

        Notes
        -----
        * `addr` is an **absolute byte offset** (same base the cells use).
        * `size` is in bytes.
        * When *coalesce_free* is True (default) consecutive free quanta are
          merged into one range – handy for allocator logic.  Set it False if
          you want one-quantum granularity everywhere.
        """
        result = []

        for c in self.cells:
            cell_info = {"label": c.label,
                         "stride": c.stride,
                         "used":  [],
                         "free":  []}

            q_tot = c.len // c.stride if c.stride else 0
            if q_tot == 0:           # empty cell
                result.append(cell_info)
                continue

            # obtain bitmap (all-solvent if obj_map is None)
            raw = (ctypes.string_at(c.obj_map, c.len)
                   if c.obj_map else b"\x00" * c.len)

            # walk quanta left→right
            run_start = None
            run_used  = None  # None = no run, True = object run, False = solvent run

            def flush_run(idx):
                if run_start is None:
                    return
                addr  = c.left + run_start * c.stride
                size  = (idx - run_start) * c.stride
                key   = "used" if run_used else "free"
                cell_info[key].append((addr, size))

            for q in range(q_tot):
                chunk = raw[q * c.stride:(q + 1) * c.stride]
                this_used = any(chunk)          # True ⇒ object byte present
                if run_start is None:           # start first run
                    run_start, run_used = q, this_used
                elif this_used != run_used or not coalesce_free:
                    flush_run(q)
                    run_start, run_used = q, this_used

            flush_run(q_tot)  # last run
            result.append(cell_info)

        return result
    class Cell(ctypes.LittleEndianStructure):
        def __init__(self, memory_graph=None, left=None, right=None, offset=None, size=None, initial_position=None, label=None, min=0, max=0, len=0, stride=1, obj_template=None, bitmap_fn=None, obj_fn=None, density_fn=None, relocate_hook=None, l_flags=0, l_solvent_permiability=0.0, r_flags=0, r_solvent_permiability=0.0, obj_flags=0, cell_pressure_fn=None, cell_force_fn=None, obj_force_fn=None):
            self.left = int(initial_position[0] if initial_position else left if left else 0)
            self.right = int(initial_position[1] if initial_position else right if right else 0)
            self.memory_graph = memory_graph
            self.left = int(offset) if offset else self.left
            self.right = int(size - offset) if size else self.right
            self.min = int(min)
            self.max = int(max)
            self.len = int(len)
            self.obj_template = obj_template() if obj_template else None
            self.object_size = ctypes.sizeof(self.obj_template) if self.obj_template else 0
            self.stride = lcm(self.object_size, stride) if self.object_size else stride
            # bitmap_fn should accept (memory_graph, label) -> raw bytes
            self.bitmap_fn = bitmap_fn if bitmap_fn else None
            self.density_fn = density_fn if density_fn else None
            self.relocate_hook = relocate_hook if relocate_hook else None
            self.l_flags = l_flags
            self.l_solvent_permiability = l_solvent_permiability
            self.r_flags = r_flags
            self.r_solvent_permiability = r_solvent_permiability
            self.obj_flags = obj_flags
            self.cell_pressure_fn = cell_pressure_fn() if cell_pressure_fn else None
            self.cell_force_fn = cell_force_fn() if cell_force_fn else None
            self.obj_force_fn = obj_force_fn() if obj_force_fn else None
            self.memory_graph = memory_graph
            # ensure label is bytes for c_char_p field
            self.label = label if label is not None else -1
            # initialize obj_map with region slice
            self.obj_map = None
            self.solvent_leftmost = solvent_leftmost
            self.shift_right      = lambda c: shift_right(c.obj_map, c.stride)
            self.shift_left       = lambda c: shift_left (c.obj_map, c.stride)
            print(f"[LinearCells.Cell] Created cell {self.label} with left={self.left}, right={self.right}, len={self.len}, stride={self.stride}, flags={self.flags}")
        @property
        def flags(self) -> int:
            "Union of L/R/object flags – lets legacy code keep working."
            return self.l_flags | self.r_flags | self.obj_flags

        _fields_ = [
            ("left",    ctypes.c_uint64),  # left boundary (byte offset)
            ("right",   ctypes.c_uint64),  # right boundary (byte offset)
            ("label",   LEN),
            ("obj_map", MAPPTR),
            ("min",     LEN),
            ("max",     LEN),
            ("len",     LEN),
            ("stride",  LEN),
            
            ("obj_template", ctypes.c_void_p),
            ("obj_fn", ctypes.c_void_p),
            ("density_fn", ctypes.c_void_p),
            ("relocate_hook", ctypes.c_void_p),
            ("l_flags", FLAG),  # left flags
            ("l_solvent_permiability", FLT),
            ("r_flags", FLAG),  # right flags
            ("r_solvent_permiability", FLT),
            ("obj_flags",   FLAG),
            ("cell_pressure_fn", ctypes.c_void_p),
            ("cell_force_fn", ctypes.c_void_p),
            ("pres", ctypes.c_int32),  # pressure
            ("force", ctypes.c_int32),  # force
            ("obj_force_fn", ctypes.c_void_p),
        ]

    LOCK      = 1 << 0
    SLIP_OK   = 1 << 1
    ONE_WAY_L = 1 << 2
    ONE_WAY_R = 1 << 3
    ELASTIC   = 1 << 4
    IMMUTABLE = 1 << 5

    FORCE_THRESH = 512
    def get_state(self) -> list[dict]:
        """Return for each cell: label, left/right boundaries and final bitmap."""
        out = []
        for c in self.cells:
            out.append({
                'label':  c.label,
                'left':   c.left,
                'right':  c.right,
                'bitmap': c.obj_map
            })
        return out
    # 2. helpers ────────────────────────────────────────────────────
    # ─────────────── interface helpers ────────────────
    def _interface_permeability(self, left:'LinearCells.Cell',
                                 right:'LinearCells.Cell') -> float:
        """
        Effective solvent permeability of the shared wall.
        Each side contributes 0‥1; stacking is the *sum*
        (cap at 1.0 for determinism).
        """
        return min(
            left .r_solvent_permiability +
            right.l_solvent_permiability , 1.0)

    def _interface_blocks_flow(self,
                               left:'LinearCells.Cell',
                               right:'LinearCells.Cell',
                               left_to_right:bool) -> bool:
        """
        Check one-way valves on the interface.

        Flow L→R is blocked if *right* wall has ONE_WAY_L.  
        Flow R→L is blocked if *left*  wall has ONE_WAY_R.
        """
        if left_to_right:
            return bool(right.l_flags & self.ONE_WAY_L)
        else:
            return bool(left .r_flags & self.ONE_WAY_R)

    def quanta(self, c: 'LinearCells.Cell') -> int:
        return c.len // c.stride

    def _solvent_leftmost(self, cell):
        if not cell.obj_map:
            return False
        buf = ctypes.string_at(cell.obj_map, cell.len)
        return solvent_leftmost(buf)


    # 3. solvent stealing helper ───────────────────────────────────

    def _edge_transfer(self, donor:'LinearCells.Cell', recv:'LinearCells.Cell'):
        """Move one solvent quantum right-wards (donor ⇒ recv)."""
        self._shift_right(donor)
        donor.len -= donor.stride
        donor.right = donor.left + donor.len
        self._shift_left(recv)
        recv.len  += donor.stride
        recv.left -= donor.stride
        donor.right = donor.left + donor.len
        # ---- relocation hooks ----------------------------------
        if callable(donor.relocate_hook):
            donor.relocate_hook(donor, +1)   # +1 ⇒ gave mass to the right
        if callable(recv .relocate_hook):
            recv .relocate_hook(recv , -1)   # -1 ⇒ received mass from left
    def _shift_right(self, cell: 'LinearCells.Cell'):
        if cell.len == 0 or not cell.obj_map:
            return
        raw = ctypes.string_at(cell.obj_map, cell.len)
        arr = bytearray(raw)
        print(f"Shifting right: {cell.label} {cell.left}..{cell.right}, len={cell.len}, arr len={len(arr)}, stride={cell.stride}")
        shift_right(arr, cell.stride)
        buf = (ctypes.c_char * len(arr)).from_buffer(arr)
        ctypes.memmove(cell.obj_map, buf, len(arr))

    def _shift_left(self, cell: 'LinearCells.Cell'):
        if cell.len == 0 or not cell.obj_map:
            return
        raw = ctypes.string_at(cell.obj_map, cell.len)
        arr = bytearray(raw)
        shift_left(arr, cell.stride)
        buf = (ctypes.c_char * len(arr)).from_buffer(arr)
        ctypes.memmove(cell.obj_map, buf, len(arr))



    def count_from(self, bytes, offset, direction, n=64, m=64, k=1, j=8): #n is depth, m is switches, j is the stride, k is the desired units of empty space
        """Count bits from a side n deep and report pattern in tuples, none for missing instances."""
        return_pattern = []
        whole_spaces = set()  # track whole spaces
        h = 0  # counter for consecutive zeros
        
        for i in range(n):
            most_recent_value = None
            if direction == 'left':
                idx = offset - i
                bit = (bytes[idx // 8] >> (idx % 8)) & 1
            elif direction == 'right':
                idx = offset + i
                bit = (bytes[idx // 8] >> (8 - (idx % 8))) & 1
            else:
                raise ValueError("Direction must be 'left' or 'right'")
            if idx < 0 or idx >= len(bytes):
                return return_pattern
            if not bit:
                k -= 1
                if return_pattern[-1][0] == 0:
                    h += 1
                    if h % j == 0:
                        
                        whole_spaces.add(idx)

            else:
                h = 0
            if bit == most_recent_value:
                return_pattern[-1][1] += 1
            else:
                return_pattern.append((bit, 1))
                m -= 1
                most_recent_value = bit
            n -= 1
            if n <= 0 or k // j > 0 or m <= 0:
                break
        
        return return_pattern, whole_spaces
        


    def minimize(self, cells):
        # you take the object that's going to get hit by the boundary move and you throw it back into the first empty spot in its line, you track four positions at every  frame of consideration, the first  objects and the first holes after the objects from either side, when none of some category exist you know the closed system needs an ejection of solvent or an increase in size

        #all we need to do is mod every cell by its stride and not cross boundaries

        for i, cell in enumerate([None] + cells + [None]):
            #calculate forces into pressures
            #add to volumetric pressure
            #keep the metaphor loose because this is actually a simple swap algorithm
            if cell is None:
                continue
            if cell.obj_map is None:
                continue
            # load the bitmap into a bytearray
            raw = ctypes.string_at(cell.obj_map, cell.len)

            if cell.left != cell.right:
                left_pattern, left_gaps = self.count_from(raw, cell.left, 'right', 4)
                right_pattern, right_gaps = self.count_from(raw, cell.right - 1, 'left', 4)


                if len(left_pattern) == 0:
                    cell.compressible = raw[0] == 0

                if cell.compressible == 0:
                    presure = 0
                    cell.l_flags = cell.l_flags | self.LOCK
                    cell.r_flags = cell.r_flags | self.LOCK
                else:
                    left_resistive_force = len(left_pattern) * cell.l_solvent_permiability
                    right_resistive_force = len(right_pattern) * cell.r_solvent_permiability
                    presure += left_resistive_force + right_resistive_force

                    left_neighbor_stride_equiv = (cells[i-1].stride + cell.stride - 1) // cell.stride
                    right_neighbor_stride_equiv = (cells[i+1].stride + cell.stride - 1) // cell.stride

                    if right_neighbor_stride_equiv < len(right_gaps):
                        cell.r_flags = cell.r_flags | self.ELASTIC
                        presure -= len(right_gaps) / right_neighbor_stride_equiv

                    if left_neighbor_stride_equiv < len(left_gaps):
                        cell.l_flags = cell.l_flags | self.ELASTIC
                        presure -= len(left_gaps) / left_neighbor_stride_equiv
                known_gaps = set(left_gaps) | set(right_gaps)
                pressure += cell.injection_queue - len(known_gaps)

                pressure *= cell.stride

                if abs(pressure) > self.FORCE_THRESH:
                    cell.resize_queue = int(pressure / self.FORCE_THRESH)

                system_pressure += pressure




    def steal_solvent(self, cells, src: int, dst: int):
        """Propagate exactly **one** solvent quantum from *src* to *dst* by walking
        edge‑to‑edge. src < dst assumed. Returns True if successful."""
        if src == dst:
            return False
        # verify src has solvent at wall (create by internal swap if needed)
        if not self._solvent_leftmost(cells[src]):
            return False
        for i in range(src, dst):
            if cells[i+1].len + cells[i].stride > cells[i+1].max:
                return False  # receiver cannot grow
            self._edge_transfer(cells[i], cells[i+1])
        return True

    # 4. injection helpers ─────────────────────────────────────────
    _inject_seq = itertools.count(1)  # global ID generator


    def inject_item(self, cells, inject_map, idx: int):
        """Inject one object quantum into cell *idx*.
        Strategy:
        1) If target has solvent, convert one solvent to object.
        2) Else search outward (left then right) for a cell that *currently has a
            solvent quantum at its boundary* and propagate it, edge‑to‑edge, toward
            the target (stealing). Only one quantum is moved.
        3) If no solvent exists anywhere, *inflate* the target by one stride.
        """
        c_tgt = cells[idx]
        stride = c_tgt.stride
        # number of quanta in the cell
        q_tot  = self.quanta(c_tgt)
        # set bits-per-quantum (default 1, adjust as needed)
        bits_per_quantum = getattr(self, "bits_per_quantum", 1)
        total_bits = q_tot * bits_per_quantum
        num_bytes = (total_bits + 7) // 8

        # Read the current bitmap from shared memory
        raw = ctypes.string_at(c_tgt.obj_map, num_bytes)
        bmap = int.from_bytes(raw, 'little')

        mask = (1 << total_bits) - 1
        # solvent_bits: bits still zero indicate solvent quanta.
        solvent_bits = (~bmap) & mask
        if solvent_bits:
            # convert the right‑most solvent quantum to object for determinism.
            k = solvent_bits.bit_length() - 1
            bmap |= (1 << k)
            new_raw = bmap.to_bytes(num_bytes, 'little')
            ctypes.memmove(c_tgt.obj_map, new_raw, num_bytes)
        else:
            # attempt to steal from left then right …
            stole = False
            # 4.a look left
            for j in range(idx-1, -1, -1):
                if self._solvent_leftmost(cells[j]) and self.steal_solvent(cells, j, idx):
                    stole = True
                    break
            # 4.b look right (mirror logic)
            if not stole:
                for j in range(idx+1, len(cells)):
                    if not self._solvent_leftmost(cells[j]):
                        continue
                    # move solvent left by chaining transfers
                    for k in range(j, idx, -1):
                        if cells[k-1].len + stride > cells[k-1].max:
                            break
                        self._edge_transfer(cells[k], cells[k-1])
                    else:
                        stole = True
                        break
            # 4.c if still no solvent, inflate the target cell
            if not stole:
                c_tgt.len += stride
                c_tgt.right = c_tgt.left + c_tgt.len
                c_tgt.max = max(c_tgt.max, c_tgt.len)
                # recompute quanta, bits, and memory length after inflation
                new_q_tot = self.quanta(c_tgt)
                new_total_bits = new_q_tot * bits_per_quantum
                new_num_bytes = (new_total_bits + 7) // 8
                # Read existing bitmap, extend if necessary (assumed zero-filled beyond original length)
                raw = ctypes.string_at(c_tgt.obj_map, num_bytes)
                bmap = int.from_bytes(raw, 'little')
                # Set the bit corresponding to the new quantum (at previous q_tot position)
                bmap |= (1 << (q_tot * bits_per_quantum))
                new_raw = bmap.to_bytes(new_num_bytes, 'little')
                ctypes.memmove(c_tgt.obj_map, new_raw, new_num_bytes)
            else:
                # after steal, target now has a solvent quantum at the left edge;
                # convert it (treat as above)
                self.inject_item(cells, inject_map, idx)
                return  # avoid double counting
        inj_id = next(self._inject_seq)
        inject_map[idx].append(inj_id)
        return inj_id

    def _swap_cells(self, a: 'LinearCells.Cell', b: 'LinearCells.Cell'):
        tmp = self.Cell()
        ctypes.memmove(ctypes.byref(tmp), ctypes.byref(a), ctypes.sizeof(self.Cell))
        ctypes.memmove(ctypes.byref(a), ctypes.byref(b), ctypes.sizeof(self.Cell))
        ctypes.memmove(ctypes.byref(b), ctypes.byref(tmp), ctypes.sizeof(self.Cell))



    def pretty_print_state(self, *, width=80, show_labels=True, show_metrics=True):
        term_width = shutil.get_terminal_size((width, 20)).columns
        total_bytes = sum(cell.right - cell.left for cell in self.cells)
        scale = term_width / total_bytes

        # Map one line of visualization
        def render_cell(cell: LinearCells.Cell):
            cell_bytes = cell.right - cell.left
            vis_width = max(1, int(cell_bytes * scale))

            bar = []
            # create a memoryview over the bitmap pointer if available
            if cell.obj_map:
                # region size in bytes determines bitmap length
                region_size = cell.right - cell.left
                try:
                    mv = memoryview(ctypes.string_at(cell.obj_map, region_size))
                except Exception:
                    mv = None
            else:
                mv = None
            
            stride = cell.stride
            if stride == 0:
                stride = 1
            for i in range(0, vis_width * stride, stride):
                if mv is None or i >= len(mv):
                    block = "░"  # unknown
                    color = "\033[90m"  # dim
                elif all(b == 0 for b in mv[i:i+stride]):
                    block = "░"  # solvent
                    color = "\033[94m"  # blue
                else:
                    block = "█"  # object
                    color = "\033[92m"  # green
                bar.append(f"{color}{block}\033[0m")

            return ''.join(bar)

        print("\n" + "=" * term_width)
        print(f"\033[1mMemory Layout Snapshot — Tick {self._tick}\033[0m")
        print("-" * term_width)

        for cell in self.cells:
            label = cell.label.decode() if isinstance(cell.label, bytes) else str(cell.label)
            bar = render_cell(cell)
            metrics = f" len={cell.len} min={cell.min} max={cell.max} Δ={cell.max - cell.len}"
            if show_labels:
                print(f"{label:<16} {bar}")
            if show_metrics:
                print(f"{'':<16} \033[90m{metrics}\033[0m")

        print("=" * term_width + "\n")

    def micro_tick(self, cells: "ctypes.Array['LinearCells.Cell']") -> bool:
        """Run one physics micro‑tick.
        Rules
        -----
        • Unlimited solvent stride‑moves.
        • At most **one internal object↔solvent swap** _and_ at most **one whole‑cell
        slip_ per tick.
        • If an elastic wall wants to move solvent but finds an object on the wall
        it may perform **one internal swap** to bring a solvent quantum to the
        border (counts toward the object‑swap budget).
        Returns
        -------
        True if anything changed this tick.
        """

        changed       = False
        n             = len(cells)
        obj_swapped   = False   # internal object<->solvent swap budget (1/tick)
        cell_swapped  = False   # whole‑cell slip budget (1/tick)

        # 1. pressure ----------------------------------------------
        for c in cells:
            c.pres = -(c.min - c.len) if c.len < c.min else (c.len - c.max if c.len > c.max else 0)

        self.pretty_print_state(show_labels=False, show_metrics=True)
        # 2. elastic solvent transfer ------------------------------
        for i in range(n-1):
            A, B = cells[i], cells[i+1]
            dP   = A.pres - B.pres
            if dP == 0:
                continue

            left_to_right = dP > 0
            donor, recv   = (A, B) if left_to_right else (B, A)

            # ----- eligibility checks -----------------------------
            if not (donor.flags & self.ELASTIC and recv.flags & self.ELASTIC):
                continue
            if self._interface_blocks_flow(A, B, left_to_right):
                continue
            if self._interface_permeability(A, B) == 0.0:
                continue
            if donor.len - donor.stride < donor.min:
                continue
            if recv.len  + donor.stride > recv .max:
                continue

            # ensure wall has solvent (may use 1 swap budget)
            if not self.solvent_leftmost(donor):
                if obj_swapped:
                    continue
                mask  = (1 << self.quanta(donor)) - 1
                free  = (~donor.obj_map) & mask
                if not free:
                    continue
                k = (free & -free).bit_length() - 1
                donor.obj_map ^= (1 << 0) | (1 << k)
                obj_swapped = True
                changed     = True

            # ----- permeability scaling: probabilistic gate -------
            if random.random() >= self._interface_permeability(A, B):
                continue  # wall “resisted” this tick

            # ----- perform the transfer ---------------------------
            self._edge_transfer(donor, recv)
            changed = True
        # 3. whole-cell slip ---------------------------------------
        for i in range(n-1):
            if cell_swapped:
                break
            A, B = cells[i], cells[i+1]
            if abs(A.force - B.force) < self.FORCE_THRESH:
                continue
            if not (A.flags & self.SLIP_OK and B.flags & self.SLIP_OK):
                continue
            self._swap_cells(A, B)
            if callable(A.relocate_hook):
                A.relocate_hook(A, +0)  # 0 ⇒ full-cell move
            if callable(B.relocate_hook):
                B.relocate_hook(B, +0)
            changed      = True
            cell_swapped = True


# ────────────────────────────────────────────────────────────────
# Live Pygame Visualiser for LinearCells
# ----------------------------------------------------------------
import pygame, sys, time

VISUALISE   = True          # master toggle
SCALE_X     = 1.0           # pixels per byte  (auto-scaled below)
ROW_H       = 28            # pixels per cell row
GRID_COLOUR = (180,180,180) # light grey grid lines
COL_SOLVENT = (200,225,255) # pale blue
COL_DATA    = ( 30,144,255) # bright blue
FPS         = 60

# ────────────────────────────────────────────────────────────────
# Live injection driver (manual keys 0-7 or auto every N seconds)
# ----------------------------------------------------------------
INJECT_KEYS = {pygame.K_0: 0, pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3,
               pygame.K_4: 4, pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7}
AUTO_INJECT_EVERY = 0.10          # seconds (set 0 to disable)


class _LCVisual:
    def __init__(self, cells):
        tot_span = max(c.right for c in cells) - min(c.left for c in cells)
        global SCALE_X
        SCALE_X = 1200 / max(1, tot_span)       # fit into ~1200 px window
        w = int(tot_span * SCALE_X) + 20
        h = len(cells) * ROW_H + 20

        pygame.init()
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("LinearCells memory layout")
        self.clock  = pygame.time.Clock()
        self.cells  = cells

    def draw(self):
        self.screen.fill((0, 0, 0))
        base_left = min(c.left for c in self.cells)

        for row, c in enumerate(self.cells):
            y0 = 10 + row * ROW_H
            stride = max(1, c.stride)          # avoid div-by-zero

            # paint each quantum separately so solvent/data differ
            if c.obj_map and c.len:
                raw = (ctypes.c_ubyte * c.len).from_address(c.obj_map)
            else:
                raw = None

            n_q = c.len // stride if stride else 0
            for q in range(n_q):
                byte_off = c.left + q * stride
                x0 = 10 + int((byte_off - base_left) * SCALE_X)
                x1 = 10 + int((byte_off + stride - base_left) * SCALE_X)
                w  = max(1, x1 - x0)

                colour = COL_DATA if (raw and raw[q * stride] != 0) else COL_SOLVENT
                pygame.draw.rect(self.screen, colour,
                                 pygame.Rect(x0, y0 + 4, w, ROW_H - 8))

            # cell boundary lines (after quanta so they stay visible)
            xL = 10 + int((c.left  - base_left) * SCALE_X)
            xR = 10 + int((c.right - base_left) * SCALE_X)
            pygame.draw.line(self.screen, GRID_COLOUR, (xL, y0), (xL, y0 + ROW_H - 1))
            pygame.draw.line(self.screen, GRID_COLOUR, (xR, y0), (xR, y0 + ROW_H - 1))

            # optional: light stride grid inside region
            # comment out if busy
            if stride and stride * SCALE_X > 4:
                x = xL + int(stride * SCALE_X)
                while x < xR - 1:
                    pygame.draw.line(self.screen, (60, 60, 60),
                                     (x, y0 + 4), (x, y0 + ROW_H - 5))
                    x += int(stride * SCALE_X)

            # label at left edge
            font = pygame.font.Font(None, 18)
            self.screen.blit(font.render(str(c.label), True, (255, 255, 255)),
                             (xL + 4, y0 + 4))

        pygame.display.flip()
        self.clock.tick(FPS)


_vis = None

# ----------- wrap micro_tick so every real solver step is painted ----------
_orig_micro = LinearCells.micro_tick
def _vis_micro(self, cells):
    global _vis
    if VISUALISE and _vis is None:
        _vis = _LCVisual(cells)

    # let the solver move things first
    res = _orig_micro(self, cells)

    if VISUALISE:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        _vis.draw()

    return res
LinearCells.micro_tick = _vis_micro
# ---------------------------------------------------------------------------

# ========================================================================
#  DEMO HARNESS  – no helper overrides, uses real inject_item()
# ========================================================================
if __name__ == "__main__":
    import ctypes, pygame, sys, time, random

    # --------------------------------------------------------------------
    # 1. create spec, solver, empty obj_maps (all solvent)
    # --------------------------------------------------------------------
    specs = [
        dict(left=0,   right=128,  label=0, len=128, stride=128, flags=0, min=0, max=128),
        dict(left=128, right=256,  label=1, len=128, stride=64 , flags=0),
        dict(left=256, right=512,  label=2, len=256, stride=32 , flags=0),
        dict(left=512, right=768,  label=3, len=256, stride=16 , flags=0),
        dict(left=768, right=896,  label=4, len=128, stride=8  , flags=0),
        dict(left=896, right=1024, label=5, len=128, stride=8  , flags=0),
        dict(left=1024,right=1024, label=6, len=0  , stride=1  , flags=0),
        dict(left=1024,right=1024, label=7, len=0  , stride=0  , flags=32),
    ]

    lc = LinearCells(specs)

    # --------------------------------------------------------------------
    # 1-bis. allocate ONE shared bitmap that spans the whole address space
    # --------------------------------------------------------------------
    # give empty cells at least one quantum so .right is > .left
    for c in lc.cells:
        if c.len == 0:
            c.len   = max(1, c.stride)
            c.right = c.left + c.len

    max_extent = max(c.right for c in lc.cells)          # right-most byte
    SharedBuf  = (ctypes.c_ubyte * max_extent)
    shared_buf = SharedBuf()                             # zero-filled solvent

    # keep a reference alive so the buffer isn’t GC’d
    lc._shared_bitmap = shared_buf

    for c in lc.cells:
        # obj_map = address of the first byte that belongs to this cell
        offset = c.left
        c.obj_map = ctypes.c_void_p(ctypes.addressof(shared_buf) + offset)

    inject_map = [[] for _ in specs]         # history for inject_item()

    # --------------------------------------------------------------------
    # 2. visualiser
    # --------------------------------------------------------------------
    vis = _LCVisual(lc.cells)
    lc.relax()                               # settle once before UI

    KEY_TO_CELL = {
        pygame.K_0: 0, pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3,
        pygame.K_4: 4, pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7,
    }

    AUTO_EVERY  = 3.0
    next_auto   = time.time() + AUTO_EVERY

    while True:
        now = time.time()

        # handle window / keyboard
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key in KEY_TO_CELL:
                idx = KEY_TO_CELL[ev.key]
                lc.inject_item(lc.cells, inject_map, idx)   # real call
                lc.relax()

        # auto-inject (optional)
        if now >= next_auto:
            tgt = random.randint(0, len(specs) - 2)         # skip sentinel 7
            lc.inject_item(lc.cells, inject_map, tgt)
            lc.relax()
            next_auto = now + AUTO_EVERY

        vis.draw()