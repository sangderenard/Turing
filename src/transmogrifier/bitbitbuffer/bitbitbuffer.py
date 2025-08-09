from typing import Any, Sequence, Optional, Callable, Union, List
import logging

from .helpers.rawspan import _RawSpan
from .helpers.data_access import BitBitBufferDataAccess
from .helpers.bitbitindex import BitBitIndex
from .helpers.bitbitindexer import BitBitIndexer
from .helpers.pidbuffer import PIDBuffer
from .helpers.cell_proposal import CellProposal

class BitBitBuffer:
    def __init__(self, data_size=None, mask_size=None, bitsforbits=None, caster=None, make_pid=True):
        self.bitsforbits = bitsforbits or 8
        if data_size is None and mask_size is None:
            raise ValueError("At least one of data_size or mask_size must be specified")
        self.data_size = data_size if data_size is not None else mask_size * self.bitsforbits
        self.mask_size = mask_size if mask_size is not None else data_size // self.bitsforbits
        self.data = bytearray(self.bittobyte(self.data_size))
        self.mask = bytearray(self.bittobyte(self.mask_size))
        self._data_access = BitBitBufferDataAccess(self, caster=caster if caster else int)
        self.pid_buffers = {}
        self.object_references = set()
        # ---------------------------------------------------------------
        # internal capability token
        # ---------------------------------------------------------------
        self._bitbit_internal = object()
        self._make_pid = make_pid
        # add a buffer‐bound indexer instance
        self.indexer = BitBitIndexer()

    # defer every get, set, bytes and hex through BitBitIndexer
    def __getitem__(self, key):
        spec = BitBitIndex(self, key, mode='view')
        return self.indexer.access(spec)

    def __setitem__(self, key, value):
        spec = BitBitIndex(self, key, mode='set', value=value)
        return self.indexer.access(spec)


    def __bytes__(self):
        spec = BitBitIndex(self, slice(None), mode='get')
        return self.indexer.access(spec)

    def hex(self):
        if self is None or not isinstance(self, BitBitBuffer):
            return "Empty Buffer"
        return bytes(self).hex()

    def get_by_pid(self, buffer, pid):
        """
        Get the PIDBuffer by its UUID.
        Returns the gap index if found, otherwise None.
        """
        gap = self.pid_buffers.get(buffer).get_by_pid(pid)
        if gap is not None:
            return gap
        return None
    def register_pid_buffer(
        self,
        pid_buffer: PIDBuffer | None = None,
        *,
        cells: Sequence[Any] | None = None,
        left: int | None = None,
        right: int | None = None,
        stride: int | None = None,
        label: str | None = None,
        bit_order: str = 'msb'
    ):
        """
        Register one or more PIDBuffer domains.

        - If you pass `cells`, you must also pass `stride`.  For each cell in
          `cells`, we register a PIDBuffer(domain_left=cell.left,
          domain_right=cell.right, domain_stride=stride, label=cell.label).
          Returns a list of PIDBuffer.
        - Otherwise you must pass exactly one of:
            * an existing PIDBuffer in `pid_buffer`, or
            * `left`, `right`, `stride` (and optionally `label`)
          to create a new one.  Returns a single PIDBuffer.
        """
        # Helper to do a single registration
        self.bit_order = bit_order
        if self._make_pid is False:
            return
        def _do_register(pb: PIDBuffer) -> PIDBuffer:
            if not (0 <= pb.domain_left < pb.domain_right <= self.mask_size):
                raise ValueError("PIDBuffer domain must lie within mask bounds")
            if pb.domain_stride <= 0:
                raise ValueError("PIDBuffer stride must be positive")
            pb.parent = self
            self.pid_buffers[pb.label] = pb
            return pb

        # 1) Register by cells
        if cells is not None:

            registered: list[PIDBuffer] = []
            for cell in cells:
                pb = PIDBuffer(
                    domain_left=cell.left,
                    domain_right=cell.right,
                    domain_stride=cell.stride,
                    label=getattr(cell, "label", None) or f"pid_{cell.left}_{cell.right}"
                )
                registered.append(_do_register(pb))
            return registered

        # 2) Register an existing PIDBuffer
        if pid_buffer is not None:
            if not isinstance(pid_buffer, PIDBuffer):
                raise TypeError("pid_buffer must be a PIDBuffer instance")
            return _do_register(pid_buffer)

        # 3) Create + register a new PIDBuffer
        if left is None or right is None or stride is None:
            raise ValueError("Must pass either `pid_buffer` or (`left`, `right`, `stride`)")
        new_label = label or f"pid_{left}_{right}"
        pb = PIDBuffer(
            domain_left=left,
            domain_right=right,
            domain_stride=stride,
            label=new_label
        )
        return _do_register(pb)

    # ==============================================================
    # INTERNAL FAST‑PATH HELPERS
    # ===============================================================
    def __len__(self):
        """
        Return the total number of bits in the buffer.
        This is the size of the mask plane.
        """
        return self.mask_size

    def __repr__(self):
        spec = BitBitIndex(self, slice(0, self.mask_size), mode='repr')
        return_val = self.indexer.access(spec)
        if return_val is not None:
            return return_val
        return "Empty Buffer (__repr__)"

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        for i in range(self.mask_size):
            yield self[i]

    def move(self, src, dst, length):
        """Move a contiguous block of bits from ``src`` to ``dst``.

        This mirrors Python's list semantics: the block is removed from its
        original location and inserted at the destination index.  Both mask and
        data planes are shifted accordingly so that they remain coupled.
        """
        if length <= 0 or src == dst:
            return

        # Work on mirror Python lists for clarity then write back
        mask_list = [int(self[i]) for i in range(self.mask_size)]
        data_list = list(self._data_access[0:self.mask_size])

        block_mask = mask_list[src:src + length]
        block_data = data_list[src:src + length]
        del mask_list[src:src + length]
        del data_list[src:src + length]
        if dst > src:
            dst -= length
        mask_list[dst:dst] = block_mask
        data_list[dst:dst] = block_data

        self[0:self.mask_size] = mask_list
        self._data_access[0:self.mask_size] = bytes(data_list)

        if self._make_pid is False:
            return
        # mirror in any PID that covers this range
        for pid in self.pid_buffers.values():
            L, R = pid.domain_left, pid.domain_right
            if src >= L and src + length <= R and dst >= L and dst <= R:
                local_src = src - L
                local_dst = dst - L
                pid.pids.move(local_src, local_dst, length)

    def swap(self, src, dst, length):
        """Swap two bit ranges, keeping mask and data coupled."""
        if length <= 0 or src == dst:
            return

        # Element-wise swap to handle overlapping ranges identically to
        # the Python reference implementation used in tests.
        for k in range(length):
            s = src + k
            d = dst + k
            s_mask = int(self[s])
            d_mask = int(self[d])
            s_data = bytes(self._data_access[s:s + 1])
            d_data = bytes(self._data_access[d:d + 1])
            self[s] = d_mask
            self[d] = s_mask
            self._data_access[s:s + 1] = d_data
            self._data_access[d:d + 1] = s_data

        if self._make_pid is False:
            return
        for pid in self.pid_buffers.values():
            L, R = pid.domain_left, pid.domain_right
            if src >= L and src + length <= R and dst >= L and dst + length <= R:
                pid.pids.swap(src - L, dst - L, length)
    # -- low‑level primitive --------------------------------------------------
    @property
    def plane(self) -> str:
        """Which plane are we targeting?"""
        # default to mask; override in caller wrappers if needed
        return 'mask'
    def round(self, value: float) -> int:
        if value % 1 < 0.5:
            return int(value)
        return int(value) + 1

    def _insert_bits(self, insertion_plans: list[tuple[int, int, int]]) -> None:
        logging.debug(f"[_insert_bits] ENTER: insertion_plans={insertion_plans!r}")
        """
        Insert `mask_bits` blank bits at mask‑index `mask_off`.
        The data plane is expanded automatically (bitsforbits × mask_bits),
        and existing payload is shifted right.

        NOTE: *nothing* is written to the newly‑created span; callers decide
        whether it stays 0 or gets stamped later.
        """


        # -------- mask plane --------------------------------------------------
        def feature_spreader(src_plane: bytearray,
                             old_bits: int,
                             stride: int,
                             plans: list[tuple[int, int, int]]
                             ) -> tuple[bytearray, _RawSpan, int]:
            logging.debug(f"[feature_spreader] ENTER: old_bits={old_bits}, stride={stride}, plans={plans!r}")
            """
            Copy `src_plane` into one enlarged plane, inserting zero-filled gaps
            described by *plans*  [(orig_off, adj_off, gap_bits), …].

            * `old_bits`  – payload length (in *mask* bits, i.e. stride‐units) of the
                            source plane before expansion.
            * `stride`    – 1 for the mask plane, self.bitsforbits for the data plane.
            * returns (new_plane, sanctioned_view, new_bits)
            """
            if old_bits == 0 and plans:
                # Find max new dst_cursor
                max_slot = 0
                for orig_off, adj_off, gap_bits in plans:
                    # In an empty buffer, only gaps at offset 0 matter
                    if orig_off == 0 and gap_bits > 0:
                        max_slot = max(max_slot, gap_bits)
                if max_slot > 0:
                    new_bits = max_slot * stride
                    new_bytes = self.bittobyte(new_bits)
                    new_plane = bytearray(new_bytes)
                    view = _RawSpan(new_plane, 0, new_bits)
                    logging.debug(f"[feature_spreader] SPRING: Allocating empty plane for {max_slot} bits")
                    return new_plane, view, new_bits


            # 1) new size = old + Σ gaps
            total_gap = sum(gap for _, _, gap in plans)
            new_bits  = (old_bits + total_gap) * stride
            new_bytes = self.bittobyte(new_bits)
            new_plane = bytearray(new_bytes)
            view      = _RawSpan(new_plane, 0, new_bits)

            # 2) single forward sweep
            src_cursor = 0      # where we are reading in the old plane (mask bits)
            dst_cursor = 0      # where we are writing   in the new plane (mask bits)

            for orig_off, _adj_off, gap_bits in plans:
                # a) copy block before this gap
                chunk = orig_off - src_cursor                    # mask-bits
                if chunk:
                    chunk_bits = chunk * stride
                    logging.debug(f"[feature_spreader] copying {chunk} units ({chunk_bits} bits) from src={src_cursor} to dst={dst_cursor}")
                    segment = self.extract_bit_region(src_plane,
                                                    src_cursor * stride,
                                                    chunk_bits)
                    logging.debug(f"[feature_spreader] writing {chunk_bits} bits to dst={dst_cursor}")
                    self.write_bit_region(view,
                                       dst_cursor * stride,
                                        segment,
                                        chunk_bits)
                    logging.debug(f"[feature_spreader] copied chunk, advancing src to {src_cursor+chunk}, dst to {dst_cursor+chunk}")
                    src_cursor += chunk
                    dst_cursor += chunk

                # b) leave `gap_bits` zero-filled bits in the destination
                dst_cursor += gap_bits

            #print(f"Final src_cursor: {src_cursor}, old_bits: {old_bits}")

            # 3) tail copy (whatever is left after the last gap)
            if src_cursor < old_bits:
                tail = old_bits - src_cursor
                tail_bits = tail * stride
                logging.debug(f"[feature_spreader] tail copy of {tail} units ({tail_bits} bits) from src={src_cursor} to dst={dst_cursor}")
                segment = self.extract_bit_region(src_plane,
                                                src_cursor * stride,
                                                tail_bits)
                self.write_bit_region(view,
                                    dst_cursor * stride,
                                    segment,
                                    tail_bits)
                logging.debug(f"[feature_spreader] completed tail copy of {tail_bits} bits")
            logging.debug(f"[feature_spreader] EXIT: new_bits={new_bits}")
            return new_plane, view, new_bits


        def mask_and_data_spreader(src_buffer: 'BitBitBuffer',
                                   mask_size: int,
                                   data_stride: int,
                                   plans: list[tuple[int, int, int]],
                                   plan_ratio: int = 1,
                                   plan_offset: int = 0,
                                   ) -> tuple[bytearray, _RawSpan, int, bytearray, _RawSpan, int]:
            logging.debug(f"[mask_and_data_spreader] ENTER: mask_size={mask_size}, data_stride={data_stride}, plans={plans!r}, "
                          f"plan_ratio={plan_ratio}, plan_offset={plan_offset}")
            ratioed_plans = []
            if abs(plan_ratio - 1.0) > 0.0001:
                # Global range covered by this sub-buffer
                low_off  = plan_offset
                high_off = plan_offset + mask_size * plan_ratio

                for orig_off, adj_off, gap_bits in plans:
                    if not (low_off <= orig_off < high_off):
                        continue
                    local_orig = (orig_off - plan_offset) // plan_ratio
                    local_adj  = (adj_off  - plan_offset) // plan_ratio
                    local_gap  = gap_bits // plan_ratio   # safe because gaps are stride-aligned
                    ratioed_plans.append((local_orig, local_adj, local_gap))

                new_mask, mask_view, new_mask_bits = feature_spreader(
                    src_buffer.mask, mask_size, 1, ratioed_plans
                )
                new_data, data_view, new_data_bits = feature_spreader(
                    src_buffer.data, mask_size, data_stride, ratioed_plans
                )

            else:
                new_mask, mask_view, new_mask_bits = feature_spreader(
                    src_buffer.mask, mask_size, 1, plans
                )
                new_data, data_view, new_data_bits = feature_spreader(
                    src_buffer.data, mask_size, data_stride, plans
                )

            logging.debug(f"[mask_and_data_spreader] result new_mask_bits={new_mask_bits}, new_data_bits={new_data_bits}")
            return new_mask, mask_view, new_mask_bits, new_data, data_view, new_data_bits

        self.mask, mask_view, self.mask_size, self.data, data_view, self.data_size =                 mask_and_data_spreader(self, self.mask_size, self.bitsforbits, insertion_plans)

        # 3) update the pid buffers
        if self._make_pid is False:
            return

        for label, pid in self.pid_buffers.items():
            logging.debug(f"[_insert_bits] updating PIDBuffer '{label}': old mask_size={pid.pids.mask_size}, old data_size={pid.pids.data_size}")
            new_pid_mask, new_pid_view, new_pid_bits, new_pid_data, new_pid_data_view, new_pid_data_bits =                     mask_and_data_spreader(pid.pids, pid.pids.mask_size, pid.pids.bitsforbits,
                                       insertion_plans, plan_ratio=pid.domain_stride, plan_offset=pid.domain_left)
            logging.debug(f"[_insert_bits] updating PIDBuffer '{label}': new mask_size={new_pid_bits}, new data_size={new_pid_data_bits}")
            pid.pids.mask, pid.pids.mask_size = new_pid_mask, new_pid_bits
            pid.pids.data, pid.pids.data_size = new_pid_data, new_pid_data_bits

            running_adj = 0
            # update the pid boundaries
            for orig_off, adj_off, sz in insertion_plans:

                if pid.domain_left >= orig_off + running_adj:
                    pid.domain_left += sz
                    pid.domain_right += sz
                elif pid.domain_right >= orig_off + running_adj:
                    pid.domain_right += sz

                logging.debug(f"[_insert_bits] updating PIDBuffer {label}: domain=({pid.domain_left},{pid.domain_right}), "
                              f"orig_off={orig_off}, adj_off={adj_off}, sz={sz}, running_adj={running_adj}")

                running_adj += sz
        logging.debug(f"[_insert_bits] EXIT: mask_size={self.mask_size}, data_size={self.data_size}")

    # -- public façade --------------------------------------------------------
    def expand(self, events, cells=None, proposals=None):
        if proposals is None:
            proposals = []
            if cells is not None:
                for cell in cells:
                    proposals.append(CellProposal(cell))
        logging.debug(f"[expand] ENTER: events={events!r}, cells={cells!r}, proposals={proposals!r}")
        """
        Insert the gaps described by *events* ([(label, offset, size), …])
        in a single pass, then update all dependent metadata.

        * self._insert_bits expects a list[(offset, size)] sorted ascending.
        * Any object in *cells* or *proposals* may expose .left/.right/
        .leftmost/.rightmost and an optional cached slice ._buf.
        """
        if not events:
            return

        # 1) sort by the user-supplied offsets (ascending)
        events = sorted(events, key=lambda t: t[1])
        logging.debug(f"[expand] sorted events={events!r}")

        plan, shift = [], 0
        for _lbl, orig_off, sz in events:
            adj_off = orig_off + shift
            plan.append((orig_off, adj_off, sz))
            shift += sz
        logging.debug(f"[expand] plan={plan!r}")

        logging.debug(f"[expand] invoking _insert_bits")
        self._insert_bits(plan)
        logging.debug(f"[expand] _insert_bits complete")

        # 3) Shift coordinates for every tracked span
        affected = proposals
        logging.debug(f"[expand] shifting coordinates for affected={affected!r}")
        if affected:
            for off, _, sz in plan:
                for c in affected:
                    old_left, old_right = c.left, c.right
                    if off <= c.left:                     # gap before cell
                        c.left += sz
                        c.right += sz
                        if hasattr(c, "leftmost")  and c.leftmost  is not None:
                            c.leftmost  += sz
                        if hasattr(c, "rightmost") and c.rightmost is not None:
                            c.rightmost += sz
                    elif c.left <= off < c.right:          # gap inside cell
                        c.right += sz
                        if (hasattr(c, "rightmost") and c.rightmost is not None
                                and off < c.rightmost):
                            c.rightmost += sz
                        if hasattr(c, "leftmost") and c.leftmost is not None:
                            if off <= c.leftmost:
                                c.leftmost += sz

                    logging.debug(f"[expand] cell={getattr(c,'label',repr(c))}, off={off}, sz={sz}, "
                                  f"old=({old_left},{old_right}) new=({c.left},{c.right})")

        logging.debug(f"[expand] EXIT")

        return proposals

    def _count_runs(self, optional_data=None):
        """Count consecutive runs of a specific value in the data."""

        count = [None, 0]
        runs = []
        last_bit = None
        for bit in (optional_data if optional_data is not None else self):
            if int(bit) == last_bit:
                count[1] += 1
            else:
                last_bit = int(bit)
                if count[1] > 0:
                    runs.append(count)
                count = [last_bit, 1]
        if count[1] > 0:
            print(f"Final run: {count}")
            runs.append(count)

        from sympy import factorint
        # Factor the runs into prime factors
        print(f"Factoring runs: {runs}")
        length = sum(length for _, length in runs)
        print(f"factors: {factorint(length)}")
        return runs

    def tuplepattern(self, src, end, length, direction='left'):
        if f"{self}" == "Empty Buffer":
            print("Empty Buffer: returning empty patterns")
            return [], []
        if self is None or not isinstance(self, BitBitBuffer):
            raise TypeError("Invalid BitBitBuffer")
        print(self)
        print(src)
        print(end)
        print(length)
        print(direction)
        print(f"tuplepattern called on {self} with src={src}, end={end}, length={length}, direction={direction}")
        if src is None or end is None or length == 0 or src >= end or src < 0 or end > len(self) or length < 0 or src == end:
            return [], []
        if direction == 'left' or direction == 'bi':
            print(self)
            print(src)
            print(end)
            print(length)
            print(direction)
            
            print(f"tuplepattern called on {self} with src={src}, end={end}, length={length}, direction={direction}")
            
            reversed = self[src:end][::-1][:length]
            right_pattern = self._count_runs(reversed)

        if direction == 'right' or direction == 'bi':
            left_pattern = self._count_runs(self[src:src + length])

        if direction == 'left':
            return right_pattern
        elif direction == 'right':
            return left_pattern
        elif direction == 'bi':
            return left_pattern, right_pattern

    def to_bitstring(self):
        # Return a string representing the data bits 
        return ''.join(str(bit) for bit in self)

    def bittobyte(self, bits):
        return self.intceil(bits) // 8

    @staticmethod
    def _intceil(val, base=8):
        """Return the smallest multiple of `base` that is >= `val`."""
        return (val + base - 1) // base * base

    def intceil(self, val, base=8):
        return self._intceil(val, base)

    def build_fill_buffer(self, fill_value: int, length_bits: int) -> bytearray:
        nbytes = (length_bits + 7) // 8
        shift = nbytes * 8 - length_bits
        if fill_value == 1:
            fill_value = (2**length_bits - 1)
        buf = bytearray((fill_value << shift).to_bytes(nbytes, byteorder='big'))
        return self.mask_padding_bits(buf, length_bits)

    # NEW: Helper method to wrap raw input if not already a metadata object.
    def _ensure_metadata(self, raw, default_mask_index=0):
        if hasattr(raw, "mask_index"):
            return raw
        else:
            raise TypeError("Stamping requires a BitBitItem or BitBitSlice from self[...]")

    def stamp(self, raw, indices, default_stride, default_value=1):
        # Enforce metadata wrapping at the very beginning
        raw = self._ensure_metadata(raw, getattr(raw, "mask_index", 0))
        for entry in indices:
            if isinstance(entry, (tuple, list)):
                if len(entry) == 2:
                    gap, stride = entry
                    value = default_value
                elif len(entry) == 3:
                    gap, stride, value = entry
                else:
                    raise ValueError("Invalid entry format")
            else:
                gap = entry
                stride = default_stride
                value = default_value
            fill_buf = self.build_fill_buffer(value, stride)
            self.write_bit_region(raw, gap, fill_buf, stride)
        return raw

    def write_bit_region(self, target, start_bit: int, buf, length_bits: int) -> None:
        # Accept internal metadata objects or direct mask/data
        from .helpers.bitbititem import BitBitItem

        if isinstance(target, BitBitItem):
            backing = target.buffer.mask
            base = target.mask_index
        elif getattr(target, '_bitbit_cap', False):
            backing = target._origin if hasattr(target, '_origin') else target
            base = getattr(target, '_offset', 0)
        elif target is self.mask or target is self.data:
            backing = target
            base = 0
        else:
            raise TypeError("Target for writing must be a BitBitItem or BitBitSlice instance")

        for i in range(length_bits):
            src_byte = buf[i // 8]
            src_bit_val = (src_byte >> (7 - (i % 8))) & 1
            dest_bit = base + start_bit + i
            dest_byte_i = dest_bit // 8
            dest_bit_offset = 7 - (dest_bit % 8)
            if src_bit_val:
                backing[dest_byte_i] |= (1 << dest_bit_offset)
            else:
                backing[dest_byte_i] &= ~(1 << dest_bit_offset)

    def extract_bit_region(self, data, start_bit: int, length: int) -> bytearray:
        # Accept raw storage or capability‑marked views.
        if getattr(data, "_bitbit_cap", False):
            backing   = data._origin
            start_bit = data._offset + start_bit
            data      = backing
        elif not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("Extraction source must be bytes‑like or an internal view")
        if not isinstance(data, (bytes, bytearray)):
            data = bytes(data)
        out = bytearray((length + 7) // 8)
        for i in range(length):
            src_bit = (data[(start_bit + i) // 8] >> (7 - ((start_bit + i) % 8))) & 1
            out[i // 8] |= src_bit << (7 - (i % 8))
        return out

    def mask_padding_bits(self, buf: bytearray, length_bits: int) -> bytearray:
        total_bits = len(buf) * 8
        pad_bits = total_bits - length_bits
        if pad_bits <= 0:
            return buf
        for bit_index in range(length_bits, total_bits):
            byte_i = bit_index // 8
            bit_offset = 7 - (bit_index % 8)
            buf[byte_i] &= ~(1 << bit_offset)
        return buf
