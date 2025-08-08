from typing import Any, Sequence
from typing import Callable, Optional
import uuid
import logging
# default to INFO, we'll raise to DEBUG once enabled
logging.basicConfig(level=logging.INFO)


class _RawSpan(bytearray):
    """A zero‑copy view (slice) tied to a BitBitBuffer plane."""
    __slots__ = ("_bitbit_cap", "_origin", "_offset")
    def __init__(self, backing, start_bit=0, length_bits=None, readonly=False):
        pass
    def __new__(cls, backing, offset, length):
        # backing: original bytearray; offset, length in bits
        view = super().__new__(cls,
            backing[offset // 8 : (offset + length + 7) // 8]
        )
        view._origin = backing
        view._offset = offset
        view._bitbit_cap = True
        
        return view
    
    @property
    def plane(self):
        """Which plane are we targeting?"""
        # default to mask; override in caller wrappers if needed
        return self._plane

class BitBitItem:
    def __init__(self, buffer=None, mask_index=None, length=None, cast=None, padding=0, padding_mask=None, reversed=False, plane='mask'):
        self._plane = plane
        self.id = id(self)
        self.buffer = buffer
        self.mask_index = mask_index
        self.padding = padding
        self.padding_mask = padding_mask or 0
        self.reversed = reversed
        self.padded_length = length
        self.useful_length = length - padding
        self.cast = cast or int
        if self.mask_index is not None:
            self.data_index = self.buffer.bitsforbits * self.mask_index
        else:
            self.data_index = None
    def __len__(self):
        return self.useful_length
    
    @property
    def data_or_mask(self):
        """Default plane when indexing with slice or int."""
        return self._plane


    def __bytes__(self):
        # build index-object for full‐slice get
        spec = BitBitIndex(self, slice(None), mode='get')
        return self.buffer.indexer.access(spec)

    def __int__(self):
        # build index-object for single-bit get
        spec = BitBitIndex(self, 0, mode='get')
        raw = self.buffer.indexer.access(spec)
        return (raw[0] >> 7) & 1

    def __getitem__(self, key):
            # Allow default plane indexing with int or slice
            second_key = None
            if isinstance(key, (slice, int)):
                second_key = key
                key = self.data_or_mask
            # Determine index for spec
            idx = second_key if second_key is not None else (self.mask_index if key == 'data' else 0)
            mode = 'view' if isinstance(idx, slice) else 'get'
            
            if key == 'mask':
                spec = BitBitIndex(self, idx, mode=mode)
                result = self.buffer.indexer.access(spec)
                
                # If the mode was 'get' (for an integer index), process and return the bit value.
                if mode == 'get':
                    return int(result[0] >> 7) if isinstance(result, (bytes, bytearray)) else int(result)
                
                # Otherwise (for 'view' mode), return the new slice object directly.
                return result

            if key == 'data':
                spec = BitBitIndex(self.buffer._data_access, idx, mode=mode)
                return self.buffer.indexer.access(spec)
                
            raise KeyError("Expected 'mask' or 'data'")


    def __setitem__(self, key, value):
        # Allow default plane setting with int or slice
        second_key = None
        if isinstance(key, (slice, int)):
            second_key = key
            key = self.data_or_mask
        # Determine index for spec
        idx = second_key if second_key is not None else (self.mask_index if key == 'data' else 0)
        
        # FIX: Explicitly set the mode based on the index type.
        mode = 'view' if isinstance(idx, slice) else 'get' # This line isn't strictly necessary for setitem,
                                                             # but the spec requires a mode. We'll use 'set'.

        if key == 'mask':
            # Always use 'set' mode for __setitem__
            spec = BitBitIndex(self, idx, mode='set', value=value)
            return self.buffer.indexer.access(spec)
        if key == 'data':
            # Always use 'set' mode for __setitem__
            spec = BitBitIndex(self.buffer._data_access, idx, mode='set', value=value)
            return self.buffer.indexer.access(spec)
            
        raise KeyError("Expected 'mask' or 'data'")

    def hex(self):
        spec = BitBitIndex(self, slice(None), mode='hex')
        return self.buffer.indexer.access(spec)

    def data_hex(self):
        spec = BitBitIndex(self.buffer._data_access, self.mask_index, mode='data_hex')
        return self.buffer.indexer.access(spec)

    def __repr__(self):
        spec = BitBitIndex(self, slice(None), mode='repr')
        return self.buffer.indexer.access(spec)

    @property
    def plane(self):
        """Which plane are we targeting?"""
        return self._plane



class BitBitSlice(BitBitItem):
    """
    Immutable view on an *aligned* bit‑range.
    """
    __slots__ = ("reversed",)

    def __init__(self, buffer, start_bit, length, reversed=False, plane='mask'):
        self._plane = plane
        stride = buffer.bitsforbits
        padded = buffer.intceil(length, stride)
        padding = padded - length
        super().__init__(buffer, start_bit, padded, cast=bytearray, padding=padding, reversed=reversed)
        self.reversed = reversed

    @property
    def plane(self):
        """Which plane are we targeting?"""
        return self._plane

    def __repr__(self):
        spec = BitBitIndex(self, slice(None), mode='repr')
        return self.buffer.indexer.access(spec)

class BitBitBufferDataAccess:
    def __init__(self, buffer, caster=int):
        self.buffer = buffer
        self.caster = caster

    @property
    def plane(self):
        # tell the indexer to target the data plane
        return 'data'

    def __getitem__(self, index):
        # data-plane get via index object
        spec = BitBitIndex(self, index, mode='get')
        return self.buffer.indexer.access(spec)

    def __setitem__(self, index, value):
        # data-plane set via index object
        spec = BitBitIndex(self, index, mode='set', value=value)
        return self.buffer.indexer.access(spec)


import sys
class PIDBuffer:
    def __init__(self, domain_left, domain_right, domain_stride, label, pid_depth=128): #128 for uuid4
        self.parent = None
        self.domain_left = domain_left
        self.domain_right = domain_right
        self.domain_stride = domain_stride

        self.pids = BitBitBuffer(
            data_size=(domain_right - domain_left) // domain_stride * pid_depth,
            mask_size=(domain_right - domain_left) // domain_stride,
            bitsforbits= pid_depth,
            caster=int, make_pid=False
        )

        self.label = label
        self.active_set = None
        logging.debug(f"[PIDBuffer.__init__] domain=({domain_left},{domain_right}), stride={domain_stride}, label={label}, depth={pid_depth}")
        assert domain_left < domain_right, "domain_left must be < domain_right"
        assert domain_stride > 0, "domain_stride must be positive"
    @property
    def plane(self):
        return 'pidref'

    def get_by_pid(self, pid):
        logging.debug(f"[PIDBuffer.get_by_pid] pid={pid}")
        assert isinstance(pid, uuid.UUID), "pid must be a UUID"
        if self.active_set is None:
            self.active_set = set()
            logging.debug(f"[PIDBuffer.get_by_pid] cache miss, populating active_set")
            
            for i in range(self.pids.mask_size):

                data = self.pids._data_access[i:i+1]
                logging.debug(f"[PIDBuffer.get_by_pid] checking index={i}, bytes={data.hex()}")
                self.active_set.add((data, i))
                stored_int = int.from_bytes(data, 'big')
                logging.debug(f"[PIDBuffer.get_by_pid] stored_int={stored_int}, pid.int={pid.int}")
                if stored_int == pid.int:
                    logging.debug(f"[PIDBuffer.get_by_pid] found match at index={i}")
                    return (BitBitBuffer._intceil(self.domain_left, self.domain_stride) + (i * self.domain_stride))
                else:
                    logging.debug(f"[PIDBuffer.get_by_pid] no match: pid_int={pid.int} vs stored={stored_int}")
        else:
            logging.debug(f"[PIDBuffer.get_by_pid] cache hit, active_set_size={len(self.active_set)}")
            for active_pid, gap in self.active_set:
                # Handle cached raw bytes or UUID objects
                if isinstance(active_pid, uuid.UUID):
                    if active_pid == pid:
                        logging.debug(f"[PIDBuffer.get_by_pid] returning cached index={gap}")
                        return (gap + BitBitBuffer._intceil(self.domain_left, self.domain_stride))
                else:
                    try:
                        if int.from_bytes(active_pid, 'big') == pid.int:
                            logging.debug(f"[PIDBuffer.get_by_pid] returning cached index={gap}")
                            return (gap + BitBitBuffer._intceil(self.domain_left, self.domain_stride))
                    except (TypeError, ValueError):
                        # Skip invalid entries
                        continue

    def get_pids(self, gaps):
        logging.debug(f"[PIDBuffer.get_pids] gaps={gaps}")
        assert isinstance(gaps, (list, tuple)), "gaps must be a list or tuple"
        return_vals = []
        for gap in gaps:
            pid = self.create_id(gap)
            logging.debug(f"[PIDBuffer.get_pids] created pid={pid} for gap={gap}")
            return_vals.append(pid)
            if self.active_set is None:
                self.active_set = set()
            self.active_set.add((pid, gap))
        return return_vals

    def create_id(self, location):
        logging.debug(f"[PIDBuffer.create_id] location={location}")
        assert self.domain_left <= location < self.domain_right, "location out of domain bounds"
        uuid_id = uuid.uuid4()
        data_index = (location - BitBitBuffer._intceil(self.domain_left, self.domain_stride)) // self.domain_stride
        value = uuid_id.int.to_bytes(self.pids.bitsforbits // 8, byteorder='big')
        logging.debug(f"[PIDBuffer.create_id] writing uuid={uuid_id} at data_index={data_index}, bytes={value.hex()}")
        self.pids._data_access[data_index : data_index + 1] = value
        return uuid_id

    def __getitem__(self, key):
        logging.debug(f"[PIDBuffer.__getitem__] key={key}")
        assert self.parent is not None, "PIDBuffer must be registered to a parent"
        spec = BitBitIndex(
            caller      = self.parent._data_access,
            index       = key,
            mode        = 'get',
            index_hook  = self.get_by_pid
        )
        return self.parent.indexer.access(spec)

    def __setitem__(self, key, value):
        logging.debug(f"[PIDBuffer.__setitem__] key={key}, value={value}")
        assert self.parent is not None, "PIDBuffer must be registered to a parent"
        spec = BitBitIndex(
            caller      = self.parent._data_access,
            index       = key,
            mode        = 'set',
            value       = value,
            index_hook  = self.get_by_pid
        )
        return self.parent.indexer.access(spec)

    def __repr__(self):
        return repr(self.pids)
    
# Use a relative import to access the package-local definition.
from .cell_consts import Cell
class CellProposal(Cell):
    def __init__(self, cell):
                
        super().__init__(cell.stride, cell.left, cell.right, cell.len, leftmost=cell.leftmost, rightmost=cell.rightmost, label=cell.label)

        self.salinity = cell.salinity
        self.pressure = cell.pressure
        self.leftmost = cell.leftmost
        self.rightmost = cell.rightmost


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
        spec = BitBitIndex(self, slice(None), mode='repr')
        return self.indexer.access(spec)

    def __iter__(self):
        for i in range(self.mask_size):
            yield self[i]

    def move(self, src, dst, length):
        # 1) perform the primary shift
        self[dst:dst + length] = self[src:src + length]
        if self._make_pid is False:
            return
        # 2) mirror in any PID that covers this range
        for pid in self.pid_buffers.values():
            L, R = pid.domain_left, pid.domain_right
            if src >= L and src + length <= R:
                local_src = src - L
                local_dst = dst - L
                pid.pids.move(local_src, local_dst, length)

    def swap(self, src, dst, length):
        # 1) perform the primary swap
        src_bits = self[src:src + length]
        dst_bits = self[dst:dst + length]
        self[dst:dst + length] = src_bits
        self[src:src + length] = dst_bits
        # 2) mirror in any PID that covers both ranges
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


        def mask_and_data_spreader(src_buffer: BitBitBuffer,
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
                for orig_off, adj_off, gap_bits in plans:
                    low_off = 0
                    high_off = mask_size
                    if hasattr(src_buffer, 'domain_left') and hasattr(src_buffer, 'domain_right'):
                        low_off = src_buffer.domain_left
                        high_off = src_buffer.domain_right
                    
                    if high_off > orig_off >= low_off:
                        ratioed_plans.append(((orig_off - plan_offset) // plan_ratio, (adj_off - plan_offset) // plan_ratio, gap_bits // plan_ratio))


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

        self.mask, mask_view, self.mask_size, self.data, data_view, self.data_size = \
            mask_and_data_spreader(self, self.mask_size, self.bitsforbits, insertion_plans)

        # 3) update the pid buffers
        if self._make_pid is False:
            return

        for label, pid in self.pid_buffers.items():
            logging.debug(f"[_insert_bits] updating PIDBuffer '{label}': old mask_size={pid.pids.mask_size}, old data_size={pid.pids.data_size}")
            new_pid_mask, new_pid_view, new_pid_bits, new_pid_data, new_pid_data_view, new_pid_data_bits = \
                mask_and_data_spreader(pid.pids, pid.pids.mask_size, pid.pids.bitsforbits,
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
        if direction == 'left' or direction == 'bi':
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

from typing import Union, List
import threading

_repr_local = threading.local()

def depth_guarded_repr(obj, fallback="<...>"):
    depth = getattr(_repr_local, 'depth', 0)
    if depth > 1:
        return fallback
    try:
        _repr_local.depth = depth + 1
        return obj.__repr__()
    finally:
        _repr_local.depth = depth


class BitBitIndex:
    """
    A simple metadata struct holding an indexing request:
      - caller: the object being indexed (BitBitBuffer, BitBitItem, BitBitSlice)
      - index: raw Python index (int or slice)
      - mode: 'get' or 'set'
      - value: payload for writes
    """
    def __init__(self, caller: Any, index: Union[int, slice], mode: str = 'get', value: Any = None, inverted: bool = False, index_hook: Optional[Callable] = None):
        self.caller = caller
        self.index = index
        self.mode = mode
        self.value = value
        self.inverted = inverted
        self.index_hook = index_hook

    def __repr__(self):
        return depth_guarded_repr(self)
    def normalize(self) -> tuple[int,int,int]:
        """
        Normalize raw index into (start, stop, step) following native Python slice rules.
        """

        # Slice / Item views use their *local* length, not the parent buffer size
        if isinstance(self.caller, (BitBitSlice, BitBitItem)):
            mask_size = len(self.caller)
        else:
            buf = self.caller.buffer if hasattr(self.caller, 'buffer') else self.caller
            mask_size = buf.mask_size

        if isinstance(self.index, int):
            start, stop, step = self.index, self.index + 1, 1
        else:
            sl = self.index
            step = sl.step or 1
            # default start/stop based on step sign
            if step > 0:
                start = sl.start if sl.start is not None else 0
                stop  = sl.stop  if sl.stop  is not None else mask_size
            else:
                start = sl.start if sl.start is not None else mask_size - 1
                stop  = sl.stop  if sl.stop  is not None else -1
        return start, stop, step

    def indices(self) -> List[int]:
        """
        Compute the flat list of mask-bit positions to access.
        """
        start, stop, step = self.normalize()
        return list(range(start, stop, step))

    @property
    def plane(self) -> str:
        if hasattr(self.caller, 'plane'):
            return self.caller.plane
        return 'mask'


    @property
    def base_offset(self) -> int:
        """Bit offset of this view in the global buffer."""
        return getattr(self.caller, 'mask_index', 0)

    @property
    def bitsforbits(self) -> int:
        return self.caller.buffer.bitsforbits if hasattr(self.caller, 'buffer') else self.caller.bitsforbits

    @property
    def caster(self):
        return getattr(self.caller, 'cast', int)

class BitBitIndexer:
    """
    Centralized entry point for all BitBitBuffer get/set accesses.
    """
    # toggleable logging config
    logging_enabled = False
    verbosity = 0

    @staticmethod
    def configure(enabled: bool = True, verbosity: int = 1):
        """
        Turn on/off detailed logging and set verbosity (1=basic, 2=normalize, 3=bit-level, …).
        """
        BitBitIndexer.logging_enabled = enabled
        BitBitIndexer.verbosity = verbosity
        level = logging.DEBUG if enabled and verbosity > 0 else logging.INFO
        logging.getLogger().setLevel(level)
        logging.debug(f"[configure] enabled={enabled}, verbosity={verbosity}")

    @staticmethod
    def _invert_bits(raw: bytes, bit_length: int) -> bytes:
        """
        Given a bytestring `raw` containing at least `bit_length` bits
        in big‑endian bit order, invert the first `bit_length` bits
        and return the resulting bytes.
        """
        # Extract the first bit_length bits into a list
        bits = [(raw[i//8] >> (7 - (i % 8))) & 1 for i in range(bit_length)]
        # Invert that list
        bits = [1 - b for b in bits]
        # Pack back into bytes
        out = bytearray((bit_length + 7) // 8)
        for pos, b in enumerate(bits):
            if b:
                out[pos // 8] |= 1 << (7 - (pos % 8))
        return bytes(out)

    @staticmethod
    def _reverse_bits(raw: bytes, bit_length: int) -> bytes:
        """
        Given a bytestring `raw` containing at least `bit_length` bits
        in big‑endian bit order, reverse the first `bit_length` bits
        and return the resulting bytes.
        """
        # Extract the first bit_length bits into a list
        bits = [(raw[i//8] >> (7 - (i % 8))) & 1 for i in range(bit_length)]
        # Reverse that list
        bits.reverse()
        # Pack back into bytes
        out = bytearray((bit_length + 7) // 8)
        for pos, b in enumerate(bits):
            if b:
                out[pos // 8] |= 1 << (7 - (pos % 8))
        return bytes(out)

    @staticmethod
    def access(spec: BitBitIndex) -> Any:
        if spec.index_hook is not None:
            translated_index = spec.index_hook(spec.index)
            if translated_index is None:
                raise KeyError(f"Index hook failed to find a match for '{spec.index}'")
            spec.index = translated_index
            spec.index_hook = None
            
        if BitBitIndexer.logging_enabled:
            logging.debug("[access] ENTER: index=%r, mode=%r, value=%r, caller=<%s at 0x%x>",
              spec.index, spec.mode, spec.value,
              type(spec.caller).__name__, id(spec.caller))

            if BitBitIndexer.verbosity >= 2:
                start, stop, step = spec.normalize()
                logging.debug(f"[access] normalize -> start={start}, stop={stop}, step={step}, plane={spec.plane}, base_offset={spec.base_offset}")

        # ───────────────────────────────────────────────────────────────
        # 0 · INDEX HOOK Translate exotic keys (PID, label, etc.) into a
        #                plain Python index *before* any other handling.
        #                The hook must return an int or slice understood
        #                by the downstream logic.
        # ───────────────────────────────────────────────────────────────

        buf = spec.caller.buffer if hasattr(spec.caller, 'buffer') else spec.caller



        # ───────────────────────────────────────────────────────
        # 0. VIEW → return a BitBitItem / BitBitSlice, never mutate
        # ───────────────────────────────────────────────────────
        if spec.mode == 'view':
            idxs = spec.indices()
            # single‑bit view  → BitBitItem
            if isinstance(spec.index, int):
                return BitBitItem(
                    buffer = buf,
                    mask_index = idxs[0],
                    length = 1,
                    cast = spec.caster,
                    plane = spec.plane
                )
            # slice view → BitBitSlice (handle reverse step)
            step      = spec.index.step or 1
            reversed_ = step < 0
            start_bit = idxs[0] if step > 0 else idxs[-1]
            return BitBitSlice(
                buffer   = buf,
                start_bit= start_bit,
                length   = len(idxs),
                reversed = reversed_,
                plane    = spec.plane
            )
        # custom convenience modes
        if spec.mode == 'hex':
            # 1) pull raw mask-bits 
            raw = BitBitIndexer._get_mask(buf, spec.base_offset, spec.indices())
            # 2) invert bits if requested
            if isinstance(spec.caller, (BitBitItem, BitBitSlice)) and getattr(spec.caller, 'inverted', False):
                raw = BitBitIndexer._invert_bits(raw, spec.caller.useful_length)
            # 3) reverse bit-order if a reversed slice
            if isinstance(spec.caller, BitBitSlice) and spec.caller.reversed:
                raw = BitBitIndexer._reverse_bits(raw, spec.caller.useful_length)
            # 4) hex-string it
            result = raw.hex()
            if BitBitIndexer.logging_enabled:
                logging.debug(f"[access] hex result={result}")
            return result

        if spec.mode == 'data_hex':
            # 1) pull raw data-bytes
            raw = BitBitIndexer._get_data(buf,
                                          spec.base_offset,
                                          spec.indices(),
                                          spec.bitsforbits,
                                          spec.caster)
            # 2) reverse byte-order if reversed slice
            if isinstance(spec.caller, BitBitSlice) and spec.caller.reversed:
                raw = raw[::-1]
            # 3) invert bytes if requested
            if isinstance(spec.caller, (BitBitItem, BitBitSlice)) and getattr(spec.caller, 'inverted', False):
                raw = bytes((~b & 0xFF) for b in raw)
            # 4) hex-string it
            result = raw.hex()
            if BitBitIndexer.logging_enabled:
                logging.debug(f"[access] data_hex result={result}")
            return result

        if spec.mode == 'repr':
            if BitBitIndexer.logging_enabled:
                logging.debug("[access] mode=repr")
            # BitBitBuffer repr
            if isinstance(spec.caller, BitBitBuffer):
                full_mask = BitBitIndexer._get_mask(buf, 0, list(range(buf.mask_size)))
                result = f"BitBitBuffer(mask={full_mask.hex()}, bitsforbits={buf.bitsforbits}, mask_size={buf.mask_size})"
                if BitBitIndexer.logging_enabled:
                    logging.debug(f"[access] repr result={result}")
                return result
            # BitBitSlice repr
            if isinstance(spec.caller, BitBitSlice):
                mraw = BitBitIndexer._get_mask(buf, spec.base_offset, spec.indices())
                if spec.caller.inverted:
                    mraw = BitBitIndexer._invert_bits(mraw, spec.caller.useful_length)
                if spec.caller.reversed:
                    mraw = BitBitIndexer._reverse_bits(mraw, spec.caller.useful_length)
                draw = BitBitIndexer._get_data(buf,
                                               spec.base_offset,
                                               spec.indices(),
                                               spec.bitsforbits,
                                               spec.caster)
                if spec.caller.reversed:
                    draw = draw[::-1]
                if spec.caller.inverted:
                    draw = bytes((~b & 0xFF) for b in draw)
                result = f"BitBitSlice(mask={mraw.hex()}, data={draw.hex()})"
                if BitBitIndexer.logging_enabled:
                    logging.debug(f"[access] repr result={result}")
                return result
            # BitBitItem repr
            if isinstance(spec.caller, BitBitItem):
                mraw = BitBitIndexer._get_mask(buf, spec.base_offset, spec.indices())
                if spec.caller.inverted:
                    mraw = BitBitIndexer._invert_bits(mraw, spec.caller.useful_length)
                dbyte = BitBitIndexer._get_data(buf,
                                                spec.caller.data_index,
                                                [0],
                                                buf.bitsforbits,
                                                spec.caster)
                if spec.caller.inverted:
                    dbyte = bytes((~b & 0xFF) for b in dbyte)
                result = (f"BitBitItem(mask={mraw.hex()}, data={dbyte.hex()}, "
                          f"len={spec.caller.useful_length}, idx={spec.caller.mask_index})")
                if BitBitIndexer.logging_enabled:
                    logging.debug(f"[access] repr result={result}")
                return result
            
        # default get/set
        idxs = spec.indices()
        if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 3:
            logging.debug(f"[access] final indices={idxs}")

        if spec.mode == 'get':
            if spec.plane == 'mask':
                if BitBitIndexer.logging_enabled:
                    logging.debug("[access] get ↔ _get_mask")
                result = BitBitIndexer._get_mask(buf, spec.base_offset, idxs)
            else:
                if BitBitIndexer.logging_enabled:
                    logging.debug("[access] get ↔ _get_data")
                result = BitBitIndexer._get_data(buf, spec.base_offset, idxs, spec.bitsforbits, spec.caster)
            if BitBitIndexer.logging_enabled:
                logging.debug(f"[access] get result={result}")
            return result
        else:
            if spec.plane == 'mask':
                if BitBitIndexer.logging_enabled:
                    logging.debug(f"[access] set↔_set_mask value={spec.value}")
                BitBitIndexer._set_mask(buf, spec.base_offset, idxs, spec.value)
                if BitBitIndexer.logging_enabled:
                    logging.debug("[access] set completed (_set_mask)")
            else:
                if BitBitIndexer.logging_enabled:
                    logging.debug(f"[access] set↔_set_data value={spec.value}")
                BitBitIndexer._set_data(buf, spec.base_offset, idxs, spec.bitsforbits, spec.value)
                if BitBitIndexer.logging_enabled:
                    logging.debug("[access] set completed (_set_data)")

    @staticmethod
    def _get_mask(buf, base: int, idxs: List[int]) -> bytes:
        if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 2:
            logging.debug(f"[_get_mask] base={base}, idxs={idxs}")
        # extract individual bits and pack into bytes
        bits = []
        for i in idxs:
            bit_pos = base + i
            if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 4:
                logging.debug(f"[_get_mask] computing bit for bit_pos={bit_pos}")
                assert isinstance(bit_pos, int) and bit_pos >= 0
            byte_i = bit_pos // 8
            bit_off = 7 - (bit_pos % 8)
            if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 4:
                logging.debug(f"[_get_mask] byte_i={byte_i}, bit_off={bit_off}")
                assert 0 <= bit_off < 8
            bits.append((buf.mask[byte_i] >> bit_off) & 1)
            if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 4:
                last_bit = bits[-1]
                logging.debug(f"[_get_mask] extracted bit={last_bit}")
                assert last_bit in (0, 1)
        out = bytearray((len(bits) + 7) // 8)
        for pos, b in enumerate(bits):
            if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 4:
                logging.debug(f"[_get_mask] packing bit at pos={pos}, b={b}")
                assert b in (0, 1)
            if b:
                out[pos // 8] |= 1 << (7 - (pos % 8))
        if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 3:
            logging.debug(f"[_get_mask] out_bytes={out.hex()}")
        return bytes(out)

    @staticmethod
    def _get_data(buf, base: int, idxs: List[int], stride: int, caster) -> bytes:
        if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 2:
            logging.debug(f"[_get_data] base={base}, idxs={idxs}, stride={stride}")
        
        # Correctly handle byte-aligned strides.
        if stride % 8 == 0:
            byte_w = stride // 8
            chunks = []
            for i in idxs:
                # The data for mask index `i` starts at data byte `i * byte_w`.
                # The `base` is a bit offset for the view, which needs to be accounted for.
                start_byte = (base * byte_w) + (i * byte_w)
                chunks.append(buf.data[start_byte : start_byte + byte_w])
            return b''.join(chunks) # Return the correct result immediately.

        # The original bit-level extraction logic is flawed for non-contiguous slices.
        # A more robust implementation would iterate and extract each chunk.
        out_bits = bytearray((stride * len(idxs) + 7) // 8)
        current_bit_out = 0
        for i in idxs:
            data_start_bit = base + (i * stride)
            region = buf.extract_bit_region(buf.data, data_start_bit, stride)
            
            # Append the extracted bits to the output buffer
            for bit_in_region in range(stride):
                src_byte = region[bit_in_region // 8]
                src_bit_val = (src_byte >> (7 - (bit_in_region % 8))) & 1
                
                if src_bit_val:
                    dest_byte_i = current_bit_out // 8
                    dest_bit_offset = 7 - (current_bit_out % 8)
                    out_bits[dest_byte_i] |= (1 << dest_bit_offset)
                current_bit_out += 1
                
        return bytes(out_bits)

    @staticmethod
    def _set_mask(buf, base: int, idxs: List[int], value: Union[int, List[int]]) -> None:
        if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 2:
            logging.debug(f"[_set_mask] base={base}, idxs={idxs}, value={value}")
        # value as int (0/1) or list of bits
        if isinstance(value, int):
            bits = [value] * len(idxs)
        else:
            bits = list(value)
        for bit, i in zip(bits, idxs):
            if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 4:
                logging.debug(f"[_set_mask] setting bit={bit} at idx={i}")
                assert bit in (0, 1) and isinstance(i, int)
            bit_pos = base + i
            byte_i = bit_pos // 8
            bit_off = 7 - (bit_pos % 8)
            if bit:
                buf.mask[byte_i] |= (1 << bit_off)
            else:
                buf.mask[byte_i] &= ~(1 << bit_off)
        if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 3:
            # verify by re-reading
            post = BitBitIndexer._get_mask(buf, base, idxs)
            logging.debug(f"[_set_mask] verified bits now={post.hex()}")

    @staticmethod
    def _set_data(buf, base: int, idxs: list[int], stride: int, value) -> None:
        if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 2:
            logging.debug(f"[_set_data] base={base}, idxs={idxs}, stride={stride}, value={value}")
        """
        Write to the data‑plane for *any* stride.
        • byte‑multiple strides use fast slice assignment
        • sub‑byte strides delegate to BitBitBuffer.write_bit_region
        """
        total_bits = stride * len(idxs)

        # 1) normal whole‑byte case  (8,16,24 … bits per element)
        if stride % 8 == 0:
            byte_w = stride // 8
            # normalise *value* to contiguous bytes
            if isinstance(value, (bytes, bytearray)):
                raw = bytes(value)
                if len(raw) != byte_w * len(idxs):
                    raise ValueError("length mismatch")
            elif isinstance(value, int):
                raw = value.to_bytes(byte_w, 'big') * len(idxs)
            else:                       # iterable of ints / bytes
                chunks = []
                for v in value:
                    if isinstance(v, int):
                        chunks.append(v.to_bytes(byte_w, 'big'))
                    else:
                        chunks.append(bytes(v))
                raw = b''.join(chunks)
            # single splice into the data plane
            first = idxs[0]
            byte_off = (base + first) // 8
            buf.data[byte_off : byte_off + len(raw)] = raw
            return

        # 2) sub‑byte stride → fall back to generic bit writer
        #    convert *value* into one contiguous bytearray of `total_bits`
        if isinstance(value, int):
            raw = value.to_bytes((total_bits + 7) // 8, 'big')
        elif isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
        else:   # iterable of ints/bits
            bits = []
            for v in value:
                bits.extend([(v >> (stride - 1 - b)) & 1 for b in range(stride)])
            raw = bytearray((total_bits + 7) // 8)
            for i, bit in enumerate(bits):
                if bit:
                    raw[i//8] |= 1 << (7 - i%8)

        buf.write_bit_region(buf.data, base + idxs[0], raw, total_bits)
        objects = []
        
        referring_buffer = buf._origin if hasattr(buf, '_origin') else buf
        

        absolute_start = base + idxs[0]
        absolute_end = absolute_start + total_bits

        for pids in referring_buffer.pid_buffers.values():
            # Skip buffers whose domain is fully outside the modified range
            if absolute_end <= pids.domain_left or absolute_start >= pids.domain_right:
                continue

            # Clip to the intersection of write region and PID domain
            pid_start = max(absolute_start, pids.domain_left)
            pid_end   = min(absolute_end,   pids.domain_right)

            # Round up to nearest stride-aligned offset within domain
            rel_start = ((pid_start - pids.domain_left + pids.domain_stride - 1) //
                        pids.domain_stride) * pids.domain_stride + pids.domain_left

            for abs_loc in range(rel_start, pid_end, pids.domain_stride):
                pid = pids.create_id(abs_loc)
                objects.append((abs_loc, pid))

        referring_buffer.object_references.update(objects)

        BitBitIndexer._set_mask(buf, base, idxs, 1)

        
# 3️⃣  A fluent, slice‑only test‑bench
def main():
    # — basic mask + data round‑trip —––––––––––––––––––––––––––––––––––––––––
    BitBitIndexer.configure(enabled=True, verbosity=9)
    buf = BitBitBuffer(mask_size=16, bitsforbits=8)
    
    # mask                                   data
    buf[0:8]          = [1, 0, 1, 1, 0, 0, 1, 1]
    buf._data_access[0:8] = bytes(range(1, 9))

    assert buf[0:8].hex()            == 'b3'              # mask check
    assert buf._data_access[0:8]     == bytes(range(1, 9))  # data check

    # reversed views (mask reverses, data preserves byte order)
    assert buf[7::-1].hex()          == 'cd'
    assert buf._data_access[7::-1]   == bytes(range(8, 0, -1))

    # — stamp, expand, pid, etc. (unchanged logic, just fluent access) ——–––
    buf2 = BitBitBuffer(mask_size=9, bitsforbits=3)
    view = buf2[0:3]
    buf2.stamp(view, [0], 3)
    assert buf2[0:3].hex() == 'e0'


    from cell_consts import Cell

    # 1) Create two test cells with stride=4, covering bits [1,4) and [4,7):
    cell1 = Cell(stride=4, left=1, right=4, len=3, label='c1')
    cell2 = Cell(stride=4, left=4, right=7, len=3, label='c2')
    cells = [cell1, cell2]

    # 2) New buffer covering 8 mask‑bits with 4 data‑bits per mask‑bit:
    buf3 = BitBitBuffer(mask_size=8, bitsforbits=4)

    # 3) Register your cells for PID tracking (optional for expand):
    buf3.register_pid_buffer(cells=cells, stride=cell1.stride)

    # 4) Expand at offset=2 by inserting 4 zero bits, tagged with cell1’s label:
    events = [
        ('c1', 2, 4),   # label, insert‑at‐mask‑index, #bits
    ]
    proposals = buf3.expand(events, cells=cells)

    # 5) Verify that each cell’s left/right moved as expected:
    assert (proposals[0].left, proposals[0].right) == (1, 8), f"Cell c1 should have moved right by 4 bits {proposals[0].label}, {proposals[0].left}, {proposals[0].right}"
    assert (proposals[1].left, proposals[1].right) == (8, 11), f"Cell c2 should have moved right by 4 bits {proposals[1].label}, {proposals[1].left}, {proposals[1].right}"
    print("expand test passed!")

    for proposal in proposals:
        for cell in cells:
            if proposal.label == cell.label:
                # Update the cell with the new left/right values
                assert proposal.left >= cell.left, f"Cell {cell.label} left should not move left: {proposal.left} < {cell.left}"
                assert proposal.right >= cell.right, f"Cell {cell.label} right should not move left: {proposal.right} < {cell.right}"
                cell.left = proposal.left
                cell.right = proposal.right
                cell.leftmost = proposal.leftmost
                cell.rightmost = proposal.rightmost
    # --- PID System Round-Trip Test ---
    # 1) Get the PIDBuffer for the 'test' domain.
    pb = buf3.pid_buffers['c1']
    
    # 2) Define test values and the target index.
    #    Since bitsforbits=4, these are 4-bit values.
    initial_value = 0xA  # (binary 1010)
    new_value = 0x5      # (binary 0101)
    test_index = 4

    # 3) Write the initial data value to the main buffer at the target index.
    buf3._data_access[test_index] = initial_value

    # 4) Create a PID that points to this location.
    pid = pb.create_id(test_index)

    # 5) Use the PID to GET the initial value and verify it's correct.
    retrieved_view = pb[pid]
    value = int.from_bytes(retrieved_view, 'big')
    assert value == initial_value, f"Expected initial value {initial_value}, got {value}"
    
    # 6) Use the PID to SET a new value.
    pb[pid] = new_value

    # 7) Use the PID to GET the new value back and verify the change was successful.
    final_view = pb[pid]
    assert int(final_view) == new_value
    
    # 8) As a final check, verify the data was changed in the underlying main buffer.
    assert int(buf3._data_access[test_index]) == new_value
    print("PID system set/get round-trip test passed!")


    pbuf = pb.pids
    pbuf[0] = 1
    orig   = int(pbuf[0])
    pbuf.move(0, 2, 1);  assert int(pbuf[2]) == orig
    pbuf.swap(0, 2, 1);  assert int(pbuf[0]) == orig

    bit0 = buf[0]
    assert bit0.hex()         == buf[0:1].hex()
    assert bit0.data_hex()    == buf._data_access[0:buf.bitsforbits].hex()

    slc = buf[0:8]
    assert slc.hex()          == buf[0:8].hex()
    assert slc.data_hex()     == buf._data_access[0:8].hex()

    # repr dispatch still funnels through the indexer
    assert repr(bit0).startswith("BitBitItem(")
    assert repr(slc).startswith("BitBitSlice(")
    assert repr(buf).startswith("BitBitBuffer(")
    print("Fluent hex/data_hex/repr tests passed\nAll checks passed!")

if __name__ == "__main__":
    main()
# ───────────────────────────────────────────────────────────────────────────────