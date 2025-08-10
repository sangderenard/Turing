import uuid
import logging
from typing import Any, Sequence, Optional, Callable
from .bitbitindex import BitBitIndex
from .data_access import BitBitBufferDataAccess  # not used here, but keeps parity
from .bitbititem import BitBitItem  # parity

class PIDBuffer:
    def __init__(self, domain_left, domain_right, domain_stride, label, pid_depth=128): #128 for uuid4
        self.parent = None
        self.domain_left = domain_left
        self.domain_right = domain_right
        self.domain_stride = domain_stride

        # Local import to avoid circular dependency at module import time
        from ..bitbitbuffer import BitBitBuffer

        self.pids = BitBitBuffer(
            data_size=((domain_right - domain_left) + domain_stride - 1) // domain_stride * pid_depth,
            mask_size=((domain_right - domain_left) + domain_stride - 1) // domain_stride,
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
                    from ..bitbitbuffer import BitBitBuffer
                    return (BitBitBuffer._intceil(self.domain_left, self.domain_stride) + (i * self.domain_stride))
                else:
                    logging.debug(f"[PIDBuffer.get_by_pid] no match: pid_int={pid.int} vs stored={stored_int}")
        else:
            logging.debug(f"[PIDBuffer.get_by_pid] cache hit, active_set_size={len(self.active_set)}")
            for active_pid, gap in self.active_set:
                # `gap` is the stride index within the domain.  Convert back to
                # an absolute bit position by scaling with the domain stride and
                # offsetting by the domain left boundary.
                if isinstance(active_pid, uuid.UUID):
                    if active_pid == pid:
                        from ..bitbitbuffer import BitBitBuffer
                        logging.debug(
                            f"[PIDBuffer.get_by_pid] returning cached index={gap}"
                        )
                        return (
                            BitBitBuffer._intceil(self.domain_left, self.domain_stride)
                            + (gap * self.domain_stride)
                        )
                else:
                    try:
                        if int.from_bytes(active_pid, "big") == pid.int:
                            from ..bitbitbuffer import BitBitBuffer
                            logging.debug(
                                f"[PIDBuffer.get_by_pid] returning cached index={gap}"
                            )
                            return (
                                BitBitBuffer._intceil(
                                    self.domain_left, self.domain_stride
                                )
                                + (gap * self.domain_stride)
                            )
                    except (TypeError, ValueError):
                        # Skip invalid entries
                        continue

    def get_pids(self, gaps):
        """Return UUIDs for the given absolute bit positions ``gaps``.

        Each gap must lie within the PIDBuffer's domain boundaries; callers
        are responsible for providing system-relative (absolute) coordinates.
        """
        logging.debug(f"[PIDBuffer.get_pids] gaps={gaps}")
        assert isinstance(gaps, (list, tuple)), "gaps must be a list or tuple"
        return_vals = []
        from ..bitbitbuffer import BitBitBuffer
        for gap in gaps:
            pid = self.create_id(gap)
            logging.debug(f"[PIDBuffer.get_pids] created pid={pid} for gap={gap}")
            return_vals.append(pid)
            if self.active_set is None:
                self.active_set = set()
            # Store the stride index within the domain rather than the absolute
            # bit position so that cached lookups mirror the cold-path logic.
            data_index = (
                gap - BitBitBuffer._intceil(self.domain_left, self.domain_stride)
            ) // self.domain_stride
            self.active_set.add((pid, data_index))
        return return_vals

    def create_id(self, location):
        logging.debug(f"[PIDBuffer.create_id] location={location}")
        assert self.domain_left <= location < self.domain_right, "location out of domain bounds"
        uuid_id = uuid.uuid4()
        from ..bitbitbuffer import BitBitBuffer
        data_index = (location - BitBitBuffer._intceil(self.domain_left, self.domain_stride)) // self.domain_stride
        value = uuid_id.int.to_bytes(self.pids.bitsforbits // 8, byteorder='big')
        logging.debug(f"[PIDBuffer.create_id] writing uuid={uuid_id} at data_index={data_index}, bytes={value.hex()}")
        self.pids._data_access[data_index : data_index + 1] = value
        # mark this stride slot as occupied in the PID mask so that
        # visualisers and subsequent lookups see the entry immediately
        self.pids[data_index] = 1
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
