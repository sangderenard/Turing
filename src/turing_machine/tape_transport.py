from __future__ import annotations

"""High-level tape transport with indexed access.

This module provides :class:`TapeTransport`, a small facade over
``CassetteTapeBackend`` that centralises the coordination of seek and
play/write operations.  It exposes Python's ``[]`` indexing so callers can
use familiar slice or index syntax while the transport handles the
underlying motor movement and gating of tape positions.

Unlike a random-access array, every access must *physically traverse* the
media.  The transport therefore **always seeks to the bit-gap preceding the
requested position and then plays forward at read speed**, accumulating audio
for each bit passed.  This mirrors the behaviour of a magnetic tape pickup
where data is only available while the tape is in motion beneath the head.
"""

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from ..hardware.cassette_tape import CassetteTapeBackend
try:
    import numpy as np
    from numpy import ndarray
except ModuleNotFoundError:
    np = None


@dataclass
class TapeTransport:
    """Sequential, gate-aware view onto a :class:`CassetteTapeBackend`.

    The transport maintains a *cursor* pointing at the bit index directly
    under the head.  All reads and writes advance this cursor by **playing**
    through intermediate bits so that data is only exposed once physically
    traversed.  Rewinding is permitted, but a fresh seek is required and the
    tape must be played forward again to reach the target location.
    """

    tape: CassetteTapeBackend
    track: int = 0
    lane: int = 0
    _cursor: int = 0
    register_mode: bool = False
    _locked: bool = field(init=False)
    _op_queue: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._locked = self.register_mode

    # ------------------------------------------------------------------
    def queue_operators(self, ops: Sequence[int]) -> None:
        """Provide operator codes that unlock the transport once consumed.

        The codes themselves are placeholders for future analog mixing
        instructions.  For now they merely serve to gate access so that tests
        can verify the register lockout behaviour.
        """

        self._op_queue.extend(int(o) for o in ops)
        if ops:
            self._locked = False

    def _consume_op(self) -> None:
        if not self.register_mode:
            return
        if self._locked or not self._op_queue:
            raise PermissionError("register tape locked; operator code required")
        self._op_queue.pop(0)
        if not self._op_queue:
            self._locked = True

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Total addressable bits on the underlying tape."""
        return self.tape.total_bits

    # ------------------------------------------------------------------
    def seek(self, bit_idx: int) -> None:
        """Seek to ``bit_idx`` and place the head in the preceding gap."""
        self.tape.move_head_to_bit(bit_idx)
        self._cursor = bit_idx

    # ------------------------------------------------------------------
    def _advance_to(self, bit_idx: int) -> None:
        """Play forward until ``bit_idx`` is under the head."""
        if bit_idx < self._cursor:
            # Rewind, landing in the gap before ``bit_idx``
            self.seek(bit_idx)
        while self._cursor < bit_idx:
            # Reading discards the payload but produces the required traversal audio
            self.tape.read_wave(self.track, self.lane, self._cursor)
            self._cursor += 1

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        self._consume_op()
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step <= 0:
                raise ValueError("negative or zero step not supported")
            self._advance_to(start)
            frames: List[ndarray] = []
            for bit in range(start, stop):
                frames.append(self.tape.read_wave(self.track, self.lane, bit))
                self._cursor = bit + 1
            return frames[::step]
        else:
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError("tape index out of range")
            self._advance_to(idx)
            frame = self.tape.read_wave(self.track, self.lane, idx)
            self._cursor = idx + 1
            return frame

    # ------------------------------------------------------------------
    def __setitem__(self, idx, data: Iterable[ndarray] | ndarray) -> None:
        self._consume_op()
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step <= 0:
                raise ValueError("negative or zero step not supported")
            self._advance_to(start)
            data_iter = iter(data)  # type: ignore[arg-type]
            for bit in range(start, stop):
                if (bit - start) % step == 0:
                    try:
                        frame = next(data_iter)
                    except StopIteration:  # pragma: no cover - defensive
                        raise ValueError("not enough data to fill slice") from None
                    self.tape.write_wave(self.track, self.lane, bit, frame)
                else:
                    # traverse intermediate bits with a read to produce audio
                    self.tape.read_wave(self.track, self.lane, bit)
                self._cursor = bit + 1
        else:
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError("tape index out of range")
            self._advance_to(idx)
            self.tape.write_wave(self.track, self.lane, idx, data)  # type: ignore[arg-type]
            self._cursor = idx + 1
