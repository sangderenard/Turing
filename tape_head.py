from __future__ import annotations

"""Physical tape head with speed-gated queues.

The head mediates all data transfer to the backing store.  Reads and writes
are enqueued and only executed when the motor reports that the tape is
moving at the calibrated read/write speed **and** the head has been put into
explicit read or write mode.  This models the requirement that no data moves
without precise coordination of motor velocity and head activation.
"""

from dataclasses import dataclass, field
import queue
from typing import Optional

from analog_spec import FRAME_SAMPLES

try:  # pragma: no cover - numpy may be absent during import analysis
    import numpy as np
    _Vec = np.ndarray
except ModuleNotFoundError:  # pragma: no cover
    np = None
    _Vec = list


@dataclass
class TapeHead:
    tape: "CassetteTapeBackend"
    speed_tolerance: float = 1e-3
    _read_queue: "queue.Queue[int]" = field(default_factory=queue.Queue)
    _write_queue: "queue.Queue[tuple[int, _Vec]]" = field(default_factory=queue.Queue)
    mode: Optional[str] = None

    def enqueue_read(self, bit_idx: int) -> None:
        self._read_queue.put(bit_idx)

    def enqueue_write(self, bit_idx: int, frame: _Vec) -> None:
        self._write_queue.put((bit_idx, frame))

    # ------------------------------------------------------------------
    def activate(self, mode: str, speed: float) -> Optional[_Vec]:
        """Execute queued transfers if ``speed`` matches read/write speed.

        Returns the PCM frame for the processed read, or ``None`` otherwise.
        """
        self.mode = mode
        if abs(speed - self.tape.read_write_speed_ips) > self.speed_tolerance:
            return None

        if mode == "read" and not self._read_queue.empty():
            bit_idx = self._read_queue.get_nowait()
            current_idx = int(round(self.tape._head_pos_inches * self.tape.bits_per_inch))
            if current_idx != bit_idx:
                raise RuntimeError("head misaligned during read activation")
            return self.tape._tape_frames.get(
                bit_idx, np.zeros(FRAME_SAMPLES, dtype="f4") if np is not None else [0.0] * FRAME_SAMPLES
            )
        if mode == "write" and not self._write_queue.empty():
            bit_idx, frame = self._write_queue.get_nowait()
            current_idx = int(round(self.tape._head_pos_inches * self.tape.bits_per_inch))
            if current_idx != bit_idx:
                raise RuntimeError("head misaligned during write activation")
            self.tape._tape_frames[bit_idx] = frame.astype("f4") if np is not None else frame
        return None
