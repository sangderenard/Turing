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
from typing import Dict, Optional, Tuple

from analog_spec import FRAME_SAMPLES, BIT_FRAME_MS, WRITE_BIAS, BIAS_AMP

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
    _read_queues: Dict[int, queue.Queue[Tuple[int, int]]] = field(default_factory=dict)
    _write_queues: Dict[int, queue.Queue[Tuple[int, int, _Vec]]] = field(default_factory=dict)
    mode: Optional[str] = None

    def enqueue_read(self, track: int, lane: int, bit_idx: int) -> None:
        self._read_queues.setdefault(track, queue.Queue()).put((lane, bit_idx))

    def enqueue_write(self, track: int, lane: int, bit_idx: int, frame: _Vec) -> None:
        self._write_queues.setdefault(track, queue.Queue()).put((lane, bit_idx, frame))

    # ------------------------------------------------------------------
    def activate(self, track: int, mode: str, speed: float) -> Optional[_Vec]:
        """Execute queued transfers if ``speed`` matches read/write speed.

        Returns the PCM frame for the processed read, or ``None`` otherwise.
        """
        self.mode = mode
        if abs(speed - self.tape.read_write_speed_ips) > self.speed_tolerance:
            return None

        if mode == "read":
            q = self._read_queues.get(track)
            if q and not q.empty():
                lane, bit_idx = q.get_nowait()
                current_idx = int(round(self.tape._head_pos_inches * self.tape.bits_per_inch))
                if current_idx != bit_idx:
                    raise RuntimeError("head misaligned during read activation")
                return self.tape._tape_frames.get(
                    (track, lane, bit_idx),
                    np.zeros(FRAME_SAMPLES, dtype="f4") if np is not None else [0.0] * FRAME_SAMPLES,
                )
        if mode == "write":
            q = self._write_queues.get(track)
            if q and not q.empty():
                lane, bit_idx, frame = q.get_nowait()
                current_idx = int(round(self.tape._head_pos_inches * self.tape.bits_per_inch))
                if current_idx != bit_idx:
                    raise RuntimeError("head misaligned during write activation")
                if np is not None:
                    t = np.linspace(0, BIT_FRAME_MS / 1000.0, FRAME_SAMPLES, endpoint=False)
                    # Bias tone sits at the RF noise floor divided across sources
                    bias = BIAS_AMP * np.sin(2 * np.pi * WRITE_BIAS * t)
                    frame = frame + bias.astype("f4")
                self.tape._tape_frames[(track, lane, bit_idx)] = frame.astype("f4") if np is not None else frame
                return frame
        return None
