from __future__ import annotations

"""Tape layout utilities.

This module defines helpers that derive the fixed BIOS, instruction table
and data zone layout of virtual tapes.  Registers are modelled as **fully
independent tape systems** rather than as mere track pairs on the main tape,
reflecting the design where the CPU can drive multiple devices in
simultaneity.

Only the structural addressing rules are modelled here; no analogue physics
is emulated.  The intention is to provide deterministic digital maps that
other components can consult.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from analog_spec import (
    BiosHeader,
    BIOS_HEADER_STRUCT,
    REGISTERS,
    header_frames,
    unpack_bios_header,
)


@dataclass
class TapeMap:
    """High level layout of the main tape.

    The map divides a tape into three regions:

    * ``BIOS`` – fixed-size header at the start of the tape.
    * ``instructions`` – sequence of 16‑bit words beginning after the BIOS.
    * ``data`` – remaining frames available for runtime data.

    The class is agnostic about physical track allocation; registers use
    separate ``TapeMap`` instances to represent their own external tapes.
    """

    bios: BiosHeader
    instruction_frames: int

    bios_start: int = 0
    instr_start: int = field(init=False)
    data_start: int = field(init=False)

    def __post_init__(self) -> None:
        bios_frame_count = len(header_frames(self.bios))
        self.instr_start = self.bios_start + bios_frame_count
        self.data_start = self.instr_start + self.instruction_frames

    # ------------------------------------------------------------------
    def encode_bios(self) -> List[List[int]]:
        """Return the BIOS header as parallel bit frames."""
        return header_frames(self.bios)

    @staticmethod
    def decode_bios(frames: List[List[int]]) -> BiosHeader:
        """Reconstruct a :class:`BiosHeader` from bit ``frames``."""

        bits: List[int] = []
        for frame in frames:
            bits.extend(frame)
        byte_arr = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for b in bits[i : i + 8]:
                byte = (byte << 1) | b
            byte_arr.append(byte)
            if len(byte_arr) >= BIOS_HEADER_STRUCT.size:
                break
        return unpack_bios_header(bytes(byte_arr[: BIOS_HEADER_STRUCT.size]))


def create_register_tapes(bios: BiosHeader, n: int = REGISTERS) -> Dict[int, TapeMap]:
    """Return independent ``TapeMap`` objects for ``n`` registers.

    Each register is modelled as its own two-track tape with a BIOS header and
    data region but no instruction table.  This mirrors the original hardware
    concept where the CPU can play directly into multiple devices at once.
    """

    return {i: TapeMap(bios, instruction_frames=0) for i in range(n)}

