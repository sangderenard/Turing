from __future__ import annotations

"""Tape layout utilities.

This module defines helpers that derive the fixed BIOS, instruction table
and data zone layout of virtual tapes.  Registers are modelled as **fully
independent tape systems** rather than as mere track pairs on the main tape,
reflecting the design where the CPU can drive multiple devices in
simultaneity.  Each register stands alone and contains no nested registers.
They remain fully fledged tape units with motor simulation and noise handled
elsewhere in the hardware stack; this module only describes their digital
addressing layout.

Only the structural addressing rules are modelled here; no analogue physics
is emulated.  The intention is to provide deterministic maps that other
components can consult.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..hardware.analog_spec import (
    BiosHeader,
    BIOS_HEADER_STRUCT,
    header_frames,
    unpack_bios_header,
)
from ..hardware.constants import REGISTERS, LANES


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

    bios: Optional[BiosHeader]
    instruction_frames: int
    is_register: bool = False

    bios_start: int = 0
    instr_start: int = field(init=False)
    data_start: int = field(init=False)

    def __post_init__(self) -> None:
        if self.bios:
            bios_frame_count = len(header_frames(self.bios))
            self.instr_start = self.bios_start + bios_frame_count
        else:
            # When no BIOS is present the tape begins directly with the
            # instruction area which, for registers, is empty.
            self.instr_start = self.bios_start
        self.data_start = self.instr_start + self.instruction_frames

    # ------------------------------------------------------------------
    def encode_bios(self) -> List[List[int]]:
        """Return the BIOS header as parallel bit frames."""
        if self.bios:
            return header_frames(self.bios)
        return []

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

    # ------------------------------------------------------------------
    @staticmethod
    def get_bios_frame_count() -> int:
        """Return the number of frames occupied by a BIOS header."""

        bits = BIOS_HEADER_STRUCT.size * 8
        return (bits + LANES - 1) // LANES


def create_register_tapes(n: int = REGISTERS) -> Dict[int, TapeMap]:
    """Return independent ``TapeMap`` objects for ``n`` registers.

    Registers are modelled as simple data tapes with **no BIOS header** and no
    instruction area.  They still represent fully featured tape devices with
    motor simulation and noise sources, but structurally they expose only a
    data region beginning at frame ``0``.  ``is_register`` is marked so that
    higher layers can lock access until a valid operator sequence is supplied.
    """

    return {
        i: TapeMap(bios=None, instruction_frames=0, is_register=True) for i in range(n)
    }

