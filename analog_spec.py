"""Analogue specification skeleton based on AGENTS.md.

This module collects constants and minimal placeholder implementations for the
analogue Turing-tape system.  Each function includes TODO notes describing the
missing physical modelling required for a faithful simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np
import math

# ---------------------------------------------------------------------------
# 1. Global Parameters
LANES = 32
TRACKS = 2
REGISTERS = 3
BIT_FRAME_MS = 500
FS = 44_100
BASE_FREQ = 110.0
SEMI_RATIO = 2 ** (1 / 12)
MOTOR_CARRIER = 60.0
WRITE_BIAS = 150.0
DATA_ADSR = (50, 50, 0.8, 100)  # attack, decay, sustain, release in ms
FRAME_SAMPLES = int(FS * (BIT_FRAME_MS / 1000.0))


def lane_frequency(lane: int) -> float:
    """Return the base frequency for a lane."""
    return BASE_FREQ * (SEMI_RATIO ** lane)


def generate_bit_wave(bit: int, lane: int) -> np.ndarray:
    """Generate a placeholder PCM frame for a single bit on the given lane."""
    t = np.linspace(0, BIT_FRAME_MS / 1000.0, FRAME_SAMPLES, endpoint=False)
    if bit:
        freq = lane_frequency(lane)
        # TODO: apply full ADSR envelope instead of unity gain
        return np.sin(2 * np.pi * freq * t).astype("f4")
    return np.zeros_like(t, dtype="f4")

# ---------------------------------------------------------------------------
# 3. Instruction Word

class Opcode(Enum):
    SEEK = 0x0
    READ = 0x1
    WRITE = 0x2
    NAND = 0x3
    SIGL = 0x4
    SIGR = 0x5
    CONCAT = 0x6
    SLICE = 0x7
    MU = 0x8
    LENGTH = 0x9
    ZEROS = 0xA
    HALT = 0xF


@dataclass
class InstructionWord:
    opcode: Opcode
    reg_a: int
    reg_b: int
    dest: int
    param: int

# ---------------------------------------------------------------------------
# 4. Register Behaviour (skeletal)

@dataclass
class Register:
    """Placeholder register holding a list of bit frames."""
    frames: List[np.ndarray]

    def read(self, i: int) -> np.ndarray:
        return self.frames[i]

# ---------------------------------------------------------------------------
# 6. Primitive Operator Stubs


def nand_wave(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analogue NAND operator.

    Full analogue amplitude summing and thresholding at 1.5A has not yet been
    modelled.  No digital shortcut is permitted, so this function is left
    unimplemented until a faithful signal-level simulation is provided.
    """
    raise NotImplementedError("analogue NAND not implemented")


def sigma_L(frames: List[np.ndarray], k: int) -> List[np.ndarray]:
    """σ_L^k – append k silent frames."""
    raise NotImplementedError("σ_L not implemented")


def sigma_R(frames: List[np.ndarray], k: int) -> List[np.ndarray]:
    """σ_R^k – drop last k frames."""
    raise NotImplementedError("σ_R not implemented")


def concat(x: List[np.ndarray], y: List[np.ndarray]) -> List[np.ndarray]:
    raise NotImplementedError("concat not implemented")


def slice_frames(x: List[np.ndarray], i: int, j: int) -> List[np.ndarray]:
    raise NotImplementedError("slice not implemented")


def mu(x: List[np.ndarray], y: List[np.ndarray], sel: List[np.ndarray]) -> List[np.ndarray]:
    raise NotImplementedError("mu selector not implemented")


def length(frames: List[np.ndarray]) -> int:
    raise NotImplementedError("length not implemented")


def zeros(n: int) -> List[np.ndarray]:
    raise NotImplementedError("zeros not implemented")

# ---------------------------------------------------------------------------
# 8. Audio Event IR

@dataclass
class MidiEvent:
    start_ms: float
    duration_ms: float
    channel: int
    note: int
    velocity: float

# ---------------------------------------------------------------------------
# 9. Execution Modes (placeholder)

class ExecMode(Enum):
    LOGIC_LEADING = 1
    TAPE_LEADING = 2
    NESTED = 3


# ---------------------------------------------------------------------------
# 7 & 10 – Motor control and headers are left for future work.
# TODO: Implement motor envelopes and binary header packing.
