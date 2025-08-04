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
        a_ms, d_ms, sustain_level, r_ms = DATA_ADSR
        a_n = int(FS * (a_ms / 1000.0))
        d_n = int(FS * (d_ms / 1000.0))
        r_n = int(FS * (r_ms / 1000.0))
        s_n = FRAME_SAMPLES - (a_n + d_n + r_n)
        # ADSR envelope: attack 0→1 over ``a_ms``, decay to ``sustain_level`` over
        # ``d_ms``, sustain constant for ``s_n`` samples, then release to 0 over
        # ``r_ms``.  Durations and level come directly from AGENTS.md's
        # ``DATA_ADSR`` and fill the full ``BIT_FRAME_MS`` of 500 ms.
        env = np.concatenate(
            [
                np.linspace(0.0, 1.0, a_n, endpoint=False),
                np.linspace(1.0, sustain_level, d_n, endpoint=False),
                np.full(s_n, sustain_level),
                np.linspace(sustain_level, 0.0, r_n, endpoint=True),
            ]
        )
        sine = np.sin(2 * np.pi * freq * t)
        return (sine * env).astype("f4")
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
    """Analogue NAND operator using amplitude summing.

    The two input waves are summed and the peak amplitude is measured.  A
    result of ``0`` is represented by a silent frame when the summed peak
    exceeds the 1.5A threshold (both inputs high).  Otherwise a unity sine wave
    is emitted as a placeholder for a logical ``1``.  The lane frequency is
    currently fixed to lane ``0``; future work must analyse the actual carrier
    of ``x``/``y`` and reproduce it exactly.
    """
    summed = x + y
    peak = float(np.max(np.abs(summed)))
    if peak >= 1.5:
        return np.zeros_like(x)
    # TODO: preserve original lane frequency and amplitude
    return generate_bit_wave(1, 0)


def sigma_L(frames: List[np.ndarray], k: int) -> List[np.ndarray]:
    """σ_L^k – append ``k`` silent frames to the sequence."""
    return frames + zeros(k)


def sigma_R(frames: List[np.ndarray], k: int) -> List[np.ndarray]:
    """σ_R^k – drop the last ``k`` frames from ``frames"."""
    if k <= 0:
        return list(frames)
    return list(frames[:-k]) if k < len(frames) else []


def concat(x: List[np.ndarray], y: List[np.ndarray]) -> List[np.ndarray]:
    """Concatenate two frame lists."""
    return list(x) + list(y)


def slice_frames(x: List[np.ndarray], i: int, j: int) -> List[np.ndarray]:
    """Return frames in the half-open interval ``[i, j)``."""
    return list(x[i:j])


def mu(x: List[np.ndarray], y: List[np.ndarray], sel: List[np.ndarray]) -> List[np.ndarray]:
    """Amplitude-gated selector.

    For each index the selector frame's peak amplitude decides whether the
    output takes the frame from ``x`` (peak < 0.5) or from ``y`` (peak ≥ 0.5).
    This is a coarse placeholder for a true VCA-based implementation.
    """
    out: List[np.ndarray] = []
    for fx, fy, fs in zip(x, y, sel):
        if np.max(np.abs(fs)) >= 0.5:
            out.append(fy)
        else:
            out.append(fx)
    return out


def length(frames: List[np.ndarray]) -> int:
    """Return the number of frames.

    TODO: model mechanical timing and encode the result as PCM frames.
    """
    return len(frames)


def zeros(n: int) -> List[np.ndarray]:
    """Return ``n`` silent frames."""
    return [np.zeros(FRAME_SAMPLES, dtype="f4") for _ in range(n)]

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
