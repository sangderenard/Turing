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


@dataclass
class DominantTone:
    """Result of analysing a wave's strongest FFT bin."""

    bin: int
    freq: float
    vector: complex
    amp: float


def dominant_tone(wave: np.ndarray) -> DominantTone:
    """Return the dominant FFT bin and its complex vector for ``wave``.

    ``amp`` is reconstructed from the FFT magnitude to match the original
    peak amplitude.  Silence defaults to lane ``0``'s frequency and zero vector.
    """
    fft = np.fft.rfft(wave)
    mags = np.abs(fft)
    mags[0] = 0.0  # ignore DC component
    idx = int(np.argmax(mags))
    vector = complex(fft[idx])
    amp = float(2 * np.abs(vector) / len(wave))
    if amp == 0.0:
        return DominantTone(0, lane_frequency(0), 0j, 0.0)
    freqs = np.fft.rfftfreq(len(wave), 1 / FS)
    return DominantTone(idx, float(freqs[idx]), vector, amp)

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

    Both operands are assumed to occupy the same carrier lane.  Their waves are
    summed and the peak amplitude of the result is compared against ``1.5A``
    where ``A`` is the larger operand amplitude.  Crossing this threshold means
    both inputs were high and the NAND output is silence.  Otherwise a waveform
    for logical ``1`` is reconstructed from the louder input's dominant FFT
    vector, preserving its frequency, amplitude, and phase.  This remains a
    placeholder and ignores proper envelopes.
    """
    summed = x + y
    peak_sum = float(np.max(np.abs(summed)))
    tone_x = dominant_tone(x)
    tone_y = dominant_tone(y)
    A = max(tone_x.amp, tone_y.amp)
    if A > 0.0 and peak_sum >= 1.5 * A:
        return np.zeros_like(x)
    if A == 0.0:
        t = np.linspace(0, BIT_FRAME_MS / 1000.0, FRAME_SAMPLES, endpoint=False)
        return np.sin(2 * np.pi * lane_frequency(0) * t).astype("f4")
    tone = tone_x if tone_x.amp >= tone_y.amp else tone_y
    spectrum = np.zeros(FRAME_SAMPLES // 2 + 1, dtype=complex)
    spectrum[tone.bin] = tone.vector
    return np.fft.irfft(spectrum, n=FRAME_SAMPLES).astype("f4")


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
