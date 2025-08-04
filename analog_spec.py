"""Analogue specification skeleton based on AGENTS.md.

This module collects constants and minimal placeholder implementations for the
analogue Turing-tape system.  Each function includes TODO notes describing the
missing physical modelling required for a faithful simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List
import struct

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
MOTOR_RAMP_MS = 250  # up/down ramp duration for SEEK envelopes


def lane_frequency(lane: int) -> float:
    """Return the base frequency for a lane."""
    return BASE_FREQ * (SEMI_RATIO ** lane)


def generate_bit_wave(bit: int, lane: int, phase: float = 0.0) -> np.ndarray:
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
        sine = np.sin(2 * np.pi * freq * t + phase)
        return (sine * env).astype("f4")
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
# 7. Motor Control Simulation


@dataclass
class MotorCalibration:
    """Calibration constants for the motor system."""

    fast_wind_ms: float
    read_speed_ms: float
    drift_ms: float


def trapezoidal_motor_envelope(
    distance_frames: int, calib: MotorCalibration, speed: str = "read"
) -> np.ndarray:
    """Return a trapezoidal motor gain envelope for a SEEK movement.

    ``distance_frames`` is the number of bit frames to traverse.  ``calib``
    provides the calibration times from the BIOS block.  ``speed`` selects
    the plateau amplitude: ``"read"`` for nominal speed (amplitude=1.0) or
    ``"fast"`` for the fast-wind factor derived from calibration.

    The envelope ramps up and down over ``MOTOR_RAMP_MS`` and keeps a constant
    plateau so that the integral of the envelope equals the travel time at the
    chosen speed.  When the distance is too short for a full plateau the peak
    amplitude is scaled instead, yielding a triangular profile.
    """

    distance_ms = distance_frames * BIT_FRAME_MS
    up_ms = MOTOR_RAMP_MS
    dn_ms = MOTOR_RAMP_MS
    plateau_amp = 1.0
    if speed == "fast" and calib.fast_wind_ms > 0.0:
        plateau_amp = calib.read_speed_ms / calib.fast_wind_ms
    unit_area = 0.5 * (up_ms + dn_ms)
    max_tri_area = plateau_amp * unit_area
    if distance_ms <= max_tri_area:
        amp = distance_ms / unit_area
        up_n = int(FS * (up_ms / 1000.0))
        dn_n = int(FS * (dn_ms / 1000.0))
        env_up = np.linspace(0.0, amp, up_n, endpoint=False)
        env_dn = np.linspace(amp, 0.0, dn_n, endpoint=True)
        return np.concatenate([env_up, env_dn]).astype("f4")
    coast_ms = max(distance_ms / plateau_amp - unit_area, 0.0)
    up_n = int(FS * (up_ms / 1000.0))
    coast_n = int(FS * (coast_ms / 1000.0))
    dn_n = int(FS * (dn_ms / 1000.0))
    env_up = np.linspace(0.0, plateau_amp, up_n, endpoint=False)
    env_coast = np.full(coast_n, plateau_amp)
    env_dn = np.linspace(plateau_amp, 0.0, dn_n, endpoint=True)
    return np.concatenate([env_up, env_coast, env_dn]).astype("f4")

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
    """Analogue NAND combining lane spectra.

    For each lane the function inspects the spectral energy of ``x`` and
    ``y`` at the lane's base frequency.  If both inputs present a '1' the
    result is silence for that lane.  If exactly one input is '1' the
    corresponding spectral component (with its phase) is copied to the output.
    When both inputs are '0' a canonical ``generate_bit_wave`` tone is
    inserted for that lane.  Only the lane bins are manipulated; all other
    spectral content is discarded.
    """
    fft_x = np.fft.rfft(x)
    fft_y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(x), 1 / FS)

    amp_threshold = 0.1
    mag_threshold = amp_threshold * FRAME_SAMPLES / 2.0

    output_spectrum = np.zeros_like(fft_x)
    active_any = False
    for lane in range(LANES):
        freq_idx = np.argmin(np.abs(freqs - lane_frequency(lane)))
        mag_x = np.abs(fft_x[freq_idx])
        mag_y = np.abs(fft_y[freq_idx])
        x_is_1 = mag_x > mag_threshold
        y_is_1 = mag_y > mag_threshold
        if x_is_1 or y_is_1:
            active_any = True
            if x_is_1 and y_is_1:
                continue
            if x_is_1:
                output_spectrum[freq_idx] = fft_x[freq_idx]
            elif y_is_1:
                output_spectrum[freq_idx] = fft_y[freq_idx]
        # lanes with both 0 are ignored unless no lane is active at all

    if not active_any:
        idx0 = np.argmin(np.abs(freqs - lane_frequency(0)))
        base_fft0 = np.fft.rfft(generate_bit_wave(1, 0))
        output_spectrum[idx0] = base_fft0[idx0]

    return np.fft.irfft(output_spectrum, n=FRAME_SAMPLES).astype("f4")


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
# 10. Header & Metadata


MAGIC_ID = b"TURINGv1"


@dataclass
class BiosHeader:
    """Fixed-length BIOS header as per AGENTS.md."""

    calib_fast_ms: float
    calib_read_ms: float
    drift_ms: float
    inputs: List[int]
    outputs: List[int]
    instr_start_addr: int


BIOS_HEADER_STRUCT = struct.Struct("<8sfffB32sB32sI6s")


def pack_bios_header(h: BiosHeader, magic: bytes = MAGIC_ID) -> bytes:
    """Pack ``h`` into a fixed-length binary header."""

    inputs = bytes(h.inputs + [0xFF] * (32 - len(h.inputs)))
    outputs = bytes(h.outputs + [0xFF] * (32 - len(h.outputs)))
    return BIOS_HEADER_STRUCT.pack(
        magic,
        float(h.calib_fast_ms),
        float(h.calib_read_ms),
        float(h.drift_ms),
        len(h.inputs),
        inputs,
        len(h.outputs),
        outputs,
        int(h.instr_start_addr),
        b"\x00" * 6,
    )


def unpack_bios_header(data: bytes) -> BiosHeader:
    """Unpack a ``BiosHeader`` from ``data``."""

    unpacked = BIOS_HEADER_STRUCT.unpack(data)
    _, fast_ms, read_ms, drift_ms, n_in, in_bytes, n_out, out_bytes, addr, _ = unpacked
    inputs = [b for b in in_bytes[:n_in]]
    outputs = [b for b in out_bytes[:n_out]]
    return BiosHeader(fast_ms, read_ms, drift_ms, inputs, outputs, addr)


def header_frames(h: BiosHeader) -> List[List[int]]:
    """Serialise ``h`` across lanes as parallel frames of bits."""

    packed = pack_bios_header(h)
    bits = []
    for byte in packed:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    frames: List[List[int]] = []
    for i in range(0, len(bits), LANES):
        frame = bits[i : i + LANES]
        if len(frame) < LANES:
            frame += [0] * (LANES - len(frame))
        frames.append(frame)
    return frames
