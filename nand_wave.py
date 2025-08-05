"""Analogue NAND operator implementation.

This module follows the stand-alone implementation guide provided in the
repository instructions.  It defines the helper utilities and the
``nand_wave`` function supporting ``parallel`` and ``dominant`` modes.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 0. Global constants

FS = 44_100  # sample rate (Hz)
BIT_FRAME_MS = 500  # frame length (ms)
FRAME_SAMPLES = int(FS * BIT_FRAME_MS / 1000)
LANES = 32  # carrier lanes
SEMI_RATIO = 2 ** (1 / 12)  # 12-TET
BASE_FREQ = 110.0  # lane 0 frequency (Hz)


def lane_frequency(lane: int) -> float:
    """Return the base frequency for ``lane``."""
    return BASE_FREQ * (SEMI_RATIO ** lane)


# ---------------------------------------------------------------------------
# 1. Static lane → FFT-bin map

FREQS = np.fft.rfftfreq(FRAME_SAMPLES, 1 / FS)
BIN_IDX = {ln: int(np.argmin(np.abs(FREQS - lane_frequency(ln)))) for ln in range(LANES)}


def lane_band(lane: int, half_bw: int = 3) -> slice:
    """Return FFT-bin slice around the lane's centre bin."""
    c = BIN_IDX[lane]
    return slice(max(c - half_bw, 0), c + half_bw + 1)


# ---------------------------------------------------------------------------
# 2. Energy detectors


def lane_rms(wave: np.ndarray, lane: int, half_bw: int = 3) -> float:
    """RMS energy of ``wave`` on ``lane``."""
    spec = np.fft.rfft(wave.astype("f4"))
    band = lane_band(lane, half_bw)
    return float(np.sqrt(np.mean(np.abs(spec[band]) ** 2)))


def track_rms(wave: np.ndarray) -> float:
    """Whole-track RMS energy."""
    return float(np.sqrt(np.mean(wave.astype("f4") ** 2)))


# ---------------------------------------------------------------------------
# 3. Envelope-preserving helpers


def extract_lane(wave: np.ndarray, lane: int, half_bw: int = 3) -> np.ndarray:
    """Isolate the lane's time-domain signal without altering its envelope."""
    spec = np.fft.rfft(wave.astype("f4"))
    mask = np.zeros_like(spec)
    mask[lane_band(lane, half_bw)] = 1.0
    lane_wave = np.fft.irfft(spec * mask, n=FRAME_SAMPLES)
    zc = np.argmax((lane_wave[:-1] < 0) & (lane_wave[1:] >= 0))
    return np.roll(lane_wave, -zc).astype("f4")


def replay_envelope(env_wave: np.ndarray, lane: int) -> np.ndarray:
    """Replay ``env_wave``'s envelope on ``lane`` starting at phase 0."""
    env = np.abs(env_wave)
    t = np.arange(FRAME_SAMPLES, dtype="f4") / FS
    carrier = np.sin(2 * np.pi * lane_frequency(lane) * t)
    return (env * carrier).astype("f4")


# ---------------------------------------------------------------------------
# 4. Canonical "fresh-1" generator


def generate_bit_wave(lane: int) -> np.ndarray:
    """Return a 500 ms ADSR tone for ``lane`` starting at phase 0."""
    a, d, s_level, r = 50, 50, 0.8, 100
    aN = int(FS * a / 1000)
    dN = int(FS * d / 1000)
    rN = int(FS * r / 1000)
    sN = FRAME_SAMPLES - (aN + dN + rN)
    env = np.concatenate(
        [
            np.linspace(0, 1, aN, endpoint=False),
            np.linspace(1, s_level, dN, endpoint=False),
            np.full(sN, s_level),
            np.linspace(s_level, 0, rN, endpoint=True),
        ]
    )
    t = np.arange(FRAME_SAMPLES) / FS
    car = np.sin(2 * np.pi * lane_frequency(lane) * t)
    return (env * car).astype("f4")


# ---------------------------------------------------------------------------
# 5. nand_wave – algorithm specification


def nand_wave(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mode: str = "parallel",
    target_lane: int | None = None,
    lane_mask: int | None = None,
    energy_thresh: float = 0.01,
) -> np.ndarray:
    """Return NAND combination of ``x`` and ``y`` according to ``mode``."""

    if mode not in {"parallel", "dominant"}:
        raise ValueError("mode must be 'parallel' or 'dominant'")

    if mode == "parallel":
        out = np.zeros(FRAME_SAMPLES, dtype="f4")
        for lane in range(LANES):
            if lane_mask is not None and not (lane_mask & (1 << lane)):
                continue
            x_on = lane_rms(x, lane) > energy_thresh
            y_on = lane_rms(y, lane) > energy_thresh
            if not x_on and not y_on:
                out += generate_bit_wave(lane)
            elif x_on and not y_on:
                out += extract_lane(x, lane)
            elif y_on and not x_on:
                out += extract_lane(y, lane)
            # if both on: NAND -> 0; add nothing
        peak = float(np.max(np.abs(out)))
        if peak > 1.0:
            out *= 1.0 / peak
        return out.astype("f4")

    # dominant mode
    if target_lane is None:
        raise ValueError("target_lane required for dominant mode")

    x_on = track_rms(x) > energy_thresh
    y_on = track_rms(y) > energy_thresh
    if x_on and y_on:
        out = np.zeros(FRAME_SAMPLES, dtype="f4")
    elif x_on and not y_on:
        rms_values = [lane_rms(x, ln) for ln in range(LANES)]
        strongest = int(np.argmax(rms_values))
        env = extract_lane(x, strongest)
        out = replay_envelope(env, target_lane)
    elif y_on and not x_on:
        rms_values = [lane_rms(y, ln) for ln in range(LANES)]
        strongest = int(np.argmax(rms_values))
        env = extract_lane(y, strongest)
        out = replay_envelope(env, target_lane)
    else:  # both off
        out = generate_bit_wave(target_lane)
    peak = float(np.max(np.abs(out)))
    if peak > 1.0:
        out *= 1.0 / peak
    return out.astype("f4")

