"""Lightweight shared constants for analogue hardware modules.

Extracting these values avoids importing heavy simulation code when only
basic parameters are required.
"""
from __future__ import annotations

import math

LANES = 32
TRACKS = 2
REGISTERS = 3
BIT_FRAME_MS = 50
REFERENCE_BIT_FRAME_MS = 500.0
FS = 44_100
BASE_FREQ = 110.0
SEMI_RATIO = 2 ** (1 / 12)
MOTOR_CARRIER = 60.0
WRITE_BIAS = 150.0
DATA_ADSR = (1, 2, 5, 1, 1.0, 0.8)
FRAME_SAMPLES = int(FS * (BIT_FRAME_MS / 1000.0))
MOTOR_RAMP_MS = 75
PLATEAU_AMP = 10.0
SIMULATION_VOLUME = 0.8
ATTACK_LEVEL = DATA_ADSR[4]
SUSTAIN_LEVEL = DATA_ADSR[5]
NOISE_FLOOR_DB = -60.0
NOISE_SOURCES = 3
BIAS_AMP = 10 ** ((NOISE_FLOOR_DB - 10 * math.log10(NOISE_SOURCES)) / 20)


def lane_frequency(lane: int) -> float:
    """Return the base frequency for ``lane``."""
    return BASE_FREQ * (SEMI_RATIO ** lane)
