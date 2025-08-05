import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.hardware.cassette_tape import CassetteTapeBackend
from src.hardware.analog_spec import trapezoidal_motor_envelope, MotorCalibration, BIT_FRAME_MS, FS


def test_motor_envelope_integrates_time():
    calib = MotorCalibration(fast_wind_ms=50.0, read_speed_ms=100.0, drift_ms=0.0)
    frames = 5
    env = trapezoidal_motor_envelope(frames, calib, "read")
    expected = frames * BIT_FRAME_MS / 1000.0
    area = float(np.sum(env)) / FS
    assert pytest.approx(expected, rel=1e-3) == area
    assert env[0] == pytest.approx(0.0)
    assert env[-1] == pytest.approx(0.0)


def test_simulate_movement_writes_audio():
    tape = CassetteTapeBackend(time_scale_factor=0.0)
    tape._simulate_movement(1.0, tape.seek_speed_ips, 1, "seek")
    assert tape._audio_cursor > 0
    assert np.any(tape._audio_buffer[:tape._audio_cursor])
    tape.close()

