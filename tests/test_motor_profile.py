import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cassette_tape import CassetteTapeBackend


def test_speed_profile_integrates_distance():
    tape = CassetteTapeBackend(time_scale_factor=0.0)
    distance = 2.0
    profile = tape._generate_speed_profile(distance, tape.seek_speed_ips)
    dt = 1.0 / tape.sample_rate_hz
    travelled = float(np.sum(profile)) * dt
    assert pytest.approx(travelled, rel=1e-3) == distance
    assert profile[0] < 1e-2
    assert profile[-1] < 1e-2
    tape.close()


def test_simulate_movement_writes_audio():
    tape = CassetteTapeBackend(time_scale_factor=0.0)
    tape._simulate_movement(1.0, tape.seek_speed_ips, 1, "seek")
    assert tape._audio_cursor > 0
    assert np.any(tape._audio_buffer[:tape._audio_cursor])
    tape.close()

