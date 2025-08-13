import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from src.hardware.analog_spec import generate_bit_wave, sigma_L, sigma_R
from src.hardware.constants import FRAME_SAMPLES


def test_generate_bit_wave_shapes():
    one = generate_bit_wave(1, 0)
    zero = generate_bit_wave(0, 0)
    assert one.shape == (FRAME_SAMPLES,)
    assert zero.shape == (FRAME_SAMPLES,)
    assert np.max(np.abs(one)) > 0.0
    assert np.max(np.abs(zero)) == 0.0


def test_sigma_ops():
    frames = [generate_bit_wave(1, 0)] * 2
    appended = sigma_L(frames, 1)
    assert len(appended) == 3
    assert np.max(np.abs(appended[-1])) == 0.0
    trimmed = sigma_R(appended, 2)
    assert len(trimmed) == 1


def test_sigma_envelope_edges():
    frames = [np.ones(FRAME_SAMPLES, dtype="f4") for _ in range(2)]
    left = sigma_L(frames, 0)
    assert left[0][0] == pytest.approx(0.0, abs=1e-3)
    assert left[-1][-1] == pytest.approx(0.0, abs=1e-3)
    right = sigma_R(frames, 0)
    assert right[0][0] == pytest.approx(0.0, abs=1e-3)
    assert right[-1][-1] == pytest.approx(0.0, abs=1e-3)
