import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from analog_spec import (
    FRAME_SAMPLES,
    FS,
    generate_bit_wave,
    lane_frequency,
    dominant_tone,
    nand_wave,
    sigma_L,
    sigma_R,
)


def test_generate_bit_wave_shapes():
    one = generate_bit_wave(1, 0)
    zero = generate_bit_wave(0, 0)
    assert one.shape == (FRAME_SAMPLES,)
    assert zero.shape == (FRAME_SAMPLES,)
    assert np.max(np.abs(one)) > 0.0
    assert np.max(np.abs(zero)) == 0.0


def test_nand_wave_behaviour():
    x1 = generate_bit_wave(1, 0)
    x0 = generate_bit_wave(0, 0)
    assert np.max(np.abs(nand_wave(x1, x1))) == 0.0
    assert np.max(np.abs(nand_wave(x1, x0))) > 0.0
    assert np.max(np.abs(nand_wave(x0, x0))) > 0.0


def test_nand_wave_preserves_characteristics():
    freq = lane_frequency(3)
    t = np.linspace(0, FRAME_SAMPLES / FS, FRAME_SAMPLES, endpoint=False)
    src = (0.5 * np.sin(2 * np.pi * freq * t + 0.3)).astype("f4")
    out = nand_wave(src, np.zeros_like(src))
    src_tone = dominant_tone(src)
    out_tone = dominant_tone(out)
    assert out_tone.amp == pytest.approx(src_tone.amp, rel=1e-2)
    assert abs(out_tone.freq - src_tone.freq) < 1.0
    assert out_tone.vector == pytest.approx(src_tone.vector, rel=1e-2, abs=1e-2)


def test_sigma_ops():
    frames = [generate_bit_wave(1, 0)] * 2
    appended = sigma_L(frames, 1)
    assert len(appended) == 3
    assert np.max(np.abs(appended[-1])) == 0.0
    trimmed = sigma_R(appended, 2)
    assert len(trimmed) == 1
