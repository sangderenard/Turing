import numpy as np
import pytest
from analog_spec import FRAME_SAMPLES, generate_bit_wave, nand_wave


def test_generate_bit_wave_shapes():
    one = generate_bit_wave(1, 0)
    zero = generate_bit_wave(0, 0)
    assert one.shape == (FRAME_SAMPLES,)
    assert zero.shape == (FRAME_SAMPLES,)
    assert np.max(np.abs(one)) > 0.0
    assert np.max(np.abs(zero)) == 0.0


def test_nand_wave_unimplemented():
    x = generate_bit_wave(1, 0)
    y = generate_bit_wave(1, 0)
    with pytest.raises(NotImplementedError):
        nand_wave(x, y)
