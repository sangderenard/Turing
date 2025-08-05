import numpy as np

from src.hardware.lane_tuner import LaneTuner
from src.hardware.analog_spec import BASE_FREQ, SEMI_RATIO


def test_c_major_middle_region():
    tuner = LaneTuner()
    freqs = tuner.assign(key="C", mode="major", fundamental="middle")
    base = BASE_FREQ * 2.0
    expected_c4 = base * (SEMI_RATIO ** 3)
    expected_d4 = base * (SEMI_RATIO ** 5)
    assert np.isclose(freqs[0], expected_c4, rtol=1e-6)
    assert np.isclose(freqs[1], expected_d4, rtol=1e-6)


def test_lane_chord_major():
    tuner = LaneTuner()
    chords = tuner.assign(key="C", mode="major", fundamental="middle", lane_chords=True)
    base = BASE_FREQ * 2.0
    root = base * (SEMI_RATIO ** 3)
    major_third = base * (SEMI_RATIO ** 7)
    perfect_fifth = base * (SEMI_RATIO ** 10)
    assert np.allclose(chords[0], [root, major_third, perfect_fifth], rtol=1e-6)


def test_output_parallel_matches_assign():
    tuner = LaneTuner()
    expected = tuner.assign(key="C", mode="major", fundamental="middle")
    out = tuner.output(key="C", mode="major", fundamental="middle")
    assert np.allclose(out, expected, rtol=1e-6)


def test_serialized_output_arpeggiates_pattern():
    tuner = LaneTuner()
    pattern = [0, 2, 1]
    assignments = tuner.assign(key="C", mode="major", fundamental="middle")
    tuner.set_serial(True)
    out = tuner.output(key="C", mode="major", fundamental="middle", pattern=pattern)
    expected = [assignments[i] for i in pattern]
    assert np.allclose(out, expected, rtol=1e-6)
