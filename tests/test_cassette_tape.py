import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np

from cassette_tape import CassetteTapeBackend


def test_read_write_emits_audio():
    tape = CassetteTapeBackend(
        tape_length_inches=0.02,
        op_sine_coeffs={"read": {440.0: 1.0}, "write": {880.0: 1.0}, "motor": {60.0: 0.5}},
    )
    tape.write_bit(0, 1)
    assert tape.read_bit(0) == 1
    assert tape._audio_cursor > 0
    tape.close()


def test_move_head_to_bit_and_write():
    tape = CassetteTapeBackend(tape_length_inches=0.02)
    tape.move_head_to_bit(5)
    tape.write_bit(5, 1)
    assert tape.read_bit(5) == 1
    tape.close()
