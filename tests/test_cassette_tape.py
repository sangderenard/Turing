import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np

from cassette_tape import CassetteTapeBackend
from analog_spec import generate_bit_wave


def test_read_write_emits_audio():
    tape = CassetteTapeBackend(
        tape_length_inches=0.02,
        op_sine_coeffs={"read": {440.0: 1.0}, "write": {880.0: 1.0}, "motor": {60.0: 0.5}},
    )
    frame = generate_bit_wave(1, 0)
    tape.write_wave(0, frame)
    out = tape.read_wave(0)
    assert np.max(np.abs(out)) > 0.0
    assert tape._audio_cursor > 0
    tape.close()


def test_move_head_to_bit_and_write_wrapper():
    tape = CassetteTapeBackend(tape_length_inches=0.02)
    tape.move_head_to_bit(5)
    tape.write_bit(5, 1)
    assert tape.read_bit(5) == 1
    tape.close()


def test_head_gates_on_speed():
    tape = CassetteTapeBackend(tape_length_inches=0.02)
    # prepare data and head position
    frame = generate_bit_wave(1, 0)
    tape._tape_frames[0] = frame
    tape.move_head_to_bit(0)
    tape._head.enqueue_read(0)
    # incorrect speed -> queue remains
    assert tape._head.activate('read', 0.5) is None
    # correct speed unlocks transfer
    out = tape._head.activate('read', tape.read_write_speed_ips)
    assert np.max(np.abs(out)) > 0
    tape.close()
