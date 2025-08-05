import pytest
import numpy as np

from src.hardware.cassette_tape import CassetteTapeBackend
from src.hardware.analog_spec import generate_bit_wave, BIT_FRAME_MS, FRAME_SAMPLES, WRITE_BIAS, BIAS_AMP
from src.turing_machine.tape_transport import TapeTransport


def test_read_write_emits_audio():
    tape = CassetteTapeBackend(
        tape_length_inches=0.02,
        op_sine_coeffs={"read": {440.0: 1.0}, "write": {880.0: 1.0}, "motor": {60.0: 0.5}},
    )
    frame0 = generate_bit_wave(1, 0)
    tape.write_wave(0, 0, 0, frame0)
    out0 = tape.read_wave(0, 0, 0)
    frame1 = generate_bit_wave(1, 1)
    tape.write_wave(0, 1, 1, frame1)
    out1 = tape.read_wave(0, 1, 1)
    assert np.max(np.abs(out0)) > 0.0
    assert np.max(np.abs(out1)) > 0.0
    assert tape._audio_cursor > 0
    tape.close()


def test_move_head_to_bit_and_write_wrapper():
    tape = CassetteTapeBackend(tape_length_inches=0.02)
    tape.move_head_to_bit(5)
    tape.write_bit(0, 0, 5, 1)
    assert tape.read_bit(0, 0, 5) == 1
    tape.write_bit(0, 1, 6, 1)
    assert tape.read_bit(0, 1, 6) == 1
    assert tape.read_bit(0, 0, 6) == 0
    tape.close()


def test_head_gates_on_speed():
    tape = CassetteTapeBackend(tape_length_inches=0.02)
    # prepare data and head position
    frame = generate_bit_wave(1, 0)
    tape._tape_frames[(0, 0, 0)] = frame
    tape.move_head_to_bit(0)
    tape._head.enqueue_read(0, 0, 0)
    # incorrect speed -> queue remains
    assert tape._head.activate(0, 'read', 0.5) is None
    # correct speed unlocks transfer
    out = tape._head.activate(0, 'read', tape.read_write_speed_ips)
    assert np.max(np.abs(out)) > 0
    tape.close()


def test_write_adds_bias_tone():
    tape = CassetteTapeBackend(tape_length_inches=0.02)
    frame = generate_bit_wave(1, 0)
    tape.write_wave(0, 0, 0, frame)
    out = tape.read_wave(0, 0, 0)
    t = np.linspace(0, BIT_FRAME_MS / 1000.0, FRAME_SAMPLES, endpoint=False)
    bias_wave = BIAS_AMP * np.sin(2 * np.pi * WRITE_BIAS * t)
    diff = out - frame
    assert np.max(np.abs(diff)) > 0.0
    assert abs(np.dot(diff, bias_wave)) > 1e-5
    tape.close()


def test_status_callback_receives_updates():
    updates = []

    def cb(tape_pos, motor_pos, reading, writing):
        updates.append((tape_pos, reading, writing))

    tape = CassetteTapeBackend(tape_length_inches=0.02, status_callback=cb)
    frame = generate_bit_wave(1, 0)
    tape.write_wave(0, 0, 0, frame)
    assert updates, "callback was not invoked"
    tape_pos, reading_flag, writing_flag = updates[-1]
    assert writing_flag and not reading_flag
    assert tape_pos[1] > 0
    tape.close()


def test_transport_pythonic_index_and_slice():
    tape = CassetteTapeBackend(tape_length_inches=0.02)
    transport = TapeTransport(tape, track=0, lane=0)
    frame = generate_bit_wave(1, 0)
    data = [frame, frame, frame]
    # Pure slice assignment writes the frames sequentially
    transport[:3] = data
    before = tape._audio_cursor
    first = transport[0]
    rest = transport[1:3]
    after = tape._audio_cursor
    assert np.max(np.abs(first)) > 0.0
    assert len(rest) == 2 and np.max(np.abs(rest[1])) > 0.0
    # audio cursor advanced as the tape traversed bits 0-2
    assert after > before
    assert transport._cursor == 3
    tape.close()
