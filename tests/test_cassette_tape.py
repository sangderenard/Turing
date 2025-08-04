import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from cassette_tape import CassetteTapeBackend


def test_read_write_and_head_movement():
    tape = CassetteTapeBackend(tape_length=8)
    tape.write_bit(3, 1)
    assert tape.read_bit(3) == 1
    assert tape.read_bit(4) == 0
    tape.move_head(5)
    assert tape.head_pos == 5
    tape.move_head(100)
    assert tape.head_pos == 7
    with pytest.raises(IndexError):
        tape.read_bit(-1)


def test_export_ir_requires_analogue_mode():
    tape = CassetteTapeBackend()
    with pytest.raises(RuntimeError):
        tape.export_ir()


def test_analogue_mode_emits_ir_packets():
    np = pytest.importorskip("numpy")
    tape = CassetteTapeBackend(tape_length=32, analogue_mode=True)
    tape.write_bit(0, 1)
    tape.read_bit(0)
    tape.move_head(1)
    lanes, motor, eq = tape.export_ir()
    expected = tape.frame_samples * 3
    assert lanes.shape == (1, expected)
    assert motor.shape == (expected,)
    assert np.allclose(motor[:tape.frame_samples], tape.motor_idle_v)
    assert np.allclose(motor[tape.frame_samples:2*tape.frame_samples], tape.motor_idle_v)
    assert np.allclose(motor[2*tape.frame_samples:], tape.motor_run_v)
    assert np.all(lanes[0, 2*tape.frame_samples:] == 0)
    assert eq == {}
    tape.close()


def test_lane_configuration_updates():
    tape = CassetteTapeBackend()
    tape.set_lane_band_gain(1, 2, 0.5)
    assert tape.lane_band_gains[1][2] == 0.5
    tape.set_lane_eq(1, {"fo": 1000.0})
    assert tape.lane_eq_params[1]["fo"] == 1000.0


def test_audio_thread_generates_waveform():
    np = pytest.importorskip("numpy")
    coeffs = {"write": {1000.0: 1.0}, "read": {500.0: 1.0}, "motor": {250.0: 1.0}}
    tape = CassetteTapeBackend(
        tape_length=16,
        analogue_mode=True,
        lane_band_gains={0: {}},
        op_sine_coeffs=coeffs,
    )
    tape.write_bit(0, 1)
    tape.read_bit(0)
    tape.move_head(1)
    import time
    time.sleep(0.1)
    tape.close()
    assert len(tape._audio_frames) == 3
    t = np.arange(tape.frame_samples) / tape.sample_rate_hz
    expected = tape._env * np.sin(2 * np.pi * 1000.0 * t)
    assert np.allclose(tape._audio_frames[0], expected, atol=1e-4)


def _expected_bus_wave(bins, n):
    import numpy as np
    spec = np.zeros(n, dtype=np.complex64)
    for k in bins:
        spec[k % n] = 1.0
        if k != 0:
            spec[-k % n] = 1.0
    wave = np.fft.ifft(spec).real
    peak = np.max(np.abs(wave)) or 1.0
    return (wave / peak).astype("f4")


def test_configurable_bus_width():
    np = pytest.importorskip("numpy")
    coeffs = {
        "write": {880.0: 1.0},
        "read": {440.0: 1.0},
        "motor": {220.0: 0.5},
        "instr": {330.0: 1.0},
    }
    tape = CassetteTapeBackend(
        tape_length=16,
        analogue_mode=True,
        lane_band_gains={0: {}},
        op_sine_coeffs=coeffs,
    )
    tape.configure_bus_width([1, 2], [3])
    assert np.allclose(
        tape._data_bus_wave, _expected_bus_wave([1, 2], tape.frame_samples)
    )
    tape.execute_instruction()
    import time

    time.sleep(0.1)
    tape.close()
    t = np.arange(tape.frame_samples) / tape.sample_rate_hz
    expected = (
        tape._env
        * _expected_bus_wave([3], tape.frame_samples)
        * np.sin(2 * np.pi * 330.0 * t)
    )
    assert len(tape._audio_frames) == 1
    assert np.allclose(tape._audio_frames[0], expected, atol=1e-4)
