"""Node-local phase clocks driven by real measured operation time, and their
batched spectral decomposition.

Covers ``src/common/tensors/abstract_convolution/node_profile_phase.py``:
real profiling (via ``shell_telemetry.TelemetryChannel``, not a synthetic
clock), zero information loss per tick, and a genuinely batched
``fft(axis=-1)`` over stacked node trajectories rather than a per-node loop.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.common.tensors import AbstractTensor as AT
from src.compiler.shell_telemetry import PROFILE, TelemetryChannel
from src.common.tensors.abstract_convolution.node_profile_phase import (
    NodePhaseClock,
    local_spectrum,
    spectral_cube,
)


def _np(t) -> np.ndarray:
    return np.asarray(t.data if hasattr(t, "data") else t)


def _busy_work(size: int) -> None:
    """A real operation whose cost genuinely scales with ``size`` -- not a
    sleep, actual floating-point work, so its measured duration is a real
    signal and not an artifact of a fixed delay."""

    a = [[float(i * j) for j in range(size)] for i in range(size)]
    total = 0.0
    for row in a:
        for value in row:
            total += value * value
    return total


# --------------------------------------------------------------------------
# real measurement, zero information loss
# --------------------------------------------------------------------------

def test_tick_records_real_profile_events_on_the_shared_channel():
    channel = TelemetryChannel(name="test")
    clock = NodePhaseClock(node=5, omega=1.0, channel=channel)

    clock.tick(lambda: _busy_work(20))
    clock.tick(lambda: _busy_work(20))

    records = [r for r in channel.records if r.kind == PROFILE]
    assert len(records) == 2
    assert all(r.detail["node"] == 5 for r in records)
    assert all(r.detail["nanoseconds"] > 0 for r in records)


def test_tick_count_matches_recorded_sample_count_exactly():
    """No aggregation: N ticks means exactly N phase/intensity samples."""

    channel = TelemetryChannel(name="test")
    clock = NodePhaseClock(node=0, omega=1.0, channel=channel)
    for _ in range(7):
        clock.tick(lambda: _busy_work(5))
    assert clock.sample_count == 7


def test_different_real_costs_produce_different_measured_durations():
    """Distinct workloads must measure as distinct durations -- if every
    tick reported the same number, the clock would be reading a synthetic
    increment, not real time."""

    channel = TelemetryChannel(name="test")
    clock = NodePhaseClock(node=0, omega=1.0, channel=channel)
    clock.tick(lambda: _busy_work(5))
    clock.tick(lambda: _busy_work(80))
    durations = [r.detail["nanoseconds"] for r in channel.records if r.kind == PROFILE]
    assert durations[1] > durations[0]


def test_trajectory_undefined_with_zero_ticks():
    channel = TelemetryChannel(name="test")
    clock = NodePhaseClock(node=0, omega=1.0, channel=channel)
    with pytest.raises(ValueError, match="no ticks"):
        clock.trajectory()


# --------------------------------------------------------------------------
# phase and intensity are exactly the recorded real quantities, re-derived
# --------------------------------------------------------------------------

def test_trajectory_matches_hand_computed_phase_and_intensity():
    channel = TelemetryChannel(name="test")
    clock = NodePhaseClock(node=0, omega=3.0, channel=channel)
    for size in (3, 6, 4):
        clock.tick(lambda size=size: _busy_work(size))

    trajectory = _np(clock.trajectory())

    # Recompute independently from the clock's own recorded elapsed time,
    # matching the module's documented construction exactly.
    seconds = [r.detail["nanoseconds"] / 1e9 for r in channel.records if r.kind == PROFILE]
    elapsed = np.cumsum(seconds)
    phase = 3.0 * elapsed
    intensity = np.array(seconds)
    expected = intensity * np.cos(phase) + 1j * intensity * np.sin(phase)

    assert np.allclose(trajectory, expected, atol=1e-9)


# --------------------------------------------------------------------------
# local_spectrum: atan2/sqrt reconstruction matches an independent FFT
# --------------------------------------------------------------------------

def test_local_spectrum_matches_independent_numpy_fft():
    channel = TelemetryChannel(name="test")
    clock = NodePhaseClock(node=0, omega=2.5, channel=channel)
    for size in (2, 9, 4, 7, 3):
        clock.tick(lambda size=size: _busy_work(size))

    trajectory = clock.trajectory()
    phase, intensity = local_spectrum(trajectory)

    expected_spectrum = np.fft.fft(_np(trajectory))
    assert np.allclose(_np(intensity), np.abs(expected_spectrum), atol=1e-9)
    assert np.allclose(
        _np(phase), np.arctan2(expected_spectrum.imag, expected_spectrum.real),
        atol=1e-9,
    )


# --------------------------------------------------------------------------
# spectral_cube: genuinely batched, no cross-node collapse, no padding
# --------------------------------------------------------------------------

def test_spectral_cube_is_one_batched_transform_not_a_collapsed_average():
    channel = TelemetryChannel(name="test")
    n_nodes, n_ticks = 5, 6
    clocks = [
        NodePhaseClock(node=i, omega=1.0 + 0.7 * i, channel=channel)
        for i in range(n_nodes)
    ]
    for _ in range(n_ticks):
        for i, clock in enumerate(clocks):
            clock.tick(lambda i=i: _busy_work(3 + i))

    phase, intensity = spectral_cube(clocks)
    assert phase.shape == (n_nodes, n_ticks)
    assert intensity.shape == (n_nodes, n_ticks)

    # Cross-check against per-node serial FFT: the batched call must agree
    # exactly with transforming each node alone, proving the stack-then-
    # transform order didn't mix or average rows together.
    for i, clock in enumerate(clocks):
        solo_phase, solo_intensity = local_spectrum(clock.trajectory())
        assert np.allclose(_np(phase)[i], _np(solo_phase), atol=1e-9)
        assert np.allclose(_np(intensity)[i], _np(solo_intensity), atol=1e-9)

    # No two nodes' rows collapsed to the same values -- real phase
    # diversity survived the batch, not averaged away.
    intensity_rows = _np(intensity)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            assert not np.allclose(intensity_rows[i], intensity_rows[j])


def test_spectral_cube_refuses_mismatched_tick_counts_rather_than_pad():
    channel = TelemetryChannel(name="test")
    long_clock = NodePhaseClock(node=0, omega=1.0, channel=channel)
    short_clock = NodePhaseClock(node=1, omega=1.0, channel=channel)
    for _ in range(4):
        long_clock.tick(lambda: _busy_work(2))
    short_clock.tick(lambda: _busy_work(2))

    with pytest.raises(ValueError, match="different tick counts"):
        spectral_cube([long_clock, short_clock])


def test_spectral_cube_refuses_empty_clock_list():
    with pytest.raises(ValueError, match="at least one clock"):
        spectral_cube([])
