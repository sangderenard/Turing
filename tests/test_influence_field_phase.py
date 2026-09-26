"""Real-profiled per-node phase on InfluenceField.propagate().

Covers the opt-in ``profile_channel=``/``node_phase_clock()`` surface added to
``src/compiler/influence_field.py``: each node's relaxation step (see
``propagate()``) is timed for real via a ``NodePhaseClock`` on the same
``shell_telemetry.TelemetryChannel`` the compiler's own trace/profile
instrumentation already flows through -- continuing the phasic/profile work
in ``spectral_propagator.py``/``node_profile_phase.py`` (commit ``b9e1b52``).

The one invariant that matters most: enabling profiling must be purely
observational. It must not change what ``propagate()`` computes, only add a
side channel of real timing data alongside it.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.compiler.influence_field import (
    DYNAMIC,
    InfluenceContract,
    InfluenceField,
)
from src.compiler.shell_telemetry import PROFILE, TelemetryChannel


def _np(t) -> np.ndarray:
    return np.asarray(t.data if hasattr(t, "data") else t)


def _build_diamond(field: InfluenceField) -> None:
    """a -> b -> d, a -> c -> d: a real merge point at d."""

    field.add_sources([("a", DYNAMIC, 0, "a", "")])
    field.add_edge("a", "b")
    field.add_edge("a", "c")
    field.add_edge("b", "d")
    field.add_edge("c", "d")


# --------------------------------------------------------------------------
# disabled by default, and unaffected by being enabled elsewhere
# --------------------------------------------------------------------------

def test_profiling_is_off_by_default():
    field = InfluenceField(InfluenceContract(enabled=True, categories=(DYNAMIC,)))
    _build_diamond(field)
    field.propagate()
    assert field.node_phase_clock("a") is None
    assert field.node_phase_clock("d") is None


def test_enabling_profiling_does_not_change_what_propagate_computes():
    """The one invariant that matters most: a side channel, not a side effect."""

    contract = InfluenceContract(enabled=True, categories=(DYNAMIC,))

    plain = InfluenceField(contract)
    _build_diamond(plain)
    plain_count = plain.propagate()

    channel = TelemetryChannel(name="test")
    profiled = InfluenceField(contract, profile_channel=channel)
    _build_diamond(profiled)
    profiled_count = profiled.propagate()

    assert profiled_count == plain_count
    for key in ("a", "b", "c", "d"):
        plain_reading = plain.reading(key)
        profiled_reading = profiled.reading(key)
        for category in contract.categories:
            assert (
                plain_reading.categories[category].hue
                == pytest.approx(profiled_reading.categories[category].hue)
            )
            assert (
                plain_reading.categories[category].weight
                == pytest.approx(profiled_reading.categories[category].weight)
            )


# --------------------------------------------------------------------------
# real measurement, one tick per real pop
# --------------------------------------------------------------------------

def test_every_visited_node_gets_a_real_clock_with_real_profile_records():
    channel = TelemetryChannel(name="test")
    field = InfluenceField(
        InfluenceContract(enabled=True, categories=(DYNAMIC,)),
        profile_channel=channel,
    )
    _build_diamond(field)
    field.propagate()

    for key in ("a", "b", "c"):
        clock = field.node_phase_clock(key)
        assert clock is not None
        assert clock.sample_count >= 1

    records = [r for r in channel.records if r.kind == PROFILE]
    assert len(records) >= 3
    assert all(r.detail["nanoseconds"] > 0 for r in records)


def test_a_node_never_popped_by_propagate_has_no_clock():
    """A node propagate() never reached has no real operation to have timed --
    an isolated node added but never wired into the topology."""

    channel = TelemetryChannel(name="test")
    field = InfluenceField(
        InfluenceContract(enabled=True, categories=(DYNAMIC,)),
        profile_channel=channel,
    )
    _build_diamond(field)
    field.add_node("isolated")
    field.propagate()

    assert field.node_phase_clock("isolated") is None


def test_merge_node_accumulates_more_ticks_than_a_single_hop_node():
    """d receives from both b and c, so it is popped (and merges) at least as
    often as a plain single-parent node -- real structural difference, not an
    artifact of clock construction order."""

    channel = TelemetryChannel(name="test")
    field = InfluenceField(
        InfluenceContract(enabled=True, categories=(DYNAMIC,)),
        profile_channel=channel,
    )
    _build_diamond(field)
    field.propagate()

    clock_b = field.node_phase_clock("b")
    clock_d = field.node_phase_clock("d")
    assert clock_d.sample_count >= clock_b.sample_count


def test_node_trajectory_is_a_genuine_complex_value_from_real_timing():
    channel = TelemetryChannel(name="test")
    field = InfluenceField(
        InfluenceContract(enabled=True, categories=(DYNAMIC,)),
        profile_channel=channel,
        phase_omega=2.0,
    )
    _build_diamond(field)
    field.propagate()

    clock = field.node_phase_clock("d")
    trajectory = _np(clock.trajectory())
    assert trajectory.shape == (clock.sample_count,)
    assert np.iscomplexobj(trajectory)
    # Intensity (the modulus) must be nonzero and derived from real elapsed
    # time -- not a placeholder constant.
    assert np.all(np.abs(trajectory) > 0.0)


def test_distinct_nodes_get_distinct_phase_clock_identities():
    channel = TelemetryChannel(name="test")
    field = InfluenceField(
        InfluenceContract(enabled=True, categories=(DYNAMIC,)),
        profile_channel=channel,
    )
    _build_diamond(field)
    field.propagate()

    node_ids = {
        field.node_phase_clock(key).node for key in ("a", "b", "c", "d")
    }
    assert len(node_ids) == 4
