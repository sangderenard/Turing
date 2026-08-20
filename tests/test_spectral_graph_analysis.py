"""Spectral graph analysis over arbitrary IR topology, piecewise by control
role.

Covers ``src/compiler/spectral_graph_analysis.py``: colour-free topology
export from ``InfluenceField``, natural-loop detection seeded by real
``BACK_EDGE_ROLES`` edges (not a graph-theoretic SCC guess), per-region
FFT/eigh dispatch, and real-profile projection onto a region's own spectral
basis via the phase-clock work in ``influence_field.py``/
``node_profile_phase.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.compiler.influence_field import DYNAMIC, InfluenceContract, InfluenceField
from src.compiler.shell_telemetry import TelemetryChannel
from src.compiler.spectral_graph_analysis import (
    analyze_graph_spectrum,
    is_circulant,
    natural_loop_regions,
    profile_projection,
    symmetric_adjacency,
)
from src.common.tensors import AbstractTensor as AT


def _np(t) -> np.ndarray:
    return np.asarray(t.data if hasattr(t, "data") else t)


def _ring_field(n: int, *, tail: bool = False, profile_channel=None) -> InfluenceField:
    """An n-node ring closed by a real loop-back edge; optionally a DAG tail
    hanging off node n-1, so the whole graph is no longer circulant while
    the ring subregion still is."""

    field = InfluenceField(
        InfluenceContract(enabled=True, categories=(DYNAMIC,)),
        profile_channel=profile_channel,
    )
    field.add_sources([(0, DYNAMIC, 0, "n0", "")])
    for i in range(n):
        role = "loop-back" if i == n - 1 else "control-next"
        field.add_edge(i, (i + 1) % n, role=role)
    if tail:
        field.add_edge(n - 1, "exit", role="control-next")
    return field


# --------------------------------------------------------------------------
# topology export is colour-free and matches what was fed in
# --------------------------------------------------------------------------

def test_node_keys_and_edge_list_are_colour_free_topology_export():
    field = _ring_field(4)
    assert set(field.node_keys()) == {0, 1, 2, 3}
    edges = {(s, t, r) for s, t, r in field.edge_list()}
    assert edges == {
        (0, 1, "control-next"), (1, 2, "control-next"),
        (2, 3, "control-next"), (3, 0, "loop-back"),
    }


def test_symmetric_adjacency_matches_undirected_edge_presence():
    field = _ring_field(4)
    adjacency, index = symmetric_adjacency(field.node_keys(), field.edge_list())
    array = _np(adjacency)
    for i in range(4):
        assert array[i, i] == 0.0
    # Every ring edge appears symmetrically, once each direction.
    assert array[index[0], index[1]] == array[index[1], index[0]] == 1.0
    assert array[index[3], index[0]] == array[index[0], index[3]] == 1.0


def test_is_circulant_true_for_a_ring_false_once_a_tail_is_attached():
    field_ring = _ring_field(6)
    adjacency_ring, _ = symmetric_adjacency(field_ring.node_keys(), field_ring.edge_list())
    assert is_circulant(adjacency_ring)

    field_tail = _ring_field(6, tail=True)
    adjacency_tail, _ = symmetric_adjacency(field_tail.node_keys(), field_tail.edge_list())
    assert not is_circulant(adjacency_tail)


# --------------------------------------------------------------------------
# natural loop detection, seeded by real control-role edges
# --------------------------------------------------------------------------

def test_natural_loop_region_is_seeded_by_the_real_back_edge():
    field = _ring_field(6, tail=True)
    regions = natural_loop_regions(field)
    assert len(regions) == 1
    region = regions[0]
    assert region.header == 0
    assert region.latch == 5
    assert region.nodes == frozenset(range(6))
    assert "exit" not in region.nodes


def test_no_back_edges_means_no_loop_regions():
    field = InfluenceField(InfluenceContract(enabled=True, categories=(DYNAMIC,)))
    field.add_sources([(0, DYNAMIC, 0, "n0", "")])
    field.add_edge(0, 1, role="control-next")
    field.add_edge(1, 2, role="control-next")
    assert natural_loop_regions(field) == ()


def test_nested_loops_produce_two_overlapping_regions_not_merged():
    """Inner loop 1->2->1, outer loop 0->1->2->3->0. Node 1 and 2 belong to
    both regions -- merging would either lose the inner loop's own
    periodicity or fabricate it for the outer loop."""

    field = InfluenceField(InfluenceContract(enabled=True, categories=(DYNAMIC,)))
    field.add_sources([(0, DYNAMIC, 0, "n0", "")])
    field.add_edge(0, 1, role="control-next")
    field.add_edge(1, 2, role="control-next")
    field.add_edge(2, 1, role="loop-back")  # inner loop
    field.add_edge(2, 3, role="control-next")
    field.add_edge(3, 0, role="loop-back")  # outer loop

    regions = natural_loop_regions(field)
    assert len(regions) == 2
    inner = next(r for r in regions if r.header == 1)
    outer = next(r for r in regions if r.header == 0)
    assert inner.nodes == frozenset({1, 2})
    assert outer.nodes == frozenset({0, 1, 2, 3})
    # Both regions genuinely contain node 1 -- overlap preserved, not merged.
    assert 1 in inner.nodes and 1 in outer.nodes


# --------------------------------------------------------------------------
# piecewise dispatch: loop regions fast when circulant, whole graph general
# --------------------------------------------------------------------------

def test_ring_region_gets_fft_and_matches_the_analytic_ring_spectrum():
    field = _ring_field(6, tail=True)
    decomposition = analyze_graph_spectrum(field)

    assert len(decomposition.loop_regions) == 1
    region = decomposition.loop_regions[0]
    assert region.method == "circulant_fft"
    assert region.eigenvectors is None
    assert set(region.node_order) == set(range(6))

    measured = np.sort(_np(region.eigenvalues))
    analytic = np.sort(2.0 - 2.0 * np.cos(2.0 * np.pi * np.arange(6) / 6))
    assert np.allclose(measured, analytic, atol=1e-9)


def test_whole_graph_with_a_tail_is_not_circulant_and_uses_dense_eigh():
    field = _ring_field(6, tail=True)
    decomposition = analyze_graph_spectrum(field)

    whole = decomposition.whole_graph
    assert whole.method == "dense_eigh"
    assert whole.eigenvectors is not None
    assert len(whole.node_order) == 7  # ring + 'exit'


def test_pure_ring_with_no_tail_makes_whole_graph_and_region_agree():
    """When the whole graph IS the ring, the whole-graph spectrum and the
    loop-region spectrum must describe the same object -- via two different
    methods, cross-checking exactly as test_dec_fft_continuous.py does."""

    field = _ring_field(6)
    decomposition = analyze_graph_spectrum(field)

    whole_spectrum = np.sort(_np(decomposition.whole_graph.eigenvalues))
    region_spectrum = np.sort(_np(decomposition.loop_regions[0].eigenvalues))
    assert np.allclose(whole_spectrum, region_spectrum, atol=1e-8)
    # The whole graph IS circulant here (no tail), so it should also take
    # the fast path rather than a needless dense solve.
    assert decomposition.whole_graph.method == "circulant_fft"


def test_spectral_gap_is_the_smallest_nonzero_eigenvalue():
    field = _ring_field(8)
    decomposition = analyze_graph_spectrum(field)
    gap = decomposition.loop_regions[0].spectral_gap()
    measured = np.sort(_np(decomposition.loop_regions[0].eigenvalues))
    assert gap == pytest.approx(float(measured[1]))
    assert gap > 0.0


def test_region_for_reports_every_enclosing_loop_for_a_nested_node():
    field = InfluenceField(InfluenceContract(enabled=True, categories=(DYNAMIC,)))
    field.add_sources([(0, DYNAMIC, 0, "n0", "")])
    field.add_edge(0, 1, role="control-next")
    field.add_edge(1, 2, role="control-next")
    field.add_edge(2, 1, role="loop-back")
    field.add_edge(2, 3, role="control-next")
    field.add_edge(3, 0, role="loop-back")

    decomposition = analyze_graph_spectrum(field)
    enclosing = decomposition.region_for(1)
    assert len(enclosing) == 2
    assert {frozenset(region.node_order) for region in enclosing} == {
        frozenset({1, 2}), frozenset({0, 1, 2, 3}),
    }


# --------------------------------------------------------------------------
# real profile timing projected onto a region's own spectral basis
# --------------------------------------------------------------------------

def test_profile_projection_requires_a_profiled_field():
    field = _ring_field(4)  # no profile_channel
    decomposition = analyze_graph_spectrum(field)
    with pytest.raises(ValueError, match="profile_channel"):
        profile_projection(decomposition.loop_regions[0], field)


def test_profile_projection_uses_real_per_node_timing():
    channel = TelemetryChannel(name="test")
    field = _ring_field(4, profile_channel=channel)
    field.propagate()
    decomposition = analyze_graph_spectrum(field)

    region = decomposition.loop_regions[0]
    phase, intensity = profile_projection(region, field)
    # (node, sample) -- one synchronized real trajectory per node, matching
    # local_spectrum/spectral_cube's own documented contract.
    n_nodes = len(region.node_order)
    assert phase.shape[0] == n_nodes
    assert intensity.shape[0] == n_nodes
    assert phase.shape == intensity.shape
    # Real measured time, so genuinely positive, not a placeholder zero.
    assert bool((intensity > 0).all())
