"""Spectral graph analysis over any IR's topology, piecewise by control role.

Purpose-built, colour-free: this reads only ``InfluenceField.node_keys()``/
``edge_list()`` (its pure topology export -- see that module's docstring for
why any producer that can name its nodes and edges is already accepted:
ProcessGraph, SSA, control IR, and the C/JavaScript/LLVM/Fortran shells
alike). Nothing here touches hue, transport, or rendering; the four existing
``field_from_*`` adapters are reused unmodified to populate the topology, and
this module never calls anything past that.

Continues the DEC/FFT/Laplacian identity set pinned in
``tests/test_dec_fft_continuous.py`` (``BuildGraphLaplace``'s combinatorial
``D - A`` == the DEC incidence assembly == the continuous Laplace-Beltrami
operator on a periodic domain, and a circulant graph's spectrum is obtainable
by FFT instead of a dense eigensolve) and the phasic/profile work in
``spectral_propagator.py``/``node_profile_phase.py``.

Piecewise, not one global choice
---------------------------------
A compiler's dependency/control graph is not circulant, so "is the whole
graph circulant" is almost always false and a single global fast/slow
decision wastes the identity above on real programs. But a LOOP BODY is
exactly the structure that makes a circulant graph interesting: a cycle,
often materialised with regular per-iteration edges. The compiler's own
control-flow role vocabulary already names these edges -- ``BACK_EDGE_ROLES``
(``loop-latch``/``loop-back``) from ``influence_field.py`` -- so control code
is what decides which regions get the fast path, not a structural guess:

* every back edge seeds a **natural loop** region (standard compiler-theory
  construction, see :func:`natural_loop_regions`), and each such region gets
  its own local spectrum, FFT-fast when its subgraph is genuinely circulant;
* the WHOLE graph additionally always gets a general dense-``eigh`` spectrum,
  since ``BuildGraphLaplace``/``AT.eigh`` require symmetric input and a real
  program's skeleton is a DAG, not a ring.

Directed graphs, symmetrized
-----------------------------
``BuildGraphLaplace`` requires a symmetric adjacency, and this tree has no
general (non-symmetric) complex eigensolver. Direction and role are real
information a caller may still want, so :func:`node_keys`/``edge_list`` keep
them -- but the Laplacian used for spectral analysis is built from the
undirected skeleton (edge presence, symmetrized), which is standard practice
for spectral analysis of directed graphs and the only input these primitives
accept anyway.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from ..common.tensors import AbstractTensor
from ..common.tensors.abstract_convolution.laplace_nd import BuildGraphLaplace
from ..common.tensors.abstract_convolution.spectral_propagator import (
    circulant_spectrum,
)
from ..common.tensors.abstract_convolution.node_profile_phase import (
    NodePhaseClock,
    local_spectrum,
)
from .influence_field import BACK_EDGE_ROLES, InfluenceField


# --------------------------------------------------------------------------
# topology in, symmetric adjacency out
# --------------------------------------------------------------------------

Topology = Tuple[Tuple[Any, ...], Tuple[Tuple[Any, Any, str], ...]]


def _topology_of(source: "InfluenceField | Topology") -> Topology:
    if isinstance(source, InfluenceField):
        return source.node_keys(), source.edge_list()
    nodes, edges = source
    return tuple(nodes), tuple(edges)


def symmetric_adjacency(
    nodes: Sequence[Any], edges: Sequence[Tuple[Any, Any, str]],
) -> Tuple[AbstractTensor, Mapping[Any, int]]:
    """A symmetric weighted adjacency over ``nodes``, from a directed edge
    list. Edge weight is the number of directed edges seen between a pair
    (so a mutually-referencing pair, or a role duplicated in both
    directions, weighs more than a single edge) -- multiplicity is real
    information, silently collapsing it to a 0/1 mask would not be.
    """

    index = {key: position for position, key in enumerate(nodes)}
    n = len(nodes)
    weights = [[0.0] * n for _ in range(n)]
    for source, target, _role in edges:
        if source not in index or target not in index:
            continue
        i, j = index[source], index[target]
        if i == j:
            continue
        weights[i][j] += 1.0
        weights[j][i] += 1.0
    return AbstractTensor.get_tensor(weights), index


def is_circulant(adjacency: AbstractTensor, *, tol: float = 1e-9) -> bool:
    """Whether every row of ``adjacency`` is the previous row cyclically
    shifted by one -- the structural property that makes ``circulant_spectrum``
    (an FFT) equal to a dense eigensolve on this matrix, per the identity in
    ``tests/test_dec_fft_continuous.py``.
    """

    array = AbstractTensor.get_tensor(adjacency)
    n = int(array.shape[0])
    if n <= 1 or int(array.shape[1]) != n:
        return n <= 1
    import numpy as _np

    values = _np.asarray(array.data if hasattr(array, "data") else array)
    first = values[0]
    for offset in range(1, n):
        if not _np.allclose(values[offset], _np.roll(first, offset), atol=tol):
            return False
    return True


# --------------------------------------------------------------------------
# natural loop regions, seeded by the compiler's own back-edge roles
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LoopRegion:
    """One back edge's natural loop: ``{header}`` plus every node that can
    reach ``latch`` without first passing back through ``header`` -- the
    standard compiler-theory construction, not a graph-theoretic guess."""

    header: Any
    latch: Any
    nodes: frozenset


def natural_loop_regions(source: "InfluenceField | Topology") -> Tuple[LoopRegion, ...]:
    """One :class:`LoopRegion` per back edge (``role in BACK_EDGE_ROLES``).

    Nested loops are NOT merged: a node in an inner loop legitimately
    belongs to both its own region and every enclosing one, and each
    region gets its own local spectrum independently in
    :func:`analyze_graph_spectrum` -- collapsing them would either lose the
    inner loop's own periodicity or fabricate one for the outer loop that
    is not really there.
    """

    _nodes, edges = _topology_of(source)
    predecessors: dict[Any, list[Any]] = {}
    for source_key, target_key, _role in edges:
        predecessors.setdefault(target_key, []).append(source_key)

    regions = []
    for source_key, target_key, role in edges:
        if role not in BACK_EDGE_ROLES:
            continue
        latch, header = source_key, target_key
        loop_nodes = {header}
        worklist = [latch]
        while worklist:
            node = worklist.pop()
            if node in loop_nodes:
                continue
            loop_nodes.add(node)
            worklist.extend(predecessors.get(node, ()))
        regions.append(LoopRegion(header=header, latch=latch, nodes=frozenset(loop_nodes)))
    return tuple(regions)


# --------------------------------------------------------------------------
# spectrum, dispatched per region
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RegionSpectrum:
    """One region's spectral decomposition of its own (symmetrized)
    Laplacian -- either the whole graph (``is_loop_body=False``) or one
    natural loop region seeded by a real back edge."""

    node_order: Tuple[Any, ...]
    laplacian: AbstractTensor
    eigenvalues: AbstractTensor
    eigenvectors: "AbstractTensor | None"
    method: str  # "circulant_fft" | "dense_eigh"
    is_loop_body: bool
    loop: "LoopRegion | None" = None

    def spectral_gap(self) -> float:
        """The algebraic connectivity (Fiedler value): the smallest nonzero
        eigenvalue. Near zero means this region is close to disconnecting
        into two pieces; large means it is well-knit. Undefined (raises) for
        fewer than two nodes."""

        import numpy as _np

        if len(self.node_order) < 2:
            raise ValueError("spectral gap needs at least two nodes")
        values = _np.sort(_np.asarray(
            self.eigenvalues.data if hasattr(self.eigenvalues, "data")
            else self.eigenvalues
        ))
        return float(values[1])


@dataclass(frozen=True)
class GraphSpectralDecomposition:
    """The whole-graph general spectrum, plus one fast local spectrum per
    control-flow-marked loop region."""

    whole_graph: RegionSpectrum
    loop_regions: Tuple[RegionSpectrum, ...]

    def region_for(self, node: Any) -> Tuple[RegionSpectrum, ...]:
        """Every loop region ``node`` belongs to (possibly several, for
        nested loops), innermost first."""

        return tuple(
            region for region in self.loop_regions
            if node in region.node_order
        )


def _region_spectrum(
    node_order: Tuple[Any, ...],
    edges: Sequence[Tuple[Any, Any, str]],
    *,
    is_loop_body: bool,
    loop: "LoopRegion | None",
    circulant_tol: float,
) -> RegionSpectrum:
    adjacency, index = symmetric_adjacency(node_order, edges)
    laplacian, _coo, _info = BuildGraphLaplace(adjacency).build()

    if is_circulant(adjacency, tol=circulant_tol):
        eigenvalues = circulant_spectrum(laplacian)
        eigenvectors = None
        method = "circulant_fft"
    else:
        eigenvalues, eigenvectors = AbstractTensor.eigh(laplacian)
        method = "dense_eigh"

    return RegionSpectrum(
        node_order=node_order, laplacian=laplacian, eigenvalues=eigenvalues,
        eigenvectors=eigenvectors, method=method, is_loop_body=is_loop_body,
        loop=loop,
    )


def analyze_region_spectrum(
    source: "InfluenceField | Topology",
    node_order: Sequence[Any],
    *,
    loop: "LoopRegion | None" = None,
    circulant_tol: float = 1e-9,
) -> RegionSpectrum:
    """Analyse one explicitly selected topology region.

    This is the bounded counterpart to :func:`analyze_graph_spectrum`: a
    caller that has first inspected graph size can analyse a small or
    circulant loop region without forcing a dense eigensolve of an unrelated,
    much larger whole graph.  Edges crossing the requested boundary are
    excluded; the region is a spectrum of its induced subgraph.
    """

    nodes, edges = _topology_of(source)
    requested = tuple(node_order)
    known = set(nodes)
    unknown = [key for key in requested if key not in known]
    if unknown:
        raise ValueError(f"region contains nodes absent from topology: {unknown!r}")
    if len(set(requested)) != len(requested):
        raise ValueError("region node order contains duplicates")
    included = set(requested)
    region_edges = tuple(
        (source_key, target_key, role) for source_key, target_key, role in edges
        if source_key in included and target_key in included
    )
    return _region_spectrum(
        requested, region_edges, is_loop_body=loop is not None, loop=loop,
        circulant_tol=circulant_tol,
    )


def analyze_graph_spectrum(
    source: "InfluenceField | Topology", *, circulant_tol: float = 1e-9,
) -> GraphSpectralDecomposition:
    """The whole graph's general spectrum, plus one fast local spectrum per
    control-flow-marked loop region (see module docstring for why this is
    piecewise rather than one global fast/slow choice).
    """

    nodes, edges = _topology_of(source)
    whole_graph = _region_spectrum(
        nodes, edges, is_loop_body=False, loop=None, circulant_tol=circulant_tol,
    )

    loop_regions = []
    for loop in natural_loop_regions((nodes, edges)):
        region_nodes = tuple(key for key in nodes if key in loop.nodes)
        loop_regions.append(analyze_region_spectrum(
            (nodes, edges), region_nodes, loop=loop,
            circulant_tol=circulant_tol,
        ))

    return GraphSpectralDecomposition(
        whole_graph=whole_graph, loop_regions=tuple(loop_regions),
    )


# --------------------------------------------------------------------------
# real profile timing, projected onto a region's spectral basis
# --------------------------------------------------------------------------

def profile_projection(
    region: RegionSpectrum, field: InfluenceField,
) -> Tuple[AbstractTensor, AbstractTensor]:
    """Where a region's REAL measured execution cost concentrates in its own
    spectral structure -- ``(phase, intensity)`` over the region's own node
    order, from each node's real :class:`NodePhaseClock` (see
    ``InfluenceField.node_phase_clock`` -- requires ``field`` to have been
    built with ``profile_channel=`` and ``propagate()`` to have run).

    This is deliberately a per-node quantity (one clock per node), not a
    whole-graph transform: ``local_spectrum`` decomposes each node's OWN
    tick trajectory over time, independent of every other node's, exactly as
    ``node_profile_phase.py`` documents (batched, never cross-node-averaged).
    A caller wanting a single combined view should read one node's own real
    intensity and set it as ``u0`` for
    ``spectral_propagator.propagate(region.laplacian, u0, t)`` instead --
    that is what genuinely projects a per-node signal onto the region's own
    Laplacian eigenbasis. This function answers the different, per-node
    question "how did each site's own real timing evolve," not "how would a
    signal at these sites propagate."
    """

    clocks = [field.node_phase_clock(key) for key in region.node_order]
    missing = [
        key for key, clock in zip(region.node_order, clocks) if clock is None
    ]
    if missing:
        raise ValueError(
            "profile_projection requires every node in this region to have "
            f"a real phase clock; missing for {missing!r}. Build the field "
            "with profile_channel= and run propagate() first."
        )
    counts = {clock.sample_count for clock in clocks}
    if len(counts) > 1:
        raise ValueError(
            f"nodes in this region were popped different numbers of times "
            f"during propagate() ({sorted(counts)}); local_spectrum needs "
            "synchronized trajectories. Read node_phase_clock per node "
            "individually instead of projecting this region as a whole."
        )
    trajectories = AbstractTensor.stack([clock.trajectory() for clock in clocks])
    return local_spectrum(trajectories)


__all__ = [
    "symmetric_adjacency", "is_circulant",
    "LoopRegion", "natural_loop_regions",
    "RegionSpectrum", "GraphSpectralDecomposition", "analyze_region_spectrum",
    "analyze_graph_spectrum",
    "profile_projection",
]
