"""The ``d`` operators: oriented faces, signed ``d1``, and ``d1 @ d0 == 0``.

``tests/test_dec_fft_continuous.py`` pins ``d1 @ d0 == 0`` against a torus
complex whose ``d1`` is written out by hand in its own fixture.  Nothing in
the library could produce such a matrix: ``TransformHub.build_d_operators``
returned ``{"d0": ..., "d1": None}`` with the comment "d1 placeholder", and
``FaceMapGenerator`` returned each face as ``tuple(sorted(...))`` -- a vertex
multiset with the ring's closing vertex still duplicated inside it, from which
no orientation and therefore no sign can be recovered.

This module covers the completed operators:

* faces come back as ORIENTED rings, deduplicated under rotation and
  reversal rather than by sorting them into sets;
* ``build_face_incidence`` reproduces the hand-written torus ``d1`` exactly,
  so the library agrees with the reference the tree already trusted;
* ``d1 @ d0 == 0`` holds on detected faces, not only on hand-built ones;
* the detected face set is the lattice's own squares, and ``rank(d1)`` is the
  graph's cycle-space dimension ``E - V + C``.

Every computation runs through ``AbstractTensor``; numpy appears only in
fixture construction and in assertions, as in the companion module.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors import AbstractTensor as AT
from src.common.tensors.abstract_convolution.laplace_nd import (
    FaceMapGenerator,
    VolumeMapGenerator,
    build_face_incidence,
    build_volume_incidence,
    ring_key,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _np(t) -> np.ndarray:
    """Extract a numpy view for assertions/reporting only."""
    return np.asarray(t.data if hasattr(t, "data") else t)


def _lattice(nx: int, ny: int, nz: int):
    """A cubic lattice: positions, ``(E, 2)`` edge index, vertex count.

    Edges run only in the ``+x``, ``+y``, ``+z`` directions, so each one is
    stored once and with a definite direction -- which is what gives ``d1``
    something to disagree with when a face traverses it backwards.
    """
    index: dict[tuple[int, int, int], int] = {}
    positions: list[list[float]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                index[(i, j, k)] = len(positions)
                positions.append([float(i), float(j), float(k)])
    edges: list[list[int]] = []
    for (i, j, k), tail in index.items():
        for step in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            head = (i + step[0], j + step[1], k + step[2])
            if head in index:
                edges.append([tail, index[head]])
    return (
        AT.get_tensor(positions),
        AT.get_tensor(edges).astype(AT.long_dtype_),
        len(positions),
    )


def _d0_of(edge_index, num_vertices: int):
    """``d0`` (E, N), spelled exactly as ``build_d_operators`` spells it."""
    rows = edge_index.tolist()
    d0 = AT.zeros((len(rows), num_vertices))
    for position, (tail, head) in enumerate(rows):
        d0[position, int(tail)] = -1
        d0[position, int(head)] = 1
    return d0


def _torus_reference(n: int):
    """``d0``, ``d1``, the edge index and the face rings of a discrete torus.

    The operators here are transcribed from ``_torus_d0_d1`` in
    ``test_dec_fft_continuous.py`` -- the tree's existing hand-built
    reference -- together with the edge index and the oriented rings that
    describe the same complex to the library.  ``n >= 3`` because a 2-torus
    joins each neighbour pair twice and an undirected pair would no longer
    name one edge.
    """
    node = lambda r, c: (r % n) * n + (c % n)
    edge_h = lambda r, c: (r % n) * n + (c % n)            # (r,c) -> (r,c+1)
    edge_v = lambda r, c: n * n + (r % n) * n + (c % n)    # (r,c) -> (r+1,c)

    num_nodes, num_edges, num_faces = n * n, 2 * n * n, n * n
    d0 = np.zeros((num_edges, num_nodes))
    edge_index = [[0, 0] for _ in range(num_edges)]
    for r in range(n):
        for c in range(n):
            h = edge_h(r, c)
            d0[h, node(r, c)] = -1.0
            d0[h, node(r, c + 1)] = 1.0
            edge_index[h] = [node(r, c), node(r, c + 1)]
            v = edge_v(r, c)
            d0[v, node(r, c)] = -1.0
            d0[v, node(r + 1, c)] = 1.0
            edge_index[v] = [node(r, c), node(r + 1, c)]

    d1 = np.zeros((num_faces, num_edges))
    rings: dict[int, list[int]] = {}
    for r in range(n):
        for c in range(n):
            f = r * n + c
            d1[f, edge_h(r, c)] = 1.0
            d1[f, edge_v(r, c + 1)] = 1.0
            d1[f, edge_h(r + 1, c)] = -1.0
            d1[f, edge_v(r, c)] = -1.0
            # The same plaquette as a walk: right, up, left, down.
            rings[f] = [
                node(r, c), node(r, c + 1), node(r + 1, c + 1), node(r + 1, c)
            ]
    return d0, d1, AT.get_tensor(edge_index).astype(AT.long_dtype_), rings


# --------------------------------------------------------------------------
# a face is an oriented ring
# --------------------------------------------------------------------------

def test_a_square_is_one_oriented_ring() -> None:
    """Four edges round a square are one face, not four vertex multisets.

    The sorted-set deduplication this replaces returned ``[0,1,2,2,3]``,
    ``[0,1,2,3,3]``, ``[0,1,1,2,3]`` and ``[0,0,1,2,3]`` for this input: four
    "faces", each five entries long, none of them a cycle.
    """
    vertices = AT.get_tensor(
        [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    edges = AT.get_tensor([[0, 1], [1, 2], [2, 3], [3, 0]]).astype(AT.long_dtype_)

    faces = FaceMapGenerator(vertices, edges).generate_face_map()

    assert len(faces) == 1, f"expected one face, got {faces}"
    ring = faces[0]
    assert len(ring) == 4, f"the closing vertex must not be repeated: {ring}"
    assert ring_key(ring) == ring_key([0, 1, 2, 3])


def test_rotation_and_reversal_are_the_same_face() -> None:
    """``ring_key`` identifies the walks that differ only in where they start.

    Sorting would identify them too, and would also identify rings that are
    genuinely different faces over the same vertices -- which is the reason
    orientation had to survive deduplication.
    """
    assert ring_key([0, 1, 2, 3]) == ring_key([2, 3, 0, 1])      # rotation
    assert ring_key([0, 1, 2, 3]) == ring_key([3, 2, 1, 0])      # reversal
    assert ring_key([0, 1, 2, 3]) != ring_key([0, 2, 1, 3])      # a real change


# --------------------------------------------------------------------------
# the builder against the tree's own reference
# --------------------------------------------------------------------------

def test_build_face_incidence_reproduces_the_torus_reference() -> None:
    """The library's ``d1`` equals the hand-written one, entry for entry.

    This is the validation that matters: the torus fixture in
    ``test_dec_fft_continuous.py`` was the only correct ``d1`` in the tree,
    and it was written out by hand in a test.  Given the same complex, the
    builder must produce that matrix and not merely something that also
    satisfies ``d1 @ d0 == 0``.
    """
    n = 4
    d0_np, d1_np, edge_index, rings = _torus_reference(n)

    built = build_face_incidence(edge_index, rings)

    assert np.array_equal(_np(built), d1_np), (
        "library d1 differs from the hand-built reference:\n"
        f"  built row0     = {_np(built)[0]}\n"
        f"  reference row0 = {d1_np[0]}"
    )
    composed = AT.get_tensor(d1_np) @ AT.get_tensor(d0_np)
    assert float(AT.linalg.norm(composed)) < 1e-9


def test_faces_are_none_and_faces_are_empty_are_different() -> None:
    """A complex with undetected 2-cells has no ``d1``; one with none has rows.

    The placeholder returned ``None`` unconditionally, so these two states
    were indistinguishable, and every ``DECSpec`` in the tree passes a 0-row
    ``D1`` that satisfies ``D1 @ D0 == 0`` for no reason at all.
    """
    edges = AT.get_tensor([[0, 1], [1, 2]]).astype(AT.long_dtype_)

    assert build_face_incidence(edges, None) is None
    empty = build_face_incidence(edges, {})
    assert empty is not None and tuple(empty.shape) == (0, 2)


def test_a_walk_that_crosses_one_edge_both_ways_cancels() -> None:
    """The row is a boundary CHAIN, so signs accumulate rather than overwrite.

    A closed walk that goes out along an edge and comes back along it
    contributes nothing to that edge.  Writing instead of adding left a
    stray sign there, and that single entry was enough to break
    ``d1 @ d0 == 0`` on every lattice larger than one cell.
    """
    edges = AT.get_tensor(
        [[0, 1], [1, 2], [2, 0], [2, 3]]).astype(AT.long_dtype_)
    # 0 -> 1 -> 2 -> 3 -> 2 -> 0: edge (2,3) is crossed both ways.
    built = build_face_incidence(edges, {0: [0, 1, 2, 3, 2]})

    assert _np(built)[0, 3] == 0.0
    composed = built @ _d0_of(edges, 4)
    assert float(AT.linalg.norm(composed)) < 1e-9


# --------------------------------------------------------------------------
# detected faces on real lattices
# --------------------------------------------------------------------------

LATTICES = [(2, 2, 2), (3, 3, 1), (3, 3, 3), (4, 3, 2)]


@pytest.mark.parametrize("shape", LATTICES)
def test_boundary_of_boundary_vanishes_on_detected_faces(shape) -> None:
    """``d1 @ d0 == 0`` for faces the library found, not only hand-built ones.

    This is the identity ``fs_dec.check`` enforces and that every ``DECSpec``
    in the tree has so far satisfied only by passing an empty ``D1``.
    """
    vertices, edges, num_vertices = _lattice(*shape)
    faces = FaceMapGenerator(vertices, edges).generate_face_map()

    d1 = build_face_incidence(edges, faces)
    composed = d1 @ _d0_of(edges, num_vertices)

    norm = float(AT.linalg.norm(composed))
    assert norm < 1e-9, f"DEC violation on {shape}: ||d1 @ d0|| = {norm:.3e}"


@pytest.mark.parametrize("shape", LATTICES)
def test_detected_faces_are_the_lattice_squares(shape) -> None:
    """Every face of a cubic lattice is one unit square, and all of them are.

    The count is the number of axis-aligned unit cells' faces: for each pair
    of axes, ``(n_a - 1) * (n_b - 1)`` squares per layer, over ``n_c`` layers.
    """
    nx, ny, nz = shape
    expected = ((nx - 1) * (ny - 1) * nz
                + (nx - 1) * (nz - 1) * ny
                + (ny - 1) * (nz - 1) * nx)

    vertices, edges, _ = _lattice(*shape)
    faces = FaceMapGenerator(vertices, edges).generate_face_map()

    assert len(faces) == expected
    assert {len(ring) for ring in faces.values()} == {4}


@pytest.mark.parametrize("shape", LATTICES)
def test_no_face_visits_a_vertex_twice(shape) -> None:
    """Rings are cycles, not figures of eight.

    The walk only marks EDGES on the way down, so it can return through a
    vertex it has already used.  Such a walk closes, and used to be emitted
    as a face; it is the reason ``d1 @ d0`` came out non-zero on a plain
    3x3 grid.
    """
    vertices, edges, _ = _lattice(*shape)
    faces = FaceMapGenerator(vertices, edges).generate_face_map()

    for key, ring in faces.items():
        assert len(set(ring)) == len(ring), f"face {key} repeats a vertex: {ring}"


@pytest.mark.parametrize("shape", LATTICES)
def test_face_rank_is_the_graph_cycle_space(shape) -> None:
    """``rank(d1) == E - V + C``: the faces span the whole cycle space.

    A connected lattice has ``C == 1``.  Spanning it is the statement that no
    independent cycle was missed; the shortfall from the face COUNT is one
    relation per closed cell, which is ``d.d == 0`` on that cell and is a
    property of the complex rather than a redundant row.
    """
    vertices, edges, num_vertices = _lattice(*shape)
    faces = FaceMapGenerator(vertices, edges).generate_face_map()
    d1 = build_face_incidence(edges, faces)

    num_edges = len(edges.tolist())
    assert np.linalg.matrix_rank(_np(d1)) == num_edges - num_vertices + 1


def test_a_tetrahedron_is_four_triangles_on_a_closed_surface() -> None:
    """Triangles, which a cubic lattice never produces, and a closed surface.

    ``K4``'s three four-cycles are all chorded -- every pair of its vertices
    is joined -- so the induced-cycle filter leaves exactly the four faces.
    A closed surface's faces satisfy one relation, their signed sum, so the
    rank is one less than the count and equals ``E - V + 1``.
    """
    vertices = AT.get_tensor([
        [0., 0., 0.], [1., 0., 0.], [0.5, 0.866, 0.], [0.5, 0.289, 0.816]])
    edges = AT.get_tensor([
        [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]).astype(AT.long_dtype_)

    faces = FaceMapGenerator(vertices, edges).generate_face_map()
    d1 = build_face_incidence(edges, faces)

    assert len(faces) == 4
    assert {len(ring) for ring in faces.values()} == {3}
    assert float(AT.linalg.norm(d1 @ _d0_of(edges, 4))) < 1e-9
    assert np.linalg.matrix_rank(_np(d1)) == 6 - 4 + 1 == 3


def test_chorded_rings_are_composites_and_are_rejected() -> None:
    """Two squares' outline is a six-ring with a chord, and is their sum.

    Admitting it would put a dependent row in ``d1`` -- the rank does not
    move when the four composites are let in -- and would count the same
    area twice in the 2-form Hodge star.
    """
    vertices, edges, num_vertices = _lattice(3, 3, 1)
    generator = FaceMapGenerator(vertices, edges)

    faces = generator.generate_face_map()
    composites = generator.generate_face_map(induced_only=False)

    assert {len(ring) for ring in faces.values()} == {4}
    assert sorted(len(ring) for ring in composites.values()) == [4, 4, 4, 4, 6, 6, 6, 6]

    strict = build_face_incidence(edges, faces)
    loose = build_face_incidence(edges, composites)
    assert (np.linalg.matrix_rank(_np(loose))
            == np.linalg.matrix_rank(_np(strict)) == len(faces))
    # The composites are still legitimate closed chains; they are redundant,
    # not wrong, which is why the identity survives either way.
    assert float(AT.linalg.norm(loose @ _d0_of(edges, num_vertices))) < 1e-9


# --------------------------------------------------------------------------
# the third level: 3-cells and d2
# --------------------------------------------------------------------------

def _complex_of(shape):
    """Vertices, edges, faces, volumes and ``d1``/``d2`` for one lattice."""
    vertices, edges, num_vertices = _lattice(*shape)
    faces = FaceMapGenerator(vertices, edges).generate_face_map()
    d1 = build_face_incidence(edges, faces)
    volumes = VolumeMapGenerator(
        vertices, edges, faces).generate_volume_map(edge_index=edges)
    d2 = build_volume_incidence(d1, volumes)
    return num_vertices, edges, faces, volumes, d1, d2


@pytest.mark.parametrize("shape", LATTICES)
def test_volume_shells_are_the_lattice_cells(shape) -> None:
    """A cubic lattice's 3-cells are its unit cubes, six faces each.

    ``VolumeMapGenerator`` returned ``{}`` and printed "Volume detection
    not implemented yet"; there was no third level at all.
    """
    nx, ny, nz = shape
    _, _, _, volumes, _, _ = _complex_of(shape)

    assert len(volumes) == (nx - 1) * (ny - 1) * (nz - 1)
    assert {len(shell) for shell in volumes.values()} <= {6}


@pytest.mark.parametrize("shape", LATTICES)
def test_boundary_of_boundary_vanishes_at_the_second_level(shape) -> None:
    """``d2 @ d1 == 0``, the same identity one dimension up.

    The signs are not chosen and then checked: they are propagated from
    one face of the shell by the rule that two faces meeting at an edge
    must cancel there, so the identity is how the operator is built.
    """
    _, _, _, volumes, d1, d2 = _complex_of(shape)
    if not volumes:
        assert d2 is not None and tuple(d2.shape)[0] == 0
        return

    norm = float(AT.linalg.norm(d2 @ d1))
    assert norm < 1e-9, f"DEC violation on {shape}: ||d2 @ d1|| = {norm:.3e}"


@pytest.mark.parametrize("shape", LATTICES)
def test_euler_characteristic_of_a_solid_block_is_one(shape) -> None:
    """``V - E + F - C == 1`` for every solid block.

    This is the one statement that tests all three detectors at once: a
    miscount anywhere -- a missed face, a composite admitted, a shell
    detected twice -- moves it.  A contractible block has the Euler
    characteristic of a point.
    """
    num_vertices, edges, faces, volumes, _, _ = _complex_of(shape)

    characteristic = (num_vertices - len(edges.tolist())
                      + len(faces) - len(volumes))
    assert characteristic == 1


def test_a_tetrahedron_is_one_three_cell() -> None:
    """Four triangles closing on themselves are a solid, not four faces."""
    vertices = AT.get_tensor([
        [0., 0., 0.], [1., 0., 0.], [0.5, 0.866, 0.], [0.5, 0.289, 0.816]])
    edges = AT.get_tensor([
        [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]).astype(AT.long_dtype_)

    faces = FaceMapGenerator(vertices, edges).generate_face_map()
    d1 = build_face_incidence(edges, faces)
    volumes = VolumeMapGenerator(
        vertices, edges, faces).generate_volume_map(edge_index=edges)
    d2 = build_volume_incidence(d1, volumes)

    assert len(volumes) == 1 and len(volumes[0]) == 4
    assert float(AT.linalg.norm(d2 @ d1)) < 1e-9
    assert 4 - 6 + 4 - 1 == 1


def test_a_flat_grid_has_no_three_cells() -> None:
    """Nothing closes in a plane, and the detector says so rather than guessing."""
    _, _, faces, volumes, _, d2 = _complex_of((3, 3, 1))

    assert len(faces) == 4
    assert volumes == {}
    assert tuple(d2.shape) == (0, 4)


def test_two_cells_sharing_a_face_are_not_a_third_cell() -> None:
    """The outer surface of two stacked cubes is their sum, not a new solid.

    Same composite the chord filter rejects for faces, one dimension up:
    a shell that properly contains another shell is dropped.
    """
    _, _, _, volumes, _, _ = _complex_of((2, 2, 3))

    assert len(volumes) == 2
    assert all(len(shell) == 6 for shell in volumes.values())


# --------------------------------------------------------------------------
# through the hub
# --------------------------------------------------------------------------

def test_build_d_operators_completes_d1_when_faces_are_given() -> None:
    """``build_d_operators`` returns ``d1`` with faces and ``None`` without.

    Keeping ``None`` for the no-faces call preserves what every existing
    caller sees; the operator is completed only where the 2-cells are known.
    """
    from src.common.tensors.abstract_convolution.laplace_nd import TransformHub

    vertices, edges, num_vertices = _lattice(2, 2, 2)
    hub = TransformHub(1.0, 1.0, (True, True, True, True))
    profile = {
        "edge_index": edges,
        "num_vertices": num_vertices,
        "num_edges": len(edges.tolist()),
    }

    without = hub.build_d_operators(edges, profile)
    assert without["d1"] is None

    faces = FaceMapGenerator(vertices, edges).generate_face_map()
    with_faces = hub.build_d_operators(edges, profile, faces=faces)
    assert with_faces["d1"] is not None
    assert np.array_equal(_np(with_faces["d0"]), _np(without["d0"]))

    composed = with_faces["d1"] @ with_faces["d0"]
    assert float(AT.linalg.norm(composed)) < 1e-9


def test_calculate_geometry_publishes_a_complete_dec_package() -> None:
    """``detect_faces=True`` yields d0, d1 and all three Hodge stars.

    This path could not run at all before: face detection reached
    ``is_planar``, which crossed two bare ``(3,)`` vectors and raised.  The
    only caller in the tree, ``validate_transform_hub``, passed a line graph
    with no cycles, so the face branch was never entered.
    """
    from src.common.tensors.abstract_convolution.laplace_nd import TransformHub

    class IdentityTransform(TransformHub):
        def transform_spatial(self, U, V, W):
            return U, V, W

    resolution = 3
    axis = AT.linspace(0, 1, resolution)
    U, V, W = AT.meshgrid(axis, axis, axis)

    edges = []
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                tail = (i * resolution + j) * resolution + k
                if i + 1 < resolution:
                    edges.append([tail, ((i + 1) * resolution + j) * resolution + k])
                if j + 1 < resolution:
                    edges.append([tail, (i * resolution + j + 1) * resolution + k])
                if k + 1 < resolution:
                    edges.append([tail, (i * resolution + j) * resolution + k + 1])
    edge_index = AT.get_tensor(edges).astype(AT.long_dtype_)

    hub = IdentityTransform(1.0, 1.0, (True, True, True, True))
    dec = hub.calculate_geometry(
        U, V, W, edge_index=edge_index, detect_faces=True)["DEC"]

    d0 = dec["d_operators"]["d0"]
    d1 = dec["d_operators"]["d1"]
    assert tuple(d0.shape) == (54, 27)
    assert tuple(d1.shape) == (36, 54)
    assert len(dec["faces"]) == 36
    assert dec["hodge_stars"]["availability"] == {
        "hodge_0": True, "hodge_1": True, "hodge_2": True}
    assert float(AT.linalg.norm(d1 @ d0)) < 1e-9
