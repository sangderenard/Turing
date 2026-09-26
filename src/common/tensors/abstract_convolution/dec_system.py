"""The exterior calculus of one complex, as operators over it.

This realizes the contract ``geometry/src/geometry/dec_toolkit_header.cpp``
declared and never implemented -- ``d0, d1, star0, star1, star2, laplace0,
laplace1, gradient, divergence, curl`` -- in the AbstractTensor lane, and
generalized past the triangle meshes that header assumed to any complex
``laplace_nd`` can build.

TWO THINGS MAKE THIS DIFFERENT FROM A PILE OF MATRICES.

**The operators are applied, not materialised.**  A part's mesh has tens
of thousands of edges, and a dense ``(E, N)`` ``d0`` for one of those is
hundreds of megabytes of almost entirely zeros.  Every operator here is a
gather followed by a scatter over the complex's own index arrays, so cost
follows the number of incidences and not the square of the number of
cells.  :meth:`DECSystem.matrix` materialises one anyway, for eigensolves
and for tests, and says so.

**The metric is not reinvented.**  For a triangle mesh -- which is what a
part is -- ``riemann/mesh_laplace.py`` already computes all three Hodge
stars correctly and is already tested: ``cotangent_weights`` IS ``*1``,
``lumped_vertex_areas`` IS ``*0``, and ``triangle_areas`` gives ``*2``.
:meth:`DECSystem.from_triangle_mesh` adopts them.  That also leaves an
independent implementation of the same operator standing next to this
one, which is what :func:`laplace0` is checked against.

CONVENTIONS.  ``*k`` is diagonal and stored as its diagonal alone.  The
codifferential is ``delta = *^-1 d^T *`` and the Laplace-de Rham operator
is ``delta d + d delta``, so ``laplace0`` is POSITIVE semi-definite --
``CotangentMeshGeometry.apply`` is its negative, being the geometer's
``div grad`` rather than the analyst's ``-div grad``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..abstraction import AbstractTensor
from .laplace_nd import face_incidence_triples


def _tensor(values):
    return AbstractTensor.get_tensor(np.asarray(values, dtype=np.float64))


def _index(values):
    index = AbstractTensor.get_tensor(np.asarray(values, dtype=np.int64))
    return index.astype(AbstractTensor.long_dtype_)


@dataclass(frozen=True)
class CellComplex:
    """Which cells exist, and how each bounds the next.

    ``edges`` is ``(E, 2)`` as tail/head.  ``face_rows/columns/signs`` are
    ``d1`` in coordinate form and ``volume_*`` are ``d2``; both are what
    :func:`face_incidence_triples` and the volume builder produce, kept
    sparse because that is what they are.
    """

    vertex_count: int
    edges: np.ndarray
    face_count: int
    face_rows: np.ndarray
    face_columns: np.ndarray
    face_signs: np.ndarray
    volume_count: int = 0
    positions: np.ndarray | None = None
    volume_rows: np.ndarray | None = None
    volume_columns: np.ndarray | None = None
    volume_signs: np.ndarray | None = None

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def boundary_faces(self) -> np.ndarray:
        """Faces bounding exactly one 3-cell: the complex's own surface.

        An interior face is entered by one cell and left by its
        neighbour, so it appears twice in ``d2``.  A face appearing once
        has nothing on its far side, which is what a boundary is.
        """
        if self.volume_rows is None or self.volume_count == 0:
            raise ValueError("a complex with no 3-cells has no boundary faces")
        counts = np.bincount(self.volume_columns, minlength=self.face_count)
        return np.flatnonzero(counts == 1)

    def boundary_edges(self) -> np.ndarray:
        """Edges lying IN the boundary surface, hence tangential to it.

        These are the degrees of freedom a perfect conductor removes:
        ``n x E = 0`` on the wall says exactly that the line integral
        along any edge within the wall vanishes.
        """
        surface = np.zeros(self.face_count, dtype=bool)
        surface[self.boundary_faces()] = True
        # face_rows names the FACE and face_columns the edge it walks, so
        # the mask selects triples belonging to a boundary face and the
        # answer is the edges those triples point at.
        on_surface = surface[self.face_rows]
        if not on_surface.any():
            return np.empty(0, dtype=np.int64)
        return np.unique(self.face_columns[on_surface])

    @property
    def euler_characteristic(self) -> int:
        """``V - E + F - C``: the one number every detector must agree on."""
        return (self.vertex_count - self.edge_count
                + self.face_count - self.volume_count)

    @staticmethod
    def lattice_region(shape: Sequence[int], occupied):
        """The subcomplex a set of 3-cells induces.

        ``occupied`` is a boolean array over the ``(nx-1, ny-1, nz-1)``
        cubes, or a callable taking cube-centre coordinates in units of
        the spacing and returning that array.  A cell brings its faces,
        those faces bring their edges, and those edges bring their
        vertices; everything else is dropped and what remains is
        reindexed.

        This is the primitive a curved cavity needs -- a pillbox is a
        cylinder cut from a lattice -- and the same one that puts a part
        inside a chamber.  Keeping it as an INDUCED subcomplex is what
        preserves ``d2 d1 == 0`` and ``d1 d0 == 0``: a boundary face now
        belongs to one cell instead of two, which is what a boundary is,
        and no cell ever refers to a face that is gone.
        """
        nx, ny, nz = (int(value) for value in shape)
        tables = CellComplex._lattice_tables((nx, ny, nz))
        edges, faces, volumes, face_edges = tables

        cells = (nx - 1, ny - 1, nz - 1)
        if callable(occupied):
            centres = np.stack(np.meshgrid(
                *(np.arange(count) + 0.5 for count in cells), indexing="ij"),
                axis=-1)
            mask = np.asarray(occupied(centres), dtype=bool)
        else:
            mask = np.asarray(occupied, dtype=bool)
        if mask.shape != cells:
            raise ValueError(f"occupancy {mask.shape} does not fit cells {cells}")
        # Cubes were emitted in k, j, i order to match the vertex layout.
        keep_cells = [index for index, flag in
                      enumerate(mask.transpose(2, 1, 0).reshape(-1)) if flag]
        if not keep_cells:
            raise ValueError("the occupied region contains no cells")

        keep_faces = sorted({face for cell in keep_cells
                             for face in volumes[cell]})
        face_position = {face: position for position, face in enumerate(keep_faces)}
        keep_edges = sorted({edge for face in keep_faces
                             for edge in face_edges[face]})
        edge_position = {edge: position for position, edge in enumerate(keep_edges)}
        keep_vertices = sorted({vertex for edge in keep_edges
                                for vertex in edges[edge]})
        vertex_position = {vertex: position
                           for position, vertex in enumerate(keep_vertices)}

        grid = np.array([[i, j, k] for k in range(nz)
                         for j in range(ny) for i in range(nx)], dtype=np.float64)
        return CellComplex.from_rings(
            len(keep_vertices),
            [[vertex_position[v] for v in edges[edge]] for edge in keep_edges],
            {position: tuple(vertex_position[v] for v in faces[face])
             for position, face in enumerate(keep_faces)},
            {position: {face_position[face]: sign
                        for face, sign in volumes[cell].items()}
             for position, cell in enumerate(keep_cells)},
            positions=grid[keep_vertices],
        )

    @staticmethod
    def _lattice_tables(shape: Sequence[int]):
        """Every cell of a full cubic lattice, before any restriction."""
        complex_tables = CellComplex.lattice(shape, _tables_only=True)
        return complex_tables

    @staticmethod
    def lattice(shape: Sequence[int], *, _tables_only: bool = False):
        """The cubic complex of a grid, enumerated rather than searched for.

        A lattice knows its own cells: there is no reason to rediscover
        its squares by walking cycles, and at chamber resolutions the
        search is the expensive part.  Cells are laid out by axis so the
        ordering is stable and readable -- all x-edges, then y, then z;
        all xy-faces, then yz, then zx.

        Orientations follow the cyclic pairs ``(x,y) (y,z) (z,x)``, so
        every face's ring runs counter-clockwise about the POSITIVE
        remaining axis and every cube's boundary is ``+far - near`` on
        each axis in turn.  That is what makes ``d2 @ d1 == 0`` hold
        here for the same reason it holds anywhere: each interior face
        is entered by one cell and left by its neighbour.
        """
        nx, ny, nz = (int(value) for value in shape)
        if min(nx, ny, nz) < 1:
            raise ValueError("a lattice needs at least one vertex per axis")
        extent = (nx, ny, nz)
        vertex = lambda i, j, k: i + nx * (j + ny * k)

        # ---- edges, grouped by axis
        edges = []
        edge_of = {}
        edge_lookup = {}
        for axis, step in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
            for k in range(nz - step[2]):
                for j in range(ny - step[1]):
                    for i in range(nx - step[0]):
                        tail = vertex(i, j, k)
                        head = vertex(i + step[0], j + step[1], k + step[2])
                        edge_of[(axis, i, j, k)] = len(edges)
                        edge_lookup[(min(tail, head), max(tail, head))] = len(edges)
                        edges.append([tail, head])

        # ---- faces, grouped by the cyclic axis pair that spans them
        pairs = ((0, 1), (1, 2), (2, 0))
        unit = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        faces = {}
        face_of = {}
        for plane, (a, b) in enumerate(pairs):
            da, db = unit[a], unit[b]
            span = [extent[axis] - 1 for axis in range(3)]
            span[3 - a - b] += 1          # the axis the face does not span
            for k in range(span[2]):
                for j in range(span[1]):
                    for i in range(span[0]):
                        origin = (i, j, k)
                        corner = lambda *offset: vertex(
                            *(origin[axis] + offset[axis] for axis in range(3)))
                        face_of[(plane, i, j, k)] = len(faces)
                        faces[len(faces)] = (
                            corner(0, 0, 0),
                            corner(*da),
                            corner(*(da[axis] + db[axis] for axis in range(3))),
                            corner(*db),
                        )

        # ---- cubes: +far - near on each axis
        volumes = {}
        for k in range(nz - 1):
            for j in range(ny - 1):
                for i in range(nx - 1):
                    origin = (i, j, k)
                    shell = {}
                    for plane, (a, b) in enumerate(pairs):
                        normal = 3 - a - b
                        near = list(origin)
                        far = list(origin)
                        far[normal] += 1
                        shell[face_of[(plane, *near)]] = -1.0
                        shell[face_of[(plane, *far)]] = 1.0
                    volumes[len(volumes)] = shell

        if _tables_only:
            face_edges = {}
            for key, ring in faces.items():
                walk = []
                for position, tail in enumerate(ring):
                    head = ring[(position + 1) % len(ring)]
                    walk.append(edge_lookup[(min(tail, head), max(tail, head))])
                face_edges[key] = tuple(walk)
            return edges, faces, volumes, face_edges

        grid = np.array([[i, j, k] for k in range(nz)
                         for j in range(ny) for i in range(nx)], dtype=np.float64)
        return CellComplex.from_rings(nx * ny * nz, edges, faces, volumes,
                                      positions=grid)

    @staticmethod
    def from_rings(vertex_count: int, edges, faces: Mapping[int, Sequence[int]],
                   volumes=None, positions=None):
        """Build from an edge index and the oriented rings over it."""
        edges = np.asarray(edges, dtype=np.int64)
        rows, columns, signs, shape = face_incidence_triples(
            _index(edges), dict(faces))
        volume_rows = volume_columns = volume_signs = None
        volume_count = 0
        if volumes:
            keys = sorted(volumes)
            volume_count = len(keys)
            volume_rows, volume_columns, volume_signs = [], [], []
            # d2's signs are already resolved by the volume builder; this
            # path takes a shell whose faces are given with their signs.
            for row, key in enumerate(keys):
                for face, sign in dict(volumes[key]).items():
                    volume_rows.append(row)
                    volume_columns.append(int(face))
                    volume_signs.append(float(sign))
            volume_rows = np.asarray(volume_rows, dtype=np.int64)
            volume_columns = np.asarray(volume_columns, dtype=np.int64)
            volume_signs = np.asarray(volume_signs, dtype=np.float64)
        return CellComplex(
            vertex_count=int(vertex_count),
            edges=edges,
            face_count=shape[0],
            face_rows=np.asarray(rows, dtype=np.int64),
            face_columns=np.asarray(columns, dtype=np.int64),
            face_signs=np.asarray(signs, dtype=np.float64),
            volume_count=volume_count,
            positions=(None if positions is None
                       else np.asarray(positions, dtype=np.float64)),
            volume_rows=volume_rows,
            volume_columns=volume_columns,
            volume_signs=volume_signs,
        )


class DECSystem:
    """Exterior derivative, Hodge star, codifferential and Laplacian."""

    #: ``delta2`` divides by ``*1``.  A cotangent weight of zero is not a
    #: small number, it is a missing one: the edge's two opposite angles
    #: are right angles, which is what every diagonal of a rectangle
    #: split into two triangles has.  Measured on such a torus: 42 of 108
    #: weights within 1e-9 of zero, and ``laplace1`` lost its entire
    #: harmonic space -- reporting zero harmonic 1-forms on a surface
    #: that has two.  Negative weights are NOT this problem; they make
    #: the operator non-positive but leave the de Rham structure intact,
    #: and the same torus with rhombic quads has 64 negative weights and
    #: still counts b1 correctly.
    STAR1_DEGENERACY_TOLERANCE = 1e-9

    def __init__(self, complex: CellComplex, star0, star1, star2=None,
                 star3=None):
        self.complex = complex
        self.star0 = _tensor(star0)
        self.star1 = _tensor(star1)
        self.star2 = None if star2 is None else _tensor(star2)
        self.star3 = None if star3 is None else _tensor(star3)
        for name, star, count in (
            ("star0", self.star0, complex.vertex_count),
            ("star1", self.star1, complex.edge_count),
            ("star2", self.star2, complex.face_count),
            ("star3", self.star3, complex.volume_count),
        ):
            if star is not None and int(star.get_shape()[0]) != count:
                raise ValueError(
                    f"{name} has {int(star.get_shape()[0])} entries for "
                    f"{count} cells")
        #: World positions per vertex when the complex knows them.
        self.vertex_positions = (None if complex.positions is None
                                 else np.asarray(complex.positions, dtype=np.float64))
        self._tail = _index(complex.edges[:, 0])
        self._head = _index(complex.edges[:, 1])
        self._face_rows = _index(complex.face_rows)
        self._face_columns = _index(complex.face_columns)
        self._face_signs = _tensor(complex.face_signs)
        if complex.volume_rows is None:
            self._volume_rows = self._volume_columns = self._volume_signs = None
        else:
            self._volume_rows = _index(complex.volume_rows)
            self._volume_columns = _index(complex.volume_columns)
            self._volume_signs = _tensor(complex.volume_signs)

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_triangle_mesh(cls, vertices, triangles, *,
                           degeneracy_tolerance: float = 1e-12):
        """The sidecar constructor: a part's welded mesh, with its metric.

        The triangles ARE the 2-cells, so nothing is searched for; the
        face detector in ``laplace_nd`` exists for complexes given only
        as vertices and edges.  The stars come from
        ``build_cotangent_geometry``, which is the tree's existing, tested
        metric for exactly this case.
        """
        from ..riemann.mesh_laplace import build_cotangent_geometry

        vertices = np.asarray(vertices, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int64)
        geometry = build_cotangent_geometry(
            vertices, triangles, degeneracy_tolerance=degeneracy_tolerance)

        faces = {position: tuple(int(v) for v in triangle)
                 for position, triangle in enumerate(triangles)}
        complex = CellComplex.from_rings(
            len(vertices), geometry.edges, faces)

        areas = np.asarray(geometry.triangle_areas, dtype=np.float64)
        safe = np.where(areas > degeneracy_tolerance, areas, 1.0)
        system = cls(
            complex,
            star0=geometry.lumped_vertex_areas,
            star1=geometry.cotangent_weights,
            # *2 on a 2-cell is dual length over primal area, and the dual
            # length of a top-dimensional cell's dual vertex is 1.
            star2=np.where(areas > degeneracy_tolerance, 1.0 / safe, 0.0),
        )
        system.geometry = geometry
        return system

    @classmethod
    def from_lattice(cls, shape: Sequence[int], spacing: float = 1.0):
        """A cubic lattice with its CIRCUMCENTRIC dual, in closed form.

        A cubic lattice is its own dual, so every measure is exact and
        nothing is approximated:

            *0 = h^3 / 1     the dual cube around a vertex
            *1 = h^2 / h     the dual square across an edge
            *2 = h / h^2     the dual edge through a face
            *3 = 1 / h^3     the dual vertex inside a cube

        The consequence is the point of the whole construction.  With
        these, ``laplace0 = *0^-1 d0^T *1 d0`` reduces to
        ``(1/h^2) d0^T d0`` -- the seven-point finite-difference stencil,
        DERIVED rather than assumed.  The same code on a triangle mesh
        gives the cotangent Laplacian.  One operator, two textbooks.
        """
        complex = CellComplex.lattice(shape)
        h = float(spacing)
        system = cls(
            complex,
            star0=np.full(complex.vertex_count, h ** 3),
            star1=np.full(complex.edge_count, h),
            star2=np.full(complex.face_count, 1.0 / h),
            star3=np.full(complex.volume_count, 1.0 / h ** 3),
        )
        if system.vertex_positions is not None:
            system.vertex_positions = system.vertex_positions * h
        return system

    # ------------------------------------------------------------------
    # exterior derivative
    # ------------------------------------------------------------------

    def d0(self, scalar):
        """0-form -> 1-form: the difference along each edge."""
        values = _tensor(scalar) if not isinstance(scalar, AbstractTensor) else scalar
        return values[self._head] - values[self._tail]

    def d0_transpose(self, one_form):
        """The adjoint of :meth:`d0`, accumulating onto each endpoint."""
        result = AbstractTensor.zeros((self.complex.vertex_count,))
        result = AbstractTensor.scatter(result, self._head, one_form, dim=0)
        return AbstractTensor.scatter(result, self._tail, -one_form, dim=0)

    def d1(self, one_form):
        """1-form -> 2-form: the circulation round each face."""
        contribution = self._face_signs * one_form[self._face_columns]
        result = AbstractTensor.zeros((self.complex.face_count,))
        return AbstractTensor.scatter(result, self._face_rows, contribution, dim=0)

    def d1_transpose(self, two_form):
        contribution = self._face_signs * two_form[self._face_rows]
        result = AbstractTensor.zeros((self.complex.edge_count,))
        return AbstractTensor.scatter(
            result, self._face_columns, contribution, dim=0)

    def d2(self, two_form):
        """2-form -> 3-form: the flux out of each cell."""
        if self._volume_rows is None:
            raise ValueError("this complex has no 3-cells")
        contribution = self._volume_signs * two_form[self._volume_columns]
        result = AbstractTensor.zeros((self.complex.volume_count,))
        return AbstractTensor.scatter(
            result, self._volume_rows, contribution, dim=0)

    # ------------------------------------------------------------------
    # codifferential and Laplace-de Rham
    # ------------------------------------------------------------------

    def delta1(self, one_form):
        """1-form -> 0-form: ``*0^-1 d0^T *1``."""
        return self.d0_transpose(self.star1 * one_form) / self.star0

    @property
    def degenerate_star1(self) -> np.ndarray:
        """Edges whose ``*1`` is too near zero to divide by."""
        values = np.asarray(self.star1.data if hasattr(self.star1, "data")
                            else self.star1, dtype=np.float64)
        return np.abs(values) < self.STAR1_DEGENERACY_TOLERANCE

    def delta2(self, two_form):
        """2-form -> 1-form: ``*1^-1 d1^T *2``."""
        if self.star2 is None:
            raise ValueError("this complex has no *2")
        degenerate = self.degenerate_star1
        if degenerate.any():
            raise ValueError(
                f"*1 vanishes on {int(degenerate.sum())} of "
                f"{len(degenerate)} edges, so delta2 cannot divide by it. "
                "A cotangent weight of zero means both angles opposite that "
                "edge are right angles -- the diagonal of a rectangle split "
                "into two triangles. Retriangulate; laplace0 is unaffected.")
        return self.d1_transpose(self.star2 * two_form) / self.star1

    def laplace0(self, scalar):
        """``delta1 d0`` on 0-forms: positive semi-definite.

        ``CotangentMeshGeometry.apply`` computes the negative of this.
        """
        return self.delta1(self.d0(scalar))

    def laplace1(self, one_form):
        """``delta2 d1 + d0 delta1`` on 1-forms: curl-curl plus grad-div.

        The two halves are what the field work needs kept apart: the
        first is the curl-curl an inductive medium responds to, and the
        second is the gradient part that a divergence-free field has none
        of.
        """
        return self.delta2(self.d1(one_form)) + self.d0(self.delta1(one_form))

    # ------------------------------------------------------------------
    # the names the header gave them
    # ------------------------------------------------------------------

    def gradient(self, scalar):
        """Per-edge rise of a scalar field: ``d0``."""
        return self.d0(scalar)

    def curl(self, one_form):
        """Per-face circulation of a 1-form: ``d1``."""
        return self.d1(one_form)

    def divergence(self, one_form):
        """Per-vertex outflow of a 1-form: ``-delta1``.

        The sign is the geometer's: divergence is what a source has a
        positive amount of, while ``delta`` is the adjoint that makes
        ``laplace0`` positive.
        """
        return -self.delta1(one_form)

    # ------------------------------------------------------------------
    # materialisation, for eigensolves and for tests
    # ------------------------------------------------------------------

    def sparse(self, name: str):
        """One exterior derivative as a scipy CSR matrix, from its triples.

        ``matrix`` builds a dense operator column by column, which costs
        one apply per column and is only for small complexes.  These come
        straight out of the coordinate form the complex already stores,
        so assembling ``d1`` for a chamber-sized lattice is a copy rather
        than a computation.
        """
        from scipy.sparse import coo_matrix

        if name == "d0":
            rows = np.repeat(np.arange(self.complex.edge_count), 2)
            columns = self.complex.edges.reshape(-1)
            values = np.tile(np.array([-1.0, 1.0]), self.complex.edge_count)
            shape = (self.complex.edge_count, self.complex.vertex_count)
        elif name == "d1":
            rows = self.complex.face_rows
            columns = self.complex.face_columns
            values = self.complex.face_signs
            shape = (self.complex.face_count, self.complex.edge_count)
        elif name == "d2":
            if self.complex.volume_rows is None:
                raise ValueError("this complex has no 3-cells")
            rows = self.complex.volume_rows
            columns = self.complex.volume_columns
            values = self.complex.volume_signs
            shape = (self.complex.volume_count, self.complex.face_count)
        else:
            raise KeyError(f"no sparse operator named {name!r}")
        return coo_matrix((values, (rows, columns)), shape=shape).tocsr()

    def matrix(self, name: str):
        """One operator as a dense matrix.

        This is the expensive spelling and exists for eigensolves and for
        tests.  ``d0`` on a mesh with ``E`` edges and ``N`` vertices is
        ``E * N`` entries whether or not they are zero.
        """
        builders = {
            "d0": (self.complex.edge_count, self.complex.vertex_count, self.d0),
            "d1": (self.complex.face_count, self.complex.edge_count, self.d1),
            "d2": (self.complex.volume_count, self.complex.face_count, self.d2),
            "laplace0": (self.complex.vertex_count, self.complex.vertex_count,
                         self.laplace0),
            "laplace1": (self.complex.edge_count, self.complex.edge_count,
                         self.laplace1),
        }
        if name not in builders:
            raise KeyError(f"no operator named {name!r}")
        rows, columns, operator = builders[name]
        out = np.zeros((rows, columns), dtype=np.float64)
        basis = np.zeros(columns, dtype=np.float64)
        for column in range(columns):
            basis[column] = 1.0
            applied = operator(_tensor(basis))
            out[:, column] = np.asarray(
                applied.data if hasattr(applied, "data") else applied,
                dtype=np.float64)
            basis[column] = 0.0
        return out
