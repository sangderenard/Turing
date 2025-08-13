import numpy as np

from src.transmogrifier.softbody.engine.mesh import make_icosphere, build_adjacency
from src.transmogrifier.softbody.engine.hierarchy import Cell
from src.transmogrifier.softbody.engine.collisions import resolve_membrane_collisions


def _build_cell(offset=(0.0, 0.0, 0.0)):
    v, f = make_icosphere(subdiv=0, radius=0.1, center=offset)
    edges, bends = build_adjacency(f)
    invm = np.ones(len(v), dtype=np.float64)
    cons = {}
    return Cell(
        id="c",
        X=v.copy(),
        V=np.zeros_like(v),
        invm=invm,
        faces=f,
        edges=np.array(edges),
        bends=np.array(bends),
        constraints=cons,
        organelles=[],
    )


def _inside_triangle(p, a, b, c):
    v0 = c - a
    v1 = b - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if denom == 0.0:
        return False
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (u >= 0.0) and (v >= 0.0) and (w >= 0.0)


def _min_vertex_triangle_dist(V, F, W):
    m = np.inf
    for v in V:
        for tri in F:
            a, b, c = W[tri]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            n = n / norm
            dist = np.dot(v - a, n)
            proj = v - dist * n
            if _inside_triangle(proj, a, b, c):
                m = min(m, dist)
    return m


def test_resolve_membrane_collisions_between_cells():
    c1 = _build_cell((0.0, 0.0, 0.0))
    c2 = _build_cell((0.05, 0.0, 0.0))  # overlapping spheres
    resolve_membrane_collisions([c1, c2])
    d12 = _min_vertex_triangle_dist(c1.X, c2.faces, c2.X)
    d21 = _min_vertex_triangle_dist(c2.X, c1.faces, c1.X)
    tol = -5e-4
    assert d12 >= tol
    assert d21 >= tol
