import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .hierarchy import Cell


def _inside_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
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


def _vertex_triangle_penetration(v: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return 0.0, None
    n = n / norm
    dist = np.dot(v - a, n)
    if dist >= 0.0:
        return dist, None
    proj = v - dist * n
    if _inside_triangle(proj, a, b, c):
        return dist, n
    return dist, None


def _resolve_pair(v: np.ndarray, tri_verts: List[np.ndarray], normal: np.ndarray, depth: float):
    v += -depth * normal


def resolve_membrane_collisions(
    cells: List["Cell"], min_separation: float = 0.0, iters: int = 10
):
    """Naively separate vertex-triangle penetrations within/between cells.

    Runs a handful of relaxation iterations; each pass scatters small
    corrections to penetrating vertex/triangle pairs.  This is an
    intentionally simple placeholder for a future broad-phase + XPBD contact
    solver but already prevents visible interpenetration in small meshes.
    """
    for _ in range(iters):
        changed = False
        for idx, cell in enumerate(cells):
            V = cell.X
            F = cell.faces
            for vi, v in enumerate(V):
                for f in F:
                    if vi in f:
                        continue
                    a, b, c = V[f[0]], V[f[1]], V[f[2]]
                    dist, n = _vertex_triangle_penetration(v, a, b, c)
                    if n is not None and dist < min_separation:
                        _resolve_pair(v, [a, b, c], n, dist - min_separation)
                        changed = True

            for other in cells[idx + 1 :]:
                Vo = other.X
                Fo = other.faces
                for v in V:
                    for f in Fo:
                        a, b, c = Vo[f[0]], Vo[f[1]], Vo[f[2]]
                        dist, n = _vertex_triangle_penetration(v, a, b, c)
                        if n is not None and dist < min_separation:
                            _resolve_pair(v, [a, b, c], n, dist - min_separation)
                            changed = True
                for v in Vo:
                    for f in F:
                        a, b, c = V[f[0]], V[f[1]], V[f[2]]
                        dist, n = _vertex_triangle_penetration(v, a, b, c)
                        if n is not None and dist < min_separation:
                            _resolve_pair(v, [a, b, c], n, dist - min_separation)
                            changed = True
        if not changed:
            break
