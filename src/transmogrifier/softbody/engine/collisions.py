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


def build_self_contacts_spatial_hash(
    X: np.ndarray, faces: np.ndarray, cell_ids: np.ndarray, voxel_size: float
) -> np.ndarray:
    """Return potential vertex–triangle pairs using a 3‑D hash grid.

    Parameters
    ----------
    X : (n, 3) array
        Vertex positions.
    faces : (m, 3) array
        Triangle vertex indices.
    cell_ids : (n,) array
        Cell identifier per vertex.  Faces are assumed to belong to the cell of
        their first vertex.  Contacts are only generated within a cell.
    voxel_size : float
        Edge length of spatial hash voxels.

    Returns
    -------
    np.ndarray
        Array of ``(vertex_index, face_index)`` pairs that may collide.  The
        function performs only a broad‑phase search and does *not* check actual
        penetration depths.
    """

    if X.size == 0 or faces.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    inv_vox = 1.0 / max(voxel_size, 1e-12)

    # --- Build per-face adjacency to exclude neighbour triangles -------------
    n_faces = len(faces)
    adjacency = [set(f) for f in faces]  # start with the face's own vertices
    edge2faces = {}
    for fi, (i, j, k) in enumerate(faces):
        for e in ((i, j), (j, k), (k, i)):
            key = tuple(sorted(e))
            edge2faces.setdefault(key, []).append(fi)

    for tris in edge2faces.values():
        if len(tris) < 2:
            continue
        verts = set()
        for fi in tris:
            verts.update(faces[fi])
        for fi in tris:
            adjacency[fi].update(verts)

    # --- Hash triangles into voxel grid -------------------------------------
    tri_hash = {}
    for fi in range(n_faces):
        pts = X[faces[fi]]
        mn = np.floor(pts.min(axis=0) * inv_vox).astype(int)
        mx = np.floor(pts.max(axis=0) * inv_vox).astype(int)
        for ix in range(mn[0], mx[0] + 1):
            for iy in range(mn[1], mx[1] + 1):
                for iz in range(mn[2], mx[2] + 1):
                    tri_hash.setdefault((ix, iy, iz), []).append(fi)

    face_cell = cell_ids[faces[:, 0]]
    pairs = set()
    for vi, v in enumerate(X):
        cell = cell_ids[vi]
        voxel = np.floor(v * inv_vox).astype(int)
        for ix in range(voxel[0] - 1, voxel[0] + 2):
            for iy in range(voxel[1] - 1, voxel[1] + 2):
                for iz in range(voxel[2] - 1, voxel[2] + 2):
                    tris = tri_hash.get((ix, iy, iz))
                    if not tris:
                        continue
                    for fi in tris:
                        if face_cell[fi] != cell:
                            continue
                        if vi in adjacency[fi]:
                            continue
                        pairs.add((vi, fi))

    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    out = np.array(sorted(pairs), dtype=np.int32)
    return out


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
