# src/cells/softbody/geometry/geodesic.py
from __future__ import annotations
import math
from typing import List, Tuple, Dict, Iterable, Optional, Literal
import numpy as np

Topology = Literal["tri", "hex"]

# --------------------------- Base Icosahedron ---------------------------

def icosahedron(radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return vertices (V, 12x3) and triangular faces (F, 20x3) for a unit icosahedron scaled to `radius`."""
    phi = (1.0 + math.sqrt(5.0)) * 0.5
    a, b = 1.0, phi
    pts = np.array([
        (-a,  b,  0), ( a,  b,  0), (-a, -b,  0), ( a, -b,  0),
        ( 0, -a,  b), ( 0,  a,  b), ( 0, -a, -b), ( 0,  a, -b),
        ( b,  0, -a), ( b,  0,  a), (-b,  0, -a), (-b,  0,  a),
    ], dtype=float)
    # Normalize to radius
    pts = normalize_rows(pts) * radius

    # Standard CCW faces (20 triangles)
    F = np.array([
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7,10), (0,10,11),
        (1, 5, 9), (5,11, 4), (11,10, 2), (10,7, 6), (7,1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4,11), (6, 2,10), (8, 6, 7), (9, 8, 1),
    ], dtype=np.int64)
    return pts, F

def normalize_rows(X: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n

# --------------------------- Subdivision ---------------------------

def subdivide_icosa(V: np.ndarray, F: np.ndarray, freq: int,
                    radius: float = 1.0, center: Iterable[float] = (0,0,0),
                    dedup_tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Class-I geodesic subdivision of an icosahedron by frequency `freq` (0 = no subdiv).
    Returns (V_out, F_out) where F_out are triangles with CCW winding (outward).
    """
    if freq <= 0:
        Vout = V.copy()
        Fout = F.copy()
    else:
        center = np.asarray(center, dtype=float)
        quant = 1.0 / max(1.0, radius) * (1.0 / max(1e-18, dedup_tol))
        # Hash: round coordinates to grid to dedup across faces
        def key_of(p: np.ndarray) -> Tuple[int,int,int]:
            return tuple(np.round(p * quant).astype(np.int64).tolist())

        vert_dict: Dict[Tuple[int,int,int], int] = {}
        verts: List[Tuple[float,float,float]] = []

        def get_or_add(p: np.ndarray) -> int:
            k = key_of(p)
            idx = vert_dict.get(k)
            if idx is None:
                idx = len(verts)
                vert_dict[k] = idx
                verts.append((float(p[0]), float(p[1]), float(p[2])))
            return idx

        F_tri: List[Tuple[int,int,int]] = []

        for (ia, ib, ic) in F:
            A, B, C = V[ia], V[ib], V[ic]
            # generate barycentric grid points
            # v(i,j) = ( (i*A + j*B + (f-i-j)*C ) / f ) projected to sphere
            # local index cache for this face to avoid recompute
            idx_local = np.full((freq+1, freq+1), -1, dtype=np.int64)
            for i in range(freq+1):
                for j in range(freq+1 - i):
                    k = freq - i - j
                    p = (A*i + B*j + C*k) / float(freq)
                    p = normalize_rows(p[None, :])[0] * radius
                    p = p + center
                    idx_local[i, j] = get_or_add(p)

            # stitch small triangles
            for i in range(freq):
                for j in range(freq - i):
                    # two faces per grid cell
                    v00 = idx_local[i, j]
                    v10 = idx_local[i+1, j]
                    v01 = idx_local[i, j+1]
                    F_tri.append((v00, v10, v01))
                    if j + i + 1 < freq:
                        v11 = idx_local[i+1, j+1]
                        F_tri.append((v10, v11, v01))

        Vout = np.array(verts, dtype=float)
        Fout = np.array(F_tri, dtype=np.int64)

    # final exact normalization to sphere (in case dedup jittered)
    ctr = np.asarray(center, dtype=float)
    Vout = normalize_rows(Vout - ctr) * radius + ctr
    return Vout, Fout

# --------------------------- Edges ---------------------------

def edges_from_faces(F: np.ndarray) -> np.ndarray:
    """Return unique undirected edges (E, m x 2) from triangular faces F."""
    E = np.vstack([
        F[:, [0,1]],
        F[:, [1,2]],
        F[:, [2,0]],
    ])
    E.sort(axis=1)
    E = np.unique(E, axis=0)
    return E

# --------------------------- Dual (Goldberg / Hex-Dominant) ---------------------------

def dual_hex_mesh(V_tri: np.ndarray, F_tri: np.ndarray, radius: float = 1.0
                  ) -> Tuple[np.ndarray, List[List[int]], np.ndarray]:
    """
    Construct the dual of a triangular sphere mesh.
    - Dual vertices are the centroids of triangles, projected to the sphere.
    - Dual faces are polygons around each original vertex (mostly hexagons; 12 pentagons).
    Returns (V_hex, P_hex, F_hex_tri) where:
      - V_hex: (nf,3) vertices on sphere (triangle centroids)
      - P_hex: list of polygons (each a list of indices into V_hex) in CCW order
      - F_hex_tri: triangle fan of each polygon (useful if your pipeline needs triangles)
    """
    # Centroids -> project outward
    C = np.mean(V_tri[F_tri], axis=1)  # (nf,3)
    C = normalize_rows(C) * radius

    # Build incident-face lists per original vertex
    nv = V_tri.shape[0]
    faces_per_v: List[List[int]] = [[] for _ in range(nv)]
    for f_idx, (a,b,c) in enumerate(F_tri):
        faces_per_v[a].append(f_idx)
        faces_per_v[b].append(f_idx)
        faces_per_v[c].append(f_idx)

    # Order incident faces CCW around each vertex using tangent-plane angles
    P_hex: List[List[int]] = []
    for vidx in range(nv):
        if not faces_per_v[vidx]:
            continue
        p = V_tri[vidx]
        n = p / max(1e-18, np.linalg.norm(p))
        # build tangent basis (t1, t2)
        t1 = np.array([n[1], -n[0], 0.0])
        if np.linalg.norm(t1) < 1e-12:
            t1 = np.array([0.0, n[2], -n[1]])
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        # angle of each incident face's centroid direction from p
        inc = faces_per_v[vidx]
        angs = []
        for f_idx in inc:
            c = C[f_idx]
            d = c - p
            # project to tangent
            u = np.dot(d, t1)
            v = np.dot(d, t2)
            angs.append(math.atan2(v, u))
        order = np.argsort(angs)
        poly = [inc[i] for i in order]  # indices into C
        P_hex.append(poly)

    # Triangulate polygons (fan) for triangle pipelines
    F_hex_tri: List[Tuple[int,int,int]] = []
    for poly in P_hex:
        if len(poly) < 3:
            continue
        c0 = poly[0]
        for k in range(1, len(poly)-1):
            F_hex_tri.append((c0, poly[k], poly[k+1]))
    F_hex_tri = np.array(F_hex_tri, dtype=np.int64)

    return C, P_hex, F_hex_tri

# --------------------------- Public API ---------------------------

def geodesic_sphere(freq: int,
                    radius: float = 1.0,
                    center: Iterable[float] = (0,0,0),
                    topology: Topology = "tri",
                    dedup_tol: float = 1e-12,
                    return_tri_from_hex: bool = True
                    ) -> Dict[str, object]:
    """
    Build a geodesic sphere.

    Args:
      freq: subdivision frequency (0 = icosahedron, 1 = split once, etc.)
      radius: sphere radius
      center: sphere center
      topology: "tri" for triangulated sphere, "hex" for Goldberg (12 pent + hexes)
      dedup_tol: dedup rounding tolerance (scaled by radius)
      return_tri_from_hex: if True, also return a triangulated fan for hex polygons

    Returns a dict with:
      - 'verts': (n,3) vertices on sphere
      - if topology == 'tri':
          'faces': (m,3) triangle faces (np.int64)
          'edges': (k,2) unique edges
      - if topology == 'hex':
          'polys': list[list[int]] polygon faces (indices into 'verts')
          'faces_tri': (mt,3) triangle fan faces (if return_tri_from_hex)
          'edges': (k,2) edges from triangulated fan (if return_tri_from_hex)
    """
    V0, F0 = icosahedron(radius=radius)
    V, F = subdivide_icosa(V0, F0, freq=freq, radius=radius, center=center, dedup_tol=dedup_tol)

    if topology == "tri":
        E = edges_from_faces(F)
        return {"verts": V, "faces": F, "edges": E}

    # hex / goldberg
    C, P, Fhex_tri = dual_hex_mesh(V, F, radius=radius)
    if return_tri_from_hex:
        E = edges_from_faces(Fhex_tri)
        return {"verts": C, "polys": P, "faces_tri": Fhex_tri, "edges": E}
    else:
        return {"verts": C, "polys": P}
