
import numpy as np
from ..geometry.geodesic import geodesic_sphere

def icosahedron():
    t = (1.0 + 5 ** 0.5) / 2.0
    verts = np.array([
        [-1,  t,  0],[ 1,  t,  0],[-1, -t,  0],[ 1, -t,  0],
        [ 0, -1,  t],[ 0,  1,  t],[ 0, -1, -t],[ 0,  1, -t],
        [ t,  0, -1],[ t,  0,  1],[-t,  0, -1],[-t,  0,  1],
    ], dtype=np.float64)
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
    ], dtype=np.int32)
    verts /= np.linalg.norm(verts, axis=1)[:,None]
    return verts, faces

def subdivide(verts, faces):
    vert_list = verts.tolist()
    midpoint_cache = {}
    def midpoint(i, j):
        key = tuple(sorted((i,j)))
        if key in midpoint_cache:
            return midpoint_cache[key]
        v = (verts[i] + verts[j]) * 0.5
        v = v / np.linalg.norm(v)
        vert_list.append(v.tolist())
        idx = len(vert_list) - 1
        midpoint_cache[key] = idx
        return idx
    new_faces = []
    for (a,b,c) in faces:
        ab = midpoint(a,b)
        bc = midpoint(b,c)
        ca = midpoint(c,a)
        new_faces += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
    verts2 = np.array(vert_list, dtype=np.float64)
    faces2 = np.array(new_faces, dtype=np.int32)
    return verts2, faces2

def make_icosphere(subdiv=1, radius=0.1, center=(0,0,0)):
    """Build a geodesic sphere using the new NumPy geodesic generator.

    Keeps the existing API (subdiv, radius, center) but internally routes to
    geometry.geodesic.geodesic_sphere with triangulated topology. Returns
    (verts, faces) as float64/int32 NumPy arrays.
    """
    # Match legacy subdiv semantics where each step quarters triangles:
    # geodesic frequency f produces f^2 triangles per base face, so f=2**subdiv
    freq = int(2 ** max(0, int(subdiv)))
    out = geodesic_sphere(
        freq=freq,
        radius=float(radius),
        center=tuple(center),
        topology="tri",
    )
    v = np.asarray(out["verts"], dtype=np.float64)
    f = np.asarray(out["faces"], dtype=np.int32)
    return v, f

def make_goldberg_sphere(subdiv=1, radius=0.1, center=(0,0,0)):
    """Build a dual (Goldberg) sphere and return a triangulated fan.

    Useful if you prefer the dual hex-dominant layout but need triangles for
    the XPBD pipeline. Returns (verts, faces_tri).
    """
    freq = int(2 ** max(0, int(subdiv)))
    out = geodesic_sphere(
        freq=freq,
        radius=float(radius),
        center=tuple(center),
        topology="hex",
        return_tri_from_hex=True,
    )
    v = np.asarray(out["verts"], dtype=np.float64)
    ftri = np.asarray(out["faces_tri"], dtype=np.int32)
    return v, ftri

def mesh_volume(verts, faces):
    """Compute enclosed volume of a closed triangle mesh.

    Previous implementation iterated triangle-by-triangle in Python. By
    indexing vertex arrays with NumPy and using ``np.cross``/``np.einsum`` we
    evaluate the scalar triple product for all faces at once, letting NumPy's
    vectorised C loops handle the heavy lifting.
    """
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    V = np.einsum("ij,ij->i", a, np.cross(b, c))
    return V.sum() / 6.0

def volume_gradients(verts, faces):
    """Gradient of ``mesh_volume`` w.r.t. vertex positions.

    Uses ``np.add.at`` to accumulate per-face contributions without explicit
    Python loops.
    """
    grads = np.zeros_like(verts)
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    np.add.at(grads, faces[:, 0], np.cross(b, c))
    np.add.at(grads, faces[:, 1], np.cross(c, a))
    np.add.at(grads, faces[:, 2], np.cross(a, b))
    grads /= 6.0
    return grads


# ----- Lower-dimensional measures -----------------------------------------

def mesh_area2d(verts, faces):
    """Area of a triangulated mesh projected to the XY plane."""
    a = verts[faces[:, 0], :2]
    b = verts[faces[:, 1], :2]
    c = verts[faces[:, 2], :2]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return cross.sum() * 0.5


def area_gradients2d(verts, faces):
    """Gradient of ``mesh_area2d`` with respect to vertex positions."""
    grads = np.zeros_like(verts)
    a = verts[faces[:, 0], :2]
    b = verts[faces[:, 1], :2]
    c = verts[faces[:, 2], :2]
    pad = np.zeros((len(faces), 1))
    grad_a = np.stack([b[:, 1] - c[:, 1], c[:, 0] - b[:, 0]], axis=1) * 0.5
    grad_b = np.stack([c[:, 1] - a[:, 1], a[:, 0] - c[:, 0]], axis=1) * 0.5
    grad_c = np.stack([a[:, 1] - b[:, 1], b[:, 0] - a[:, 0]], axis=1) * 0.5
    np.add.at(grads, faces[:, 0], np.hstack([grad_a, pad]))
    np.add.at(grads, faces[:, 1], np.hstack([grad_b, pad]))
    np.add.at(grads, faces[:, 2], np.hstack([grad_c, pad]))
    return grads


def polyline_length(verts, edges, dim=3):
    """Total length of a polyline defined by ``edges``."""
    a = verts[edges[:, 0], :dim]
    b = verts[edges[:, 1], :dim]
    return float(np.linalg.norm(b - a, axis=1).sum())


def polyline_length_gradients(verts, edges, dim=3):
    """Gradient of ``polyline_length`` wrt vertex positions."""
    a = verts[edges[:, 0], :dim]
    b = verts[edges[:, 1], :dim]
    d = b - a
    L = np.linalg.norm(d, axis=1, keepdims=True)
    n = np.zeros_like(d)
    mask = L[:, 0] > 1e-12
    n[mask] = d[mask] / L[mask]
    grads = np.zeros_like(verts)
    pad = verts.shape[1] - dim
    if pad > 0:
        n_full = np.hstack([n, np.zeros((len(edges), pad))])
    else:
        n_full = n
    np.add.at(grads, edges[:, 0], -n_full)
    np.add.at(grads, edges[:, 1], n_full)
    return grads

def build_adjacency(faces):
    edge2tris = {}
    for t_idx,(i,j,k) in enumerate(faces):
        for e in [(i,j),(j,k),(k,i)]:
            e_sorted = tuple(sorted(e))
            edge2tris.setdefault(e_sorted, []).append((t_idx, (i,j,k)))
    edges = []
    bends = []  # (i,j,k,l)
    for (i,j), tris in edge2tris.items():
        edges.append((i,j))
        if len(tris)==2:
            (t1,(a1,b1,c1)), (t2,(a2,b2,c2)) = tris
            k = [v for v in (a1,b1,c1) if v not in (i,j)][0]
            l = [v for v in (a2,b2,c2) if v not in (i,j)][0]
            bends.append((i,j,k,l))
    return edges, bends
