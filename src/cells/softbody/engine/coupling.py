import numpy as np
import math
from .mesh import mesh_area2d, polyline_length


def cell_area(cell):
    """Compute total surface measure of a softbody cell."""
    dim = getattr(cell, "dim", 3)
    X = np.asarray(cell.X, dtype=np.float64)
    if dim == 2:
        F = np.asarray(cell.faces, dtype=np.int32)
        return float(mesh_area2d(X, F))
    if dim == 1:
        edges = np.asarray(cell.faces, dtype=np.int32)
        return float(polyline_length(X, edges, dim=1))
    F = np.asarray(cell.faces, dtype=np.int32)
    AB = X[F[:, 1]] - X[F[:, 0]]
    AC = X[F[:, 2]] - X[F[:, 0]]
    A = 0.5 * np.linalg.norm(np.cross(AB, AC), axis=1).sum()
    return float(A)


def _prep_refs(cell):
    if not hasattr(cell, "_A0"):
        cell._A0 = cell_area(cell)
    if not hasattr(cell, "_V0"):
        cell._V0 = abs(cell.enclosed_volume())
    if getattr(cell, "dim", 3) != 3 or hasattr(cell, "_edge_dual_area"):
        return
    if not hasattr(cell, "_edge_dual_area"):
        F = np.asarray(cell.faces, dtype=np.int32)
        X = np.asarray(cell.X, dtype=np.float64)
        edges = np.asarray(cell.edges, dtype=np.int32)
        n_verts = X.shape[0]
        n_edges = edges.shape[0]

        AB = X[F[:, 1]] - X[F[:, 0]]
        AC = X[F[:, 2]] - X[F[:, 0]]
        A = 0.5 * np.linalg.norm(np.cross(AB, AC), axis=1)

        edge_map = -np.ones((n_verts, n_verts), dtype=np.int32)
        idx = np.arange(n_edges, dtype=np.int32)
        edge_map[edges[:, 0], edges[:, 1]] = idx
        edge_map[edges[:, 1], edges[:, 0]] = idx

        face_edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        edge_idx = edge_map[face_edges[:, 0], face_edges[:, 1]]
        dual = np.zeros(n_edges, dtype=np.float64)
        np.add.at(dual, edge_idx, np.repeat(A / 3.0, 3))

        cell._edge_dual_area = dual
        cell._edge_lookup = edge_map


def _set_stretch_from_Ka(cell, Ka):
    if getattr(cell, "dim", 3) != 3:
        return
    _prep_refs(cell)
    eps = 1e-12
    sc = cell.constraints.get("stretch")
    if sc is None:
        return
    idx = np.asarray(sc["indices"], dtype=np.int32)
    i, j = idx[:, 0], idx[:, 1]
    Adual = np.maximum(eps, cell._edge_dual_area[cell._edge_lookup[i, j]])
    L0 = np.maximum(eps, sc["rest"])
    sc["compliance"][:] = 1.0 / np.maximum(eps, Ka * (Adual / L0))
    sc["lamb"][:] = 0.0


def laplace_from_Ka(cell, Ka, gamma0=0.0):
    dim = getattr(cell, "dim", 3)
    A = cell_area(cell)
    A0 = max(1e-12, getattr(cell, "_A0", A))
    V = abs(cell.enclosed_volume())
    if dim == 3:
        R = ((3.0 * max(V, 1e-12)) / (4.0 * math.pi)) ** (1.0 / 3.0)
        dP_factor = 2.0 / max(1e-6, R)
    elif dim == 2:
        R = (max(A, 1e-12) / math.pi) ** 0.5
        dP_factor = 1.0 / max(1e-6, R)
    else:
        R = max(V, 1e-12) * 0.5
        dP_factor = 2.0 / max(1e-6, V)
    eps_A = (A - A0) / A0
    gamma = gamma0 + Ka * eps_A
    dP_L = gamma * dP_factor
    return float(gamma), float(dP_L)


def harmonized_update(cell, Ka, K_bulk, P_osm_in, P_ext, gamma0=0.0):
    _prep_refs(cell)
    _set_stretch_from_Ka(cell, Ka)
    gamma, dP_L = laplace_from_Ka(cell, Ka, gamma0)
    V = abs(cell.enclosed_volume())
    P_drive = float(P_ext) + float(dP_L) - float(P_osm_in)
    dV = -V * (P_drive / max(1e-12, K_bulk))
    V_target = V + dV
    vc = cell.constraints.get("volume")
    if vc is not None:
        vc.target = float(V_target)
        vc.compliance = 1.0 / max(1e-12, K_bulk)
        vc.lamb = 0.0
    return {
        "gamma": gamma,
        "dP_L": dP_L,
        "P_drive": P_drive,
        "V_target": V_target,
    }
