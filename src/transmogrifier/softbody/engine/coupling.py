import numpy as np
import math


def cell_area(cell):
    """Compute total surface area of a softbody cell."""
    A = 0.0
    X, F = cell.X, cell.faces
    for tri in F:
        a, b, c = X[tri[0]], X[tri[1]], X[tri[2]]
        A += 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    return float(A)


def _prep_refs(cell):
    if not hasattr(cell, "_A0"):
        cell._A0 = cell_area(cell)
    if not hasattr(cell, "_V0"):
        cell._V0 = abs(cell.enclosed_volume())
    if not hasattr(cell, "_edge_dual_area"):
        dual = {}
        for (i, j) in cell.edges:
            dual[(i, j)] = 0.0
            dual[(j, i)] = 0.0
        for (i, j, k) in cell.faces:
            a = 0.5 * np.linalg.norm(np.cross(cell.X[j] - cell.X[i], cell.X[k] - cell.X[i]))
            for e in [(i, j), (j, k), (k, i)]:
                dual[e] += a / 3.0
        cell._edge_dual_area = {
            tuple(sorted(e)): (dual[e] + dual[(e[1], e[0])])
            for e in dual
            if e[0] < e[1]
        }


def _set_stretch_from_Ka(cell, Ka):
    _prep_refs(cell)
    eps = 1e-12
    for sc in cell.constraints.get("stretch", []):
        i, j = sc.i, sc.j
        Adual = max(eps, cell._edge_dual_area.get(tuple(sorted((i, j))), 0.0))
        L0 = max(eps, sc.rest)
        k_edge = Ka * (Adual / L0)
        sc.compliance = 1.0 / max(eps, k_edge)
        sc.lamb = 0.0


def laplace_from_Ka(cell, Ka, gamma0=0.0):
    A = cell_area(cell)
    A0 = max(1e-12, getattr(cell, "_A0", A))
    V = abs(cell.enclosed_volume())
    R = ((3.0 * max(V, 1e-12)) / (4.0 * math.pi)) ** (1.0 / 3.0)
    eps_A = (A - A0) / A0
    gamma = gamma0 + Ka * eps_A
    dP_L = 2.0 * gamma / max(1e-6, R)
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
