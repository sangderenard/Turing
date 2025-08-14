# membrane.py
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, List
import numpy as np


# ----------------------------- Public config types -----------------------------

@dataclass
class MembraneConfig:
    # Bending (Helfrich) energy
    bending_kappa: float = 5e-20         # [J] typical lipid bilayer scale
    spontaneous_curvature: float = 0.0   # C0, 1/m

    # Area resistance (penalties; use strong values to mimic incompressibility)
    area_k_local: float = 0.0            # per-triangle penalty [N/m]
    area_k_global: float = 1e-3          # global total-area penalty [N/m]

    # Optional XPBD-style constraint compliances (set >0 to enable small projection nudges)
    xpbd_compliance_area_local: float = 0.0
    xpbd_compliance_area_global: float = 0.0
    xpbd_compliance_volume: float = 0.0

    # Preferred shape penalty (keeps shape near a rest shell without pinning rigid motion)
    preferred_shape_k: float = 0.0       # [N] multiplies Laplacian mismatch (smooth “spring to rest-shape”)
    preferred_shape_tangent_only: bool = True  # resist tangential drift more than normal drift

    # Cortex (in-plane shearable network)
    cortex_enabled: bool = False
    cortex_shear_k: float = 0.0          # [N/m] linear edge spring
    cortex_damping_c: float = 0.0        # [N·s/m] edge dashpot

    # Fluid / environment coupling
    drag_coefficient: float = 0.0        # [N·s/m] per-vertex isotropic drag (IB-lite)
    ib_mode: str = "none"                # "none" | "ib" (drag) | "bie" (use traction callback if set)

    # Numerics
    eps_geom: float = 1e-18


@dataclass
class MembraneHooks:
    """
    External hooks the engine/cellsim can provide.

    fluid_velocity: Callable(X:(n,3), t)-> (n,3) vertex fluid velocities (IB)
    external_traction: Callable(X:(n,3), normals:(n,3), A_faces:(m,), F:(m,3), t)-> (n,3) vertex tractions (BIE)
    deltaP: Callable(volume: float, area: float, t: float)-> float  (osmotic/hydrostatic pressure jump)
    """
    fluid_velocity: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
    external_traction: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]] = None
    deltaP: Optional[Callable[[float, float, float], float]] = None


@dataclass
class MembraneState:
    """Static mesh & rest geometry."""
    F: np.ndarray                     # (m,3) triangle indices, int64
    X_ref: np.ndarray                 # (n,3) reference vertices (preferred shell)
    A_ref_local: np.ndarray           # (m,) per-triangle rest areas
    A_ref_total: float                # scalar rest total area
    V_ref: float                      # scalar rest volume
    E_edges: np.ndarray               # (k,2) unique undirected edges for cortex
    L0_edges: np.ndarray              # (k,) edge rest lengths for cortex
    N_ref: np.ndarray                 # (n,3) vertex normals at reference (for tangent-only option)


# ----------------------------- Builder / utilities -----------------------------

def normalize_rows(X: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n

def edges_from_faces(F: np.ndarray) -> np.ndarray:
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E.sort(axis=1)
    E = np.unique(E, axis=0)
    return E

def tri_areas_normals(X: np.ndarray, F: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    xi, xj, xk = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    n = np.cross(xj - xi, xk - xi)           # 2A * n_hat
    dblA = np.maximum(np.linalg.norm(n, axis=1, keepdims=True), eps)
    A = 0.5 * dblA[:, 0]
    n_hat = n / dblA
    return A, n_hat

def tri_area_grads(X: np.ndarray, F: np.ndarray, eps: float) -> np.ndarray:
    """dA/dx_i per face i,j,k — shape (m,3,3)."""
    xi, xj, xk = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    eij = xj - xi; eik = xk - xi
    n = np.cross(eij, eik)
    dblA = np.maximum(np.linalg.norm(n, axis=1, keepdims=True), eps)
    n_hat = n / dblA
    dAi = 0.5 * np.cross(xk - xj, n_hat)
    dAj = 0.5 * np.cross(xi - xk, n_hat)
    dAk = 0.5 * np.cross(xj - xi, n_hat)
    return np.stack([dAi, dAj, dAk], axis=1)

def vertex_normals_from_area_weighted(X: np.ndarray, F: np.ndarray, eps: float) -> np.ndarray:
    """Area-weighted vertex normals."""
    A, n_hat = tri_areas_normals(X, F, eps)
    n = n_hat * (A[:, None])  # weighted
    N = np.zeros_like(X)
    np.add.at(N, F[:, 0], n)
    np.add.at(N, F[:, 1], n)
    np.add.at(N, F[:, 2], n)
    N = normalize_rows(N, eps)
    return N

def volume_and_grads(X: np.ndarray, F: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Signed volume via face fan to origin; grads dV/dx_i, shape (n,3).
    V = (1/6) sum_f (x_i x x_j)·x_k
    dV/dx_i = (1/6) (x_j x x_k)
    """
    xi, xj, xk = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    V = (1.0 / 6.0) * np.sum(np.einsum("ij,ij->i", np.cross(xi, xj), xk))
    # Per-face vertex grads
    g_i = np.cross(xj, xk) / 6.0
    g_j = np.cross(xk, xi) / 6.0
    g_k = np.cross(xi, xj) / 6.0
    G = np.zeros_like(X)
    np.add.at(G, F[:, 0], g_i)
    np.add.at(G, F[:, 1], g_j)
    np.add.at(G, F[:, 2], g_k)
    return V, G

def cotan_weights_face(X: np.ndarray, i: int, j: int, k: int, eps: float) -> Tuple[float, float, float]:
    """
    For triangle (i,j,k), return cotangents at vertices opposite each edge:
      w_ij += 0.5*cot(angle at k), etc.
    """
    xi, xj, xk = X[i], X[j], X[k]
    # cot at k, opposite edge (i,j)
    a = xi - xk; b = xj - xk
    cot_k = np.dot(a, b) / max(eps, np.linalg.norm(np.cross(a, b)))
    # cot at i, opposite (j,k)
    a = xj - xi; b = xk - xi
    cot_i = np.dot(a, b) / max(eps, np.linalg.norm(np.cross(a, b)))
    # cot at j, opposite (k,i)
    a = xk - xj; b = xi - xj
    cot_j = np.dot(a, b) / max(eps, np.linalg.norm(np.cross(a, b)))
    return cot_i, cot_j, cot_k


# ----------------------------- Membrane core -----------------------------

class Membrane:
    """
    Master membrane model: builds once from mesh, then computes forces per substep.

    Use:
        mem = Membrane(F, X0, cfg, hooks)
        F_total, parts, geom = mem.forces(X, V, t)
    """
    def __init__(self, F: np.ndarray, X0: np.ndarray,
                 cfg: Optional[MembraneConfig] = None,
                 hooks: Optional[MembraneHooks] = None):
        assert F.ndim == 2 and F.shape[1] == 3
        assert X0.ndim == 2 and X0.shape[1] == 3
        self.cfg = cfg or MembraneConfig()
        self.hooks = hooks or MembraneHooks()
        self.F = F.astype(np.int64, copy=True)
        self.n = X0.shape[0]
        self.m = F.shape[0]

        # Rest / reference geometry
        A0, _ = tri_areas_normals(X0, self.F, self.cfg.eps_geom)
        V0, _ = volume_and_grads(X0, self.F)
        E = edges_from_faces(self.F)
        L0 = np.linalg.norm(X0[E[:, 1]] - X0[E[:, 0]], axis=1)
        N_ref = vertex_normals_from_area_weighted(X0, self.F, self.cfg.eps_geom)

        self.state = MembraneState(
            F=self.F,
            X_ref=X0.copy(),
            A_ref_local=A0.copy(),
            A_ref_total=float(np.sum(A0)),
            V_ref=float(V0),
            E_edges=E,
            L0_edges=L0,
            N_ref=N_ref,
        )

    # --------------------- public: compute all forces ---------------------

    def forces(self, X: np.ndarray, V: np.ndarray, t: float = 0.0
               ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
        """
        Compute per-vertex forces and return:
          F_total (n,3),
          parts: dict of named components,
          geom: dict with 'A_total','V','ΔP' etc for coupling.
        """
        cfg = self.cfg
        st = self.state
        n = self.n

        F_total = np.zeros((n, 3), dtype=float)
        parts: Dict[str, np.ndarray] = {}

        # geometry
        A_local, n_hat = tri_areas_normals(X, st.F, cfg.eps_geom)
        A_total = float(np.sum(A_local))
        V, dVdx = volume_and_grads(X, st.F)

        # ---------------- pressure (osmosis/hydro) ----------------
        dP = 0.0
        if self.hooks.deltaP is not None:
            dP = float(self.hooks.deltaP(V, A_total, t))
        if dP != 0.0:
            Fp = self._pressure_forces_from_faces(A_local, n_hat, dP, X)
            F_total += Fp; parts["pressure"] = Fp
        else:
            parts["pressure"] = np.zeros_like(F_total)

        # ---------------- area penalties --------------------------
        if cfg.area_k_local > 0.0 or cfg.area_k_global > 0.0:
            Fa = self._area_forces(X, A_local, st.A_ref_local, st.A_ref_total, cfg.area_k_local, cfg.area_k_global)
            F_total += Fa; parts["area"] = Fa
        else:
            parts["area"] = np.zeros_like(F_total)

        # ---------------- bending: Helfrich via cotan Laplacian ---
        if cfg.bending_kappa > 0.0:
            Fb = self._bending_forces_cotan(X, st.F, cfg.bending_kappa, cfg.spontaneous_curvature)
            F_total += Fb; parts["bending"] = Fb
        else:
            parts["bending"] = np.zeros_like(F_total)

        # ---------------- preferred shape (Laplacian match) -------
        if cfg.preferred_shape_k > 0.0:
            Fref = self._preferred_shape_force(X)
            if cfg.preferred_shape_tangent_only:
                # remove normal component along current normals to avoid “pin spikes”
                N = vertex_normals_from_area_weighted(X, st.F, cfg.eps_geom)
                Fref -= (np.sum(Fref * N, axis=1, keepdims=True)) * N
            F_total += Fref; parts["shape_ref"] = Fref
        else:
            parts["shape_ref"] = np.zeros_like(F_total)

        # ---------------- cortex (in-plane springs/dashpots) -------
        if self.cfg.cortex_enabled and (self.cfg.cortex_shear_k > 0.0 or self.cfg.cortex_damping_c > 0.0):
            Fc = self._cortex_forces(X, V)
            F_total += Fc; parts["cortex"] = Fc
        else:
            parts["cortex"] = np.zeros_like(F_total)

        # ---------------- fluid coupling: IB / BIE -----------------
        Fenv = np.zeros_like(F_total)
        if cfg.ib_mode == "ib":
            # simple Stokes drag to ambient fluid velocity field
            if cfg.drag_coefficient > 0.0:
                u = self.hooks.fluid_velocity(X, t) if self.hooks.fluid_velocity else 0.0
                Fenv = -cfg.drag_coefficient * (V - (u if isinstance(u, np.ndarray) else 0.0))
        elif cfg.ib_mode == "bie":
            # traction provided by boundary-integral solver (if any)
            if self.hooks.external_traction is not None:
                Fenv = self.hooks.external_traction(X, self._vertex_normals_cached(X), A_local, st.F, t)
        F_total += Fenv; parts["env"] = Fenv

        # ---------------- tiny XPBD nudges (optional) --------------
        if cfg.xpbd_compliance_area_local > 0.0 or cfg.xpbd_compliance_area_global > 0.0 or cfg.xpbd_compliance_volume > 0.0:
            Fx = self._xpbd_micro_projections(X, A_local, dVdx, A_total, V, t)
            F_total += Fx; parts["xpbd"] = Fx
        else:
            parts["xpbd"] = np.zeros_like(F_total)

        geom = {"A_total": A_total, "V": float(V), "ΔP": dP}
        return F_total, parts, geom

    # ----------------------------- components -----------------------------

    def _pressure_forces_from_faces(self, A: np.ndarray, n_hat: np.ndarray, deltaP: float, X: np.ndarray) -> np.ndarray:
        """Uniform pressure jump ΔP: equal share to triangle vertices along face normal."""
        Fv = np.zeros_like(X)
        fp = (deltaP * A / 3.0)[:, None] * n_hat
        np.add.at(Fv, self.state.F[:, 0], fp)
        np.add.at(Fv, self.state.F[:, 1], fp)
        np.add.at(Fv, self.state.F[:, 2], fp)
        return Fv

    def _area_forces(self, X: np.ndarray, A_local: np.ndarray,
                     A0_local: np.ndarray, A0_total: float,
                     k_local: float, k_global: float) -> np.ndarray:
        """Penalty forces: F = -k * (A - A0) * gradA."""
        Fv = np.zeros_like(X)
        grads = tri_area_grads(X, self.state.F, self.cfg.eps_geom)  # (m,3,3)

        # local
        if k_local > 0.0:
            dA = (A_local - A0_local)[:, None, None]
            Fl = -k_local * dA * grads
            # scatter
            np.add.at(Fv, self.state.F[:, 0], Fl[:, 0, :])
            np.add.at(Fv, self.state.F[:, 1], Fl[:, 1, :])
            np.add.at(Fv, self.state.F[:, 2], Fl[:, 2, :])

        # global
        if k_global > 0.0:
            dA_tot = (np.sum(A_local) - A0_total)
            # dA_tot/dx = sum_f gradA_f
            Gsum = np.zeros_like(X)
            np.add.at(Gsum, self.state.F[:, 0], grads[:, 0, :])
            np.add.at(Gsum, self.state.F[:, 1], grads[:, 1, :])
            np.add.at(Gsum, self.state.F[:, 2], grads[:, 2, :])
            Fv += -k_global * dA_tot * Gsum

        return Fv

    def _bending_forces_cotan(self, X: np.ndarray, F: np.ndarray,
                              kappa: float, C0: float) -> np.ndarray:
        """
        Discrete mean-curvature force: F = -kappa * (L X - C0 * something).
        Here we implement the classic cotangent Laplacian L acting on vertex positions:
            (L X)_i = sum_j w_ij (X_i - X_j), w_ij = 0.5*(cot α + cot β)
        The spontaneous curvature C0 term is often modeled as a pressure/tension adjustment;
        for simplicity we omit explicit C0 gradient here or absorb it into ΔP.
        """
        n = X.shape[0]
        Fv = np.zeros_like(X)
        # accumulate per-face contributions
        for (ia, ib, ic) in F:
            ci, cj, ck = cotan_weights_face(X, ia, ib, ic, self.cfg.eps_geom)
            # contributions to each edge pair
            # edge (i,j): weight from opposite k is 0.5*cot_k
            w_ij = 0.5 * ck
            w_jk = 0.5 * ci
            w_ki = 0.5 * cj

            xi, xj, xk = X[ia], X[ib], X[ic]

            # i <-> j
            fij = w_ij * (xi - xj)
            Fv[ia] += -kappa * fij
            Fv[ib] +=  kappa * fij

            # j <-> k
            fjk = w_jk * (xj - xk)
            Fv[ib] += -kappa * fjk
            Fv[ic] +=  kappa * fjk

            # k <-> i
            fki = w_ki * (xk - xi)
            Fv[ic] += -kappa * fki
            Fv[ia] +=  kappa * fki

        return Fv

    def _preferred_shape_force(self, X: np.ndarray) -> np.ndarray:
        """
        Smooth “spring to rest-shape”: Laplacian of (X - X_ref).
        F = -k_ref * L_ref (X - X_ref), with L_ref assembled on reference mesh implicitly via cotans.
        """
        k = self.cfg.preferred_shape_k
        Xref = self.state.X_ref
        n = X.shape[0]
        Fv = np.zeros_like(X)

        # Use reference cotan weights for stability
        for (ia, ib, ic) in self.state.F:
            ci, cj, ck = cotan_weights_face(self.state.X_ref, ia, ib, ic, self.cfg.eps_geom)
            w_ij = 0.5 * ck
            w_jk = 0.5 * ci
            w_ki = 0.5 * cj

            # Laplacian action on (X - X_ref)
            di = X[ia] - Xref[ia]
            dj = X[ib] - Xref[ib]
            dk = X[ic] - Xref[ic]

            # i <-> j
            lij = w_ij * (di - dj)
            Fv[ia] += -k * lij
            Fv[ib] +=  k * lij

            # j <-> k
            ljk = w_jk * (dj - dk)
            Fv[ib] += -k * ljk
            Fv[ic] +=  k * ljk

            # k <-> i
            lki = w_ki * (dk - di)
            Fv[ic] += -k * lki
            Fv[ia] +=  k * lki

        return Fv

    def _cortex_forces(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Linear edge springs + dashpots on the cortex graph."""
        k = self.cfg.cortex_shear_k
        c = self.cfg.cortex_damping_c
        E = self.state.E_edges
        L0 = self.state.L0_edges

        Fv = np.zeros_like(X)
        if E.size == 0:
            return Fv

        vi = E[:, 0]; vj = E[:, 1]
        dij = X[vj] - X[vi]
        L = np.maximum(np.linalg.norm(dij, axis=1, keepdims=True), self.cfg.eps_geom)
        dir_ = dij / L

        if k > 0.0:
            fmag = k * (L - L0[:, None])          # (k,1)
            f = fmag * dir_
            np.add.at(Fv, vi,  f)
            np.add.at(Fv, vj, -f)

        if c > 0.0:
            vrel = (V[vj] - V[vi])
            s = np.sum(vrel * dir_, axis=1, keepdims=True)
            fd = c * s * dir_
            np.add.at(Fv, vi,  fd)
            np.add.at(Fv, vj, -fd)

        return Fv

    def _vertex_normals_cached(self, X: np.ndarray) -> np.ndarray:
        return vertex_normals_from_area_weighted(X, self.state.F, self.cfg.eps_geom)

    def _xpbd_micro_projections(self, X: np.ndarray, A_local: np.ndarray, dVdx: np.ndarray,
                                A_total: float, V: float, t: float) -> np.ndarray:
        """
        Very small “nudge” forces that mimic one XPBD projection step without mutating X.
        Intended to be tiny stabilizers; keep compliances small.
        """
        Fv = np.zeros_like(X)
        grads = tri_area_grads(X, self.state.F, self.cfg.eps_geom)

        # Local area
        alpha_l = self.cfg.xpbd_compliance_area_local
        if alpha_l > 0.0:
            C = (A_local - self.state.A_ref_local)  # (m,)
            # Jacobian norm squared (sum over the 3 vertex grads)
            J2 = np.sum(grads ** 2, axis=(1, 2)) + 1e-18
            lam = -C / (J2 * (1.0 + alpha_l))      # damped scalar per face
            # force ~ -λ * gradC
            Fl = -(lam[:, None, None]) * grads
            np.add.at(Fv, self.state.F[:, 0], Fl[:, 0, :])
            np.add.at(Fv, self.state.F[:, 1], Fl[:, 1, :])
            np.add.at(Fv, self.state.F[:, 2], Fl[:, 2, :])

        # Global area
        alpha_g = self.cfg.xpbd_compliance_area_global
        if alpha_g > 0.0:
            Cg = (A_total - self.state.A_ref_total)
            Gsum = np.zeros_like(X)
            np.add.at(Gsum, self.state.F[:, 0], grads[:, 0, :])
            np.add.at(Gsum, self.state.F[:, 1], grads[:, 1, :])
            np.add.at(Gsum, self.state.F[:, 2], grads[:, 2, :])
            J2 = np.sum(Gsum ** 2) + 1e-18
            lam = -Cg / (J2 * (1.0 + alpha_g))
            Fv += -lam * Gsum

        # Volume (tiny correction)
        alpha_v = self.cfg.xpbd_compliance_volume
        if alpha_v > 0.0:
            Cv = (V - self.state.V_ref)
            J2 = np.sum(dVdx ** 2) + 1e-18
            lam = -Cv / (J2 * (1.0 + alpha_v))
            Fv += -lam * dVdx

        return Fv

    # ----------------------------- helpers / mutations -----------------------------

    def set_reference_shape(self, X_ref: np.ndarray) -> None:
        """Update preferred shell to a new rest shape (e.g., turgor remodeling)."""
        A0, _ = tri_areas_normals(X_ref, self.state.F, self.cfg.eps_geom)
        V0, _ = volume_and_grads(X_ref, self.state.F)
        self.state.X_ref = X_ref.copy()
        self.state.A_ref_local = A0.copy()
        self.state.A_ref_total = float(np.sum(A0))
        self.state.V_ref = float(V0)
        self.state.N_ref = vertex_normals_from_area_weighted(X_ref, self.state.F, self.cfg.eps_geom)

    def update_edge_rest_lengths(self, X_for_cortex: np.ndarray) -> None:
        """Slow remodeling: e.g., cortex rest-length adaptation."""
        E = self.state.E_edges
        self.state.L0_edges = np.linalg.norm(X_for_cortex[E[:, 1]] - X_for_cortex[E[:, 0]], axis=1)

    # ----------------------------- convenience for coupling -----------------------------

    def measure_geometry(self, X: np.ndarray) -> Dict[str, float]:
        A, _ = tri_areas_normals(X, self.state.F, self.cfg.eps_geom)
        V, _ = volume_and_grads(X, self.state.F)
        return {"A_total": float(np.sum(A)), "V": float(V)}

    def set_hooks(self, hooks: MembraneHooks) -> None:
        self.hooks = hooks
