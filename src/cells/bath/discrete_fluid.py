# discrete_fluid.py
# -*- coding: utf-8 -*-
"""
Discrete particle fluid simulator (WCSPH) for the Bath engine.

Design goals
------------
- Physics-predictive Smoothed Particle Hydrodynamics (SPH) core (Weakly-Compressible SPH).
- NumPy-only (no numba), with careful batching to avoid memory blow-ups.
- Clean API for coupling with an external solver ("Bath") via:
    * source terms (mass/solute injection or extraction at points),
    * sampling fields (pressure, velocity, salinity, temperature),
    * per-step callbacks (e.g., osmotic filters).
- Deterministic, reproducible results for the same inputs (no RNG use here).
- Units: MKS (meter, kilogram, second). Keep parameters consistent.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Iterable

import numpy as np
import copy
from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.dt_controller import Targets, STController, step_with_dt_control
from src.common.dt_system.debug import dbg, is_enabled


# ------------------------------ Kernels -------------------------------------

@dataclass
class SPHKernel:
    """Standard cubic kernels for 3D SPH."""
    h: float  # smoothing length (support radius ~ h)
    h2: float = field(init=False)
    c_poly6: float = field(init=False)
    c_spiky: float = field(init=False)
    c_visc: float = field(init=False)

    def __post_init__(self):
        self.h2 = self.h * self.h
        # 3D normalization constants
        # poly6: W(r) = C * (h^2 - r^2)^3 for r < h
        self.c_poly6 = 315.0 / (64.0 * np.pi * self.h ** 9)
        # spiky gradient magnitude: |∇W| = C * (h - r)^2, direction -r_hat
        self.c_spiky = 45.0 / (np.pi * self.h ** 6)
        # viscosity laplacian: ∇^2 W = C * (h - r)
        self.c_visc = 45.0 / (np.pi * self.h ** 6)

    # Original safe versions (kept for compatibility)
    def W(self, r: np.ndarray) -> np.ndarray:
        """Scalar kernel value for distances r (shape: (...,))."""
        q2 = np.clip(self.h2 - r*r, 0.0, None)
        return self.c_poly6 * q2 * q2 * q2

    def gradW(self, r_vec: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Gradient ∇W as a vector with shape (..., 3).
        Uses the spiky kernel derivative for improved stability.
        For r=0, returns 0.
        """
        mask = (r > 0) & (r < self.h)
        coeff = np.zeros_like(r)
        dr = (self.h - r[mask])
        coeff[mask] = -self.c_spiky * dr * dr / np.maximum(r[mask], 1e-20)  # -(C*(h - r)^2) * r_hat
        return r_vec * coeff[:, None]

    def laplaceW_visc(self, r: np.ndarray) -> np.ndarray:
        """Viscosity Laplacian ∇^2 W for distances r (shape (...,))."""
        out = np.zeros_like(r)
        mask = (r >= 0) & (r < self.h)
        out[mask] = self.c_visc * (self.h - r[mask])
        return out

    # Hot-path masked variants (callers guarantee r< h etc.)
    def W_from_r2_masked(self, r2: np.ndarray) -> np.ndarray:
        q2 = self.h2 - r2
        return self.c_poly6 * q2 * q2 * q2

    def gradW_masked(self, r_vec: np.ndarray, r: np.ndarray) -> np.ndarray:
        coeff = -self.c_spiky * ((self.h - r) ** 2) / np.maximum(r, 1e-20)
        return r_vec * coeff[:, None]

    def laplaceW_visc_masked(self, r: np.ndarray) -> np.ndarray:
        return self.c_visc * (self.h - r)


# --------------------------- Physical Parameters ----------------------------

@dataclass
class FluidParams:
    # Core SPH
    rest_density: float = 1000.0       # ρ0 (kg/m^3) e.g., water
    particle_mass: float = 0.02        # m (kg)
    smoothing_length: float = 0.1      # h (m), support radius
    gamma: float = 7.0                 # EOS exponent
    bulk_modulus: float = 2.2e9        # K (Pa) ~ water
    viscosity_nu: float = 1.0e-6       # kinematic viscosity ν (m^2/s), ~water 1e-6
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)

    # Diffusion
    thermal_diffusivity: float = 1.4e-7   # κ (m^2/s) ~ water
    solute_diffusivity: float = 1.0e-9    # D_s (m^2/s) typical ion diffusion

    # Surface tension (optional; set to 0 to disable)
    surface_tension: float = 0.0          # σ (N/m)
    color_field_eps: float = 1e-5

    # Buoyancy (Boussinesq approx): ρ = ρ0 (1 - β_T (T-T0) - β_S (S-S0))
    T0: float = 293.15                    # reference temperature (K)
    beta_T: float = 2.07e-4               # thermal expansion (1/K)
    S0: float = 0.0                       # reference salinity (kg/kg)
    beta_S: float = 7.6e-4                # haline contraction (1)

    # Time stepping / stabilization
    xsph_eps: float = 0.0                 # XSPH velocity blending (0..0.5 typical)
    cfl_number: float = 0.25

    # Boundary
    bounce_damping: float = 0.0           # velocity damping on boundary collision [0..1]

    # Source term relaxation (0..1, 1 = immediate)
    source_relaxation: float = 1.0


# ----------------------------- Discrete Fluid -------------------------------

@dataclass
class DropletParticle:
    """Lightweight particle with higher surface tension and smaller mass."""
    mass: float = 0.005
    surface_tension: float = 0.072

class DiscreteFluid:
    """
    Weakly-Compressible SPH (WCSPH) with:
      - Pressure (Tait EOS)
      - Viscosity (Laplacian)
      - Gravity + Boussinesq buoyancy (T, S)
      - Optional surface tension (color field curvature model)
      - Advection by velocity (semi-implicit)
      - Diffusion of temperature and salinity by SPH Laplacian
      - Axis-aligned box boundaries with inelastic bounce + damping

    State arrays (N particles):
      pos : (N,3) position [m]
      vel : (N,3) velocity [m/s]
      rho : (N,)  density [kg/m^3]
      P   : (N,)  pressure [Pa]
      T   : (N,)  temperature [K]
      S   : (N,)  salinity (mass fraction kg/kg)
    """
    def __init__(
        self,
        positions: np.ndarray,                 # (N,3)
        velocities: Optional[np.ndarray],
        temperature: Optional[np.ndarray],
        salinity: Optional[np.ndarray],
        params: FluidParams,
        bounds_min: Tuple[float, float, float] = (-np.inf, -np.inf, -np.inf),
        bounds_max: Tuple[float, float, float] = ( np.inf,  np.inf,  np.inf),
        droplet_indices: Optional[Iterable[int]] = None,
        droplet_particle: Optional[DropletParticle] = None,
    ) -> None:
        assert positions.ndim == 2 and positions.shape[1] == 3
        self.N = positions.shape[0]

        self.p = positions.astype(np.float64).copy()
        self.v = (velocities.astype(np.float64).copy()
                  if velocities is not None else np.zeros_like(self.p))
        self.rho = np.full(self.N, params.rest_density, dtype=np.float64)
        self.P = np.zeros(self.N, dtype=np.float64)
        self.T = (temperature.astype(np.float64).copy()
                  if temperature is not None else
                  np.full(self.N, params.T0, dtype=np.float64))
        self.S = (salinity.astype(np.float64).copy()
                  if salinity is not None else
                  np.zeros(self.N, dtype=np.float64))

        # Mass and solute tracking
        self.m = np.full(self.N, params.particle_mass, dtype=np.float64)
        self.solute_mass = self.m * self.S
        self.m_target = self.m.copy()
        self.solute_mass_target = self.solute_mass.copy()

        # Surface tension per particle and droplet tagging
        self.sigma = np.full(self.N, params.surface_tension, dtype=np.float64)
        self.is_droplet = np.zeros(self.N, dtype=bool)
        if droplet_particle is None:
            droplet_particle = DropletParticle()
        if droplet_indices is not None:
            droplet_indices = np.asarray(list(droplet_indices), dtype=int)
            self.is_droplet[droplet_indices] = True
            self.m[droplet_indices] = droplet_particle.mass
            self.solute_mass[droplet_indices] = self.m[droplet_indices] * self.S[droplet_indices]
            self.m_target[droplet_indices] = self.m[droplet_indices]
            self.solute_mass_target[droplet_indices] = self.solute_mass[droplet_indices]
            self.sigma[droplet_indices] = droplet_particle.surface_tension

        self.params = params
        self.kernel = SPHKernel(params.smoothing_length)

        self.bounds_min = np.array(bounds_min, dtype=np.float64)
        self.bounds_max = np.array(bounds_max, dtype=np.float64)

        # Grid for neighbor search (hash -> particle indices)
        self._grid_cell = None          # (N,3) int64 indices
        self._grid_keys = None          # (N,) int64 hash
        self._cell_index = None         # sorted indices into particles
        self._cell_starts = None        # start offsets per unique key
        self._cell_keys = None          # unique keys
        self._cell_span_map = None      # dict: key -> (start,end)
        self._neighbor_offsets = self._build_neighbor_offsets()

        # Cached constants
        self._g = np.array(params.gravity, dtype=np.float64)

        # Detached droplet particles
        self.droplet_p = np.zeros((0, 3), dtype=np.float64)
        self.droplet_v = np.zeros((0, 3), dtype=np.float64)
        # Each ballistic droplet uses a fixed mass taken from ``droplet_particle``
        # for momentum conservation when merging back into the fluid.
        self.droplet_mass = float(droplet_particle.mass)
        # Threshold speed for automatic emission; np.inf disables auto mode
        self.droplet_threshold = np.inf
        # Linear drag coefficient for ballistic droplets
        self.droplet_drag = 0.0

    # ------------------------- Public API ------------------------------------

    def step(self, dt: float, substeps: int = 1, *, hooks=None) -> None:
        """Advance the fluid by exactly ``dt`` seconds without self-capping.

        Internal stability is now governed by the dt controller via metrics,
        so this method performs a single substep of size ``dt``. Any desired
        pre/post hooks are still honored around the substep.
        """
        from src.common.sim_hooks import SimHooks

        hooks = hooks or SimHooks()
        if is_enabled():
            try:
                vmax0 = float(np.max(np.linalg.norm(self.v, axis=1))) if self.N > 0 else 0.0
            except Exception:
                vmax0 = 0.0
            dbg("eng.discrete").debug(
                f"step: dt={float(dt):.6g} N={self.N} h={self.kernel.h:.3g} vmax0={vmax0:.3e}"
            )
        hooks.run_pre(self, float(dt))
        self._substep(float(dt))
        hooks.run_post(self, float(dt))
        if is_enabled():
            try:
                vmax1 = float(np.max(np.linalg.norm(self.v, axis=1))) if self.N > 0 else 0.0
                rho_min = float(np.min(self.rho)) if self.N > 0 else 0.0
                rho_max = float(np.max(self.rho)) if self.N > 0 else 0.0
                p_min = float(np.min(self.P)) if self.N > 0 else 0.0
                p_max = float(np.max(self.P)) if self.N > 0 else 0.0
                n_drop = int(self.droplet_p.shape[0])
            except Exception:
                vmax1 = rho_min = rho_max = p_min = p_max = 0.0
                n_drop = 0
            dbg("eng.discrete").debug(
                f"done: vmax={vmax1:.3e} rho=[{rho_min:.3e},{rho_max:.3e}] P=[{p_min:.3e},{p_max:.3e}] droplets={n_drop}"
            )

    def copy_shallow(self):
        """Return a shallow copy for rollback in adaptive stepping."""
        return copy.deepcopy(self)

    def restore(self, saved) -> None:
        """Restore state from :func:`copy_shallow`."""
        self.__dict__.update(copy.deepcopy(saved.__dict__))

    def step_with_controller(
        self,
        dt: float,
        ctrl: STController,
        targets: Targets,
    ) -> tuple[Metrics, float]:
        """Advance with :class:`STController` adaptive timestep."""

        dx = self.kernel.h

        def advance(state: "DiscreteFluid", dt_step: float):
            prev_mass = float(np.sum(state.m))
            state._substep(dt_step)
            vmax = float(np.max(np.linalg.norm(state.v, axis=1))) if state.N > 0 else 0.0
            mass_now = float(np.sum(state.m))
            mass_err = abs(mass_now - prev_mass) / max(prev_mass, 1e-12)
            metrics = Metrics(
                max_vel=vmax,
                max_flux=vmax,
                div_inf=0.0,
                mass_err=mass_err,
            )
            return True, metrics

        metrics, dt_next = step_with_dt_control(self, dt, dx, targets, ctrl, advance)
        return metrics, dt_next

    def sample_at(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        SPH interpolation of fields at arbitrary points.
        Returns dict with 'rho','P','v','T','S' arrays of suitable shapes.
        """
        assert points.ndim == 2 and points.shape[1] == 3
        self._build_grid()
        pairs_iter = self._pairs_points(points)
        h = self.kernel.h

        rho = np.zeros(points.shape[0]); P = np.zeros_like(rho)
        v = np.zeros_like(points); T = np.zeros(points.shape[0]); S = np.zeros_like(T)
        denom = np.zeros(points.shape[0])

        for (pi, pj, rvec, r, W) in pairs_iter:
            m_over_rho = (self.m[pj] / self.rho[pj])
            w = W * m_over_rho

            rho[pi]  += self.m[pj] * W
            denom[pi] += w
            P[pi]    += self.P[pj] * w
            T[pi]    += self.T[pj] * w
            S[pi]    += self.S[pj] * w
            v[pi]    += self.v[pj] * w[:, None]

        eps = 1e-12
        denom = np.maximum(denom, eps)
        P /= denom; T /= denom; S /= denom; v /= denom[:, None]
        return {"rho": rho, "P": P, "v": v, "T": T, "S": S}

    def export_vertices(self) -> np.ndarray:
        """Return a copy of particle positions for visualization."""
        return self.p.copy()

    def export_positions_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return copies of particle positions and velocity vectors."""
        return self.p.copy(), self.v.copy()

    def emit_droplets(
        self,
        threshold: Optional[float] = None,
        indices: Optional[Iterable[int]] = None,
    ) -> None:
        """Spawn ballistic droplets from fluid particles."""
        if indices is None:
            if threshold is None:
                threshold = self.droplet_threshold
            if not np.isfinite(threshold):
                return
            speeds = np.linalg.norm(self.v, axis=1)
            indices = np.nonzero(speeds > threshold)[0]
        else:
            indices = np.asarray(list(indices), dtype=int)

        if indices.size == 0:
            return

        self.droplet_p = np.vstack([self.droplet_p, self.p[indices]])
        self.droplet_v = np.vstack([self.droplet_v, self.v[indices]])

    def apply_sources(self, centers: np.ndarray, dM: np.ndarray, dS_mass: np.ndarray,
                      radius: float) -> Dict[str, np.ndarray]:
        """
        Distribute mass and solute sources around ``centers`` within ``radius`` using
        kernel weights.  Mass and solute are tracked separately via target particle
        masses and solute masses.  Realized amounts are returned per center after
        clamping to keep per-particle quantities non-negative.
        """
        assert centers.ndim == 2 and centers.shape[1] == 3
        assert dM.shape == (centers.shape[0],)
        assert dS_mass.shape == (centers.shape[0],)

        self._build_grid()
        out_M = np.zeros_like(dM)
        out_Sm = np.zeros_like(dS_mass)

        h_src = max(radius, self.kernel.h)
        k_src = SPHKernel(h_src)

        for ci in range(centers.shape[0]):
            c = centers[ci:ci+1, :]
            pi, pj, rvec, r, W = next(self._pairs_points(c, custom_kernel=k_src, yield_once=True))
            if pj.size == 0:
                continue
            w = W / (W.sum() + 1e-12)

            dM_j = dM[ci] * w
            dSm_j = dS_mass[ci] * w

            m_new = self.m_target[pj] + dM_j
            neg_mask = m_new < 1e-12
            if np.any(neg_mask):
                dM_j[neg_mask] = 1e-12 - self.m_target[pj][neg_mask]
                m_new = self.m_target[pj] + dM_j
            self.m_target[pj] = m_new

            sm_new = self.solute_mass_target[pj] + dSm_j
            neg_sm = sm_new < 0.0
            if np.any(neg_sm):
                dSm_j[neg_sm] = -self.solute_mass_target[pj][neg_sm]
                sm_new = self.solute_mass_target[pj] + dSm_j
            self.solute_mass_target[pj] = sm_new

            out_M[ci] = dM_j.sum()
            out_Sm[ci] = dSm_j.sum()

        return {"dM": out_M, "dS_mass": out_Sm}

    # --------------------------- Core substep ---------------------------------

    def _substep(self, dt: float) -> None:
        if is_enabled():
            dbg("eng.discrete").debug(f"    _substep dt={float(dt):.6g}")
        # Emit droplets based on velocity threshold if requested
        self.emit_droplets()

        # Relax any pending source targets before computing new densities
        self._relax_sources()

        # Neighbor grid
        self._build_grid()

        # Density & pressure (EOS)
        self._compute_density()
        self._compute_pressure()
        if is_enabled():
            try:
                rho_min = float(np.min(self.rho)) if self.N > 0 else 0.0
                rho_max = float(np.max(self.rho)) if self.N > 0 else 0.0
                p_min = float(np.min(self.P)) if self.N > 0 else 0.0
                p_max = float(np.max(self.P)) if self.N > 0 else 0.0
            except Exception:
                rho_min = rho_max = p_min = p_max = 0.0
            dbg("eng.discrete").debug(
                f"      density/pressure: rho=[{rho_min:.3e},{rho_max:.3e}] P=[{p_min:.3e},{p_max:.3e}]"
            )

        # Forces
        f = self._pressure_forces() + self._viscosity_forces() + self._body_forces()
        if np.any(self.sigma > 0.0):
            f += self._surface_tension_forces()

        # Integrate velocities and positions (semi-implicit / symplectic Euler)
        self.v += dt * (f / np.maximum(self.rho[:, None], 1e-12))

        # XSPH velocity blending (optional)
        if self.params.xsph_eps > 0.0:
            self.v = (1.0 - self.params.xsph_eps) * self.v + self.params.xsph_eps * self._xsph_velocity()

        # Position update
        self.p += dt * self.v

        # Boundaries (box)
        self._resolve_boundaries()

        # Diffusion steps (explicit; can be substepped if needed)
        if self.params.thermal_diffusivity > 0.0 or self.params.solute_diffusivity > 0.0:
            self._diffuse_scalars(dt)

        # Advance ballistic droplets
        if self.droplet_p.size:
            self._integrate_droplets(dt)
            # After moving droplets, check for re-entry into the main fluid
            self._merge_droplets()

        # Anomaly checks
        if is_enabled():
            try:
                bad = (not np.all(np.isfinite(self.p))) or (not np.all(np.isfinite(self.v)))
            except Exception:
                bad = False
            if bad:
                dbg("eng.discrete").debug("      anomaly: non-finite positions or velocities detected")

    def _relax_sources(self) -> None:
        """Relax particle mass and solute mass toward targets."""
        r = np.clip(self.params.source_relaxation, 0.0, 1.0)
        if r <= 0.0:
            return
        dm = self.m_target - self.m
        self.m += r * dm
        dsm = self.solute_mass_target - self.solute_mass
        self.solute_mass += r * dsm
        self.S = np.clip(self.solute_mass / np.maximum(self.m, 1e-12), 0.0, 1.0)

    def _integrate_droplets(self, dt: float) -> None:
        """Advance detached droplets under gravity and drag."""
        g = self._g[None, :]
        self.droplet_v += dt * (g - self.droplet_drag * self.droplet_v)
        self.droplet_p += dt * self.droplet_v

    def _merge_droplets(self) -> None:
        """Merge ballistic droplets back into nearby fluid particles."""
        self._build_grid()
        M = self.droplet_p.shape[0]
        if M == 0:
            return
        pi, pj, rvec, r, W = next(self._pairs_points(self.droplet_p, yield_once=True))
        if pj.size == 0:
            return
        merged = np.zeros(M, dtype=bool)
        for di in np.unique(pi):
            mask = (pi == di)
            nb = pj[mask]
            if nb.size == 0:
                continue
            weights = W[mask] * self.sigma[nb]
            wsum = weights.sum()
            if wsum <= 0.0:
                weights = W[mask]
                wsum = weights.sum()
            if wsum <= 0.0:
                continue
            weights /= wsum
            dm = self.droplet_mass * weights
            m_old = self.m[nb]
            self.v[nb] = (self.v[nb] * m_old[:, None] + dm[:, None] * self.droplet_v[di]) / (m_old[:, None] + dm[:, None])
            self.m[nb] = m_old + dm
            self.m_target[nb] += dm
            self.solute_mass_target[nb] += 0.0
            self.solute_mass[nb] += 0.0
            self.S[nb] = np.clip(self.solute_mass[nb] / np.maximum(self.m[nb], 1e-12), 0.0, 1.0)
            merged[di] = True
        keep = ~merged
        self.droplet_p = self.droplet_p[keep]
        self.droplet_v = self.droplet_v[keep]

    # --------------------------- Density & Pressure ---------------------------

    def _compute_density(self, include_self: bool = True) -> None:
        """ρ_i = m * ∑_j W_ij (with r²-culling and bincount scatter)."""
        N = self.N
        rho = np.zeros(N, dtype=np.float64)

        if include_self:
            rho += self.m * float(self.kernel.W(np.array([0.0]))[0])

        h2 = self.kernel.h2
        for (i, j, rvec, r2) in self._pairs_particles():
            mask = r2 < h2
            if not np.any(mask):
                continue
            i = i[mask]; j = j[mask]; r2 = r2[mask]

            W = self.kernel.W_from_r2_masked(r2)
            # symmetric contributions via bincount
            rho += np.bincount(i, weights=self.m[j] * W, minlength=N)
            rho += np.bincount(j, weights=self.m[i] * W, minlength=N)

        self.rho = np.maximum(rho, 1e-6)

    def _compute_pressure(self) -> None:
        """Tait equation: P = K [ (ρ/ρ0)^γ - 1 ]."""
        p = self.params
        ratio = self.rho / p.rest_density
        self.P = p.bulk_modulus * (np.power(np.clip(ratio, 1e-6, None), p.gamma) - 1.0)

    # ------------------------------- Forces ----------------------------------

    @staticmethod
    def _scatter_add_vec_pm_(f: np.ndarray, i: np.ndarray, j: np.ndarray, vec_ij: np.ndarray) -> None:
        """In-place: add +vec_ij to rows i and -vec_ij to rows j using bincount per axis."""
        N = f.shape[0]
        for k in range(3):
            if vec_ij.shape[0] == 0:
                continue
            f[:, k] += np.bincount(i, weights=vec_ij[:, k], minlength=N)
            f[:, k] -= np.bincount(j, weights=vec_ij[:, k], minlength=N)

    def _pressure_forces(self, bulk_modulus: Optional[float] = None) -> np.ndarray:
        """Compute pairwise pressure forces with r²-culling and bincount scatter."""
        N = self.N
        f = np.zeros((N, 3), dtype=np.float64)
        p = self.params

        if bulk_modulus is None:
            P_over_rho2 = self.P / np.maximum(self.rho, 1e-12) ** 2
        else:
            ratio = self.rho / p.rest_density
            P_tmp = bulk_modulus * (np.power(np.clip(ratio, 1e-6, None), p.gamma) - 1.0)
            P_over_rho2 = P_tmp / np.maximum(self.rho, 1e-12) ** 2

        h2 = self.kernel.h2
        for (i, j, rvec, r2) in self._pairs_particles():
            mask = r2 < h2
            if not np.any(mask):
                continue
            i = i[mask]; j = j[mask]; rvec = rvec[mask]; r = np.sqrt(r2[mask])

            gW = self.kernel.gradW_masked(rvec, r)
            c = self.m[i] * self.m[j] * (P_over_rho2[i] + P_over_rho2[j])
            fij = c[:, None] * gW
            self._scatter_add_vec_pm_(f, i, j, fij)

        return f

    def _viscosity_forces(self) -> np.ndarray:
        """
        Viscosity (Morris formulation):
        f_i += μ m^2 (v_j - v_i)/ (ρ_i ρ_j) ∇^2 W_ij
        Here μ is dynamic viscosity = ρ0 * ν (use rest density).
        """
        N = self.N
        nu = self.params.viscosity_nu
        mu = self.params.rest_density * nu
        f = np.zeros((N, 3), dtype=np.float64)

        inv_rho = 1.0 / np.maximum(self.rho, 1e-12)
        h2 = self.kernel.h2

        for (i, j, rvec, r2) in self._pairs_particles():
            mask = r2 < h2
            if not np.any(mask):
                continue
            i = i[mask]; j = j[mask]; r = np.sqrt(r2[mask])

            lapW = self.kernel.laplaceW_visc_masked(r)
            coeff = mu * self.m[i] * self.m[j] * (lapW * (inv_rho[i] * inv_rho[j]))
            dv = (self.v[j] - self.v[i])
            fij = coeff[:, None] * dv
            self._scatter_add_vec_pm_(f, i, j, fij)

        return f

    def _body_forces(self) -> np.ndarray:
        """
        Gravity + buoyancy (Boussinesq):
        ρ = ρ0 (1 - β_T (T-T0) - β_S (S-S0)) -> body force per particle:
        f = ρ * g ≈ ρ0 (1 - β_T ΔT - β_S ΔS) g
        """
        p = self.params
        deltaT = (self.T - p.T0)
        deltaS = (self.S - p.S0)
        rho_eff = p.rest_density * (1.0 - p.beta_T * deltaT - p.beta_S * deltaS)
        return rho_eff[:, None] * self._g[None, :]

    def _surface_tension_forces(self) -> np.ndarray:
        """
        Continuum surface force using color field gradient and curvature.
        """
        sigma = self.sigma
        if not np.any(sigma > 0.0):
            return np.zeros_like(self.p)

        n_vec = np.zeros_like(self.p)
        h2 = self.kernel.h2

        # color gradient
        for (i, j, rvec, r2) in self._pairs_particles():
            mask = r2 < h2
            if not np.any(mask): continue
            i = i[mask]; j = j[mask]; rvec = rvec[mask]; r = np.sqrt(r2[mask])

            gW = self.kernel.gradW_masked(rvec, r)
            m_over_rho_j = self.m[j] / np.maximum(self.rho[j], 1e-12)
            m_over_rho_i = self.m[i] / np.maximum(self.rho[i], 1e-12)

            # n_i += sum_j m_j/ρ_j ∇_i W_ij ; n_j -= m_i/ρ_i ∇_j W_ji = -(...)∇_i W_ij
            self._scatter_add_vec_pm_(n_vec, i, j, m_over_rho_j[:, None] * gW)
            self._scatter_add_vec_pm_(n_vec, j, i, m_over_rho_i[:, None] * gW)

        n_norm = np.linalg.norm(n_vec, axis=1) + self.params.color_field_eps
        n_hat = n_vec / n_norm[:, None]

        # curvature κ = -∇·(n̂)
        kappa = np.zeros(self.N)
        for (i, j, rvec, r2) in self._pairs_particles():
            mask = r2 < h2
            if not np.any(mask): continue
            i = i[mask]; j = j[mask]; rvec = rvec[mask]; r = np.sqrt(r2[mask])
            gW = self.kernel.gradW_masked(rvec, r)
            m_over_rho_j = self.m[j] / np.maximum(self.rho[j], 1e-12)
            m_over_rho_i = self.m[i] / np.maximum(self.rho[i], 1e-12)

            # ∇·A_i ≈ sum_j m/ρ_j (A_j - A_i) · ∇W_ij
            kappa_i = m_over_rho_j * np.einsum('ij,ij->i', (n_hat[j] - n_hat[i]), gW)
            kappa_j = m_over_rho_i * np.einsum('ij,ij->i', (n_hat[i] - n_hat[j]), -gW)
            kappa += np.bincount(i, weights=kappa_i, minlength=self.N)
            kappa += np.bincount(j, weights=kappa_j, minlength=self.N)

        f_st = sigma[:, None] * (kappa[:, None] * n_hat)
        return f_st

    def _xsph_velocity(self) -> np.ndarray:
        """
        XSPH velocity smoothing:
        v_i' = v_i + eps * sum_j (m/ρ_avg) (v_j - v_i) W_ij  (here W factor omitted per your original;
        if you want kernel-weighted XSPH, reintroduce W_ij.)
        """
        eps = self.params.xsph_eps
        if eps <= 0.0:
            return self.v

        N = self.N
        dv_accum = np.zeros_like(self.v)
        rho = self.rho

        for (i, j, _rvec, _r2) in self._pairs_particles():
            inv_rho_avg = 2.0 / np.maximum(rho[i] + rho[j], 1e-12)
            w_ij = self.m[j] * inv_rho_avg
            w_ji = self.m[i] * inv_rho_avg

            dv = self.v[j] - self.v[i]
            self._scatter_add_vec_pm_(dv_accum, i, j, dv * w_ij[:, None])
            self._scatter_add_vec_pm_(dv_accum, j, i, dv * w_ji[:, None])

        return self.v + eps * dv_accum

    # ------------------------------- Diffusion --------------------------------

    def _diffuse_scalars(self, dt: float) -> None:
        """
        Explicit diffusion of temperature T and salinity S via SPH Laplacian.
        Stable if dt <= h^2 / (2 * d).
        """
        kappa = self.params.thermal_diffusivity
        D_s = self.params.solute_diffusivity
        if kappa <= 0.0 and D_s <= 0.0:
            return

        N = self.N
        dT = np.zeros(N); dS = np.zeros(N)
        inv_rho = 1.0 / np.maximum(self.rho, 1e-12)
        h2 = self.kernel.h2

        for (i, j, rvec, r2) in self._pairs_particles():
            mask = r2 < h2
            if not np.any(mask): continue
            i = i[mask]; j = j[mask]; r = np.sqrt(r2[mask])

            lapW = self.kernel.laplaceW_visc_masked(r)
            m_over_rho_j = self.m[j] * inv_rho[j]
            m_over_rho_i = self.m[i] * inv_rho[i]

            if kappa > 0:
                inc_i = kappa * m_over_rho_j * (self.T[j] - self.T[i]) * lapW
                inc_j = kappa * m_over_rho_i * (self.T[i] - self.T[j]) * lapW
                dT += np.bincount(i, weights=inc_i, minlength=N)
                dT += np.bincount(j, weights=inc_j, minlength=N)

            if D_s > 0:
                inc_i = D_s * m_over_rho_j * (self.S[j] - self.S[i]) * lapW
                inc_j = D_s * m_over_rho_i * (self.S[i] - self.S[j]) * lapW
                dS += np.bincount(i, weights=inc_i, minlength=N)
                dS += np.bincount(j, weights=inc_j, minlength=N)

        self.T += dt * dT
        self.S = np.clip(self.S + dt * dS, 0.0, 1.0)

    # ----------------------------- Boundaries ---------------------------------

    def _resolve_boundaries(self) -> None:
        """Axis-aligned box with restitution-less bounce and optional damping."""
        damp = np.clip(self.params.bounce_damping, 0.0, 1.0)
        pmin, pmax = self.bounds_min, self.bounds_max

        for axis in range(3):
            # low side
            mask = self.p[:, axis] < pmin[axis]
            if np.any(mask):
                self.p[mask, axis] = pmin[axis]
                self.v[mask, axis] = - (1.0 - damp) * self.v[mask, axis]
            # high side
            mask = self.p[:, axis] > pmax[axis]
            if np.any(mask):
                self.p[mask, axis] = pmax[axis]
                self.v[mask, axis] = - (1.0 - damp) * self.v[mask, axis]

    # --------------------------- Neighbor machinery ---------------------------

    def _build_grid(self) -> None:
        """Hash grid for neighbor search (linked-cell)."""
        h = self.kernel.h
        inv_h = 1.0 / max(h, 1e-12)
        # integer cell coords
        cell = np.floor(self.p * inv_h).astype(np.int64)  # (N,3)
        key = self._hash3(cell[:, 0], cell[:, 1], cell[:, 2])

        order = np.argsort(key, kind='mergesort')
        key_sorted = key[order]
        cell_sorted = cell[order]

        uniq_keys, starts = np.unique(key_sorted, return_index=True)
        starts = np.concatenate([starts, np.array([len(order)], dtype=np.int64)])

        self._grid_cell = cell_sorted
        self._grid_keys = key_sorted
        self._cell_index = order
        self._cell_starts = starts
        self._cell_keys = uniq_keys
        # Build span map for fast lookups in _pairs_points
        self._cell_span_map = {int(k): (int(self._cell_starts[i]), int(self._cell_starts[i+1]))
                               for i, k in enumerate(self._cell_keys)}

    @staticmethod
    def _hash3(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
        """Stable 64-bit mix (works with negatives)."""
        ix = ix.astype(np.int64); iy = iy.astype(np.int64); iz = iz.astype(np.int64)
        return (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)

    @staticmethod
    def _build_neighbor_offsets() -> np.ndarray:
        """27 neighbor offsets in a 3x3x3 cube."""
        d = np.array(np.meshgrid([-1,0,1], [-1,0,1], [-1,0,1], indexing='ij')).reshape(3, -1).T
        return d.astype(np.int64)

    # -------------------------- Pair iterators --------------------------------
    # NOTE: hot-path now yields (i, j, rvec, r2) with r²-culling inside.
    # Callers decide whether to sqrt(r2) or compute kernels from r2.

    def _pairs_particles(self, max_pairs_chunk: int = 5_000_000
                         ) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Yield particle-particle neighbor interactions in chunks.
        Each yield returns arrays ``(i, j, rvec, r2)`` already culled by r² < h².

        rvec is defined as ``p_j - p_i`` (vector from particle i to j).
        We only generate each unordered pair once.
        """
        cell = self._grid_cell
        keys = self._grid_keys
        starts = self._cell_starts
        uniq_keys = self._cell_keys
        order = self._cell_index
        h2 = self.kernel.h2

        for ui, key in enumerate(uniq_keys):
            a0 = starts[ui]; a1 = starts[ui+1]
            idxA = order[a0:a1]
            cellA = cell[a0]

            for off in self._neighbor_offsets:
                nb_cell = cellA + off
                nb_key = self._hash3(nb_cell[0], nb_cell[1], nb_cell[2])

                rel = nb_key - key
                if rel < 0:
                    continue

                # binary search for neighbor key
                pos = np.searchsorted(uniq_keys, nb_key)
                if pos >= uniq_keys.size or uniq_keys[pos] != nb_key:
                    continue

                b0 = starts[pos]; b1 = starts[pos+1]
                idxB = order[b0:b1]

                if nb_key == key:
                    if idxA.size < 2:
                        continue
                    n = idxA.size
                    approx_pairs = n*(n-1)//2
                    if approx_pairs <= max_pairs_chunk:
                        I, J = np.triu_indices(n, k=1)
                        i = idxA[I]; j = idxA[J]
                        rvec = self.p[j] - self.p[i]
                        r2 = np.einsum('ij,ij->i', rvec, rvec)
                        mask = r2 < h2
                        if not np.any(mask):
                            continue
                        yield i[mask], j[mask], rvec[mask], r2[mask]
                    else:
                        row_start = 0
                        while row_start < n-1:
                            w = max(1, min(n - row_start - 1, max_pairs_chunk // max(1, n - row_start - 1)))
                            I = []; J = []
                            for rr in range(row_start, min(n-1, row_start + w)):
                                cnt = n - rr - 1
                                I.append(np.full(cnt, rr, dtype=np.int64))
                                J.append(np.arange(rr+1, n, dtype=np.int64))
                            I = np.concatenate(I); J = np.concatenate(J)
                            i = idxA[I]; j = idxA[J]
                            rvec = self.p[j] - self.p[i]
                            r2 = np.einsum('ij,ij->i', rvec, rvec)
                            mask = r2 < h2
                            if np.any(mask):
                                yield i[mask], j[mask], rvec[mask], r2[mask]
                            row_start += w
                else:
                    na = idxA.size; nb = idxB.size
                    if na == 0 or nb == 0:
                        continue
                    approx = na * nb
                    if approx <= max_pairs_chunk:
                        i = np.repeat(idxA, nb)
                        j = np.tile(idxB, na)
                        rvec = self.p[j] - self.p[i]
                        r2 = np.einsum('ij,ij->i', rvec, rvec)
                        mask = r2 < h2
                        if not np.any(mask):
                            continue
                        yield i[mask], j[mask], rvec[mask], r2[mask]
                    else:
                        if na >= nb:
                            stride = max(1, max_pairs_chunk // max(1, nb))
                            for a0b in range(0, na, stride):
                                aa = idxA[a0b:a0b+stride]
                                if aa.size == 0: continue
                                i = np.repeat(aa, nb)
                                j = np.tile(idxB, aa.size)
                                rvec = self.p[j] - self.p[i]
                                r2 = np.einsum('ij,ij->i', rvec, rvec)
                                mask = r2 < h2
                                if np.any(mask):
                                    yield i[mask], j[mask], rvec[mask], r2[mask]
                        else:
                            stride = max(1, max_pairs_chunk // max(1, na))
                            for b0b in range(0, nb, stride):
                                bb = idxB[b0b:b0b+stride]
                                if bb.size == 0: continue
                                i = np.repeat(idxA, bb.size)
                                j = np.tile(bb, idxA.size)
                                rvec = self.p[j] - self.p[i]
                                r2 = np.einsum('ij,ij->i', rvec, rvec)
                                mask = r2 < h2
                                if np.any(mask):
                                    yield i[mask], j[mask], rvec[mask], r2[mask]

    def _pairs_points(self, points: np.ndarray, custom_kernel: Optional[SPHKernel] = None,
                      yield_once: bool = False
                      ) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Yield interactions between arbitrary query points and particles.
        Returns tuples ``(pi, pj, rvec, r, W)`` analogous to the particle version.
        """
        kernel = custom_kernel if custom_kernel is not None else self.kernel
        h = kernel.h
        inv_h = 1.0 / max(h, 1e-12)

        # build grid indices for points
        cell_q = np.floor(points * inv_h).astype(np.int64)   # (M,3)
        key_q = self._hash3(cell_q[:,0], cell_q[:,1], cell_q[:,2])

        span_map = self._cell_span_map  # pre-built dict from _build_grid()

        M = points.shape[0]
        if yield_once:
            pi_list = []; pj_list = []; rvec_list = []; r_list = []; W_list = []
            for qi in range(M):
                for off in self._neighbor_offsets:
                    nb_cell = cell_q[qi] + off
                    nb_key = int(self._hash3(nb_cell[0], nb_cell[1], nb_cell[2]))
                    span = span_map.get(nb_key)
                    if span is None:
                        continue
                    b0, b1 = span
                    idxB = self._cell_index[b0:b1]
                    if idxB.size == 0: continue
                    pj = idxB
                    pi = np.full(pj.size, qi, dtype=np.int64)
                    rvec = self.p[pj] - points[pi]
                    r = np.linalg.norm(rvec, axis=1)
                    mask = r < h
                    if np.any(mask):
                        pj_list.append(pj[mask]); pi_list.append(pi[mask])
                        rv = rvec[mask]; rr = r[mask]; WW = kernel.W(rr)
                        rvec_list.append(rv); r_list.append(rr); W_list.append(WW)
            if len(pj_list) == 0:
                yield np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), \
                      np.empty((0,3)), np.empty(0), np.empty(0)
            else:
                pi = np.concatenate(pi_list); pj = np.concatenate(pj_list)
                rvec = np.concatenate(rvec_list); r = np.concatenate(r_list); W = np.concatenate(W_list)
                yield pi, pj, rvec, r, W
        else:
            for qi in range(M):
                pi_list = []; pj_list = []; rvec_list = []; r_list = []; W_list = []
                for off in self._neighbor_offsets:
                    nb_cell = cell_q[qi] + off
                    nb_key = int(self._hash3(nb_cell[0], nb_cell[1], nb_cell[2]))
                    span = span_map.get(nb_key)
                    if span is None:
                        continue
                    b0, b1 = span
                    idxB = self._cell_index[b0:b1]
                    if idxB.size == 0: continue
                    pj = idxB
                    pi = np.full(pj.size, qi, dtype=np.int64)
                    rvec = self.p[pj] - points[pi]
                    r = np.linalg.norm(rvec, axis=1)
                    mask = r < h
                    if np.any(mask):
                        pj_list.append(pj[mask]); pi_list.append(pi[mask])
                        rv = rvec[mask]; rr = r[mask]; WW = kernel.W(rr)
                        rvec_list.append(rv); r_list.append(rr); W_list.append(WW)
                if len(pj_list) == 0:
                    yield np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), \
                          np.empty((0,3)), np.empty(0), np.empty(0)
                else:
                    pi = np.concatenate(pi_list); pj = np.concatenate(pj_list)
                    rvec = np.concatenate(rvec_list); r = np.concatenate(r_list); W = np.concatenate(W_list)
                    yield pi, pj, rvec, r, W

    # ------------------------------- Utilities --------------------------------

    def _particle_volume_estimate(self, idx: np.ndarray) -> np.ndarray:
        """Estimate per-particle volume as m / ρ (vectorized)."""
        return self.m[idx] / np.maximum(self.rho[idx], 1e-12)

    def _stable_dt(self) -> float:
        """
        CFL-like timestep from:
          - speed of sound (from EOS): c = sqrt(dP/dρ) at ρ0 ≈ sqrt(γ K / ρ0)
          - max velocity magnitude
          - diffusion stability: dt <= h^2/(2 max(κ, D_s))
        """
        p = self.params
        c = np.sqrt(p.gamma * p.bulk_modulus / max(p.rest_density, 1e-12))
        vmax = float(np.max(np.linalg.norm(self.v, axis=1))) if self.N > 0 else 0.0
        adv = self.kernel.h / max(c + vmax, 1e-6)
        diff = np.inf
        dmax = max(p.thermal_diffusivity, p.solute_diffusivity)
        if dmax > 0:
            diff = (self.kernel.h ** 2) / (2.0 * dmax)
        dt = p.cfl_number * min(adv, diff)
        return dt

    # ------------------------------ Diagnostics -------------------------------

    def kinetic_energy(self) -> float:
        return 0.5 * float(np.sum(self.m * np.einsum('ij,ij->i', self.v, self.v)))

    def potential_energy(self) -> float:
        g = self._g
        if np.allclose(g, 0.0):
            return 0.0
        gnorm = np.linalg.norm(g)
        gh = (self.p @ (g / gnorm))
        return float(np.sum(self.m * gnorm * gh))

    # ------------------------------ Debug/demo --------------------------------

    @staticmethod
    def demo_dam_break(n_x=10, n_y=20, n_z=10, h=0.08) -> "DiscreteFluid":
        """
        Quick starter: rectangular block of water above ground (dam break).
        Returns a DiscreteFluid instance.
        """
        dx = h * 0.9
        xs = np.arange(n_x) * dx
        ys = np.arange(n_y) * dx
        zs = np.arange(n_z) * dx
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        pos[:, 1] += 0.2

        params = FluidParams(smoothing_length=h, particle_mass=0.02, bounce_damping=0.2)
        fluid = DiscreteFluid(pos, None, None, None, params,
                              bounds_min=(0.0, 0.0, 0.0), bounds_max=(2.0, 2.0, 2.0))
        return fluid


if __name__ == "__main__":
    # Minimal smoke test (no rendering):
    fluid = DiscreteFluid.demo_dam_break(n_x=8, n_y=12, n_z=8, h=0.05)
    t, dt = 0.0, 5e-4
    for step in range(50):
        fluid.step(dt)
        t += dt
    print("KE:", fluid.kinetic_energy(), "PE:", fluid.potential_energy(), "rho_mean:", float(fluid.rho.mean()))
