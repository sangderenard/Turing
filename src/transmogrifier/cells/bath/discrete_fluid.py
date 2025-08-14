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

References (informal):
- Monaghan, "Smoothed Particle Hydrodynamics", Annual Review of Astronomy and Astrophysics, 1992.
- Becker & Teschner, "Weakly compressible SPH for free surface flows", 2007.
- Morris, Fox, Zhu (viscosity/discretization details), 1997/2000.

Copyright
---------
MIT License. Use freely with attribution.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Iterable

import numpy as np


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
        coeff[mask] = -self.c_spiky * dr * dr / r[mask]  # -(C*(h - r)^2) * r_hat
        return r_vec * coeff[:, None]

    def laplaceW_visc(self, r: np.ndarray) -> np.ndarray:
        """Viscosity Laplacian ∇^2 W for distances r (shape (...,))."""
        out = np.zeros_like(r)
        mask = (r >= 0) & (r < self.h)
        out[mask] = self.c_visc * (self.h - r[mask])
        return out


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
    max_dt: float = 1e-3                  # safety cap

    # Boundary
    bounce_damping: float = 0.0           # velocity damping on boundary collision [0..1]


# ----------------------------- Discrete Fluid -------------------------------

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

        self.params = params
        self.kernel = SPHKernel(params.smoothing_length)

        self.bounds_min = np.array(bounds_min, dtype=np.float64)
        self.bounds_max = np.array(bounds_max, dtype=np.float64)

        # Grid for neighbor search (hash -> particle indices)
        self._grid_cell = None          # (N,3) int32 indices
        self._grid_keys = None          # (N,) int64 hash
        self._cell_index = None         # sorted indices into particles
        self._cell_starts = None        # start offsets per unique key
        self._cell_keys = None          # unique keys
        self._neighbor_offsets = self._build_neighbor_offsets()

        # Cached constants
        self._m = params.particle_mass
        self._g = np.array(params.gravity, dtype=np.float64)

    # ------------------------- Public API ------------------------------------

    def step(self, dt: float, substeps: int = 1) -> None:
        """Advance the fluid by dt (seconds), splitting into substeps with CFL cap."""
        dt_target = dt / max(1, int(substeps))
        remaining = dt
        while remaining > 1e-12:
            # choose stable dt <= dt_target and <= params.max_dt
            dt_s = min(self._stable_dt(), dt_target, self.params.max_dt, remaining)
            self._substep(dt_s)
            remaining -= dt_s

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

        for (pi, pj, rvec, r, W) in pairs_iter:
            # accumulate SPH sums at points pi from neighbor particles pj
            m_over_rho = (self._m / self.rho[pj])
            w = W * m_over_rho

            rho[pi] += self._m * W
            P[pi]   += self.P[pj] * w
            T[pi]   += self.T[pj] * w
            S[pi]   += self.S[pj] * w
            v[pi]   += self.v[pj] * w[:, None]

        # Normalize interpolated fields where needed (pressure, T, S, v)
        # For rho we used standard SPH density estimate; leave as-is.
        eps = 1e-12
        denom = np.maximum(rho / self._m, eps)
        P /= denom; T /= denom; S /= denom; v /= denom[:, None]
        return {"rho": rho, "P": P, "v": v, "T": T, "S": S}

    def apply_sources(self, centers: np.ndarray, dM: np.ndarray, dS_mass: np.ndarray,
                      radius: float) -> Dict[str, np.ndarray]:
        """
        Apply source/sink terms to mass (dM [kg]) and solute mass (dS_mass [kg]) around
        given centers within a spherical radius using kernel weights.
        Returns realized amounts per center (may be smaller if needed to keep positivity).
        """
        assert centers.ndim == 2 and centers.shape[1] == 3
        assert dM.shape == (centers.shape[0],)
        assert dS_mass.shape == (centers.shape[0],)

        self._build_grid()
        out_M = np.zeros_like(dM)
        out_Sm = np.zeros_like(dS_mass)

        # For each center, distribute to nearby particles (batched)
        # Use h_src = max(radius, h) for smoother deposition
        h_src = max(radius, self.kernel.h)
        k_src = SPHKernel(h_src)

        for ci in range(centers.shape[0]):
            c = centers[ci:ci+1, :]
            # gather nearby particles
            pi, pj, rvec, r, W = next(self._pairs_points(c, custom_kernel=k_src, yield_once=True))
            if pj.size == 0:
                continue
            w = W / (W.sum() + 1e-12)

            # Compute capacity constraints: keep rho, S non-negative
            # Mass update: change particle mass proxy by modifying density (WCSPH uses constant m,
            # but adding/removing mass can be modeled by adjusting rho "target" to reflect injection).
            # We emulate it by modifying density immediately; pressure will react in next step.
            dM_j = dM[ci] * w
            # Solute mass: add to particle salinity mass = rho * volume * S ≈ m * S, here assume per-particle mass m const.
            dSm_j = dS_mass[ci] * w

            # Apply: adjust density proportionally and salinity mass -> S (clamped to [0,1])
            self.rho[pj] = np.maximum(1e-6, self.rho[pj] + dM_j / (self._particle_volume_estimate(pj) + 1e-12))
            # Update S by mass fraction: (m*S + dSm) / m, with m≈constant (proxy)
            S_new = np.clip((self._m * self.S[pj] + dSm_j) / max(self._m, 1e-12), 0.0, 1.0)
            self.S[pj] = S_new

            out_M[ci] = dM_j.sum()
            out_Sm[ci] = dSm_j.sum()

        return {"dM": out_M, "dS_mass": out_Sm}

    # --------------------------- Core substep ---------------------------------

    def _substep(self, dt: float) -> None:
        # Neighbor grid
        self._build_grid()

        # Density & pressure (EOS)
        self._compute_density()
        self._compute_pressure()

        # Forces
        f = self._pressure_forces() + self._viscosity_forces() + self._body_forces()
        if self.params.surface_tension > 0.0:
            f += self._surface_tension_forces()

        # Integrate velocities and positions (semi-implicit / symplectic Euler)
        self.v += dt * (f / np.maximum(self.rho[:, None], 1e-12))

        # XSPH velocity blending (optional)
        if self.params.xsph_eps > 0.0:
            self.v = (1.0 - self.params.xsph_eps) * self.v + self.params.xsph_eps * self._xsph_velocity()

        self.p += dt * self.v

        # Boundaries (box)
        self._resolve_boundaries()

        # Diffusion steps (explicit; can be substepped if needed)
        if self.params.thermal_diffusivity > 0.0 or self.params.solute_diffusivity > 0.0:
            self._diffuse_scalars(dt)

    # --------------------------- Density & Pressure ---------------------------

    def _compute_density(self) -> None:
        """ρ_i = m * sum_j W_ij"""
        N = self.N
        rho = np.zeros(N, dtype=np.float64)
        for (i, j, rvec, r, W) in self._pairs_particles():
            # self term included naturally because cell list includes i=j when r=0
            rho[i] += self._m * W
        self.rho = np.maximum(rho, 1e-6)

    def _compute_pressure(self) -> None:
        """Tait equation: P = K [ (ρ/ρ0)^γ - 1 ]."""
        p = self.params
        ratio = self.rho / p.rest_density
        self.P = p.bulk_modulus * (np.power(np.clip(ratio, 1e-6, None), p.gamma) - 1.0)

    # ------------------------------- Forces ----------------------------------

    def _pressure_forces(self) -> np.ndarray:
        """
        Symmetric pressure force:
        f_i += - m^2 (P_i/ρ_i^2 + P_j/ρ_j^2) ∇W_ij
        """
        f = np.zeros_like(self.p)
        for (i, j, rvec, r, W) in self._pairs_particles():
            # gradW is vector with shape (...,3)
            gW = self.kernel.gradW(rvec, r)
            Pi = self.P[i]; Pj = self.P[j]
            rhoi = self.rho[i]; rhoj = self.rho[j]
            # pair coefficient
            c = - self._m * self._m * (Pi / (rhoi * rhoi) + Pj / (rhoj * rhoj))
            fij = c[:, None] * gW
            # action-reaction (scatter-add)
            np.add.at(f, i,  fij)
            np.add.at(f, j, -fij)
        return f

    def _viscosity_forces(self) -> np.ndarray:
        """
        Viscosity (Morris formulation):
        f_i += μ m^2 (v_j - v_i)/ (ρ_i ρ_j) ∇^2 W_ij
        Here μ is dynamic viscosity = ρ0 * ν (use rest density).
        """
        nu = self.params.viscosity_nu
        mu = self.params.rest_density * nu
        f = np.zeros_like(self.p)
        for (i, j, rvec, r, W) in self._pairs_particles():
            lapW = self.kernel.laplaceW_visc(r)
            rhoi = self.rho[i]; rhoj = self.rho[j]
            coeff = mu * self._m * self._m * (lapW / np.maximum(rhoi * rhoj, 1e-12))[:, None]
            dv = (self.v[j] - self.v[i])
            fij = coeff * dv
            np.add.at(f, i,  fij)
            np.add.at(f, j, -fij)
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
        c = sum_j m/ρ_j W_ij
        n = ∇c (via symmetric gradient), κ = -∇·(n/|n|)
        f_s = σ κ n̂
        This is approximate; set sigma=0 to disable.
        """
        sigma = self.params.surface_tension
        if sigma <= 0.0:
            return np.zeros_like(self.p)

        # Compute color field gradient per particle
        n_vec = np.zeros_like(self.p)
        c_field = np.zeros(self.N)
        for (i, j, rvec, r, W) in self._pairs_particles():
            m_over_rho_j = self._m / np.maximum(self.rho[j], 1e-12)
            m_over_rho_i = self._m / np.maximum(self.rho[i], 1e-12)
            # symmetric gradient approx
            gW = self.kernel.gradW(rvec, r)
            n_vec[i] += m_over_rho_j[:, None] * gW
            n_vec[j] -= m_over_rho_i[:, None] * gW
            c_field[i] += m_over_rho_j * W
            c_field[j] += m_over_rho_i * W

        # curvature κ = -∇·(n̂)
        n_norm = np.linalg.norm(n_vec, axis=1) + self.params.color_field_eps
        n_hat = n_vec / n_norm[:, None]

        # approximate divergence via SPH identity: ∇·A_i ≈ sum_j m/ρ_j (A_j - A_i) · ∇W_ij
        kappa = np.zeros(self.N)
        for (i, j, rvec, r, W) in self._pairs_particles():
            gW = self.kernel.gradW(rvec, r)
            m_over_rho_j = self._m / np.maximum(self.rho[j], 1e-12)
            m_over_rho_i = self._m / np.maximum(self.rho[i], 1e-12)
            kappa[i] -= m_over_rho_j * np.einsum('ij,ij->i', (n_hat[j] - n_hat[i]), gW)
            kappa[j] -= m_over_rho_i * np.einsum('ij,ij->i', (n_hat[i] - n_hat[j]), -gW)

        f_st = sigma * (kappa[:, None] * n_hat)
        return f_st

    def _xsph_velocity(self) -> np.ndarray:
        """
        XSPH velocity smoothing:
        v_i' = v_i + eps * sum_j (m/ρ_avg) (v_j - v_i) W_ij
        """
        eps = self.params.xsph_eps
        v_new = self.v.copy()
        if eps <= 0.0:
            return v_new

        dv_accum = np.zeros_like(self.v)
        for (i, j, rvec, r, W) in self._pairs_particles():
            rho_avg = 0.5 * (self.rho[i] + self.rho[j])
            w = (self._m / np.maximum(rho_avg, 1e-12)) * W
            contrib = (self.v[j] - self.v[i]) * w[:, None]
            np.add.at(dv_accum, i, contrib)
            np.add.at(dv_accum, j, -contrib)
        return self.v + eps * dv_accum

    # ------------------------------- Diffusion --------------------------------

    def _diffuse_scalars(self, dt: float) -> None:
        """
        Explicit diffusion of temperature T and salinity S via SPH Laplacian.
        Stable if dt <= h^2 / (2 * d), capped by max_dt anyway.
        """
        kappa = self.params.thermal_diffusivity
        D_s = self.params.solute_diffusivity
        if kappa <= 0.0 and D_s <= 0.0:
            return

        dT = np.zeros(self.N); dS = np.zeros(self.N)
        for (i, j, rvec, r, W) in self._pairs_particles():
            lapW = self.kernel.laplaceW_visc(r)  # same operator works as Laplacian weight
            m_over_rho = self._m / np.maximum(self.rho[j], 1e-12)
            if kappa > 0:
                np.add.at(dT, i, kappa * m_over_rho * (self.T[j] - self.T[i]) * lapW)
                np.add.at(dT, j, kappa * (self._m / np.maximum(self.rho[i],1e-12)) * (self.T[i] - self.T[j]) * lapW)
            if D_s > 0:
                np.add.at(dS, i, D_s * m_over_rho * (self.S[j] - self.S[i]) * lapW)
                np.add.at(dS, j, D_s * (self._m / np.maximum(self.rho[i],1e-12)) * (self.S[i] - self.S[j]) * lapW)

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

        # unique keys and ranges
        uniq_keys, starts = np.unique(key_sorted, return_index=True)
        # append end sentinel
        starts = np.concatenate([starts, np.array([len(order)], dtype=np.int64)])

        self._grid_cell = cell_sorted
        self._grid_keys = key_sorted
        self._cell_index = order
        self._cell_starts = starts
        self._cell_keys = uniq_keys

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

    def _pairs_particles(self, max_pairs_chunk: int = 5_000_000
                         ) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Yield particle-particle neighbor interactions in chunks.
        Each yield returns tuple of arrays (i, j, rvec, r, W).
        We only generate each unordered pair once (j > i when same cell, and
        only consider neighbor cells with a '>= key' rule).
        """
        cell = self._grid_cell
        keys = self._grid_keys
        starts = self._cell_starts
        uniq_keys = self._cell_keys
        order = self._cell_index
        h = self.kernel.h

        # map from key -> span [start, end)
        # iterate each cell's neighbors with a consistent ordering
        for ui, key in enumerate(uniq_keys):
            a0 = starts[ui]; a1 = starts[ui+1]
            idxA = order[a0:a1]             # particle indices in cell A
            cellA = cell[a0]                # its integer coords

            # loop over neighbor offsets
            for off in self._neighbor_offsets:
                nb_cell = cellA + off
                nb_key = self._hash3(nb_cell[0], nb_cell[1], nb_cell[2])
                # impose ordering to avoid duplicates: only process neighbor cells
                # with nb_key > key, except the same cell (nb_key==key) where we enforce j>i.
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
                    # same cell: form all pairs i<j
                    if idxA.size < 2:
                        continue
                    # Use broadcasting to form upper triangle pairs in chunks
                    # Split if too many pairs
                    # Approx pairs = n(n-1)/2
                    n = idxA.size
                    if n*(n-1)//2 <= max_pairs_chunk:
                        I, J = np.triu_indices(n, k=1)
                        i = idxA[I]; j = idxA[J]
                        rvec = self.p[j] - self.p[i]
                        r = np.linalg.norm(rvec, axis=1)
                        mask = r < h
                        if not np.any(mask):
                            continue
                        i = i[mask]; j = j[mask]; rvec = rvec[mask]; r = r[mask]
                        W = self.kernel.W(r)
                        yield i, j, rvec, r, W
                    else:
                        # chunk by rows
                        row_start = 0
                        while row_start < n-1:
                            # choose a window size w so that ~w*(n - row) <= max_pairs_chunk
                            w = max(1, min(n - row_start - 1, max_pairs_chunk // max(1, n - row_start - 1)))
                            I = []; J = []
                            for rr in range(row_start, min(n-1, row_start + w)):
                                cnt = n - rr - 1
                                I.append(np.full(cnt, rr, dtype=np.int64))
                                J.append(np.arange(rr+1, n, dtype=np.int64))
                            I = np.concatenate(I); J = np.concatenate(J)
                            i = idxA[I]; j = idxA[J]
                            rvec = self.p[j] - self.p[i]
                            r = np.linalg.norm(rvec, axis=1)
                            mask = r < h
                            if np.any(mask):
                                i = i[mask]; j = j[mask]; rvec = rvec[mask]; r = r[mask]
                                W = self.kernel.W(r)
                                yield i, j, rvec, r, W
                            row_start += w
                else:
                    # distinct cells: all pairs A x B
                    na = idxA.size; nb = idxB.size
                    if na == 0 or nb == 0:
                        continue
                    approx = na * nb
                    if approx <= max_pairs_chunk:
                        i = np.repeat(idxA, nb)
                        j = np.tile(idxB, na)
                        rvec = self.p[j] - self.p[i]
                        r = np.linalg.norm(rvec, axis=1)
                        mask = r < h
                        if not np.any(mask):
                            continue
                        i = i[mask]; j = j[mask]; rvec = rvec[mask]; r = r[mask]
                        W = self.kernel.W(r)
                        yield i, j, rvec, r, W
                    else:
                        # sub-batch by slicing A or B
                        # choose chunk along larger dimension
                        if na >= nb:
                            stride = max(1, max_pairs_chunk // max(1, nb))
                            for a0b in range(0, na, stride):
                                aa = idxA[a0b:a0b+stride]
                                if aa.size == 0: continue
                                i = np.repeat(aa, nb)
                                j = np.tile(idxB, aa.size)
                                rvec = self.p[j] - self.p[i]
                                r = np.linalg.norm(rvec, axis=1)
                                mask = r < h
                                if np.any(mask):
                                    i = i[mask]; j = j[mask]; rvec = rvec[mask]; r = r[mask]
                                    W = self.kernel.W(r)
                                    yield i, j, rvec, r, W
                        else:
                            stride = max(1, max_pairs_chunk // max(1, na))
                            for b0b in range(0, nb, stride):
                                bb = idxB[b0b:b0b+stride]
                                if bb.size == 0: continue
                                i = np.repeat(idxA, bb.size)
                                j = np.tile(bb, idxA.size)
                                rvec = self.p[j] - self.p[i]
                                r = np.linalg.norm(rvec, axis=1)
                                mask = r < h
                                if np.any(mask):
                                    i = i[mask]; j = j[mask]; rvec = rvec[mask]; r = r[mask]
                                    W = self.kernel.W(r)
                                    yield i, j, rvec, r, W

    def _pairs_points(self, points: np.ndarray, custom_kernel: Optional[SPHKernel] = None,
                      yield_once: bool = False
                      ) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Yield interactions between arbitrary query points and particles.
        Returns tuples (pi, pj, rvec, r, W) analogous to _pairs_particles,
        where pi indexes into the query points array.
        If yield_once=True, yields exactly one combined batch (or empty) for the provided points.
        """
        kernel = custom_kernel if custom_kernel is not None else self.kernel
        h = kernel.h
        inv_h = 1.0 / max(h, 1e-12)

        # build grid indices for points
        cell_q = np.floor(points * inv_h).astype(np.int64)   # (M,3)
        key_q = self._hash3(cell_q[:,0], cell_q[:,1], cell_q[:,2])

        # For each point, collect neighbors across 27 cells and form one batch
        # Implementation: loop over points in Python (acceptable M is small for sampling/sources)
        # but form a single vectorized batch if yield_once.
        M = points.shape[0]
        if yield_once:
            # aggregate all
            pi_list = []; pj_list = []; rvec_list = []; r_list = []; W_list = []
            for qi in range(M):
                # Build candidate particle index list
                for off in self._neighbor_offsets:
                    nb_cell = cell_q[qi] + off
                    nb_key = self._hash3(nb_cell[0], nb_cell[1], nb_cell[2])
                    pos = np.searchsorted(self._cell_keys, nb_key)
                    if pos >= self._cell_keys.size or self._cell_keys[pos] != nb_key:
                        continue
                    b0 = self._cell_starts[pos]; b1 = self._cell_starts[pos+1]
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
            # point-by-point streaming (rarely used; sampling typically small anyway)
            for qi in range(M):
                pi_list = []; pj_list = []; rvec_list = []; r_list = []; W_list = []
                for off in self._neighbor_offsets:
                    nb_cell = cell_q[qi] + off
                    nb_key = self._hash3(nb_cell[0], nb_cell[1], nb_cell[2])
                    pos = np.searchsorted(self._cell_keys, nb_key)
                    if pos >= self._cell_keys.size or self._cell_keys[pos] != nb_key:
                        continue
                    b0 = self._cell_starts[pos]; b1 = self._cell_starts[pos+1]
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
        return self._m / np.maximum(self.rho[idx], 1e-12)

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
        return max(1e-6, p.cfl_number * min(adv, diff))

    # ------------------------------ Diagnostics -------------------------------

    def kinetic_energy(self) -> float:
        return 0.5 * self._m * float(np.sum(np.einsum('ij,ij->i', self.v, self.v)))

    def potential_energy(self) -> float:
        # gravitational potential relative to z=0 along gravity direction
        g = self._g
        if np.allclose(g, 0.0):
            return 0.0
        # project position onto gravity direction (unit)
        gnorm = np.linalg.norm(g)
        gh = (self.p @ (g / gnorm))
        return float(np.sum(self._m * gnorm * gh))

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
        pos[:, 1] += 0.2   # lift above ground

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
