# -*- coding: utf-8 -*-
"""
Hybrid particle–grid fluid simulator (Terraria-style condense↔shatter)
=====================================================================

This module brings the hybrid fluid to feature parity with the discrete (SPH)
and voxel (MAC) solvers by combining a MAC grid (for fully occupied regions)
with particles (for partially filled / spray regions). It supports 1D, 2D, and
3D by interpreting "missing" dimensions as size-1 slabs.

Core ideas
----------
- Maintain a **MAC grid** (via VoxelMACFluid) and a **particle set**.
- **Deposit** particles to a temporary liquid fraction field ``phi_tilde`` and to
  provisional grid momentum.
- **Condense**: where ``phi_tilde >= phi_condense`` (with hysteresis), remove the
  contributing particles, mark the cell(s) as grid-owned fluid (``phi=1``), and
  keep momentum on the grid.
- **Shatter**: where ``phi <= phi_shatter`` and pressure is not high (``p <=
  p_shatter_max``), spawn droplets/particles from that cell with mass equal to
  the cell’s fluid content and reset the cell fluid fraction/momentum.
- Particle advection is **PIC/FLIP-blended**: ``v_p <- alpha*FLIP + (1-alpha)*PIC``
  where PIC samples grid velocity at particle positions, and FLIP adds the grid
  velocity change since the last step.

Public API (selected)
---------------------
- ``HybridFluid(shape, dx, n_particles, ...)``: create hybrid solver.
- ``step(dt, substeps=1)``: advance with per-substep stability capping.
- ``seed_block(...)``: helper to initialize a liquid block as particles and/or
  grid fluid.
- ``export_grid()`` / ``export_particles()`` / ``export_vector_field()`` for
  visualization plumbing consistent with sibling solvers.

Notes
-----
- Uses only NumPy and the sibling ``voxel_fluid`` module. Keeps the interface
  intentionally lean for coupling with higher-level demos/runners.
- Designed for game-robust behavior first; tuning thresholds and the PIC/FLIP
  blending stabilizes the look.

MIT License.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, List, Optional

import numpy as np

try:
    # Local sibling import; adjust path if packaging differs
    from .voxel_fluid import VoxelMACFluid, VoxelFluidParams
except Exception as e:  # pragma: no cover
    VoxelMACFluid = None  # type: ignore
    VoxelFluidParams = None  # type: ignore


# ---------------------------------------------------------------------------
# Kernel constants & helpers (dimension-aware)
# ---------------------------------------------------------------------------
# Normalization constants for standard SPH kernels grouped by dimension.
# The ``poly6`` kernel appears in density estimates while the ``spiky``
# gradient is used for pressure forces.  Storing them in a single dictionary
# keyed by ``dim`` keeps all dimension-specific factors together and makes it
# trivial to extend to other kernels later.
KERNEL_NORMALIZERS = {
    1: {"poly6_norm": 35.0 / 32.0, "spiky_norm": 15.0 / 16.0},
    2: {"poly6_norm": 4.0 / np.pi, "spiky_norm": 10.0 / np.pi},
    3: {"poly6_norm": 315.0 / (64.0 * np.pi), "spiky_norm": 15.0 / np.pi},
}


def poly6_W(r: np.ndarray, h: float, dim: int) -> np.ndarray:
    c = KERNEL_NORMALIZERS[dim]["poly6_norm"] / (h ** (dim + 2))
    q2 = np.clip(h * h - r * r, 0.0, None)
    return c * q2 * q2 * q2


def cone_kernel(r: np.ndarray, R: float) -> np.ndarray:
    """Simple compact linear kernel used for deposition (dimension-agnostic)."""
    if R <= 0:  # pragma: no cover
        return np.zeros_like(r)
    w = np.maximum(0.0, 1.0 - (r / R))
    return w


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class HybridParams:
    # Grid geometry/physics (subset forwarded to VoxelMACFluid)
    dx: float = 0.02
    rho0: float = 1000.0
    nu: float = 1.0e-6
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    cfl: float = 0.5
    max_dt: float = 1e-3

    # Particle side
    particle_mass: float = 0.02
    smoothing_length: float = 0.08  # influences deposition radius

    # Hybrid thresholds (with hysteresis)
    phi_condense: float = 0.85
    phi_shatter: float = 0.25
    p_shatter_max: float = 0.0

    # Flux emission gate
    phi_full: float = 0.95
    p_low: float = -50.0
    emit_fraction: float = 0.35
    max_particles_per_face_step: int = 8

    # Pressure/full pause (surface tension gate)
    p_high: float = 100.0
    sigma: float = 0.072
    k_tau: float = 0.02
    tau_min: float = 5e-3
    tau_max: float = 80e-3
    pause_damping: float = 0.85

    # Minimum-velocity condensation
    v_min: float = 0.05
    v_hyst: float = 0.015
    tau_slow: float = 0.0
    slow_damping: float = 0.9

    # PIC/FLIP blending
    flip_alpha: float = 0.95   # 1 = pure FLIP, 0 = pure PIC

    # Shatter/condense controls
    shatter_n_per_cell: int = 8
    max_cells_flip_per_step: Optional[int] = None  # limit transitions per step

    # Rendering/scalars
    scalar_names: Iterable[str] | None = None


# ---------------------------------------------------------------------------
# Hybrid solver
# ---------------------------------------------------------------------------
@dataclass
class HybridFluid:
    shape: Tuple[int, ...]
    n_particles: int
    params: HybridParams = field(default_factory=HybridParams)

    # --- internal state (filled in __post_init__) ---
    dim: int = field(init=False)
    grid: VoxelMACFluid | None = field(init=False)
    phi: np.ndarray = field(init=False)       # liquid fraction per cell [0,1]
    solid: np.ndarray = field(init=False)     # cell-centered solid mask

    # particles
    x: np.ndarray = field(init=False)
    v: np.ndarray = field(init=False)
    m: np.ndarray = field(init=False)
    rho: np.ndarray = field(init=False)
    phase: np.ndarray = field(init=False)
    pause_t: np.ndarray = field(init=False)

    # caches
    _last_grid_u: List[np.ndarray] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.dim = len(self.shape)

        # Build a 3D MAC grid regardless of dim by lifting to (nx,ny,nz)
        nx = self.shape[0]
        ny = self.shape[1] if self.dim >= 2 else 1
        nz = self.shape[2] if self.dim >= 3 else 1
        if VoxelMACFluid is None:
            raise ImportError("voxel_fluid.VoxelMACFluid not available; ensure sibling module is on PYTHONPATH")
        vp = VoxelFluidParams(nx=nx, ny=ny, nz=nz, dx=self.params.dx, rho0=self.params.rho0,
                              nu=self.params.nu, gravity=self.params.gravity,
                              cfl=self.params.cfl, max_dt=self.params.max_dt)
        self.grid = VoxelMACFluid(vp)
        self.phi = np.zeros((nx, ny, nz), dtype=np.float64)
        self.solid = np.zeros_like(self.phi, dtype=bool)

        # Particles (in lifted D, but we keep unused dims at 0)
        self.x = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        self.v = np.zeros_like(self.x)
        self.m = np.full(self.n_particles, self.params.particle_mass, dtype=np.float64)
        self.rho = np.full(self.n_particles, self.params.rho0, dtype=np.float64)
        self.phase = np.zeros(self.n_particles, dtype=np.int8)
        self.pause_t = np.zeros(self.n_particles, dtype=np.float64)

        self._last_grid_u = [comp.copy() for comp in (self.grid.u, self.grid.v, self.grid.w)]

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def seed_block(self, lo: Tuple[float, ...], hi: Tuple[float, ...], mode: str = "particles",
                   jitter: float = 0.3, volume_fill: float = 1.0) -> None:
        """Seed a rectangular block either with particles, grid phi, or both.

        Parameters
        ----------
        lo, hi: tuples in world units (meters) of size ``dim``
        mode: 'particles' | 'grid' | 'both'
        jitter: particle jitter inside each cell (0..1)
        volume_fill: if mode includes 'grid', set phi in covered cells to this
        """
        dim = self.dim; dx = self.params.dx
        lo = np.array(list(lo) + [0]*(3-dim), dtype=np.float64)
        hi = np.array(list(hi) + [0]*(3-dim), dtype=np.float64)

        # grid coverage
        i0 = np.maximum(np.floor(lo / dx - 0.5).astype(int), 0)
        i1 = np.minimum(np.ceil (hi / dx - 0.5).astype(int), np.array(self.grid.pr.shape)-1)
        Ii, Jj, Kk = np.meshgrid(np.arange(i0[0], i1[0]+1),
                                 np.arange(i0[1], i1[1]+1),
                                 np.arange(i0[2], i1[2]+1), indexing='ij')
        cells = np.stack([Ii.ravel(), Jj.ravel(), Kk.ravel()], axis=-1)

        if mode in ("grid", "both"):
            self.phi[Ii, Jj, Kk] = np.clip(volume_fill, 0.0, 1.0)

        if mode in ("particles", "both") and self.n_particles > 0:
            # place one particle per eligible cell by default (repeat if needed)
            needed = len(cells)
            count = min(self.n_particles, needed)
            idx = np.random.choice(needed, size=count, replace=needed < count)
            sel = cells[idx]
            centers = (sel + 0.5) * dx
            jitter_amt = (np.random.rand(count, 3) - 0.5) * jitter * dx
            X = centers + jitter_amt
            # project down to dim
            self.x[:count, :dim] = X[:, :dim]

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, dt: float, substeps: int = 1) -> None:
        dt_target = dt / max(1, int(substeps))
        remaining = dt
        while remaining > 1e-12:
            dt_s = min(self._stable_dt(), dt_target, self.params.max_dt, remaining)
            self._substep(dt_s)
            remaining -= dt_s

    # ------------------------------------------------------------------
    # Substep
    # ------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        assert self.grid is not None
        # 1) Deposit particles to provisional fields
        phi_tilde, mom_u, mom_v, mom_w = self._deposit_particles(radius=max(self.params.smoothing_length, self.params.dx*1.25))

        # 2) Blend provisional momentum into grid faces (weak coupling)
        self._apply_momentum_deposition(mom_u, mom_v, mom_w)

        # 3) Advance grid (gravity, viscosity, advection, projection, scalars)
        last_u = [self.grid.u.copy(), self.grid.v.copy(), self.grid.w.copy()]
        self.grid.step(dt)

        # 4) PIC/FLIP update for particles from grid
        self._advect_particles_from_grid(dt, last_u, (self.grid.u, self.grid.v, self.grid.w))

        # 5) Flux-triggered emission (voxel -> particles)
        self._flux_to_particles(dt)

        # 6) Update paused/min-velocity particles and condense if needed
        self._update_pausing_particles(dt)

        # 7) Condense and shatter based on liquid fraction
        self._condense(phi_tilde)
        self._shatter()

        # 8) Clamp particles to domain (lifted dims)
        self._constrain_particles_to_domain()

        # 9) Cache grid velocity for next FLIP
        self._last_grid_u = [self.grid.u.copy(), self.grid.v.copy(), self.grid.w.copy()]

    # ------------------------------------------------------------------
    # Particle ↔ Grid helpers
    # ------------------------------------------------------------------
    def _deposit_particles(self, radius: float):
        """Return (phi_tilde, mom_u, mom_v, mom_w) accumulated from particles.
        Uses a cone kernel for robustness.
        """
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        dx = self.params.dx
        phi_tilde = np.zeros((nx, ny, nz), dtype=np.float64)
        mom_u = np.zeros_like(self.grid.u)
        mom_v = np.zeros_like(self.grid.v)
        mom_w = np.zeros_like(self.grid.w)

        if self.x.size == 0:
            return phi_tilde, mom_u, mom_v, mom_w

        # Cell-centered deposition for phi (liquid fraction)
        # Interpret particle mass per cell as rho0 * phi * cell_volume
        # => phi += m / (rho0 * dx^3)
        cell_centers = (np.stack(np.meshgrid(np.arange(nx)+0.5,
                                             np.arange(ny)+0.5,
                                             np.arange(nz)+0.5, indexing='ij'), axis=-1) * dx).reshape(-1, 3)
        P = self._lift_positions_to3d(self.x)
        for c in cell_centers.reshape(-1, 3)[::max(1,int( (nx*ny*nz)/4096 ))]:
            # chunked distance eval to limit memory
            r = np.linalg.norm(P - c[None, :], axis=1)
            w = cone_kernel(r, radius)
            if w.sum() <= 0: continue
            contrib = (self.m / (self.params.rho0 * (dx**3))) * w
            # find the index of this center
            i = int(round((c[0]/dx) - 0.5)); j = int(round((c[1]/dx) - 0.5)); k = int(round((c[2]/dx) - 0.5))
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                phi_tilde[i, j, k] += float(contrib.sum())

        # Face-centered deposition for momentum (simple nearest-face sampling)
        for axis in range(3):
            if axis == 0:
                # u faces at (i, j+1/2, k+1/2)
                I, J, K = np.meshgrid(np.arange(nx+1), np.arange(ny), np.arange(nz), indexing='ij')
                centers = np.stack([I, J+0.5, K+0.5], axis=-1).reshape(-1, 3) * dx
                comp = 0
            elif axis == 1:
                I, J, K = np.meshgrid(np.arange(nx), np.arange(ny+1), np.arange(nz), indexing='ij')
                centers = np.stack([I+0.5, J, K+0.5], axis=-1).reshape(-1, 3) * dx
                comp = 1
            else:
                I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz+1), indexing='ij')
                centers = np.stack([I+0.5, J+0.5, K], axis=-1).reshape(-1, 3) * dx
                comp = 2

            mom = np.zeros(centers.shape[0], dtype=np.float64)
            r = np.linalg.norm(self._lift_positions_to3d(self.x) - centers[:, None, :], axis=2)
            w = cone_kernel(r, radius)
            wsum = w.sum(axis=1) + 1e-12
            # weighted average of particle momentum along component
            pv = self._lift_velocities_to3d(self.v)[:, comp]
            mom = (w @ (self.m * pv)) / wsum
            if axis == 0:
                mom_u[...] = mom.reshape(self.grid.u.shape)
            elif axis == 1:
                mom_v[...] = mom.reshape(self.grid.v.shape)
            else:
                mom_w[...] = mom.reshape(self.grid.w.shape)

        return phi_tilde, mom_u, mom_v, mom_w

    def _apply_momentum_deposition(self, mom_u, mom_v, mom_w):
        # weakly blend into the grid (acts like jets/forcing)
        blend = 0.1
        self.grid.u = (1.0 - blend) * self.grid.u + blend * mom_u
        self.grid.v = (1.0 - blend) * self.grid.v + blend * mom_v
        self.grid.w = (1.0 - blend) * self.grid.w + blend * mom_w

    def _advect_particles_from_grid(self, dt: float, last_u, new_u) -> None:
        # PIC velocity from new grid
        V_pic = self._sample_grid_velocity(self.x)
        # FLIP delta from grid change
        V_last = self._sample_grid_velocity(self.x, faces=last_u)
        V_new = self._sample_grid_velocity(self.x, faces=new_u)
        dV = V_new - V_last
        alpha = np.clip(self.params.flip_alpha, 0.0, 1.0)
        self.v = self.v + alpha * dV + (1.0 - alpha) * (V_pic - self.v)
        self.x = self.x + dt * self.v

    # ------------------------------------------------------------------
    # Emission / absorption helpers
    # ------------------------------------------------------------------
    def _condense_particles(self, mask: np.ndarray, cell_idx: np.ndarray) -> int:
        """Condense selected particles into grid cells.

        Parameters
        ----------
        mask: boolean array selecting particles to condense
        cell_idx: integer cell coordinates for all particles (shape (N,3))

        Returns
        -------
        int
            Number of particles condensed.
        """
        if not np.any(mask):
            return 0
        dx = self.params.dx
        rho0 = self.params.rho0
        cells = cell_idx[mask]
        masses = self.m[mask]
        for (ci, cj, ck), dm in zip(cells, masses):
            self.phi[ci, cj, ck] = min(1.0, self.phi[ci, cj, ck] + dm / (rho0 * (dx ** 3)))
        keep = ~mask
        self.x = self.x[keep]
        self.v = self.v[keep]
        self.m = self.m[keep]
        self.rho = self.rho[keep]
        self.phase = self.phase[keep]
        self.pause_t = self.pause_t[keep]
        return int(np.sum(mask))

    def _flux_to_particles(self, dt: float) -> None:
        """Emit particles from grid faces based on flux and pressure gates."""
        dx = self.params.dx
        rho0 = self.params.rho0
        alpha = self.params.emit_fraction
        m_p = self.params.particle_mass
        phi_full = self.params.phi_full
        p_low = self.params.p_low
        max_n = self.params.max_particles_per_face_step

        dim = self.dim
        area = dx ** max(1, dim - 1)

        if self.grid is None:
            return

        new_x = []
        new_v = []
        n_spawn = 0

        # iterate over axes
        for axis, faces in enumerate([self.grid.u, self.grid.v, self.grid.w]):
            if (axis == 1 and dim < 2) or (axis == 2 and dim < 3):
                continue
            shape = faces.shape
            # interior faces
            it = np.ndindex(shape)
            for idx in it:
                u_f = faces[idx]
                if u_f <= 0:
                    continue
                # determine source and target cell indices
                if axis == 0:
                    i, j, k = idx
                    if i >= self.grid.nx:
                        continue
                    src = (i - 1, j, k)
                    tgt = (i, j, k)
                elif axis == 1:
                    i, j, k = idx
                    if j >= self.grid.ny:
                        continue
                    src = (i, j - 1, k)
                    tgt = (i, j, k)
                else:
                    i, j, k = idx
                    if k >= self.grid.nz:
                        continue
                    src = (i, j, k - 1)
                    tgt = (i, j, k)

                if min(src) < 0:
                    continue

                if self.grid.pr[tgt] > p_low:
                    continue
                if self.phi[tgt] >= phi_full:
                    continue

                m_emit = alpha * rho0 * u_f * area * dt
                n = int(np.ceil(m_emit / m_p))
                if n <= 0:
                    continue
                n = min(n, max_n)
                m_emit = n * m_p

                # spawn particles at target cell center with small jitter
                center = np.array([(tgt[0] + 0.5), (tgt[1] + 0.5), (tgt[2] + 0.5)]) * dx
                jitter = (np.random.rand(n, 3) - 0.5) * 0.25 * dx
                X = center[None, :] + jitter
                V = np.zeros((n, 3))
                V[:, axis] = u_f
                new_x.append(X[:, :dim])
                new_v.append(V[:, :dim])
                n_spawn += n

                # subtract mass from source cell
                self.phi[src] = max(0.0, self.phi[src] - m_emit / (rho0 * (dx ** 3)))

        if n_spawn > 0:
            new_x_arr = np.vstack(new_x)
            new_v_arr = np.vstack(new_v)
            self.x = np.vstack([self.x, new_x_arr]) if self.x.size else new_x_arr
            self.v = np.vstack([self.v, new_v_arr]) if self.v.size else new_v_arr
            self.m = np.hstack([self.m, m_p * np.ones(n_spawn)]) if self.m.size else (m_p * np.ones(n_spawn))
            self.rho = np.hstack([self.rho, rho0 * np.ones(n_spawn)]) if self.rho.size else (rho0 * np.ones(n_spawn))
            self.phase = np.hstack([self.phase, np.zeros(n_spawn, dtype=np.int8)]) if self.phase.size else np.zeros(n_spawn, dtype=np.int8)
            self.pause_t = np.hstack([self.pause_t, np.zeros(n_spawn, dtype=np.float64)]) if self.pause_t.size else np.zeros(n_spawn, dtype=np.float64)

    def _update_pausing_particles(self, dt: float) -> None:
        """Apply pressure/full pauses and minimum-velocity condensation."""
        if self.x.size == 0:
            return

        dx = self.params.dx
        dim = self.dim
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        X3 = self._lift_positions_to3d(self.x)
        cell = np.floor(X3 / dx - 0.5).astype(int)
        cell[:, 0] = np.clip(cell[:, 0], 0, nx - 1)
        if dim >= 2:
            cell[:, 1] = np.clip(cell[:, 1], 0, ny - 1)
        else:
            cell[:, 1] = 0
        if dim >= 3:
            cell[:, 2] = np.clip(cell[:, 2], 0, nz - 1)
        else:
            cell[:, 2] = 0

        # Rule A: pressure/full dwell
        pr = self.grid.pr[tuple(cell.T)]
        phi = self.phi[tuple(cell.T)]
        enterA = ((pr >= self.params.p_high) | (phi >= self.params.phi_full)) & (self.phase == 0)
        if np.any(enterA):
            tau = self.params.k_tau * self.params.sigma / np.maximum(1e-12, pr[enterA] - 0.0)
            tau = np.clip(tau, self.params.tau_min, self.params.tau_max)
            self.pause_t[enterA] = tau
            self.phase[enterA] = 1

        # Rule B: min-velocity dwell
        speed = np.linalg.norm(self.v, axis=1)
        if self.params.tau_slow == 0.0:
            condense_now = speed < self.params.v_min
            self._condense_particles(condense_now, cell)
            # update local arrays after removal
            if self.x.size == 0:
                return
            X3 = self._lift_positions_to3d(self.x)
            cell = np.floor(X3 / dx - 0.5).astype(int)
            cell[:, 0] = np.clip(cell[:, 0], 0, nx - 1)
            if dim >= 2:
                cell[:, 1] = np.clip(cell[:, 1], 0, ny - 1)
            if dim >= 3:
                cell[:, 2] = np.clip(cell[:, 2], 0, nz - 1)
            speed = np.linalg.norm(self.v, axis=1)
        else:
            enterB = (speed < self.params.v_min) & (self.phase == 0)
            self.phase[enterB] = 2
            self.pause_t[enterB] = self.params.tau_slow
            slow = self.phase == 2
            if np.any(slow):
                self.v[slow] *= self.params.slow_damping
                self.pause_t[slow] -= dt
                unpause = slow & (np.linalg.norm(self.v[slow], axis=1) > self.params.v_hyst)
                self.phase[unpause] = 0
                self.pause_t[unpause] = 0.0
                cond = slow & (self.pause_t <= 0)
                self._condense_particles(cond, cell)
                if self.x.size == 0:
                    return
                X3 = self._lift_positions_to3d(self.x)
                cell = np.floor(X3 / dx - 0.5).astype(int)
                cell[:, 0] = np.clip(cell[:, 0], 0, nx - 1)
                if dim >= 2:
                    cell[:, 1] = np.clip(cell[:, 1], 0, ny - 1)
                if dim >= 3:
                    cell[:, 2] = np.clip(cell[:, 2], 0, nz - 1)

        # Paused by Rule A
        paused = self.phase == 1
        if np.any(paused):
            self.v[paused] *= self.params.pause_damping
            self.pause_t[paused] -= dt
            cond = paused & (self.pause_t <= 0)
            self._condense_particles(cond, cell)

    def _condense(self, phi_tilde: np.ndarray) -> None:
        # Hysteresis: condense where phi_tilde is high
        mask = phi_tilde >= self.params.phi_condense
        if not np.any(mask):
            return
        self.phi[mask] = 1.0
        # Remove particles strongly owned by condensed cells (nearest-cell heuristic)
        dx = self.params.dx
        cell = np.floor(self._lift_positions_to3d(self.x) / dx - 0.5).astype(int)
        i, j, k = cell[:,0], cell[:,1] if self.dim>=2 else np.zeros_like(cell[:,0]), cell[:,2] if self.dim>=3 else np.zeros_like(cell[:,0])
        i = np.clip(i, 0, self.grid.nx-1); j = np.clip(j, 0, self.grid.ny-1); k = np.clip(k, 0, self.grid.nz-1)
        owned = mask[i, j, k]
        keep = ~owned
        self.x = self.x[keep]
        self.v = self.v[keep]
        self.m = self.m[keep]
        self.rho = self.rho[keep]
        self.phase = self.phase[keep]
        self.pause_t = self.pause_t[keep]

    def _shatter(self) -> None:
        # Shatter select low-fraction cells into droplets (limit count per step)
        dx = self.params.dx
        low = (self.phi <= self.params.phi_shatter)
        if self.params.p_shatter_max < np.inf:
            # use pressure to gate shatter (<= threshold)
            pr = self.grid.pr
            low &= (pr <= self.params.p_shatter_max)
        cand = np.argwhere(low)
        if cand.size == 0:
            return
        if self.params.max_cells_flip_per_step:
            np.random.shuffle(cand)
            cand = cand[: self.params.max_cells_flip_per_step]
        # spawn droplets
        new_x = []
        new_v = []
        new_m = []
        for (i,j,k) in cand:
            M = float(self.params.rho0 * self.phi[i,j,k] * (dx**3))
            if M <= 0:  # nothing to do
                continue
            n = max(1, int(self.params.shatter_n_per_cell))
            # positions jittered inside the cell
            base = (np.array([i+0.5, j+0.5, k+0.5]) * dx)[:max(3,self.dim)]
            jitter = (np.random.rand(n, 3) - 0.5) * 0.5 * dx
            X = base[None, :3] + jitter
            Vc = self._sample_grid_velocity((X[:, :self.dim]).reshape(-1, self.dim))
            m = (M / n) * np.ones(n)
            new_x.append(X[:, :self.dim])
            new_v.append(Vc)
            new_m.append(m)
            # clear cell fraction (mass moved to particles)
            self.phi[i,j,k] = 0.0
        if new_x:
            total_new = sum(len(m) for m in new_m)
            self.x = np.vstack([self.x, np.vstack(new_x)])
            self.v = np.vstack([self.v, np.vstack(new_v)])
            self.m = np.hstack([self.m, np.hstack(new_m)])
            self.rho = np.hstack([self.rho, np.full(total_new, self.params.rho0)])
            self.phase = np.hstack([self.phase, np.zeros(n, dtype=np.int8)]) if self.phase.size else np.zeros(n, dtype=np.int8)
            self.pause_t = np.hstack([self.pause_t, np.zeros(n, dtype=np.float64)]) if self.pause_t.size else np.zeros(n, dtype=np.float64)

    # ------------------------------------------------------------------
    # Sampling & export
    # ------------------------------------------------------------------
    def sample_at(self, points_world: np.ndarray) -> Dict[str, np.ndarray]:
        """Sample hybrid velocity/pressure at given points.
        Currently uses the grid fields for (v, P) and ignores particle kernels
        for speed/stability; extend as needed.
        """
        v = self._sample_grid_velocity(points_world)
        P = self.grid._sample_scalar_cc(self.grid.pr, points_world)  # type: ignore[attr-defined]
        return {"v": v, "P": P}

    def export_grid(self) -> Dict[str, np.ndarray]:
        data = {
            "phi": self.phi,
            "solid": self.solid,
            "p": self.grid.pr,
            "u": self.grid.u,
            "v": self.grid.v,
            "w": self.grid.w,
        }
        return data

    def export_particles(self) -> Dict[str, np.ndarray]:
        return {"x": self.x.copy(), "v": self.v.copy(), "m": self.m.copy(), "rho": self.rho.copy()}

    def export_vector_field(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.grid.export_vector_field()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _sample_grid_velocity(self, X: np.ndarray, faces: Optional[List[np.ndarray]] = None) -> np.ndarray:
        # Lift to 3D and reuse VoxelMACFluid sampling path
        X3 = self._lift_positions_to3d(X)
        if faces is None:
            return self.grid._sample_velocity(X3)  # type: ignore[attr-defined]
        u, v, w = faces
        return self.grid._sample_velocity_from(u, v, w, X3)  # type: ignore[attr-defined]

    def _lift_positions_to3d(self, X: np.ndarray) -> np.ndarray:
        if self.dim == 3:
            return X
        out = np.zeros((X.shape[0], 3), dtype=np.float64)
        out[:, :self.dim] = X
        return out

    def _lift_velocities_to3d(self, V: np.ndarray) -> np.ndarray:
        if self.dim == 3:
            return V
        out = np.zeros((V.shape[0], 3), dtype=np.float64)
        out[:, :self.dim] = V
        return out

    def _constrain_particles_to_domain(self) -> None:
        dx = self.params.dx
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        bounds = np.array([nx*dx, ny*dx, nz*dx])
        X3 = self._lift_positions_to3d(self.x)
        X3 = np.clip(X3, 0.0, bounds)
        self.x[:, :self.dim] = X3[:, :self.dim]

    def _stable_dt(self) -> float:
        # min(grid CFL, particle CFL)
        dt_grid = self.grid._stable_dt()  # type: ignore[attr-defined]
        vmax = float(np.max(np.linalg.norm(self.v, axis=1))) if self.v.size else 0.0
        adv = np.inf if vmax == 0 else self.params.cfl * self.params.dx / vmax
        return max(1e-6, min(dt_grid, adv))

    def total_mass(self) -> float:
        """Total mass of fluid in grid plus particles."""
        cell_vol = self.params.dx ** 3
        return self.params.rho0 * cell_vol * float(self.phi.sum()) + float(self.m.sum())


# Convenience factory
def demo_hybrid_dambreak(nx=48, ny=32, nz=1, dx=0.03, n_particles=4000) -> HybridFluid:
    params = HybridParams(dx=dx, smoothing_length=dx*1.5, phi_condense=0.85, phi_shatter=0.25,
                          flip_alpha=0.95, shatter_n_per_cell=8)
    sim = HybridFluid(shape=(nx, ny, nz), n_particles=n_particles, params=params)
    # seed a block in the top-left
    sim.seed_block(lo=(0.2, 0.4)[:max(2,sim.dim)], hi=(0.6, 0.8)[:max(2,sim.dim)], mode="both", volume_fill=1.0)
    return sim
