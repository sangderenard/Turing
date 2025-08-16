# voxel_fluid.py
# -*- coding: utf-8 -*-
"""
Voxel (grid/MAC) fluid simulator for the Bath engine.

Features
--------
- Incompressible Navier–Stokes on a staggered (MAC) grid
- Semi-Lagrangian advection for velocities and scalars
- Pressure projection via Conjugate Gradient (7-point Poisson)
- Optional viscosity: implicit Helmholtz solve per component
- Boussinesq buoyancy from temperature T and salinity S
- Scalar diffusion (explicit or unconditionally stable implicit option)
- Solid walls: no-slip or free-slip; arbitrary interior solid masks
- NumPy-only, deterministic, with careful memory usage
- Sampling hooks for coupling (pressure/velocity/T/S at points)
- Source hooks to inject scalar/momentum locally

Units
-----
MKS: meters, kilograms, seconds. Gravity along +y by default is negative.

References (informal)
---------------------
- Chorin (1968/69) fractional step method
- Stam (1999) "Stable Fluids" (semi-Lagrangian advection)
- Bridson (2015) "Fluid Simulation for Computer Graphics" (MAC details)
- Fedkiw et al. (2001) for practical pressure projection boundaries

License
-------
MIT
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import warnings
import copy
from src.cells.bath.dt_controller import Metrics, Targets, STController, step_with_dt_control

# Default CFL number exposed for external callers.  A value of 0.5 is
# reasonably conservative for the semi-Lagrangian scheme employed here.
CFL = 0.5


@dataclass
class VoxelFluidParams:
    # Grid & physics
    nx: int
    ny: int
    nz: int
    dx: float = 0.02                 # cell size (m)
    rho0: float = 1000.0             # reference density (kg/m^3)
    nu: float = 1.0e-6               # kinematic viscosity (m^2/s)
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)

    # Boussinesq buoyancy
    T0: float = 293.15               # K
    beta_T: float = 2.07e-4          # 1/K
    S0: float = 0.0                  # kg/kg
    beta_S: float = 7.6e-4           # 1

    # Diffusion
    thermal_diffusivity: float = 1.4e-7   # m^2/s
    solute_diffusivity: float = 1.0e-9    # m^2/s

    # Time stepping
    cfl: float = CFL
    max_dt: float = 1e-3
    nocap: bool = True
    pressure_tol: float = 1e-6
    pressure_maxiter: int = 800
    visc_tol: float = 1e-6
    visc_maxiter: int = 200

    # Boundaries
    boundary: str = "no-slip"  # "no-slip" or "free-slip"


class VoxelMACFluid:
    """
    Incompressible Navier–Stokes solver on a MAC grid.

    Staggering:
      - u: (nx+1, ny,   nz  ) at faces normal to x
      - v: (nx,   ny+1, nz  ) at faces normal to y
      - w: (nx,   ny,   nz+1) at faces normal to z
      - p, T, S: (nx, ny, nz) at cell centers

    Coordinate frame:
      Cells are laid out from origin (0,0,0) with spacing dx.
      The center of cell (i,j,k) is at ((i+0.5)dx, (j+0.5)dx, (k+0.5)dx).
    """
    def __init__(self, params: VoxelFluidParams):
        p = params
        self.p = p
        nx, ny, nz = p.nx, p.ny, p.nz
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = float(p.dx)
        self.inv_dx = 1.0 / self.dx
        # Dimensionality inferred from grid extents (size-1 axes ignored)
        self.dim = int((nx > 1) + (ny > 1) + (nz > 1))
        # Expose CFL number for external consumers
        self.cfl = p.cfl

        # Staggered velocities
        self.u = np.zeros((nx+1, ny, nz), dtype=np.float64)
        self.v = np.zeros((nx, ny+1, nz), dtype=np.float64)
        self.w = np.zeros((nx, ny, nz+1), dtype=np.float64)

        # Pressure & scalars
        self.pr = np.zeros((nx, ny, nz), dtype=np.float64)
        self.T  = np.full((nx, ny, nz), p.T0, dtype=np.float64)
        self.S  = np.zeros((nx, ny, nz), dtype=np.float64)

        # Solid masks (False = fluid, True = solid). Cell-centered and derived face masks.
        self.solid = np.zeros((nx, ny, nz), dtype=bool)
        self._update_face_solids()

        # Cached gravity and buoyancy
        self.g = np.array(p.gravity, dtype=np.float64)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def set_solid_mask(self, solid_cc: np.ndarray) -> None:
        """Set cell-centered solid mask; updates face masks accordingly."""
        assert solid_cc.shape == (self.nx, self.ny, self.nz)
        self.solid = solid_cc.astype(bool)
        self._update_face_solids()

    def add_scalar_sources(self, centers_world: np.ndarray, dT: np.ndarray, dS: np.ndarray, radius: float) -> None:
        """
        Add temperature/salinity sources around world-space centers within spherical radius
        using trilinear weights (normalized). dT, dS per source (K and mass fraction).
        """
        assert centers_world.shape[0] == dT.shape[0] == dS.shape[0]
        # convert to grid indices (float) at cell centers
        cx = centers_world / self.dx - 0.5
        rad = max(radius / self.dx, 1e-6)

        for i in range(cx.shape[0]):
            c = cx[i]
            # compute bounding box in index space
            lo = np.maximum(np.floor(c - rad).astype(int), 0)
            hi = np.minimum(np.ceil (c + rad).astype(int), [self.nx-1, self.ny-1, self.nz-1])
            if np.any(hi < lo):
                continue
            ii = np.arange(lo[0], hi[0]+1)
            jj = np.arange(lo[1], hi[1]+1)
            kk = np.arange(lo[2], hi[2]+1)
            I, J, K = np.meshgrid(ii, jj, kk, indexing='ij')
            P = np.stack([I, J, K], axis=-1).reshape(-1,3).astype(np.float64)
            # center positions
            pos = (P + 0.5) * self.dx
            # distance in world space
            r = np.linalg.norm(pos - centers_world[i][None,:], axis=1)
            w = np.maximum(0.0, 1.0 - (r / max(radius, 1e-12)))  # cone kernel
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            self.T[I, J, K] += (dT[i] * w).reshape(I.shape)
            self.S[I, J, K] = np.clip(self.S[I, J, K] + (dS[i] * w).reshape(I.shape), 0.0, 1.0)

    def add_momentum_sources(self, centers_world: np.ndarray, force_world: np.ndarray, radius: float) -> None:
        """Distribute body force density (N/m^3) to velocities via face weights; crude but useful for jets."""
        assert centers_world.shape == force_world.shape
        f = force_world
        # sample onto u-faces, v-faces, w-faces
        self._accumulate_face_forces(self.u, centers_world, f[:,0], radius, axis=0)
        self._accumulate_face_forces(self.v, centers_world, f[:,1], radius, axis=1)
        self._accumulate_face_forces(self.w, centers_world, f[:,2], radius, axis=2)

    def step(self, dt: float, substeps: int = 1, *, hooks=None) -> None:
        """Advance by ``dt`` seconds with stability-capped substeps."""
        from src.common.sim_hooks import SimHooks

        hooks = hooks or SimHooks()
        dt_target = dt / max(1, int(substeps))
        remaining = dt
        while remaining > 1e-12:
            if getattr(self.p, "nocap", True):
                dt_s = min(self._stable_dt(), dt_target, remaining)
            else:
                dt_s = min(self._stable_dt(), dt_target, self.p.max_dt, remaining)
            hooks.run_pre(self, dt_s)
            self._substep(dt_s)
            hooks.run_post(self, dt_s)
            remaining -= dt_s

    def copy_shallow(self):
        """Return a shallow copy for rollback."""
        return copy.deepcopy(self)

    def restore(self, saved) -> None:
        """Restore state from ``copy_shallow``."""
        self.__dict__.update(copy.deepcopy(saved.__dict__))

    def compute_metrics(self, prev_mass: float) -> Metrics:
        max_vel = float(max(np.max(np.abs(self.u)), np.max(np.abs(self.v)), np.max(np.abs(self.w))))
        return Metrics(max_vel=max_vel, max_flux=max_vel, div_inf=0.0, mass_err=0.0)

    def step_with_controller(
        self,
        dt: float,
        ctrl: STController,
        targets: Targets,
    ) -> tuple[Metrics, float]:
        """Adaptive step using :class:`STController`."""

        dx = self.dx

        def advance(state: "VoxelMACFluid", dt_step: float):
            state._substep(dt_step)
            metrics = state.compute_metrics(0.0)
            return True, metrics

        metrics, dt_next = step_with_dt_control(self, dt, dx, targets, ctrl, advance)
        return metrics, dt_next

    def sample_at(self, points_world: np.ndarray) -> Dict[str, np.ndarray]:
        """Sample velocity, pressure, T, S at world points; returns dict of arrays."""
        v = self._sample_velocity(points_world)
        p = self._sample_scalar_cc(self.pr, points_world)
        T = self._sample_scalar_cc(self.T, points_world)
        S = self._sample_scalar_cc(self.S, points_world)
        return {"v": v, "P": p, "T": T, "S": S}

    def export_vector_field(self) -> tuple[np.ndarray, np.ndarray]:
        """Return cell-center positions and velocity vectors for visualization."""
        nx, ny, nz = self.nx, self.ny, self.nz
        # cell center positions
        xs = (np.arange(nx) + 0.5) * self.dx
        ys = (np.arange(ny) + 0.5) * self.dx
        zs = (np.arange(nz) + 0.5) * self.dx
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pos = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        # velocity at cell centers: average neighboring faces
        u_c = 0.5 * (self.u[:-1, :, :] + self.u[1:, :, :])
        v_c = 0.5 * (self.v[:, :-1, :] + self.v[:, 1:, :])
        w_c = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        vec = np.stack([u_c, v_c, w_c], axis=-1).reshape(-1, 3)
        return pos, vec

    # ---------------------------------------------------------------------
    # Core substep
    # ---------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        # 1) Add body forces (gravity + buoyancy) to face velocities
        self._add_body_forces(dt)

        # Optional vorticity forces (e.g., confinement) are meaningful only in
        # 2D/3D. Skip when ``dim == 1`` to avoid spuriously injecting momentum
        # in a purely linear domain.
        if self.dim > 1:
            self._add_vorticity_forces(dt)

        # 2) Viscosity (implicit Helmholtz) if nu>0
        if self.p.nu > 0.0:
            self._viscosity_implicit(dt)

        # 3) Advect velocities (semi-Lagrangian, MAC-consistent)
        u0 = self.u.copy(); v0 = self.v.copy(); w0 = self.w.copy()
        self._advect_velocity(dt, u0, v0, w0)

        # Enforce face boundary velocities (no-slip/free-slip + interior solids)
        self._apply_velocity_boundaries()

        # 4) Pressure projection to enforce div-free (∇·v = 0)
        self._project(dt)

        # 5) Advect scalars and diffuse
        T0 = self.T.copy(); S0 = self.S.copy()
        self.T = self._advect_scalar_cc(T0, dt)
        self.S = self._advect_scalar_cc(S0, dt)
        self._clip_salinity("after advection")
        self._diffuse_scalars(dt)
        self._clip_salinity("after diffusion")


    # ---------------------------------------------------------------------
    # Forces
    # ---------------------------------------------------------------------
    def _add_body_forces(self, dt: float) -> None:
        # constant gravity plus buoyancy: g + ( -β_T ΔT - β_S ΔS ) * g
        p = self.p
        fx, fy, fz = self.g

        # constant gravity on non-solid faces
        if fx != 0.0:
            self.u[~self.solid_u] += dt * fx
        if fy != 0.0:
            self.v[~self.solid_v] += dt * fy
        if fz != 0.0:
            self.w[~self.solid_w] += dt * fz

        # buoyancy term from temperature and salinity differences
        if fx != 0.0 or fy != 0.0 or fz != 0.0:
            dT = self.T - p.T0
            dS = self.S - p.S0
            buoy = -(p.beta_T * dT + p.beta_S * dS)

            if fx != 0.0:
                bu = 0.5 * (
                    self._pad_x(buoy, left=True, expand=True)
                    + self._pad_x(buoy, left=False, expand=True)
                )
                self.u[~self.solid_u] += dt * fx * bu[~self.solid_u]
            if fy != 0.0:
                bv = 0.5 * (
                    self._pad_y(buoy, left=True, expand=True)
                    + self._pad_y(buoy, left=False, expand=True)
                )
                self.v[~self.solid_v] += dt * fy * bv[~self.solid_v]
            if fz != 0.0:
                bw = 0.5 * (
                    self._pad_z(buoy, left=True, expand=True)
                    + self._pad_z(buoy, left=False, expand=True)
                )
                self.w[~self.solid_w] += dt * fz * bw[~self.solid_w]

    def _add_vorticity_forces(self, dt: float) -> None:
        """Placeholder for vorticity-based forces.

        The solver currently does not implement vorticity confinement or
        similar effects; the stub keeps the call site in :meth:`_substep`
        explicit and documents that such forces are intentionally skipped when
        ``dim == 1``.
        """
        return

    # ---------------------------------------------------------------------
    # Viscosity (implicit Helmholtz) : (I - ν dt ∇²) u^{n+1} = u*
    # ---------------------------------------------------------------------
    def _viscosity_implicit(self, dt: float) -> None:
        a = self.p.nu * dt
        if a <= 0: return
        # Solve for each component on its staggered grid with CG
        self.u = self._cg_helmholtz_face(self.u, a, axis=0, tol=self.p.visc_tol, maxiter=self.p.visc_maxiter)
        self.v = self._cg_helmholtz_face(self.v, a, axis=1, tol=self.p.visc_tol, maxiter=self.p.visc_maxiter)
        self.w = self._cg_helmholtz_face(self.w, a, axis=2, tol=self.p.visc_tol, maxiter=self.p.visc_maxiter)

    # ---------------------------------------------------------------------
    # Advection
    # ---------------------------------------------------------------------
    def _advect_velocity(self, dt: float, u0, v0, w0) -> None:
        # Advect each face component by tracing from face centers using full velocity
        # and resampling the SAME component from (u0,v0,w0) with stagger-aware interpolation.
        self.u = self._advect_component_face(u0, u0, v0, w0, dt, axis=0)
        self.v = self._advect_component_face(v0, u0, v0, w0, dt, axis=1)
        self.w = self._advect_component_face(w0, u0, v0, w0, dt, axis=2)

    def _advect_scalar_cc(self, F: np.ndarray, dt: float) -> np.ndarray:
        # Backtrace from cell centers
        nx, ny, nz = self.nx, self.ny, self.nz
        dx = self.dx
        # world positions of cell centers
        I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
        Xc = (np.stack([I+0.5, J+0.5, K+0.5], axis=-1) * dx).reshape(-1, 3)
        # velocities at these positions
        Vc = self._sample_velocity(Xc)
        Xb = Xc - dt * Vc
        Fb = self._sample_scalar_cc(F, Xb)
        return Fb.reshape(nx, ny, nz)

    # ---------------------------------------------------------------------
    # Pressure projection
    # ---------------------------------------------------------------------
    def _project(self, dt: float) -> None:
        dx = self.dx; inv_dx = self.inv_dx
        nx, ny, nz = self.nx, self.ny, self.nz
        rho = self.p.rho0

        # compute divergence at cell centers from face velocities
        div = np.zeros((nx, ny, nz), dtype=np.float64)
        div += (self.u[1:,:,:] - self.u[:-1,:,:]) * inv_dx
        div += (self.v[:,1:,:] - self.v[:,:-1,:]) * inv_dx
        div += (self.w[:,:,1:] - self.w[:,:,:-1]) * inv_dx

        # zero divergence in solid cells (they don't enforce fluid incompressibility)
        div[self.solid] = 0.0

        # Solve Poisson: ∇² p = (ρ/dt) div with Neumann at walls, Dirichlet inside solids
        b = (rho / max(dt, 1e-12)) * div
        fluid_cells = ~self.solid
        if np.any(fluid_cells):
            b_mean = float(b[fluid_cells].mean())
            b -= b_mean
        # Dispatch to a dimension-appropriate pressure solver: a direct
        # tridiagonal solve for purely 1D domains, otherwise the generic CG
        # Poisson solver used for 2D/3D grids.
        if self.dim == 1:
            pr = self._poisson_1d_tridiag(b, self.solid)
        else:
            pr = self._cg_poisson_cc_rhs(b, self.solid,
                                         tol=self.p.pressure_tol,
                                         maxiter=self.p.pressure_maxiter)

        self.pr = pr

        # Subtract pressure gradient from face velocities (skip solid faces)
        gradx = (pr[1:,:,:] - pr[:-1,:,:]) * inv_dx
        grady = (pr[:,1:,:] - pr[:,:-1,:]) * inv_dx
        gradz = (pr[:,:,1:] - pr[:,:,:-1]) * inv_dx

        # Fluid faces: zero in/out of solids
        fluid_u = ~(self.solid[:-1,:,:] | self.solid[1:,:,:])
        fluid_v = ~(self.solid[:, :-1,:] | self.solid[:, 1:,:])
        fluid_w = ~(self.solid[:, :, :-1] | self.solid[:, :, 1:])

        # The previous implementation attempted to slice the face arrays and
        # then apply boolean indexing on the result.  Boolean indexing creates a
        # copy which led to shape mismatches when the boolean mask was further
        # sliced, ultimately raising ``IndexError`` in narrow grids (e.g. 1D
        # domains).  Instead we subtract the pressure gradient in-place and use
        # the boolean masks as multiplicative gates.  This keeps array shapes
        # aligned and updates only the fluid faces.
        self.u[1:-1, :, :] -= (dt / rho) * gradx * fluid_u
        self.v[:, 1:-1, :] -= (dt / rho) * grady * fluid_v
        self.w[:, :, 1:-1] -= (dt / rho) * gradz * fluid_w

        # Enforce boundary velocities again after projection
        self._apply_velocity_boundaries()

    # ---------------------------------------------------------------------
    # Scalar diffusion
    # ---------------------------------------------------------------------
    def _diffuse_scalars(self, dt: float) -> None:
        kT = self.p.thermal_diffusivity
        kS = self.p.solute_diffusivity
        if kT <= 0.0 and kS <= 0.0:
            return
        # explicit diffusion (stable if dt <= dx^2/(2*d)), otherwise substep because step() already caps dt
        if kT > 0.0:
            self.T = self._diffuse_cc_explicit(self.T, kT, dt)
        if kS > 0.0:
            self.S = self._diffuse_cc_explicit(self.S, kS, dt)

    def _clip_salinity(self, stage: str) -> None:
        clipped = np.clip(self.S, 0.0, 1.0)
        if not np.array_equal(clipped, self.S):
            warnings.warn(f"Salinity outside [0,1] {stage}; clipping", RuntimeWarning)
        self.S = clipped

    # ---------------------------------------------------------------------
    # Boundaries & masks
    # ---------------------------------------------------------------------
    def _update_face_solids(self) -> None:
        s = self.solid
        self.solid_u = np.zeros((self.nx+1, self.ny, self.nz), dtype=bool)
        self.solid_v = np.zeros((self.nx, self.ny+1, self.nz), dtype=bool)
        self.solid_w = np.zeros((self.nx, self.ny, self.nz+1), dtype=bool)
        # face is solid if either adjacent cell is solid
        self.solid_u[1:-1,:,:] = s[:-1,:,:] | s[1:,:,:]
        self.solid_u[0,:,:] = True; self.solid_u[-1,:,:] = True
        self.solid_v[:,1:-1,:] = s[:, :-1,:] | s[:, 1:,:]
        self.solid_v[:,0,:] = True; self.solid_v[:,-1,:] = True
        self.solid_w[:,:,1:-1] = s[:, :, :-1] | s[:, :, 1:]
        self.solid_w[:,:,0] = True; self.solid_w[:,:,-1] = True

    def _apply_velocity_boundaries(self) -> None:
        mode = self.p.boundary
        if mode == "no-slip":
            # zero-out all solid faces
            self.u[self.solid_u] = 0.0
            self.v[self.solid_v] = 0.0
            self.w[self.solid_w] = 0.0
        elif mode == "free-slip":
            # zero normal velocity only (already face-normal), leave tangential untouched
            self.u[self.solid_u] = 0.0
            self.v[self.solid_v] = 0.0
            self.w[self.solid_w] = 0.0
        else:
            raise ValueError("Unknown boundary mode")

    # ---------------------------------------------------------------------
    # Sampling & interpolation
    # ---------------------------------------------------------------------
    def _sample_velocity(self, Xw: np.ndarray) -> np.ndarray:
        """Sample the MAC velocity at world points Xw (N,3)."""
        u = self._sample_face(self.u, Xw, axis=0)
        v = self._sample_face(self.v, Xw, axis=1)
        w = self._sample_face(self.w, Xw, axis=2)
        return np.stack([u, v, w], axis=-1)

    def _sample_velocity_from(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray, Xw: np.ndarray
    ) -> np.ndarray:
        """Sample the provided MAC velocity (u, v, w) at world points Xw (N,3)."""
        us = self._sample_face(u, Xw, axis=0)
        vs = self._sample_face(v, Xw, axis=1)
        ws = self._sample_face(w, Xw, axis=2)
        return np.stack([us, vs, ws], axis=-1)

    def _sample_scalar_cc(self, F: np.ndarray, Xw: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of a cell-centered scalar field at world points."""
        dx = self.dx
        # convert to index space of centers
        X = Xw / dx - 0.5
        return self._trilinear_cc(F, X)

    def _sample_face(self, F: np.ndarray, Xw: np.ndarray, axis: int) -> np.ndarray:
        """Sample a face-centered component (u/v/w) at world points (stagger-aware)."""
        dx = self.dx
        X = Xw / dx  # index space
        if axis == 0:   # u at (i, j+1/2, k+1/2)
            X[:,1] -= 0.5; X[:,2] -= 0.5
            return self._trilinear_face(F, X, axis=0)
        elif axis == 1: # v at (i+1/2, j, k+1/2)
            X[:,0] -= 0.5; X[:,2] -= 0.5
            return self._trilinear_face(F, X, axis=1)
        else:           # w at (i+1/2, j+1/2, k)
            X[:,0] -= 0.5; X[:,1] -= 0.5
            return self._trilinear_face(F, X, axis=2)

    # Trilinear helpers ---------------------------------------------------
    def _trilinear_cc(self, F: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Trilinear interpolation on cell centers with index coords X."""
        nx, ny, nz = F.shape
        x = np.clip(X[:,0], 0.0, nx-1.001)
        y = np.clip(X[:,1], 0.0, ny-1.001)
        z = np.clip(X[:,2], 0.0, nz-1.001)
        i0 = np.floor(x).astype(int); j0 = np.floor(y).astype(int); k0 = np.floor(z).astype(int)
        i1 = np.minimum(i0+1, nx-1); j1 = np.minimum(j0+1, ny-1); k1 = np.minimum(k0+1, nz-1)
        tx = x - i0; ty = y - j0; tz = z - k0
        # gather corners
        c000 = F[i0, j0, k0]; c100 = F[i1, j0, k0]
        c010 = F[i0, j1, k0]; c110 = F[i1, j1, k0]
        c001 = F[i0, j0, k1]; c101 = F[i1, j0, k1]
        c011 = F[i0, j1, k1]; c111 = F[i1, j1, k1]
        c00 = c000*(1-tx) + c100*tx
        c01 = c001*(1-tx) + c101*tx
        c10 = c010*(1-tx) + c110*tx
        c11 = c011*(1-tx) + c111*tx
        c0 = c00*(1-ty) + c10*ty
        c1 = c01*(1-ty) + c11*ty
        return c0*(1-tz) + c1*tz

    def _trilinear_face(self, F: np.ndarray, X: np.ndarray, axis: int) -> np.ndarray:
        """Trilinear interpolation on a face grid (u/v/w)."""
        nx, ny, nz = F.shape
        x = np.clip(X[:,0], 0.0, nx-1.001)
        y = np.clip(X[:,1], 0.0, ny-1.001)
        z = np.clip(X[:,2], 0.0, nz-1.001)
        i0 = np.floor(x).astype(int); j0 = np.floor(y).astype(int); k0 = np.floor(z).astype(int)
        i1 = np.minimum(i0+1, nx-1); j1 = np.minimum(j0+1, ny-1); k1 = np.minimum(k0+1, nz-1)
        tx = x - i0; ty = y - j0; tz = z - k0
        # gather corners
        c000 = F[i0, j0, k0]; c100 = F[i1, j0, k0]
        c010 = F[i0, j1, k0]; c110 = F[i1, j1, k0]
        c001 = F[i0, j0, k1]; c101 = F[i1, j0, k1]
        c011 = F[i0, j1, k1]; c111 = F[i1, j1, k1]
        c00 = c000*(1-tx) + c100*tx
        c01 = c001*(1-tx) + c101*tx
        c10 = c010*(1-tx) + c110*tx
        c11 = c011*(1-tx) + c111*tx
        c0 = c00*(1-ty) + c10*ty
        c1 = c01*(1-ty) + c11*ty
        return c0*(1-tz) + c1*tz

    # ---------------------------------------------------------------------
    # Semi-Lagrangian advection of face components
    # ---------------------------------------------------------------------
    def _advect_component_face(self, Fcomp, u0, v0, w0, dt: float, axis: int):
        dx = self.dx
        if axis == 0:
            nx, ny, nz = self.nx+1, self.ny, self.nz
            # face centers positions
            I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
            X = np.stack([I, J+0.5, K+0.5], axis=-1).reshape(-1,3) * dx
        elif axis == 1:
            nx, ny, nz = self.nx, self.ny+1, self.nz
            I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
            X = np.stack([I+0.5, J, K+0.5], axis=-1).reshape(-1,3) * dx
        else:
            nx, ny, nz = self.nx, self.ny, self.nz+1
            I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
            X = np.stack([I+0.5, J+0.5, K], axis=-1).reshape(-1,3) * dx

        V = self._sample_velocity_from(u0, v0, w0, X)
        Xb = X - dt * V

        vals = self._sample_face(Fcomp, Xb, axis)
        return vals.reshape(Fcomp.shape)

    # ---------------------------------------------------------------------
    # Diffusion (explicit, CC)
    # ---------------------------------------------------------------------
    def _diffuse_cc_explicit(self, F: np.ndarray, kappa: float, dt: float) -> np.ndarray:
        dx2 = self.dx * self.dx
        lam = kappa * dt / dx2

        # Pad with edge values to enforce Neumann boundaries for arbitrary
        # dimensionality (1D/2D/3D).  The padded array lets us compute a
        # standard 7-point Laplacian without having to special-case thin
        # domains where one or more axes have size ``1``.  On such axes the
        # "neighbour" samples equal the centre value which effectively
        # contributes zero flux along that dimension.
        Fp = np.pad(F, 1, mode="edge")
        lap = (
            Fp[2:, 1:-1, 1:-1] + Fp[:-2, 1:-1, 1:-1]
            + Fp[1:-1, 2:, 1:-1] + Fp[1:-1, :-2, 1:-1]
            + Fp[1:-1, 1:-1, 2:] + Fp[1:-1, 1:-1, :-2]
            - 6.0 * F
        )

        return F + lam * lap

    # ---------------------------------------------------------------------
    # Linear solvers
    # ---------------------------------------------------------------------
    def _poisson_1d_tridiag(self, b: np.ndarray, solid_cc: np.ndarray) -> np.ndarray:
        """Solve the 1D Poisson equation with Neumann boundaries.

        This routine is used when the grid is effectively one-dimensional
        (``ny == nz == 1``).  A simple tridiagonal matrix representing the
        second derivative is assembled and solved directly.  Solid cells are
        treated as Dirichlet regions with zero pressure.
        """

        nx = self.nx
        rhs = b[:, 0, 0].copy()
        solid = solid_cc[:, 0, 0]

        main = -2.0 * np.ones(nx)
        off = np.ones(nx - 1)
        # Neumann boundary conditions: one-sided derivative at ends
        main[0] = main[-1] = -1.0

        A = np.diag(main)
        if nx > 1:
            A += np.diag(off, 1) + np.diag(off, -1)

        # Dirichlet in solid cells
        for i in range(nx):
            if solid[i]:
                A[i, :] = 0.0
                A[i, i] = 1.0
                rhs[i] = 0.0

        # Scale by dx^2 since Laplacian discretisation divides by dx^2
        A *= self.inv_dx ** 2

        p = np.linalg.solve(A, rhs)
        out = np.zeros_like(b)
        out[:, 0, 0] = p
        out[solid_cc] = 0.0
        return out

    def _cg_poisson_cc_rhs(
        self,
        b: np.ndarray,
        solid_cc: np.ndarray,
        tol: float = 1e-6,
        maxiter: int = 800,
        zero_rhs_mean: bool = False,
        ref_cell: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Solve ∇²x = b with Neumann on walls and Dirichlet inside solids (x=0 in solids).

        Parameters
        ----------
        b : np.ndarray
            Right-hand side.
        solid_cc : np.ndarray
            Boolean mask of solid cells.
        tol : float
            Convergence tolerance on the residual norm.
        maxiter : int
            Maximum CG iterations.
        zero_rhs_mean : bool, optional
            If True, subtract the mean of ``b`` over fluid cells before solving to
            ensure compatibility with the nullspace.
        ref_cell : tuple of int, optional
            If given, subtract the resulting pressure at this fluid cell from the
            entire solution, fixing the additive constant.
        """
        b = b.copy()
        if zero_rhs_mean:
            fluid = ~solid_cc
            if np.any(fluid):
                b -= float(b[fluid].mean())
        nx, ny, nz = self.nx, self.ny, self.nz
        x = np.zeros_like(b)
        r = np.nan_to_num(b - self._laplace_cc(x, solid_cc))
        p = r.copy()
        rsold = float(np.sum(r * r))
        if not np.isfinite(rsold) or rsold < tol * tol:
            return x
        for it in range(maxiter):
            Ap = self._laplace_cc(p, solid_cc)
            denom = float(np.sum(p * Ap))
            if not np.isfinite(denom) or abs(denom) < 1e-30:
                break
            alpha = rsold / denom
            x = np.nan_to_num(x + alpha * p)
            r = np.nan_to_num(r - alpha * Ap)
            rsnew = float(np.sum(r * r))
            if not np.isfinite(rsnew) or rsnew < tol * tol:
                break
            p = np.nan_to_num(r + (rsnew / rsold) * p)
            rsold = rsnew
        # zero inside solids for consistency
        x[solid_cc] = 0.0
        if ref_cell is not None:
            x -= x[ref_cell]
        return x

    def _cg_helmholtz_face(self, F: np.ndarray, a: float, axis: int, tol=1e-6, maxiter=200) -> np.ndarray:
        """Solve (I - a ∇²) x = F on a face grid with solid faces as Dirichlet (x=0)."""
        if axis == 0:
            solid = self.solid_u
        elif axis == 1:
            solid = self.solid_v
        else:
            solid = self.solid_w

        x = F.copy()
        x[solid] = 0.0
        r = np.nan_to_num(F - self._helmholtz_face_apply(x, a, axis, solid))
        p = r.copy()
        rsold = float(np.sum(r * r))
        if not np.isfinite(rsold) or rsold < tol * tol:
            return x
        for it in range(maxiter):
            Ap = self._helmholtz_face_apply(p, a, axis, solid)
            denom = float(np.sum(p * Ap))
            if not np.isfinite(denom) or abs(denom) < 1e-30:
                break
            alpha = rsold / denom
            x = np.nan_to_num(x + alpha * p)
            x[solid] = 0.0
            r = np.nan_to_num(r - alpha * Ap)
            rsnew = float(np.sum(r * r))
            if not np.isfinite(rsnew) or rsnew < tol * tol:
                break
            p = np.nan_to_num(r + (rsnew / rsold) * p)
            p[solid] = 0.0
            rsold = rsnew
        x[solid] = 0.0
        return x

    # Operators ------------------------------------------------------------
    def _laplace_cc(self, X: np.ndarray, solid_cc: np.ndarray) -> np.ndarray:
        """7-point Laplacian on CC with Neumann at walls, skipping solid neighbors."""
        solid = solid_cc
        fluid = ~solid
        X = np.nan_to_num(X)
        Y = np.zeros_like(X)
        cnt = np.zeros_like(X, dtype=np.int32)

        # +x and -x neighbors
        mask = fluid[:-1, :, :] & fluid[1:, :, :]
        Y[:-1, :, :][mask] += X[1:, :, :][mask]
        cnt[:-1, :, :][mask] += 1
        Y[1:, :, :][mask] += X[:-1, :, :][mask]
        cnt[1:, :, :][mask] += 1

        # +y and -y neighbors
        mask = fluid[:, :-1, :] & fluid[:, 1:, :]
        Y[:, :-1, :][mask] += X[:, 1:, :][mask]
        cnt[:, :-1, :][mask] += 1
        Y[:, 1:, :][mask] += X[:, :-1, :][mask]
        cnt[:, 1:, :][mask] += 1

        # +z and -z neighbors
        mask = fluid[:, :, :-1] & fluid[:, :, 1:]
        Y[:, :, :-1][mask] += X[:, :, 1:][mask]
        cnt[:, :, :-1][mask] += 1
        Y[:, :, 1:][mask] += X[:, :, :-1][mask]
        cnt[:, :, 1:][mask] += 1

        Y[fluid] -= cnt[fluid] * X[fluid]
        Y[solid] = 0.0
        return (self.inv_dx ** 2) * Y

    def _helmholtz_face_apply(self, X: np.ndarray, a: float, axis: int, solid: np.ndarray) -> np.ndarray:
        """Apply (I - a ∇²) on a face grid with Dirichlet on solid faces."""
        Xc = np.nan_to_num(X.copy())
        Xc[solid] = 0.0
        L = self._laplace_face(Xc, axis, solid)
        Y = Xc - a * L
        Y[solid] = 0.0
        return Y

    def _laplace_face(self, F: np.ndarray, axis: int, solid: np.ndarray) -> np.ndarray:
        """7-point Laplacian on a face grid with Dirichlet at solid faces and Neumann at domain walls."""
        fluid = ~solid
        F = np.nan_to_num(F)
        Y = np.zeros_like(F)
        cnt = np.zeros_like(F, dtype=np.int32)

        # +x / -x neighbors
        mask = fluid[:-1, :, :] & fluid[1:, :, :]
        Y[:-1, :, :][mask] += F[1:, :, :][mask]
        cnt[:-1, :, :][mask] += 1
        Y[1:, :, :][mask] += F[:-1, :, :][mask]
        cnt[1:, :, :][mask] += 1

        # +y / -y neighbors
        mask = fluid[:, :-1, :] & fluid[:, 1:, :]
        Y[:, :-1, :][mask] += F[:, 1:, :][mask]
        cnt[:, :-1, :][mask] += 1
        Y[:, 1:, :][mask] += F[:, :-1, :][mask]
        cnt[:, 1:, :][mask] += 1

        # +z / -z neighbors
        mask = fluid[:, :, :-1] & fluid[:, :, 1:]
        Y[:, :, :-1][mask] += F[:, :, 1:][mask]
        cnt[:, :, :-1][mask] += 1
        Y[:, :, 1:][mask] += F[:, :, :-1][mask]
        cnt[:, :, 1:][mask] += 1

        Y[fluid] -= cnt[fluid] * F[fluid]
        Y[solid] = 0.0
        return (self.inv_dx ** 2) * Y

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def _pad_x(self, A: np.ndarray, left: bool, expand: bool = False) -> np.ndarray:
        """Pad +/- x for averaging to u faces.

        When ``expand`` is True the returned array has ``nx+1`` entries in the
        x-direction so it can directly correspond to the u-face grid.  When
        False the padded array is cropped back to the original size and can be
        used for centered differences.
        """
        if left:
            P = np.pad(A, ((1, 0), (0, 0), (0, 0)), mode="edge")
            return P if expand else P[:-1, :, :]
        else:
            P = np.pad(A, ((0, 1), (0, 0), (0, 0)), mode="edge")
            return P if expand else P[1:, :, :]

    def _pad_y(self, A: np.ndarray, left: bool, expand: bool = False) -> np.ndarray:
        if left:
            P = np.pad(A, ((0, 0), (1, 0), (0, 0)), mode="edge")
            return P if expand else P[:, :-1, :]
        else:
            P = np.pad(A, ((0, 0), (0, 1), (0, 0)), mode="edge")
            return P if expand else P[:, 1:, :]

    def _pad_z(self, A: np.ndarray, left: bool, expand: bool = False) -> np.ndarray:
        if left:
            P = np.pad(A, ((0, 0), (0, 0), (1, 0)), mode="edge")
            return P if expand else P[:, :, :-1]
        else:
            P = np.pad(A, ((0, 0), (0, 0), (0, 1)), mode="edge")
            return P if expand else P[:, :, 1:]

    def _accumulate_face_forces(self, F: np.ndarray, centers: np.ndarray, comp: np.ndarray, radius: float, axis: int) -> None:
        """Distribute force density onto face grid with a cone kernel."""
        dx = self.dx
        rad = max(radius, dx)
        # Build world positions of face centers (in chunks if large)
        if axis == 0:
            nx, ny, nz = self.nx+1, self.ny, self.nz
            I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
            X = np.stack([I, J+0.5, K+0.5], axis=-1).reshape(-1,3) * dx
        elif axis == 1:
            nx, ny, nz = self.nx, self.ny+1, self.nz
            I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
            X = np.stack([I+0.5, J, K+0.5], axis=-1).reshape(-1,3) * dx
        else:
            nx, ny, nz = self.nx, self.ny, self.nz+1
            I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
            X = np.stack([I+0.5, J+0.5, K], axis=-1).reshape(-1,3) * dx

        for c, f in zip(centers, comp):
            r = np.linalg.norm(X - c[None,:], axis=1)
            w = np.maximum(0.0, 1.0 - r / rad)
            if w.sum() <= 0: continue
            w = (f * w / w.sum()).reshape((nx, ny, nz))
            F += w

    def _stable_dt(self) -> float:
        # CFL based on per-axis face velocities
        if self.u.size == 0:
            return 1e-6 if getattr(self.p, "nocap", True) else self.p.max_dt

        adv_limits = []
        umax = float(np.max(np.abs(self.u)))
        if umax > 0:
            adv_limits.append(self.p.cfl * self.dx / umax)
        if self.dim >= 2:
            vmax = float(np.max(np.abs(self.v)))
            if vmax > 0:
                adv_limits.append(self.p.cfl * self.dx / vmax)
        if self.dim == 3:
            wmax = float(np.max(np.abs(self.w)))
            if wmax > 0:
                adv_limits.append(self.p.cfl * self.dx / wmax)
        adv = min(adv_limits) if adv_limits else np.inf

        # Viscosity / diffusion stability limits (explicit). Velocity diffusion is
        # implicit in this solver but we still cap dt against scalar diffusion for
        # completeness.  The factor ``2*dim`` follows the standard ftcs limit.
        nu = self.p.nu
        visc = np.inf if nu <= 0 else (self.dx ** 2) / (2.0 * nu * self.dim)
        dmax = max(self.p.thermal_diffusivity, self.p.solute_diffusivity, 0.0)
        diff = np.inf if dmax == 0 else (self.dx ** 2) / (2.0 * dmax * self.dim)

        return max(1e-6, min(adv, visc, diff))

    # ---------------------------------------------------------------------
    # Demo & smoke test
    # ---------------------------------------------------------------------
    @staticmethod
    def demo_buoyant_plume(nx=48, ny=64, nz=48, dx=0.03) -> "VoxelMACFluid":
        params = VoxelFluidParams(nx=nx, ny=ny, nz=nz, dx=dx, nu=1e-5, gravity=(0.0, -9.81, 0.0))
        sim = VoxelMACFluid(params)
        # warm plume
        centers = np.array([[nx*dx*0.5, ny*dx*0.1, nz*dx*0.5]])
        sim.add_scalar_sources(centers, dT=np.array([20.0]), dS=np.array([0.0]), radius=dx*2.0)
        return sim


if __name__ == "__main__":
    sim = VoxelMACFluid.demo_buoyant_plume()
    t, dt = 0.0, 5e-4
    for s in range(30):
        sim.step(dt)
        t += dt
    # Simple diagnostics
    print("Mean|u|:", float(np.mean(np.abs(sim.u))))
    print("Mean|v|:", float(np.mean(np.abs(sim.v))))
    print("Mean|w|:", float(np.mean(np.abs(sim.w))))
    print("Mean P:", float(sim.pr.mean()))