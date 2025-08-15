"""Hybrid particle–grid fluid prototype.

This module defines :class:`HybridFluid`, a minimal hybrid fluid simulator
combining a MAC grid and discrete particles.  It is intended as a light-weight
placeholder for more sophisticated FLIP/MPM style solvers.  The class exposes a
small state consisting of grid fields, particle arrays and a few thresholds used
by phase-change logic in higher level simulations.

The implementation here focuses on providing a clean API rather than full
physics.  The :meth:`step` method currently advects particles by their velocity
and keeps them inside the grid domain.  Future versions may add momentum
exchange and pressure projection between particles and grid.

Kernel Constants
----------------
``Kernels`` provides precomputed normalisation constants for commonly used SPH
kernels (poly6 and spiky) in one, two and three spatial dimensions.  These
constants assume a unit smoothing length; callers should scale by ``h`` with the
appropriate exponent when evaluating kernel functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, List

import numpy as np

# ---------------------------------------------------------------------------
# Kernel constants
# ---------------------------------------------------------------------------
# Normalisation constants for unit smoothing length.  Actual kernel values
# require scaling by ``h`` to the appropriate power.
_poly6_const = {
    1: 35.0 / 32.0,
    2: 4.0 / np.pi,
    3: 315.0 / (64.0 * np.pi),
}
_spiky_const = {
    1: 15.0 / 16.0,
    2: 10.0 / np.pi,
    3: 15.0 / np.pi,
}
Kernels: Dict[int, Dict[str, float]] = {
    D: {"poly6": _poly6_const[D], "spiky": _spiky_const[D]} for D in (1, 2, 3)
}


# ---------------------------------------------------------------------------
# Hybrid fluid
# ---------------------------------------------------------------------------
@dataclass
class HybridFluid:
    """Hybrid particle–grid fluid simulator.

    Parameters
    ----------
    shape:
        Grid shape ``(nx, ny, nz)`` for 3-D, ``(nx, ny)`` for 2-D, etc.
    n_particles:
        Number of discrete particles maintained by the simulator.
    scalar_names:
        Optional iterable with names of additional cell-centred scalar fields.
    phi_cond, phi_shat, p_shat_max:
        Threshold parameters controlling phase-change heuristics.  They default
        to zero and can be adjusted after construction.
    """

    shape: Tuple[int, ...]
    n_particles: int
    scalar_names: Iterable[str] | None = None
    phi_cond: float = 0.0
    phi_shat: float = 0.0
    p_shat_max: float = 0.0

    dim: int = field(init=False)
    u: List[np.ndarray] = field(init=False)
    p: np.ndarray = field(init=False)
    scalars: Dict[str, np.ndarray] = field(init=False)
    phi: np.ndarray = field(init=False)
    solid: np.ndarray = field(init=False)
    x: np.ndarray = field(init=False)
    v: np.ndarray = field(init=False)
    m: np.ndarray = field(init=False)
    rho: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.dim = len(self.shape)
        self.p = np.zeros(self.shape, dtype=np.float64)
        self.phi = np.zeros(self.shape, dtype=np.float64)
        self.solid = np.zeros(self.shape, dtype=bool)

        # Optional scalar fields
        names = tuple(self.scalar_names) if self.scalar_names is not None else ()
        self.scalars = {name: np.zeros(self.shape, dtype=np.float64) for name in names}

        # Staggered MAC velocity components (u[axis])
        self.u = []
        for axis, n in enumerate(self.shape):
            vel_shape = list(self.shape)
            vel_shape[axis] = n + 1
            self.u.append(np.zeros(tuple(vel_shape), dtype=np.float64))

        # Particle arrays
        self.x = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        self.v = np.zeros_like(self.x)
        self.m = np.ones(self.n_particles, dtype=np.float64)
        self.rho = np.ones(self.n_particles, dtype=np.float64)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        """Advance the hybrid fluid by ``dt`` seconds.

        The current implementation performs a very simple explicit Euler
        advection of particle positions by their velocities and constrains them
        to the grid bounds.  This acts as a placeholder until a full particle-
        grid coupling scheme is implemented.
        """

        self.x += self.v * dt
        for axis, n in enumerate(self.shape):
            self.x[:, axis] = np.clip(self.x[:, axis], 0.0, float(n))

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def export_grid(self) -> Dict[str, np.ndarray]:
        """Return a dictionary with grid fields."""
        data: Dict[str, np.ndarray] = {
            "p": self.p,
            "phi": self.phi,
            "solid": self.solid,
        }
        for axis, comp in enumerate(self.u):
            data[f"u{axis}"] = comp
        data.update(self.scalars)
        return data

    def export_particles(self) -> Dict[str, np.ndarray]:
        """Return a dictionary with particle arrays."""
        return {
            "x": self.x,
            "v": self.v,
            "m": self.m,
            "rho": self.rho,
        }


__all__ = ["HybridFluid", "Kernels"]
