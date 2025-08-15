"""Adapters bridging Bath to specific fluid backends.

These thin wrappers provide a common interface for sampling fields,
depositing sources, and advancing the simulation regardless of the
underlying fluid representation. They are intentionally lightweight so
that higher level orchestration code can stream patch batches without
branching on the backend type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np

from .discrete_fluid import DiscreteFluid
from .voxel_fluid import VoxelMACFluid
from .hybrid_fluid import HybridFluid

try:  # Surface animation is optional; allow import failure
    from .surface_animator import SurfaceAnimator
except Exception:  # pragma: no cover
    SurfaceAnimator = None  # type: ignore


class BathAdapter:
    """Abstract interface over a fluid simulator used by :class:`Bath`."""

    def sample(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """Sample fluid fields at ``points`` (shape: ``(N,3)``)."""
        raise NotImplementedError

    def deposit(
        self,
        centers: np.ndarray,
        dV: np.ndarray,
        dS: np.ndarray,
        radius: float,
    ) -> Dict[str, np.ndarray]:
        """Deposit volume/solute sources around ``centers``.

        Returns a dictionary describing realized amounts. Implementations
        may ignore certain fields if unsupported by the backend.
        """
        raise NotImplementedError

    def step(self, dt: float) -> None:
        """Advance the underlying simulator by ``dt`` seconds."""
        raise NotImplementedError

    def visualization_state(self) -> Dict[str, np.ndarray]:
        """Return data useful for visualization (positions, vectors, etc.)."""
        raise NotImplementedError


@dataclass
class SPHAdapter(BathAdapter):
    """Adapter for :class:`DiscreteFluid` (weakly-compressible SPH)."""

    sim: DiscreteFluid

    def __post_init__(self) -> None:
        # rest_density is used to convert volume sources to mass sources
        self._rho0 = self.sim.params.rest_density

    def sample(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        return self.sim.sample_at(points)

    def deposit(
        self,
        centers: np.ndarray,
        dV: np.ndarray,
        dS: np.ndarray,
        radius: float,
    ) -> Dict[str, np.ndarray]:
        dV = np.asarray(dV, dtype=float)
        dS = np.asarray(dS, dtype=float)
        # Convert volume change to mass using rest density
        dM = self._rho0 * dV
        dS_mass = self._rho0 * dS
        return self.sim.apply_sources(centers, dM=dM, dS_mass=dS_mass, radius=radius)

    def step(self, dt: float) -> None:
        self.sim.step(dt)

    def visualization_state(self) -> Dict[str, np.ndarray | List]:
        pos, vec = self.sim.export_positions_vectors()
        return {"positions": pos, "vectors": vec, "surface_batches": []}


@dataclass
class MACAdapter(BathAdapter):
    """Adapter for :class:`VoxelMACFluid` (incompressible grid solver)."""

    sim: VoxelMACFluid
    animator: Optional[SurfaceAnimator] = None
    _time: float = 0.0

    def sample(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        return self.sim.sample_at(points)

    def deposit(
        self,
        centers: np.ndarray,
        dV: np.ndarray,
        dS: np.ndarray,
        radius: float,
    ) -> Dict[str, np.ndarray]:
        # Incompressible grid: only scalar sources are supported here.
        dS = np.asarray(dS, dtype=float)
        self.sim.add_scalar_sources(centers, dT=np.zeros_like(dS), dS=dS, radius=radius)
        return {"dV": np.zeros_like(dV), "dS": dS}

    def step(self, dt: float) -> None:
        self.sim.step(dt)
        self._time += dt
        if self.animator is not None:
            try:
                self.animator.update(self.sim, self._time)
            except Exception:
                pass

    def visualization_state(self) -> Dict[str, np.ndarray | List]:
        pos, vec = self.sim.export_vector_field()
        state: Dict[str, np.ndarray | List] = {"positions": pos, "vectors": vec, "surface_batches": []}
        if self.animator is not None:
            try:
                state["surface_batches"] = self.animator.instance_batches()
            except Exception:
                state["surface_batches"] = []
        return state


@dataclass
class HybridAdapter(BathAdapter):
    """Adapter for :class:`HybridFluid` (particle–grid solver)."""

    sim: HybridFluid
    animator: Optional[SurfaceAnimator] = None
    _time: float = 0.0

    def sample(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        return self.sim.sample_at(points)

    def deposit(
        self,
        centers: np.ndarray,
        dV: np.ndarray,
        dS: np.ndarray,
        radius: float,
    ) -> Dict[str, np.ndarray]:
        dS = np.asarray(dS, dtype=float)
        # Forward scalar sources to underlying grid; volume change ignored.
        self.sim.grid.add_scalar_sources(centers, dT=np.zeros_like(dS), dS=dS, radius=radius)
        return {"dV": np.zeros_like(dV), "dS": dS}

    def step(self, dt: float) -> None:
        self.sim.step(dt)
        self._time += dt
        if self.animator is not None:
            try:
                self.animator.update(self.sim.grid, self._time)
            except Exception:
                pass

    def visualization_state(self) -> Dict[str, np.ndarray | List]:
        pos, vec = self.sim.export_vector_field()
        state: Dict[str, np.ndarray | List] = {"positions": pos, "vectors": vec, "surface_batches": []}
        if self.animator is not None:
            try:
                state["surface_batches"] = self.animator.instance_batches()
            except Exception:
                state["surface_batches"] = []
        return state
