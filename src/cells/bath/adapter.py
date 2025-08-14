"""Adapters bridging Bath to specific fluid backends.

These thin wrappers provide a common interface for sampling fields,
depositing sources, and advancing the simulation regardless of the
underlying fluid representation. They are intentionally lightweight so
that higher level orchestration code can stream patch batches without
branching on the backend type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .discrete_fluid import DiscreteFluid
from .voxel_fluid import VoxelMACFluid


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


@dataclass
class MACAdapter(BathAdapter):
    """Adapter for :class:`VoxelMACFluid` (incompressible grid solver)."""

    sim: VoxelMACFluid

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
