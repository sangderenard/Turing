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
        return {
            "positions": pos,
            "point_vectors": vec,
            "vector_positions": pos,
            "vectors": vec,
            "surface_batches": [],
        }


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
        state: Dict[str, np.ndarray | List] = {
            "positions": np.zeros((0, 3), dtype=float),
            "point_vectors": np.zeros((0, 3), dtype=float),
            "vector_positions": pos,
            "vectors": vec,
            "surface_batches": [],
        }
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
        parts = self.sim.export_particles()
        p_pos = parts.get("x", np.zeros((0, 3), dtype=float))
        p_vel = parts.get("v", np.zeros_like(p_pos))
        v_pos, v_vec = self.sim.export_vector_field()
        state: Dict[str, np.ndarray | List] = {
            "positions": p_pos,
            "point_vectors": p_vel,
            "vector_positions": v_pos,
            "vectors": v_vec,
            "surface_batches": [],
        }
        if self.animator is not None:
            try:
                state["surface_batches"] = self.animator.instance_batches()
            except Exception:
                state["surface_batches"] = []
        return state


def run_headless(adapter: BathAdapter, steps: int, dt: float) -> List[Dict[str, np.ndarray | List]]:
    """Advance ``adapter`` for ``steps`` without drawing.

    The function returns a list of visualization states gathered after each
    step, allowing callers (including tests or CLI tools) to post-process the
    data.  It performs no rendering and is safe for headless environments.
    """

    frames: List[Dict[str, np.ndarray | List]] = []
    for _ in range(int(steps)):
        adapter.step(dt)
        frames.append(adapter.visualization_state())
    return frames


def run_opengl(
    adapter: BathAdapter,
    steps: int,
    dt: float,
    *,
    draw: str = "points+vectors",
    scale: float | None = None,
) -> List[Dict[str, Dict[str, np.ndarray]]]:
    """Prepare OpenGL-friendly primitives for ``adapter`` over ``steps``.

    Parameters
    ----------
    adapter:
        Any :class:`BathAdapter` instance.
    steps, dt:
        Simulation steps and time step.
    draw:
        Drawing mode – ``"points"``, ``"vectors"`` or ``"points+vectors"``.
    scale:
        Optional vector scale.  Defaults to ``0.5*dx`` when available.

    Returns
    -------
    list of dict
        One entry per step containing "points" and/or "vectors" geometries.
        Each geometry dictionary holds NumPy arrays ready for GL buffers.
    """

    modes = {m.strip() for m in draw.split("+") if m.strip()}
    frames: List[Dict[str, Dict[str, np.ndarray]]] = []

    for _ in range(int(steps)):
        adapter.step(dt)
        state = adapter.visualization_state()

        p_pos = np.asarray(state.get("positions", np.zeros((0, 3))), dtype=float)
        p_vec = state.get("point_vectors")
        if p_vec is not None:
            p_vec = np.asarray(p_vec, dtype=float)
        v_pos = np.asarray(state.get("vector_positions", p_pos), dtype=float)
        v_vec = np.asarray(state.get("vectors", np.zeros_like(v_pos)), dtype=float)

        frame: Dict[str, Dict[str, np.ndarray]] = {}

        # Points -------------------------------------------------------------
        if "points" in modes and p_pos.size:
            if p_vec is not None and p_vec.shape == p_pos.shape:
                speed = np.linalg.norm(p_vec, axis=1)
            else:
                speed = np.zeros(p_pos.shape[0], dtype=float)
            size = 5.0 + 5.0 * speed
            frame["points"] = {"positions": p_pos, "size": size}

        # Vectors ------------------------------------------------------------
        if "vectors" in modes and v_vec.size:
            if scale is None:
                dx_val = None
                if hasattr(adapter, "sim"):
                    sim = getattr(adapter, "sim")
                    if hasattr(sim, "params") and hasattr(sim.params, "dx"):
                        dx_val = float(getattr(sim.params, "dx"))
                    elif hasattr(sim, "dx"):
                        dx_val = float(getattr(sim, "dx"))
                vec_scale = 0.5 * dx_val if dx_val is not None else 1.0
            else:
                vec_scale = scale

            start = v_pos
            end = v_pos + v_vec * vec_scale
            colors = np.ones((start.shape[0], 4), dtype=float)
            base_alpha = 0.25 if isinstance(adapter, HybridAdapter) else 1.0
            if start.shape[0] and start.shape[1] >= 3:
                z = start[:, 2]
                z_norm = (z - z.min()) / (np.ptp(z) + 1e-9)
                colors[:, 3] = base_alpha * (1.0 - z_norm)
            else:
                colors[:, 3] = base_alpha
            frame["vectors"] = {"start": start, "end": end, "color": colors}

        frames.append(frame)

    return frames


__all__ = [
    "BathAdapter",
    "SPHAdapter",
    "MACAdapter",
    "HybridAdapter",
    "run_headless",
    "run_opengl",
]
