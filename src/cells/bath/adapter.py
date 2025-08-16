"""Adapters bridging Bath to specific fluid backends.

These thin wrappers provide a common interface for sampling fields,
depositing sources, and advancing the simulation regardless of the
underlying fluid representation. They are intentionally lightweight so
that higher level orchestration code can stream patch batches without
branching on the backend type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, TYPE_CHECKING

import numpy as np

from .discrete_fluid import DiscreteFluid
from .voxel_fluid import VoxelMACFluid
from .hybrid_fluid import HybridFluid
from .dt_controller import STController, Targets, run_superstep_plan
from src.common.dt import SuperstepPlan, SuperstepResult

if TYPE_CHECKING:  # import only for types
    from .surface_animator import SurfaceAnimator


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
    # Adaptive dt controller state
    dt_ctrl: STController = field(default_factory=STController)
    dt_targets: Targets | None = None
    _dt_curr: float = 0.0
    _time: float = 0.0

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
        # Initialize targets lazily to favor CFL-only scaling against kernel h
        if self.dt_targets is None:
            # Use the SPH cfl_number as the target CFL; relax other penalties
            cfl = float(getattr(self.sim.params, "cfl_number", 0.25) or 0.25)
            self.dt_targets = Targets(cfl=cfl, div_max=1e30, mass_max=1e30)
        # Determine current dt from controller; use provided dt as seed on first call
        if self._dt_curr <= 0.0:
            self._dt_curr = float(dt)
        # Advance using adaptive dt controller (velocity-scaled)
        try:
            _, dt_next = self.sim.step_with_controller(self._dt_curr, self.dt_ctrl, self.dt_targets)
        except Exception:
            # Fallback to fixed stepping if controller path fails for any reason
            self.sim.step(self._dt_curr)
            dt_next = self._dt_curr
        self._time += self._dt_curr
        self._dt_curr = float(dt_next)
        # Optionally allow subsequent callers to pass dt_next
        return None

    def visualization_state(self) -> Dict[str, np.ndarray | List]:
        pos, vec = self.sim.export_positions_vectors()
        return {
            "positions": pos,
            "point_vectors": vec,
            "vector_positions": pos,
            "vectors": vec,
            "surface_batches": [],
        }

    def step_super(self, round_max: float, allow_increase_mid_round: bool = False) -> SuperstepResult:
        """Advance exactly one frame window with a non-increasing dt policy.

        Returns a :class:`~src.common.dt.SuperstepResult` and updates internal
        controller state and time accumulator.
        """
        # Ensure targets
        if self.dt_targets is None:
            cfl = float(getattr(self.sim.params, "cfl_number", 0.25) or 0.25)
            self.dt_targets = Targets(cfl=cfl, div_max=1e30, mass_max=1e30)
        if self._dt_curr <= 0.0:
            # Seed from engine stability estimate to avoid artificial 1e-3 caps
            dt0 = getattr(self.sim, "_stable_dt", None)
            self._dt_curr = float(dt0() if callable(dt0) else 1e-6)
        def advance(state, dt_step):
            # Advance once and compute metrics à la DiscreteFluid.step_with_controller
            prev_mass = float(np.sum(state.m))
            state._substep(dt_step)
            vmax = float(np.max(np.linalg.norm(state.v, axis=1))) if state.N > 0 else 0.0
            mass_now = float(np.sum(state.m))
            mass_err = abs(mass_now - prev_mass) / max(prev_mass, 1e-12)
            metrics = type("M", (), {"max_vel": vmax, "max_flux": vmax, "div_inf": 0.0, "mass_err": mass_err, "osc_flag": False, "stiff_flag": False})()
            return True, metrics
        plan = SuperstepPlan(round_max=float(round_max), dt_init=float(self._dt_curr or 1e-6), allow_increase_mid_round=bool(allow_increase_mid_round))
        # For SPH, use smoothing length as spatial scale for CFL
        dx_val = float(getattr(self.sim, "kernel", type("K", (), {"h": 1.0})()).h)
        res = run_superstep_plan(self.sim, plan, dx_val, self.dt_targets, self.dt_ctrl, advance)
        self._time += float(res.advanced)
        self._dt_curr = float(res.dt_next)
        return res


@dataclass
class MACAdapter(BathAdapter):
    """Adapter for :class:`VoxelMACFluid` (incompressible grid solver)."""

    sim: VoxelMACFluid
    animator: Optional["SurfaceAnimator"] = None
    # Adaptive dt controller state
    dt_ctrl: STController = field(default_factory=STController)
    dt_targets: Targets | None = None
    _dt_curr: float = 0.0
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
        # Lazy init of targets: use grid CFL and relax constraints so CFL dominates
        if self.dt_targets is None:
            cfl = float(getattr(self.sim, "cfl", getattr(self.sim.p, "cfl", 0.5)) or 0.5)
            self.dt_targets = Targets(cfl=cfl, div_max=1e30, mass_max=1e30)
        if self._dt_curr <= 0.0:
            self._dt_curr = float(dt)
        try:
            _, dt_next = self.sim.step_with_controller(self._dt_curr, self.dt_ctrl, self.dt_targets)
        except Exception:
            self.sim.step(self._dt_curr)
            dt_next = self._dt_curr
        self._time += self._dt_curr
        self._dt_curr = float(dt_next)
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

    def step_super(self, round_max: float, allow_increase_mid_round: bool = False) -> SuperstepResult:
        if self.dt_targets is None:
            cfl = float(getattr(self.sim, "cfl", getattr(self.sim.p, "cfl", 0.5)) or 0.5)
            self.dt_targets = Targets(cfl=cfl, div_max=1e30, mass_max=1e30)
        if self._dt_curr <= 0.0:
            dt0 = getattr(self.sim, "_stable_dt", None)
            self._dt_curr = float(dt0() if callable(dt0) else 1e-6)
        def advance(state, dt_step):
            # Run the solver step directly; gather simple velocity metric
            saved = state.copy_shallow()
            try:
                vel0 = getattr(state, "max_velocity", lambda: 0.0)()
                state.step(dt_step)
                vel1 = getattr(state, "max_velocity", lambda: 0.0)()
                max_vel = max(vel0, vel1)
                metrics = type("M", (), {"max_vel": max_vel, "max_flux": max_vel, "div_inf": 0.0, "mass_err": 0.0, "osc_flag": False, "stiff_flag": False})()
                return True, metrics
            except Exception:
                state.restore(saved)
                return False, type("M", (), {"max_vel": 0.0, "max_flux": 0.0, "div_inf": 0.0, "mass_err": 1.0, "osc_flag": False, "stiff_flag": True})()
        plan = SuperstepPlan(round_max=float(round_max), dt_init=float(self._dt_curr or 1e-6), allow_increase_mid_round=bool(allow_increase_mid_round))
        res = run_superstep_plan(self.sim, plan, getattr(self.sim, "dx", 1.0), self.dt_targets, self.dt_ctrl, advance)
        self._time += float(res.advanced)
        self._dt_curr = float(res.dt_next)
        if self.animator is not None:
            try:
                self.animator.update(self.sim, self._time)
            except Exception:
                pass
        return res


@dataclass
class HybridAdapter(BathAdapter):
    """Adapter for :class:`HybridFluid` (particle–grid solver)."""

    sim: HybridFluid
    animator: Optional["SurfaceAnimator"] = None
    # Adaptive dt controller state
    dt_ctrl: STController = field(default_factory=STController)
    dt_targets: Targets | None = None
    _dt_curr: float = 0.0
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
        if self.dt_targets is None:
            # Use grid CFL from hybrid params; relax other penalties
            cfl = float(getattr(self.sim.params, "cfl", 0.5) or 0.5)
            self.dt_targets = Targets(cfl=cfl, div_max=1e30, mass_max=1e30)
        if self._dt_curr <= 0.0:
            self._dt_curr = float(dt)
        try:
            _, dt_next = self.sim.step_with_controller(self._dt_curr, self.dt_ctrl, self.dt_targets)
        except Exception:
            self.sim.step(self._dt_curr)
            dt_next = self._dt_curr
        self._time += self._dt_curr
        self._dt_curr = float(dt_next)
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

    def step_super(self, round_max: float, allow_increase_mid_round: bool = False) -> SuperstepResult:
        if self.dt_targets is None:
            cfl = float(getattr(self.sim.params, "cfl", 0.5) or 0.5)
            self.dt_targets = Targets(cfl=cfl, div_max=1e30, mass_max=1e30)
        if self._dt_curr <= 0.0:
            dt0 = getattr(self.sim, "_stable_dt", None)
            self._dt_curr = float(dt0() if callable(dt0) else 1e-6)
        def advance(state, dt_step):
            saved = state.copy_shallow()
            try:
                v0 = getattr(state, "max_velocity", lambda: 0.0)()
                state.step(dt_step)
                v1 = getattr(state, "max_velocity", lambda: 0.0)()
                vmax = max(v0, v1)
                metrics = type("M", (), {"max_vel": vmax, "max_flux": vmax, "div_inf": 0.0, "mass_err": 0.0, "osc_flag": False, "stiff_flag": False})()
                return True, metrics
            except Exception:
                state.restore(saved)
                return False, type("M", (), {"max_vel": 0.0, "max_flux": 0.0, "div_inf": 0.0, "mass_err": 1.0, "osc_flag": False, "stiff_flag": True})()
        plan = SuperstepPlan(round_max=float(round_max), dt_init=float(self._dt_curr or 1e-6), allow_increase_mid_round=bool(allow_increase_mid_round))
        res = run_superstep_plan(self.sim, plan, getattr(self.sim.params, "dx", 1.0), self.dt_targets, self.dt_ctrl, advance)
        self._time += float(res.advanced)
        self._dt_curr = float(res.dt_next)
        if self.animator is not None:
            try:
                self.animator.update(self.sim.grid, self._time)
            except Exception:
                pass
        return res


def run_headless(adapter: BathAdapter, steps: int, dt: float) -> List[Dict[str, np.ndarray | List]]:
    """Advance ``adapter`` for ``steps`` without drawing.

    The function returns a list of visualization states gathered after each
    step, allowing callers (including tests or CLI tools) to post-process the
    data.  It performs no rendering and is safe for headless environments.
    """

    frames: List[Dict[str, np.ndarray | List]] = []
    for _ in range(int(steps)):
        # Prefer exact landing via superstep when available
        step_super = getattr(adapter, "step_super", None)
        if callable(step_super):
            try:
                _res = step_super(float(dt))
            except Exception:
                adapter.step(dt)
        else:
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
        # Prefer exact landing via superstep when available
        step_super = getattr(adapter, "step_super", None)
        if callable(step_super):
            try:
                _res = step_super(float(dt))
            except Exception:
                adapter.step(dt)
        else:
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
