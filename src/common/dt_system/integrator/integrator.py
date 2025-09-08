# -*- coding: utf-8 -*-
"""
Integrator module for dt_system: research-grade integration and differentiation.

This module introduces the Integrator class, which provides advanced integration
methods and a poetic API for summing over infinitesimals.
"""

from typing import Callable, Any, Optional, Sequence
import math
from ..engine_api import DtCompatibleEngine
from ..state_table_archive import StateTableArchive
from ...tensors.abstraction import AbstractTensor

class IntegrationAlgorithm:
    def step(self, f: Callable, t: float, x: float, dt: float) -> float:
        raise NotImplementedError

class EulerIntegrator(IntegrationAlgorithm):
    def step(self, f, t, x, dt):
        """Advance ``x`` by one Euler step.

        The demo integrator occasionally operates on non-numeric states
        (e.g. dictionaries representing structured values).  In that case
        addition with ``x`` is undefined; we fall back to returning the
        derivative directly.
        """

        y = f(t, x)
        try:
            return x + y * dt
        except TypeError:
            return y

class VelocityVerletIntegrator(IntegrationAlgorithm):
    def step(self, f, t, x, dt, v=0.0):
        # For 2nd order ODEs: x'' = f(t, x), v is velocity
        # Here, f returns acceleration
        a0 = f(t, x)
        x_new = x + v * dt + 0.5 * a0 * dt * dt
        a1 = f(t + dt, x_new)
        v_new = v + 0.5 * (a0 + a1) * dt
        return x_new, v_new

class RK2Integrator(IntegrationAlgorithm):
    def step(self, f, t, x, dt):
        k1 = f(t, x)
        k2 = f(t + dt, x + k1 * dt)
        return x + 0.5 * (k1 + k2) * dt

class RK4Integrator(IntegrationAlgorithm):
    def step(self, f, t, x, dt):
        k1 = f(t, x)
        k2 = f(t + 0.5 * dt, x + 0.5 * k1 * dt)
        k3 = f(t + 0.5 * dt, x + 0.5 * k2 * dt)
        k4 = f(t + dt, x + k3 * dt)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

class Integrator(DtCompatibleEngine):

    def __init__(
        self,
        dynamics: Callable[[float, Any], float] | Sequence[Callable[[float, Any], float]] | None = None,
        algorithm: str = "euler",
        archive: Optional[StateTableArchive] = None,
        *,
        min_trans: float | None = None,
        max_trans: float | None = None,
        spectral_damp: float | None = None,
        axis_mask: Optional[Sequence[bool]] = None,
        quantize: float | None = None,
    ):
        """Create an ``Integrator``.

        Parameters
        ----------
        dynamics:
            Either a single derivative function or a sequence of functions.  If a
            sequence is provided their contributions are summed, allowing multiple
            physics simulators to influence the motion.
        algorithm:
            One of ``'euler'``, ``'verlet'``, ``'rk2'`` or ``'rk4'``.
        min_trans / max_trans:
            Optional minimum and maximum per-step translocation.  When set, the
            change in state per step is clamped into ``[min_trans, max_trans]``.
        spectral_damp:
            Exponential smoothing factor ``[0,1]`` applied to motion deltas.  A
            value of ``0`` leaves motion untouched; higher values increasingly
            damp rapid oscillations.
        axis_mask:
            Optional boolean mask indicating which axes are allowed to move.  A
            ``False`` entry freezes the corresponding component, enabling
            independent-axis integration (e.g. isolating the Y axis).
        quantize:
            If provided, motion deltas are quantised to multiples of this value
            to emulate a sticky, discrete movement grid.
        """

        super().__init__()
        self.dynamics = dynamics if dynamics is not None else (lambda t, x: 0.0)
        self.dt_graph = None
        self.algorithm = algorithm.lower()
        self._algorithms = {
            "euler": EulerIntegrator(),
            "verlet": VelocityVerletIntegrator(),
            "rk2": RK2Integrator(),
            "rk4": RK4Integrator(),
        }
        self._state = None
        self.archive = archive or StateTableArchive()

        self.min_trans = min_trans
        self.max_trans = max_trans
        self.spectral_damp = spectral_damp
        self.axis_mask = list(axis_mask) if axis_mask is not None else None
        self.quantize = quantize
        self._prev_delta = None

    def register_dt_graph(self, dt_graph: object):
        """
        Register a dt_graph as the dynamics provider. The dt_graph must provide an 'evaluate(t, x)' method.
        """
        self.dt_graph = dt_graph
        self.dynamics = lambda t, x: self.dt_graph.evaluate(t, x)

    # ------------------------------------------------------------------
    # Internal helpers
    def _eval_dynamics(self, t: float, x: Any) -> Any:
        """Evaluate dynamics, supporting multiple derivative providers."""
        dyn = self.dynamics
        if isinstance(dyn, (list, tuple)):
            it = iter(dyn)
            try:
                total = next(it)(t, x)
            except StopIteration:
                return 0
            for f in it:
                total += f(t, x)
            return total
        return dyn(t, x)

    def _apply_post(self, prev: Any, new: Any) -> Any:
        """Apply clamping, damping and quantisation to motion."""
        if self.min_trans is None and self.max_trans is None and \
           self.spectral_damp is None and self.axis_mask is None and \
           self.quantize is None:
            return new

        if isinstance(prev, AbstractTensor) or isinstance(new, AbstractTensor):
            p = prev if isinstance(prev, AbstractTensor) else AbstractTensor.tensor(prev)
            n = new if isinstance(new, AbstractTensor) else AbstractTensor.tensor(new)
            delta = n - p
            if self.spectral_damp is not None:
                if self._prev_delta is None:
                    self._prev_delta = delta * 0
                delta = (1.0 - self.spectral_damp) * delta + self.spectral_damp * self._prev_delta
                self._prev_delta = delta
            if self.quantize:
                if self.quantize > 0:
                    q = self.quantize
                    delta = (delta / q).round() * q
            if self.max_trans is not None:
                delta = delta.clamp(min=-self.max_trans, max=self.max_trans)
            if self.min_trans is not None:
                m = self.min_trans
                delta = AbstractTensor.where(delta.abs() < m,
                                             delta.sign() * m if self.quantize else AbstractTensor.tensor(0.0),
                                             delta)
            state = p + delta
            if self.axis_mask is not None:
                mask = AbstractTensor.tensor(self.axis_mask)
                state = AbstractTensor.where(mask, state, p)
            return state

        def _process(p, n, idx=None):
            delta = n - p
            if self.spectral_damp is not None:
                if self._prev_delta is None:
                    self._prev_delta = [0.0] * (len(prev) if isinstance(prev, (list, tuple)) else 1)
                pd = self._prev_delta[idx or 0]
                delta = (1.0 - self.spectral_damp) * delta + self.spectral_damp * pd
                self._prev_delta[idx or 0] = delta
            if self.quantize:
                if self.quantize > 0:
                    delta = round(delta / self.quantize) * self.quantize
            if self.max_trans is not None and abs(delta) > self.max_trans:
                delta = math.copysign(self.max_trans, delta)
            if self.min_trans is not None and 0 < abs(delta) < self.min_trans:
                delta = math.copysign(self.min_trans, delta) if self.quantize else 0.0
            return p + delta

        if isinstance(new, (list, tuple)):
            out = []
            prev_list = list(prev)
            for i, (p, n) in enumerate(zip(prev_list, list(new))):
                if self.axis_mask is not None and i < len(self.axis_mask) and not self.axis_mask[i]:
                    out.append(p)
                    continue
                out.append(_process(p, n, i))
            return type(new)(out) if isinstance(new, tuple) else out
        else:
            if self.axis_mask is not None and len(self.axis_mask) > 0 and not self.axis_mask[0]:
                return prev
            return _process(prev, new, 0)

    def summa_cum_infinitesimalibus(self, a: float, b: float, dt_graph: Any, v0: float = 0.0) -> float:
        t = a
        x = 0.0
        v = v0
        alg = self._algorithms.get(self.algorithm, EulerIntegrator())
        # Use self.dynamics if set, else try to use self.dt_graph
        dynamics = self.dynamics
        if dynamics is None and self.dt_graph is not None:
            dynamics = lambda t, x: self.dt_graph.evaluate(t, x)
        if dynamics is None:
            raise ValueError("No dynamics or dt_graph registered for integration.")
        while t < b:
            dt = dt_graph.get_dt(t, x) if hasattr(dt_graph, 'get_dt') else 0.01
            if t + dt > b:
                dt = b - t
            x_prev = x
            if self.algorithm == "verlet":
                x, v = alg.step(self._eval_dynamics, t, x, dt, v)
            else:
                x = alg.step(self._eval_dynamics, t, x, dt)
            x = self._apply_post(x_prev, x)
            t += dt
        self._state = x
        return x

    # Alias for poetic and practical use
    integral = summa_cum_infinitesimalibus

    def step(self, dt: float, state=None, state_table=None):
        # For dt_system compatibility: advance by dt from current state
        # Set self._state to the input state each time
        x = state if state is not None else (self._state if self._state is not None else 0.0)
        prev_state = x
        self._state = x
        t = getattr(self, 'world_time', 0.0)
        alg = self._algorithms.get(self.algorithm, EulerIntegrator())
        if self.algorithm == "verlet":
            v = getattr(self, '_v', 0.0)
            x, v = alg.step(self._eval_dynamics, t, x, dt, v)
            self._v = v
        else:
            x = alg.step(self._eval_dynamics, t, x, dt)
        x = self._apply_post(prev_state, x)
        self._state = x
        self.world_time = t + dt
        metrics = None  # Could be extended to return integration error, etc.
        if state_table is not None:
            # Attach archive to state_table for external access
            if getattr(state_table, "archive", None) is None:
                state_table.archive = self.archive
            t_vec = (self.world_time,)
            try:
                self.archive.insert(t_vec, state_table)
            except Exception:
                pass
        return True, metrics, self._state

    def get_state(self, state=None) -> object:
        # Return the current state (x)
        return self._state
