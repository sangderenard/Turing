# -*- coding: utf-8 -*-
"""
Integrator module for dt_system: research-grade integration and differentiation.

This module introduces the Integrator class, which provides advanced integration
methods and a poetic API for summing over infinitesimals.
"""

from typing import Callable, Any, Optional
from ..engine_api import DtCompatibleEngine

class IntegrationAlgorithm:
    def step(self, f: Callable, t: float, x: float, dt: float) -> float:
        raise NotImplementedError

class EulerIntegrator(IntegrationAlgorithm):
    def step(self, f, t, x, dt):
        return x + f(t, x) * dt

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

    def __init__(self, dynamics: Callable[[float, Any], float] = None, algorithm: str = "euler"):
        """
        dynamics: a function f(t, x) returning the derivative at time t and state x, or None for deferred registration.
        algorithm: one of 'euler', 'verlet', 'rk2', 'rk4'
        """
        super().__init__()
        self.dynamics = dynamics
        self.dt_graph = None
        self.algorithm = algorithm.lower()
        self._algorithms = {
            "euler": EulerIntegrator(),
            "verlet": VelocityVerletIntegrator(),
            "rk2": RK2Integrator(),
            "rk4": RK4Integrator(),
        }
        self._state = None

    def register_dt_graph(self, dt_graph: object):
        """
        Register a dt_graph as the dynamics provider. The dt_graph must provide an 'evaluate(t, x)' method.
        """
        self.dt_graph = dt_graph
        self.dynamics = lambda t, x: self.dt_graph.evaluate(t, x)

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
            if self.algorithm == "verlet":
                x, v = alg.step(dynamics, t, x, dt, v)
            else:
                x = alg.step(dynamics, t, x, dt)
            t += dt
        self._state = x
        return x

    # Alias for poetic and practical use
    integral = summa_cum_infinitesimalibus

    def step(self, dt: float, state=None, state_table=None):
        # For dt_system compatibility: advance by dt from current state
        # Set self._state to the input state each time
        x = state if state is not None else (self._state if self._state is not None else 0.0)
        self._state = x
        t = getattr(self, 'world_time', 0.0)
        alg = self._algorithms.get(self.algorithm, EulerIntegrator())
        if self.algorithm == "verlet":
            v = getattr(self, '_v', 0.0)
            x, v = alg.step(self.dynamics, t, x, dt, v)
            self._v = v
        else:
            x = alg.step(self.dynamics, t, x, dt)
        self._state = x
        self.world_time = t + dt
        metrics = None  # Could be extended to return integration error, etc.
        return True, metrics, self._state

    def get_state(self, state=None) -> object:
        # Return the current state (x)
        return self._state
