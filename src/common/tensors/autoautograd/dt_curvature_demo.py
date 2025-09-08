"""Autoautograd demo: dt_system integrator with curvature-modulated spring.

A particle is attached to a fixed anchor via a spring.  The spring force is
modulated by a curvature term derived from an implied hexagonal face on the
edge.  Motion is integrated with the dt_system ``Integrator`` using a high
quality Runge–Kutta 4 scheme and the shared spectral dampener.
"""
from __future__ import annotations

from ..abstraction import AbstractTensor
from ...dt_system.integrator.integrator import Integrator
from ...dt_system.curvature import hex_face_curvature


def demo() -> None:
    anchor = AbstractTensor.tensor([0.0, 0.0])
    rest_len = 1.0
    stiffness = 4.0
    mass = 1.0
    dt = 0.05

    def dynamics(t: float, x: AbstractTensor) -> AbstractTensor:
        p = x[:2]
        v = x[2:]
        edge = p - anchor
        length = AbstractTensor.linalg.norm(edge) + 1e-12
        curvature = hex_face_curvature(anchor, p)
        field = (length - rest_len) + curvature
        force_mag = stiffness * AbstractTensor.tanh(field)
        force = -force_mag * edge / length
        a = force / mass
        return AbstractTensor.tensor([v[0], v[1], a[0], a[1]])

    integ = Integrator(dynamics=dynamics, algorithm="rk4", spectral_damp=0.05)
    state = AbstractTensor.tensor([1.3, 0.0, 0.0, 0.0])
    for _ in range(10):
        _, _, state = integ.step(dt, state=state)
        print(f"t={integ.world_time:.2f} p={state[:2].tolist()} v={state[2:].tolist()}")


if __name__ == "__main__":
    demo()
