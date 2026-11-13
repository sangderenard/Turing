"""One batched, fixed-capacity spring-network step for sheets and ropes.

The equations are the numerical authority.  A row owns its position and
velocity columns; this law advances one externally chosen dt.  The batch axis
flattens (network, fixed node slot, xyz), so all occupied nodes run together.
Four neighbor slots serve a sheet; a rope uses two and masks the others.
Unoccupied slots retain their previous state exactly.
"""

from __future__ import annotations

import sympy as sp

from .symbolic_equation_compiler import compile_symbolic_program


INPUT_NAMES = (
    "position", "velocity", "contact_enabled", "contact_height",
    "neighbor_0", "neighbor_1", "neighbor_2", "neighbor_3",
    "rest_0", "rest_1", "rest_2", "rest_3", "edge_0", "edge_1", "edge_2", "edge_3",
    "stiffness", "inverse_mass", "drag", "external_acceleration", "dt",
    "occupied", "pinned", "anchor",
)
OUTPUT_NAMES = ("position_next", "velocity_next")


def spring_network_equations():
    """Symplectic Euler with implicit linear drag and fixed-neighbor springs.

    Every quantity is a scalar batch column.  Neighbor positions are gathered
    from the row's private state before the call.  Rest offsets are signed per
    axis; opposite edges therefore cancel at the authored rest shape.
    """
    s = {name: sp.Symbol(name, real=True) for name in INPUT_NAMES}
    force = sum(
        s[f"edge_{i}"] * (s[f"neighbor_{i}"] - s["position"] - s[f"rest_{i}"])
        for i in range(4)
    )
    acceleration = s["external_acceleration"] + s["inverse_mass"] * s["stiffness"] * force
    velocity_free = (s["velocity"] + s["dt"] * acceleration) / (1 + s["dt"] * s["drag"])
    free = s["occupied"] * (1 - s["pinned"])
    candidate_position=s["position"]+s["dt"]*velocity_free
    penetration=s["contact_enabled"]*sp.Max(s["contact_height"]-candidate_position,0)
    accepted_position=candidate_position+penetration
    accepted_velocity=velocity_free+penetration/s["dt"]
    velocity_next = free * accepted_velocity + (1 - s["occupied"]) * s["velocity"]
    position_next = (free * accepted_position
                     + s["occupied"] * s["pinned"] * s["anchor"]
                     + (1 - s["occupied"]) * s["position"])
    equations = (
        sp.Eq(sp.Symbol("position_next"), position_next, evaluate=False),
        sp.Eq(sp.Symbol("velocity_next"), velocity_next, evaluate=False),
    )
    return equations, s


def compile_spring_network_step():
    return compile_symbolic_program(
        spring_network_equations, name="spring_network_step",
    )


__all__ = ["INPUT_NAMES", "OUTPUT_NAMES", "spring_network_equations", "compile_spring_network_step"]
