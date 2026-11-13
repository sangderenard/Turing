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


def compile_spring_network_compute_shader(*, local_size: int = 64):
    """Lower the authored scalar law to the deployable f64 shader ABI.

    One invocation advances one occupied or masked scalar node coordinate.
    The host still gathers neighbors from its previous state and swaps output
    columns only after a completed dispatch, as the native row does.
    """
    from .ssa_spirv_backend import emit_deployable_compute_shader

    compiled = compile_spring_network_step()
    function = compiled.function
    results = {
        instruction.res.id: instruction.res
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    }
    outputs = tuple(results[compiled.output_ids[name]] for name in OUTPUT_NAMES)
    return emit_deployable_compute_shader(
        function, outputs=outputs, dtype="float64", local_size=local_size,
        name="spring_network_step", module=compiled.module,
    )


def emit_spring_network_resident_glsl(*, local_size: int = 64) -> str:
    """Read the four neighbor feeds from the current device position column.

    The numerical operations still come from the same scalar SSA law. Masks
    prevent reads across sheet borders; the control's row stride is width*3
    scalar lanes. Dispatch-to-dispatch barriers make the next state visible.
    """
    from .ssa_glsl_compute_backend import emit_ssa_scalar_to_glsl_compute

    reads = {
        11: "(feed5.data[element] != 0.0 ? feed17.data[element - 3u] : 0.0)",
        12: "(feed6.data[element] != 0.0 ? feed17.data[element + 3u] : 0.0)",
        13: "(feed7.data[element] != 0.0 ? feed17.data[element - control.row_stride] : 0.0)",
        14: "(feed8.data[element] != 0.0 ? feed17.data[element + control.row_stride] : 0.0)",
    }
    return emit_ssa_scalar_to_glsl_compute(
        compile_spring_network_step().function,
        name="spring_network_step_resident", local_size=local_size,
        feed_reads=reads, control_fields=("uint row_stride;",),
    )


__all__ = ["INPUT_NAMES", "OUTPUT_NAMES", "spring_network_equations", "compile_spring_network_step", "compile_spring_network_compute_shader", "emit_spring_network_resident_glsl"]
