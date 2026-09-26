"""Compact parametric world physics authored as simultaneous SymPy equations.

The equations are the only numerical authority.  They lower through the
repository's canonical SymPy -> ProcessGraph -> SSA path and then directly to
WebAssembly.  Hosts provide parameter values in the published arena ABI, so
gravity, drag, contact softness, bounds, and portal transforms remain live
editable without recompiling the module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import sympy

from .ssa_wasm_backend import SSAWasmArtifact, emit_ssa_function_to_wasm
from .symbolic_equation_compiler import (
    SymbolicEquationCompilation,
    SymbolicPublication,
    compile_sympy_equations,
)
from .abstract_ui_world import WorldWasmPlugin


ABSTRACT_UI_PHYSICS_VERSION = "abstract-ui-symbolic-world-physics-v0"
PHYSICS_STATE_OUTPUTS = (
    "position_x_next", "position_y_next", "position_z_next",
    "velocity_x_next", "velocity_y_next", "velocity_z_next",
)


@dataclass(frozen=True, slots=True)
class PhysicsParameter:
    name: str
    default: float
    unit: str
    group: str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    live_editable: bool = True

    def to_data(self) -> dict[str, Any]:
        result = {
            "name": self.name, "default": self.default, "unit": self.unit,
            "group": self.group, "live_editable": self.live_editable,
        }
        for name in ("minimum", "maximum", "step"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class SymbolicWorldPhysics:
    equations: tuple[sympy.Equality, ...]
    symbols: dict[str, sympy.Symbol]
    parameters: tuple[PhysicsParameter, ...]
    publications: tuple[SymbolicPublication, ...]

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": ABSTRACT_UI_PHYSICS_VERSION,
            "source_language": "sympy-equation-set",
            "lowering": [
                "sympy-expressions", "canonical-process-graph",
                "compiler-ssa", "webassembly",
            ],
            "equations": [str(equation) for equation in self.equations],
            "equation_srepr": [sympy.srepr(equation) for equation in self.equations],
            "state_outputs": list(PHYSICS_STATE_OUTPUTS),
            "parameters": [parameter.to_data() for parameter in self.parameters],
            "constraints": [
                "dt > 0", "minimum_axis <= maximum_axis",
                "0 <= contact_softness <= 1",
                "portal_cos^2 + portal_sin^2 = 1",
                "portal_active is normally 0 or 1; fractions interpolate",
            ],
            "integration": "semi-implicit-euler-with-implicit-linear-drag",
            "contact": "unilateral-aabb-projection-with-editable-compliance",
            "traversal": "source-relative-yaw-transposition-then-target-translation",
        }


def _symbols() -> dict[str, sympy.Symbol]:
    names = (
        "position_x position_y position_z velocity_x velocity_y velocity_z dt "
        "gravity_x gravity_y gravity_z force_x force_y force_z inverse_mass "
        "linear_drag contact_softness radius minimum_x minimum_y minimum_z "
        "maximum_x maximum_y maximum_z portal_active portal_source_x "
        "portal_source_y portal_source_z portal_target_x portal_target_y "
        "portal_target_z portal_cos portal_sin obstacle_active "
        "obstacle_normal_x obstacle_normal_z obstacle_plane"
    )
    return {name: sympy.Symbol(name, real=True) for name in names.split()}


def _parameter_surface() -> tuple[PhysicsParameter, ...]:
    return (
        PhysicsParameter("gravity_x", 0.0, "m/s^2", "forces", -40.0, 40.0, 0.01),
        PhysicsParameter("gravity_y", -9.81, "m/s^2", "forces", -40.0, 40.0, 0.01),
        PhysicsParameter("gravity_z", 0.0, "m/s^2", "forces", -40.0, 40.0, 0.01),
        PhysicsParameter("force_x", 0.0, "N", "forces", -100.0, 100.0, 0.05),
        PhysicsParameter("force_y", 0.0, "N", "forces", -100.0, 100.0, 0.05),
        PhysicsParameter("force_z", 0.0, "N", "forces", -100.0, 100.0, 0.05),
        PhysicsParameter("inverse_mass", 1.0, "1/kg", "body", 0.0, 20.0, 0.01),
        PhysicsParameter("linear_drag", 0.8, "1/s", "body", 0.0, 20.0, 0.01),
        PhysicsParameter("contact_softness", 1.0, "1", "contact", 0.0, 1.0, 0.01),
        PhysicsParameter("radius", 0.0625, "m", "body", 0.001, 4.0, 0.001),
        PhysicsParameter("minimum_x", -10.0, "m", "bounds", -10000.0, 10000.0, 0.1),
        PhysicsParameter("minimum_y", 0.225, "m", "bounds", -10000.0, 10000.0, 0.01),
        PhysicsParameter("minimum_z", -10.0, "m", "bounds", -10000.0, 10000.0, 0.1),
        PhysicsParameter("maximum_x", 10.0, "m", "bounds", -10000.0, 10000.0, 0.1),
        PhysicsParameter("maximum_y", 100.0, "m", "bounds", -10000.0, 10000.0, 0.1),
        PhysicsParameter("maximum_z", 10.0, "m", "bounds", -10000.0, 10000.0, 0.1),
        PhysicsParameter("obstacle_active", 0.0, "1", "contact", live_editable=False),
        PhysicsParameter("obstacle_normal_x", 0.0, "1", "contact", live_editable=False),
        PhysicsParameter("obstacle_normal_z", 0.0, "1", "contact", live_editable=False),
        PhysicsParameter("obstacle_plane", 0.0, "m", "contact", live_editable=False),
        PhysicsParameter("portal_active", 0.0, "1", "traversal", 0.0, 1.0, 1.0),
        PhysicsParameter("portal_source_x", 0.0, "m", "traversal", step=0.1),
        PhysicsParameter("portal_source_y", 0.0, "m", "traversal", step=0.1),
        PhysicsParameter("portal_source_z", 0.0, "m", "traversal", step=0.1),
        PhysicsParameter("portal_target_x", 0.0, "m", "traversal", step=0.1),
        PhysicsParameter("portal_target_y", 0.0, "m", "traversal", step=0.1),
        PhysicsParameter("portal_target_z", 0.0, "m", "traversal", step=0.1),
        PhysicsParameter("portal_cos", 1.0, "1", "traversal", -1.0, 1.0, 0.001),
        PhysicsParameter("portal_sin", 0.0, "1", "traversal", -1.0, 1.0, 0.001),
    )


@lru_cache(maxsize=1)
def symbolic_world_physics_equations() -> SymbolicWorldPhysics:
    """Return one compact gravity/contact/traversal state transition."""

    s = _symbols()
    dt = s["dt"]
    drag_denominator = 1 + s["linear_drag"] * dt

    def free_velocity(axis: str) -> sympy.Basic:
        return (
            s[f"velocity_{axis}"]
            + dt * (
                s[f"gravity_{axis}"]
                + s[f"force_{axis}"] * s["inverse_mass"]
            )
        ) / drag_denominator

    free_v = {axis: free_velocity(axis) for axis in "xyz"}
    trial = {
        axis: s[f"position_{axis}"] + dt * free_v[axis]
        for axis in "xyz"
    }

    def correction(axis: str) -> sympy.Basic:
        return (
            sympy.Max(0, s[f"minimum_{axis}"] + s["radius"] - trial[axis])
            - sympy.Max(0, trial[axis] + s["radius"] - s[f"maximum_{axis}"])
        )

    correction_by_axis = {axis: correction(axis) for axis in "xyz"}
    contacted_position = {
        axis: trial[axis] + s["contact_softness"] * correction_by_axis[axis]
        for axis in "xyz"
    }
    contacted_velocity = {
        axis: free_v[axis]
        + s["contact_softness"] * correction_by_axis[axis] / dt
        for axis in "xyz"
    }

    obstacle_penetration = sympy.Max(
        0,
        s["obstacle_normal_x"] * contacted_position["x"]
        + s["obstacle_normal_z"] * contacted_position["z"]
        + s["radius"] - s["obstacle_plane"],
    ) * s["obstacle_active"]
    obstacle_correction = {
        "x": -s["obstacle_normal_x"] * obstacle_penetration,
        "y": sympy.Integer(0),
        "z": -s["obstacle_normal_z"] * obstacle_penetration,
    }
    solid_position = {
        axis: contacted_position[axis]
        + s["contact_softness"] * obstacle_correction[axis]
        for axis in "xyz"
    }
    solid_velocity = {
        axis: contacted_velocity[axis]
        + s["contact_softness"] * obstacle_correction[axis] / dt
        for axis in "xyz"
    }

    relative_x = solid_position["x"] - s["portal_source_x"]
    relative_y = solid_position["y"] - s["portal_source_y"]
    relative_z = solid_position["z"] - s["portal_source_z"]
    transposed_position = {
        "x": s["portal_target_x"] + s["portal_cos"] * relative_x
        - s["portal_sin"] * relative_z,
        "y": s["portal_target_y"] + relative_y,
        "z": s["portal_target_z"] + s["portal_sin"] * relative_x
        + s["portal_cos"] * relative_z,
    }
    transposed_velocity = {
        "x": s["portal_cos"] * solid_velocity["x"]
        - s["portal_sin"] * solid_velocity["z"],
        "y": solid_velocity["y"],
        "z": s["portal_sin"] * solid_velocity["x"]
        + s["portal_cos"] * solid_velocity["z"],
    }
    keep = 1 - s["portal_active"]
    expressions = {
        **{
            f"position_{axis}_next": keep * solid_position[axis]
            + s["portal_active"] * transposed_position[axis]
            for axis in "xyz"
        },
        **{
            f"velocity_{axis}_next": keep * solid_velocity[axis]
            + s["portal_active"] * transposed_velocity[axis]
            for axis in "xyz"
        },
        "contact_penetration": (
            sum(map(sympy.Abs, correction_by_axis.values()))
            + obstacle_penetration
        ),
        "specific_kinetic_energy": sympy.Rational(1, 2) * sum(
            value * value for value in transposed_velocity.values()
        ),
    }
    equations = tuple(
        sympy.Eq(sympy.Symbol(name, real=True), expression, evaluate=False)
        for name, expression in expressions.items()
    )
    publications = tuple(
        SymbolicPublication(name, f"world.physics.{name}")
        for name in PHYSICS_STATE_OUTPUTS
    ) + (
        SymbolicPublication(
            "contact_penetration", "world.physics.contact_penetration", "metric", "m",
        ),
        SymbolicPublication(
            "specific_kinetic_energy", "world.physics.specific_kinetic_energy",
            "metric", "m^2/s^2",
        ),
    )
    return SymbolicWorldPhysics(
        equations, s, _parameter_surface(), publications,
    )


@lru_cache(maxsize=1)
def compile_symbolic_world_physics(
    *, name: str = "abstract_ui_world_physics_step",
) -> SymbolicEquationCompilation:
    model = symbolic_world_physics_equations()
    return compile_sympy_equations(
        model.equations, name=name, publications=model.publications,
    )


@lru_cache(maxsize=1)
def compile_symbolic_world_physics_wasm() -> SSAWasmArtifact:
    compiled = compile_symbolic_world_physics()
    artifact = emit_ssa_function_to_wasm(compiled.module, compiled.function.name)
    if not artifact.complete:
        reasons = "; ".join(item.reason for item in artifact.shortfalls)
        raise RuntimeError(f"symbolic world physics does not lower to WASM: {reasons}")
    return artifact


@lru_cache(maxsize=1)
def symbolic_world_physics_wasm_plugin() -> WorldWasmPlugin:
    """Publish the direct SSA WASM artifact through the common world ABI."""

    model = symbolic_world_physics_equations()
    compiled = compile_symbolic_world_physics()
    artifact = compile_symbolic_world_physics_wasm()
    return WorldWasmPlugin(
        "abstract-ui/plugins/symbolic-world-physics",
        "integrate-gravity-contact-and-boundary-traversal",
        artifact.binary,
        artifact.name,
        ({"name": "io", "role": "arena-base", "dtype": "int32"},),
        "\n".join(str(equation) for equation in model.equations),
        source_language="sympy",
        capability="physics",
        operation_count=sum(
            len(block.instrs) for block in compiled.function.blocks.values()
        ),
        reserved_bytes=max((*artifact.input_offsets, *artifact.output_offsets)) + 8,
        abi={
            "kind": "ssa-scalar-arena-v0",
            "invocation": "arena-base-pointer",
            "dtype": "float64",
            "input_names": list(artifact.input_names),
            "output_names": list(artifact.output_names),
            "input_offsets": list(artifact.input_offsets),
            "output_offsets": list(artifact.output_offsets),
            "equation_schema": ABSTRACT_UI_PHYSICS_VERSION,
        },
    )


__all__ = [
    "ABSTRACT_UI_PHYSICS_VERSION", "PHYSICS_STATE_OUTPUTS", "PhysicsParameter",
    "SymbolicWorldPhysics", "compile_symbolic_world_physics",
    "compile_symbolic_world_physics_wasm", "symbolic_world_physics_equations",
    "symbolic_world_physics_wasm_plugin",
]
