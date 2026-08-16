"""Pure-SymPy viscous shallow-water current and wave equations.

This is a depth-averaged free-surface Navier--Stokes model.  Its executable
finite-volume stencil is authored entirely as SymPy equations; callers may
compile and invoke it, but there is no parallel Python numerical update.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy

from .symbolic_equation_compiler import (
    SymbolicEquationCompilation,
    SymbolicPublication,
    compile_sympy_equations,
)


STATE_FIELDS = ("height", "momentum_x", "momentum_y", "tracer")
NEIGHBORS = ("center", "east", "west", "north", "south")


@dataclass(frozen=True, slots=True)
class SymbolicFluidModel:
    """The exact authored mathematics and its named state/publication surface."""

    equations: tuple[sympy.Equality, ...]
    symbols: dict[str, sympy.Symbol]
    state_outputs: tuple[str, ...]
    publications: tuple[SymbolicPublication, ...]


def _symbols() -> dict[str, sympy.Symbol]:
    names = [
        f"{field}_{neighbor}"
        for field in STATE_FIELDS
        for neighbor in NEIGHBORS
    ]
    names.extend((
        "dt", "dx", "gravity", "viscosity", "tracer_diffusivity",
        "linear_drag", "coriolis", "minimum_height",
    ))
    return {name: sympy.Symbol(name, real=True) for name in names}


def symbolic_viscous_shallow_water_equations() -> SymbolicFluidModel:
    """Return one conservative current/wave update as pure SymPy equations.

    The fluxes are the two-dimensional viscous shallow-water equations, the
    standard depth-integrated free-surface form of Navier--Stokes.  A local
    Rusanov finite-volume flux supplies conservative upwind transport;
    centered Laplacians supply momentum viscosity and tracer diffusion. The
    update is deliberately unclamped: compiled diagnostic outputs let the
    managed dt system reject and retry a step that violates physical bounds.
    """

    s = _symbols()
    dt, dx = s["dt"], s["dx"]
    g = s["gravity"]
    nu = s["viscosity"]
    kappa = s["tracer_diffusivity"]
    drag = s["linear_drag"]
    coriolis = s["coriolis"]
    h_floor = s["minimum_height"]

    def state(field: str, neighbor: str) -> sympy.Symbol:
        return s[f"{field}_{neighbor}"]

    def flux_x(field: str, neighbor: str) -> sympy.Basic:
        h = state("height", neighbor)
        mx = state("momentum_x", neighbor)
        my = state("momentum_y", neighbor)
        tracer = state("tracer", neighbor)
        return {
            "height": mx,
            "momentum_x": mx * mx / h + sympy.Rational(1, 2) * g * h * h,
            "momentum_y": mx * my / h,
            "tracer": tracer * mx / h,
        }[field]

    def flux_y(field: str, neighbor: str) -> sympy.Basic:
        h = state("height", neighbor)
        mx = state("momentum_x", neighbor)
        my = state("momentum_y", neighbor)
        tracer = state("tracer", neighbor)
        return {
            "height": my,
            "momentum_x": mx * my / h,
            "momentum_y": my * my / h + sympy.Rational(1, 2) * g * h * h,
            "tracer": tracer * my / h,
        }[field]

    def laplacian(field: str) -> sympy.Basic:
        return (
            state(field, "east") + state(field, "west")
            + state(field, "north") + state(field, "south")
            - 4 * state(field, "center")
        ) / (dx * dx)

    def gravity_wave_speed(neighbor: str, axis: str) -> sympy.Basic:
        momentum = state(
            "momentum_x" if axis == "x" else "momentum_y", neighbor,
        )
        h = state("height", neighbor)
        return sympy.Abs(momentum / h) + sympy.sqrt(g * h)

    def rusanov_face(
        field: str, left: str, right: str, axis: str,
    ) -> sympy.Basic:
        flux = flux_x if axis == "x" else flux_y
        speed = sympy.Max(
            gravity_wave_speed(left, axis),
            gravity_wave_speed(right, axis),
        )
        return (
            sympy.Rational(1, 2) * (flux(field, left) + flux(field, right))
            - sympy.Rational(1, 2) * speed
            * (state(field, right) - state(field, left))
        )

    def time_derivative(field: str) -> sympy.Basic:
        east_flux = rusanov_face(field, "center", "east", "x")
        west_flux = rusanov_face(field, "west", "center", "x")
        north_flux = rusanov_face(field, "center", "north", "y")
        south_flux = rusanov_face(field, "south", "center", "y")
        return -(east_flux - west_flux + north_flux - south_flux) / dx

    height_next = state("height", "center") + dt * time_derivative("height")
    momentum_x_next = state("momentum_x", "center") + dt * (
        time_derivative("momentum_x")
        + nu * laplacian("momentum_x")
        - drag * state("momentum_x", "center")
        + coriolis * state("momentum_y", "center")
    )
    momentum_y_next = state("momentum_y", "center") + dt * (
        time_derivative("momentum_y")
        + nu * laplacian("momentum_y")
        - drag * state("momentum_y", "center")
        - coriolis * state("momentum_x", "center")
    )
    tracer_next = state("tracer", "center") + dt * (
        time_derivative("tracer") + kappa * laplacian("tracer")
    )

    center_h = state("height", "center")
    velocity_x = state("momentum_x", "center") / center_h
    velocity_y = state("momentum_y", "center") / center_h
    vorticity = (
        state("momentum_y", "east") / state("height", "east")
        - state("momentum_y", "west") / state("height", "west")
        - state("momentum_x", "north") / state("height", "north")
        + state("momentum_x", "south") / state("height", "south")
    ) / (2 * dx)
    speed = sympy.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)
    wave_speed = sympy.Max(*(
        gravity_wave_speed(neighbor, axis)
        for neighbor in NEIGHBORS
        for axis in ("x", "y")
    ))
    height_violation = sympy.Max(sympy.Integer(0), h_floor - height_next)
    tracer_violation = sympy.Max(
        sympy.Integer(0),
        -tracer_next,
        tracer_next - height_next,
    )

    expressions = {
        "height_next": height_next,
        "momentum_x_next": momentum_x_next,
        "momentum_y_next": momentum_y_next,
        "tracer_next": tracer_next,
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "vorticity": vorticity,
        "speed": speed,
        "wave_speed": wave_speed,
        "height_violation": height_violation,
        "tracer_violation": tracer_violation,
    }
    equations = tuple(
        sympy.Eq(sympy.Symbol(name, real=True), expression, evaluate=False)
        for name, expression in expressions.items()
    )
    publications = (
        SymbolicPublication("height_next", "fluid.free_surface.height", "field", "m"),
        SymbolicPublication("tracer_next", "fluid.tracer.concentration", "field", "1"),
        SymbolicPublication("velocity_x", "fluid.velocity.x", "vector-component", "m/s"),
        SymbolicPublication("velocity_y", "fluid.velocity.y", "vector-component", "m/s"),
        SymbolicPublication("vorticity", "fluid.vorticity", "field", "1/s"),
        SymbolicPublication("speed", "fluid.speed", "field", "m/s"),
        SymbolicPublication("wave_speed", "dt.metric.wave_speed", "metric", "m/s"),
        SymbolicPublication("height_violation", "dt.error.height_positivity", "metric", "m"),
        SymbolicPublication("tracer_violation", "dt.error.tracer_bounds", "metric", "m"),
    )
    return SymbolicFluidModel(
        equations=equations,
        symbols=s,
        state_outputs=(
            "height_next", "momentum_x_next", "momentum_y_next", "tracer_next",
        ),
        publications=publications,
    )


def compile_symbolic_fluid_step(
    *, name: str = "symbolic_fluid_step",
) -> SymbolicEquationCompilation:
    """Compile the authored SymPy current/wave equations to repository SSA."""

    model = symbolic_viscous_shallow_water_equations()
    return compile_sympy_equations(
        model.equations,
        name=name,
        publications=model.publications,
    )


__all__ = [
    "NEIGHBORS",
    "STATE_FIELDS",
    "SymbolicFluidModel",
    "compile_symbolic_fluid_step",
    "symbolic_viscous_shallow_water_equations",
]
