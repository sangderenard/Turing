"""The "reduced" tire-fidelity fallback: a real integral of the tire's own
rest geometry against the ground, not a tuned spring.

This is the bottom rung of the fidelity ladder (reduced / spectral / green /
fine, see docs/PLAN_TIRE_FIDELITY_LADDER.md): the cheap mode every ordinary
rig/validator run should use, reserving the full deformable mesh ("fine")
for the offline runs that build the spectral/Green's-function models. It
must still feel like a real tire, so it is derived from the same 4-station
ring cross-section already used for the Pappus volume law
(vehicle_tire_ring_model.py), not an arbitrary constant.

Physics: at axial position z, the tire's rest cross-section (from the same
four ring stations, revolved about the spin axis) is a solid disk of radius
r(z), piecewise-linear across the three station-to-station segments (bead
-> inner shoulder -> outer shoulder -> bead). Where that disk extends a
depth H below a flat ground plane, its chord width there is
``2*sqrt(r(z)**2 - H**2)`` (0 when r(z) <= H, i.e. no contact at that
slice). Integrating that chord width over the axial extent gives the real
contact-patch area implied by the tire's own rest geometry; multiplying by
the tire's own internal pressure gives the normal load
``force = pressure * contact_patch_area`` -- the standard pneumatic-tire
load relation (load capacity tables are built the same way: contact area
times pressure), not a hand-picked spring constant.

The axial integral has no simple closed form once clamped by ``Max(...,
0)`` inside a square root, so each of the three linear segments is
integrated by a fixed 5-point Gauss-Legendre quadrature, unrolled into a
closed-form finite sum -- the same pattern already used for the Goertzel
and ring-volume laws in this compiler. Verified against
``scipy.integrate.quad`` (an independent reference) in
tools/verify_reduced_contact_law.py before being trusted anywhere: matches
to ~1e-13 relative error for any compression depth up to a station's own
radius (the realistic operating range -- a tire compressing partway toward
its shoulder). ``Max(r(z)**2 - depth**2, 0)`` is genuinely non-smooth at the
z where the ground plane crosses a station-to-station taper, so once
``depth`` approaches a *bead* radius (the tire compressed nearly onto its
own rim -- not a realistic operating point for this fallback law) the fixed
5-point-per-segment quadrature underestimates near that kink (~1e-3
relative error at 0.30 m depth against a 0.28 m bead radius in the
verification fixture). This law is not meant to be trusted that deep; a
real build should clamp ``compression_depth_m`` well inside the tire's own
station radii before calling it.
"""

from __future__ import annotations

import sympy

from .symbolic_equation_compiler import (
    SymbolicEquationCompilation, SymbolicPublication, compile_sympy_equations)

_STATIONS = ("bead_inboard", "shoulder_inboard", "shoulder_outboard", "bead_outboard")

# 5-point Gauss-Legendre on [-1, 1]: nodes and weights, standard tabulated
# values (exact for polynomials up to degree 9 -- ample for a smooth
# sqrt-clamped integrand sampled per linear segment).
_GAUSS5_NODES = (
    0.0,
    0.5384693101056831,
    -0.5384693101056831,
    0.9061798459386640,
    -0.9061798459386640,
)
_GAUSS5_WEIGHTS = (
    0.5688888888888889,
    0.4786286704993665,
    0.4786286704993665,
    0.2369268850561891,
    0.2369268850561891,
)


def _symbols(names: str) -> dict[str, sympy.Symbol]:
    return {name: sympy.Symbol(name, real=True) for name in names.split()}


def _segment_contact_width_integral(
    r0: sympy.Expr, z0: sympy.Expr, r1: sympy.Expr, z1: sympy.Expr, depth: sympy.Expr,
) -> sympy.Expr:
    """Gauss-Legendre quadrature of the below-ground chord width over one
    linear station-to-station segment, mapped from [-1, 1] to [z0, z1].
    """

    half_length = (z1 - z0) / 2
    total = sympy.Integer(0)
    for node, weight in zip(_GAUSS5_NODES, _GAUSS5_WEIGHTS):
        t = (node + 1) / 2  # map [-1, 1] -> [0, 1]
        radius_here = r0 + t * (r1 - r0)
        clamped = sympy.Max(radius_here ** 2 - depth ** 2, 0)
        chord_width = 2 * sympy.sqrt(clamped)
        total += weight * chord_width
    return total * half_length


def symbolic_reduced_contact_law_equations(
) -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """Contact-patch area and normal force from the tire's rest ring
    geometry pressed a depth ``compression_depth_m`` into flat ground, at
    internal pressure ``gas_pressure_pa``.
    """

    s = _symbols(" ".join(f"{name}_r {name}_z" for name in _STATIONS))
    s.update(_symbols("compression_depth_m gas_pressure_pa"))
    depth = s["compression_depth_m"]
    points = [(s[f"{name}_r"], s[f"{name}_z"]) for name in _STATIONS]
    area = sympy.Integer(0)
    for index in range(len(points) - 1):
        r0, z0 = points[index]
        r1, z1 = points[index + 1]
        area += _segment_contact_width_integral(r0, z0, r1, z1, depth)
    force = s["gas_pressure_pa"] * area
    equations = (
        sympy.Eq(sympy.Symbol("reduced_contact_patch_area_m2", real=True), area, evaluate=False),
        sympy.Eq(sympy.Symbol("reduced_contact_force_n", real=True), force, evaluate=False),
    )
    return equations, s


def compile_reduced_contact_law_ssa() -> SymbolicEquationCompilation:
    equations, _ = symbolic_reduced_contact_law_equations()
    return compile_sympy_equations(
        equations, name="tire_reduced_contact_law",
        publications=tuple(
            SymbolicPublication(str(equation.lhs), f"world.vehicle.tire.reduced.{equation.lhs}")
            for equation in equations
        ),
        dtype="float64",
    )


__all__ = [
    "symbolic_reduced_contact_law_equations", "compile_reduced_contact_law_ssa",
]
