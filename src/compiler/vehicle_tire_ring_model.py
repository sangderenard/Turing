"""Coarse tire cross-section as four ring stations: the non-transient scale.

Bead / inner-shoulder / outer-shoulder / bead in the (radial, axial) half
plane, revolved fully around the wheel spin axis, is a closed quadrilateral.
Its enclosed volume is the Pappus solid-of-revolution of that quadrilateral,
in closed form -- no per-triangle integration, no mesh.  This is the
"scientific non-transient portion" of the tire: the slow, nearly-equilibrium
gas volume that the fine mesh's local contact deformation perturbs, not the
membrane's fast elastic modes.

The revolved-polygon volume identity used here,

    V = (pi/3) * sum_i (r_i^2 + r_i*r_{i+1} + r_{i+1}^2) * (z_{i+1} - z_i)

was verified numerically before being authored: against the Pappus
centroid formula (V = 2*pi*r_centroid*area) and against a fine numeric
disk-integration of the same quadrilateral, both exact to float precision.
A first transcription of this identity was off by a factor of two; it is
recorded here only after that check, not from memory.
"""

from __future__ import annotations

import sympy

from .symbolic_equation_compiler import (
    SymbolicEquationCompilation, SymbolicPublication, compile_sympy_equations)

_STATIONS = ("bead_inboard", "shoulder_inboard", "shoulder_outboard", "bead_outboard")
#: Four corners only.  A bulge-ring (mid-sidewall, mid-tread) variant closed
#: about a third of the volume gap against the real mesh at 1.6x the cost,
#: but a hand-picked station count is the wrong place to spend that budget:
#: an online-fitted correction against the running fine mesh (see the
#: coupling discussion around 2026-09-03) can absorb the same error for
#: less, and stays correct as the mesh's own material parameters change,
#: which added stations do not.  Kept simple on purpose.


def _symbols(names: str) -> dict[str, sympy.Symbol]:
    return {name: sympy.Symbol(name, real=True) for name in names.split()}


def symbolic_ring_volume_equations(
) -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """Enclosed volume and cross-section area of the four-station ring."""

    s = _symbols(" ".join(f"{name}_r {name}_z" for name in _STATIONS))
    points = [(s[f"{name}_r"], s[f"{name}_z"]) for name in _STATIONS]
    volume = sympy.Integer(0)
    area = sympy.Integer(0)
    for index in range(len(points)):
        r0, z0 = points[index]
        r1, z1 = points[(index + 1) % len(points)]
        volume += (r0 ** 2 + r0 * r1 + r1 ** 2) * (z1 - z0)
        area += r0 * z1 - r1 * z0
    volume = sympy.pi * volume / 3
    area = area / 2
    equations = (
        sympy.Eq(sympy.Symbol("ring_volume_m3", real=True), volume, evaluate=False),
        sympy.Eq(sympy.Symbol("ring_cross_section_area_m2", real=True), area, evaluate=False),
    )
    return equations, s


def compile_ring_volume_ssa() -> SymbolicEquationCompilation:
    equations, _ = symbolic_ring_volume_equations()
    return compile_sympy_equations(
        equations, name="tire_ring_volume",
        publications=tuple(
            SymbolicPublication(str(equation.lhs), f"world.vehicle.tire.ring.{equation.lhs}")
            for equation in equations
        ),
        dtype="float64",
    )


__all__ = ["symbolic_ring_volume_equations", "compile_ring_volume_ssa"]
