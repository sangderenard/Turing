"""Compiler-owned elastic/plastic/fracture transition for vehicle graph members.

The equation is deliberately geometry-agnostic.  A beam/tube graph operator
supplies axial, outer-fibre bending and engineering shear/torsion strains and
their rates.  Material and section values remain runtime parameters.  The
same transition therefore serves structural tube members, suspension links,
and sacrificial junction cartridges without a host-authored damage analogue.
"""

from __future__ import annotations

from functools import lru_cache

import sympy

from .ssa_c_backend import CFunctionArtifact, emit_ssa_function_to_c
from .ssa_wasm_backend import SSAWasmArtifact, emit_ssa_function_to_wasm
from .symbolic_equation_compiler import (
    SymbolicEquationCompilation,
    SymbolicPublication,
    compile_symbolic_program,
)


# Output names in the exact order the authored return map emits them, so the
# persistent symbolic cache can be keyed without constructing the equations.
MEMBER_MATERIAL_OUTPUTS = (
    "plastic_axial_next",
    "plastic_bending_next",
    "plastic_shear_next",
    "accumulated_plastic_strain_next",
    "work_hardening_next",
    "remaining_ductility_next",
    "failed_next",
    "axial_stress_pa",
    "bending_stress_pa",
    "shear_stress_pa",
    "equivalent_trial_stress_pa",
    "current_yield_stress_pa",
    "fracture_demand",
    "elastic_energy_j",
    "damping_power_w",
    "plastic_work_increment_j",
    "dissipated_energy_next",
)


def _positive(value: sympy.Basic) -> sympy.Basic:
    """``max(value, 0)`` spelled so every backend evaluates it exactly.

    ``(value + Abs(value)) / 2`` is the same function in real arithmetic, but
    it relies on ``value`` and ``Abs(value)`` cancelling to exactly zero.
    SymPy's automatic evaluation distributes the halving and re-spells the
    operands, so the compiled schedule adds two differently rounded copies
    of ``value`` and keeps a residue of a few ULP; the 1e12 fracture gate
    below then turns that residue into a phantom failure fraction (measured:
    exactly 2**-14, scaling every stress by 2**-13).  ``Max`` is a single
    exact relational select in every backend.
    """

    return sympy.Max(value, 0)


def _clamp(value: sympy.Basic, lower: sympy.Basic, upper: sympy.Basic) -> sympy.Basic:
    return sympy.Min(sympy.Max(value, lower), upper)


def symbolic_vehicle_member_material_equations(
) -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """Return a small-strain J2-like beam/tube material return map.

    Bending strain is the signed outer-fibre strain derived by the graph's
    beam kinematics.  ``shear_strain`` combines transverse shear and torsional
    surface strain.  Plastic flow is a radial return in this three-component
    stress space.  Isotropic work hardening raises the next yield surface but
    also raises the fracture demand, representing the loss of remaining
    ductility requested for repeatedly worked vehicle members.
    """

    names = (
        "dt axial_strain bending_strain shear_strain "
        "axial_strain_rate bending_strain_rate shear_strain_rate "
        "plastic_axial_previous plastic_bending_previous plastic_shear_previous "
        "accumulated_plastic_strain_previous dissipated_energy_previous failed_previous "
        "youngs_modulus_pa shear_modulus_pa initial_yield_stress_pa ultimate_stress_pa "
        "hardening_modulus_pa fracture_plastic_strain hardening_fragility "
        "material_volume_m3 axial_viscosity_pa_s bending_viscosity_pa_s shear_viscosity_pa_s"
    )
    s = {name: sympy.Symbol(name, real=True) for name in names.split()}
    tiny = sympy.Float("1e-30")
    active = 1 - _clamp(s["failed_previous"], 0, 1)
    elastic_axial_trial = s["axial_strain"] - s["plastic_axial_previous"]
    elastic_bending_trial = s["bending_strain"] - s["plastic_bending_previous"]
    elastic_shear_trial = s["shear_strain"] - s["plastic_shear_previous"]
    sigma_axial_trial = s["youngs_modulus_pa"] * elastic_axial_trial
    sigma_bending_trial = s["youngs_modulus_pa"] * elastic_bending_trial
    tau_trial = s["shear_modulus_pa"] * elastic_shear_trial
    equivalent_trial = sympy.sqrt(
        sigma_axial_trial ** 2 + sigma_bending_trial ** 2 + 3 * tau_trial ** 2 + tiny)
    previous_accumulated = sympy.Max(0, s["accumulated_plastic_strain_previous"])
    current_yield = (s["initial_yield_stress_pa"]
                     + s["hardening_modulus_pa"] * previous_accumulated)
    plastic_multiplier = active * _positive(equivalent_trial - current_yield) / (
        s["youngs_modulus_pa"] + s["hardening_modulus_pa"] + tiny)
    axial_direction = sigma_axial_trial / equivalent_trial
    bending_direction = sigma_bending_trial / equivalent_trial
    shear_direction = 3 * tau_trial / equivalent_trial
    plastic_axial = s["plastic_axial_previous"] + plastic_multiplier * axial_direction
    plastic_bending = s["plastic_bending_previous"] + plastic_multiplier * bending_direction
    plastic_shear = s["plastic_shear_previous"] + plastic_multiplier * shear_direction
    accumulated = previous_accumulated + plastic_multiplier
    work_hardening = s["hardening_modulus_pa"] * accumulated / (
        s["initial_yield_stress_pa"] + tiny)
    remaining_ductility = s["fracture_plastic_strain"] / (
        1 + s["hardening_fragility"] * work_hardening)
    fracture_demand = sympy.Max(
        equivalent_trial / (s["ultimate_stress_pa"] + tiny),
        accumulated / (remaining_ductility + tiny),
    )
    # Repository SSA currently has no scalar relational-select spelling in
    # the C backend.  This near-discontinuous algebraic gate keeps the branch
    # inside the compiled equation; the transition width is far below any
    # representable vehicle material tolerance.  Hosts treat >= .5 as open.
    fractured_now = _clamp(
        _positive(fracture_demand - 1) * sympy.Float("1e12"), 0, 1)
    failed = sympy.Max(_clamp(s["failed_previous"], 0, 1), fractured_now)
    surviving = 1 - failed
    elastic_axial = surviving * (s["axial_strain"] - plastic_axial)
    elastic_bending = surviving * (s["bending_strain"] - plastic_bending)
    elastic_shear = surviving * (s["shear_strain"] - plastic_shear)
    sigma_axial = surviving * s["youngs_modulus_pa"] * elastic_axial
    sigma_bending = surviving * s["youngs_modulus_pa"] * elastic_bending
    tau = surviving * s["shear_modulus_pa"] * elastic_shear
    elastic_energy = surviving * s["material_volume_m3"] * (
        sympy.Rational(1, 2) * s["youngs_modulus_pa"]
        * (elastic_axial ** 2 + elastic_bending ** 2)
        + sympy.Rational(1, 2) * s["shear_modulus_pa"] * elastic_shear ** 2)
    damping_power = active * s["material_volume_m3"] * (
        s["axial_viscosity_pa_s"] * s["axial_strain_rate"] ** 2
        + s["bending_viscosity_pa_s"] * s["bending_strain_rate"] ** 2
        + s["shear_viscosity_pa_s"] * s["shear_strain_rate"] ** 2)
    plastic_work = active * current_yield * plastic_multiplier * s["material_volume_m3"]
    dissipated_energy = (s["dissipated_energy_previous"] + plastic_work
                          + sympy.Max(0, s["dt"]) * damping_power)
    values = {
        "plastic_axial_next": plastic_axial,
        "plastic_bending_next": plastic_bending,
        "plastic_shear_next": plastic_shear,
        "accumulated_plastic_strain_next": accumulated,
        "work_hardening_next": work_hardening,
        "remaining_ductility_next": remaining_ductility,
        "failed_next": failed,
        "axial_stress_pa": sigma_axial,
        "bending_stress_pa": sigma_bending,
        "shear_stress_pa": tau,
        "equivalent_trial_stress_pa": equivalent_trial,
        "current_yield_stress_pa": current_yield,
        "fracture_demand": fracture_demand,
        "elastic_energy_j": elastic_energy,
        "damping_power_w": damping_power,
        "plastic_work_increment_j": plastic_work,
        "dissipated_energy_next": dissipated_energy,
    }
    assert tuple(values) == MEMBER_MATERIAL_OUTPUTS
    equations = tuple(sympy.Eq(sympy.Symbol(name, real=True), expression, evaluate=False)
                      for name, expression in values.items())
    return equations, s


@lru_cache(maxsize=1)
def compile_vehicle_member_material_ssa() -> SymbolicEquationCompilation:
    return compile_symbolic_program(
        symbolic_vehicle_member_material_equations,
        name="vehicle_member_material_step",
        publications=tuple(
            SymbolicPublication(name, f"world.vehicle.member.{name}")
            for name in MEMBER_MATERIAL_OUTPUTS
        ),
        dtype="float64",
    )


@lru_cache(maxsize=1)
def compile_vehicle_member_material_c() -> CFunctionArtifact:
    compiled = compile_vehicle_member_material_ssa()
    artifact = emit_ssa_function_to_c(
        compiled.module, compiled.function.name, entry_name="vehicle_member_material_step")
    if not artifact.complete:
        reasons = "; ".join(shortfall.reason for shortfall in artifact.shortfalls)
        raise RuntimeError(f"vehicle member material law does not lower to C: {reasons}")
    return artifact


@lru_cache(maxsize=1)
def compile_vehicle_member_material_wasm() -> SSAWasmArtifact:
    compiled = compile_vehicle_member_material_ssa()
    artifact = emit_ssa_function_to_wasm(
        compiled.module, compiled.function.name, work_contract="deploy")
    if not artifact.complete:
        reasons = "; ".join(shortfall.reason for shortfall in artifact.shortfalls)
        raise RuntimeError(f"vehicle member material law does not lower to Wasm: {reasons}")
    return artifact
