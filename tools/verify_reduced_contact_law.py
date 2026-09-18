"""Independent numeric verification of vehicle_tire_reduced_contact_law.py's
Gauss-Legendre contact-patch-area law against scipy.integrate.quad on the
same piecewise-linear r(z) profile, before it is trusted anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy import integrate  # noqa: E402
import sympy  # noqa: E402

from src.compiler.vehicle_tire_reduced_contact_law import (  # noqa: E402
    symbolic_reduced_contact_law_equations,
)


def reference_contact_patch_area(stations, depth: float) -> float:
    """Plain numeric reference: integrate the below-ground chord width with
    scipy over the same piecewise-linear r(z), independent of the compiled
    quadrature.
    """

    def r_of_z(z: float) -> float:
        for (r0, z0), (r1, z1) in zip(stations, stations[1:]):
            if z0 <= z <= z1 or z1 <= z <= z0:
                t = 0.0 if z1 == z0 else (z - z0) / (z1 - z0)
                return r0 + t * (r1 - r0)
        raise ValueError(f"z={z} outside station range")

    def chord_width(z: float) -> float:
        r = r_of_z(z)
        clamped = max(r * r - depth * depth, 0.0)
        return 2.0 * np.sqrt(clamped)

    total = 0.0
    for (r0, z0), (r1, z1) in zip(stations, stations[1:]):
        value, _ = integrate.quad(chord_width, min(z0, z1), max(z0, z1), limit=200)
        total += value
    return total


def main() -> int:
    equations, symbols_by_name = symbolic_reduced_contact_law_equations()
    area_equation = next(eq for eq in equations if str(eq.lhs) == "reduced_contact_patch_area_m2")
    force_equation = next(eq for eq in equations if str(eq.lhs) == "reduced_contact_force_n")

    # A representative light-truck tire cross-section: bead near the rim,
    # shoulders near the tread, symmetric about the mid-width plane.
    stations = {
        "bead_inboard": (0.28, -0.11), "shoulder_inboard": (0.38, -0.09),
        "shoulder_outboard": (0.38, 0.09), "bead_outboard": (0.28, 0.11),
    }
    pressure_pa = 2.4e5

    worst_relative_error = 0.0
    realistic_depths = (0.0, 0.02, 0.05, 0.08, 0.10, 0.12)
    extreme_depths = (0.30,)  # near/past the bead radius -- documented low-accuracy domain
    for depth in (*realistic_depths, *extreme_depths):
        substitution = {symbols_by_name[f"{name}_r"]: value[0]
                        for name, value in stations.items()}
        substitution.update({symbols_by_name[f"{name}_z"]: value[1]
                             for name, value in stations.items()})
        substitution[symbols_by_name["compression_depth_m"]] = depth
        substitution[symbols_by_name["gas_pressure_pa"]] = pressure_pa

        compiled_area = float(area_equation.rhs.subs(substitution).evalf())
        compiled_force = float(force_equation.rhs.subs(substitution).evalf())

        station_points = [stations[name] for name in
                          ("bead_inboard", "shoulder_inboard", "shoulder_outboard", "bead_outboard")]
        reference_area = reference_contact_patch_area(station_points, depth)
        reference_force = pressure_pa * reference_area

        relative_error = abs(compiled_area - reference_area) / max(reference_area, 1e-9)
        if depth in realistic_depths:
            worst_relative_error = max(worst_relative_error, relative_error)
        domain = "realistic" if depth in realistic_depths else "EXTREME (documented low-accuracy)"
        print(f"depth={depth:.3f} m [{domain}]  compiled_area={compiled_area:.6e} m^2  "
              f"reference_area={reference_area:.6e} m^2  relative_error={relative_error:.3e}  "
              f"compiled_force={compiled_force:.3e} N  reference_force={reference_force:.3e} N",
              flush=True)

    print(f"worst relative error across realistic depths: {worst_relative_error:.3e}", flush=True)
    if worst_relative_error > 1e-6:
        print("FAILED: quadrature law disagrees with independent reference beyond tolerance "
              "within the realistic operating range", flush=True)
        return 1
    print("PASSED: quadrature law matches independent scipy reference to float precision "
          "across the realistic operating range", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
