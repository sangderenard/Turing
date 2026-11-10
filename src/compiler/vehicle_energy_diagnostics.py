"""Named energy compartments published by the browser vehicle equation owner.

All values sample the same pre-transition state. These are partial compartments,
not a closed energy budget: contact/internal work, thermal/chemical stores,
unsprung translation and compliant deformation are not included in their sum.
"""
from __future__ import annotations
import sympy

CORNERS = ("front_left", "front_right", "rear_left", "rear_right")
ENERGY_OUTPUTS = (
    "energy_chassis_translation_j", "energy_chassis_rotation_j",
    "energy_chassis_gravity_j", "energy_engine_rotation_j",
    *(f"energy_wheel_rotation_{corner}_j" for corner in CORNERS),
    "energy_differential_rotation_front_j", "energy_differential_rotation_rear_j",
)
ENERGY_DIAGNOSTICS_CONTRACT = {
    "schema": "browser-vehicle-compiled-energy-compartments-v1",
    "authority": "same-symbolic-integration-input-state-and-inertias",
    "sample_phase": "before-transition", "units": "J", "complete_budget": False,
    "outputs": list(ENERGY_OUTPUTS),
    "node_outputs": {
        "powertrain.engine": ["energy_engine_rotation_j"],
        **{f"suspension.{corner}.hub": [f"energy_wheel_rotation_{corner}_j"] for corner in CORNERS},
        **{f"powertrain.{axle}_differential": [f"energy_differential_rotation_{axle}_j"] for axle in ("front", "rear")},
    },
    "gravity_reference": "world y=0; signed potential -mass*gravity*y",
    "excluded": ["unsprung translation", "spring and tire deformation",
                 "chemical and thermal stores", "external work and integrated losses",
                 "rotating-frame cross terms"],
}


def vehicle_energy_expressions(s, effective_engine_inertia):
    """Use the integration owner's engine inertia, not an independently rebuilt one."""
    half = sympy.Rational(1, 2)
    expressions = {
        "energy_chassis_translation_j": half * sum(s[f"velocity_{a}"]**2 for a in "xyz") / s["inverse_mass"],
        "energy_chassis_rotation_j": half * sum(s[f"{a}_velocity"]**2 / s[f"inverse_inertia_{a}"] for a in ("roll", "pitch", "yaw")),
        "energy_chassis_gravity_j": -s["gravity"] * s["position_y"] / s["inverse_mass"],
        "energy_engine_rotation_j": half * effective_engine_inertia * s["engine_angular_speed"]**2,
        **{f"energy_wheel_rotation_{corner}_j": half * s["wheel_inertia"] * s[f"wheel_omega_{corner}"]**2 for corner in CORNERS},
        **{f"energy_differential_rotation_{axle}_j": half *
           (s["differential_brake_rotor_inertia"] + s[f"external_differential_inertia_{axle}"]) *
           s[f"differential_wrench_shaft_omega_{axle}"]**2 for axle in ("front", "rear")},
    }
    assert tuple(expressions) == ENERGY_OUTPUTS
    return expressions
