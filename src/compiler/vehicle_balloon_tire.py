"""Compiler-owned pressurized membrane tyre kernels.

Runtime tyre geometry is the closed triangle skin carried by ``positions``
and ``velocities``. Material, gas, bead, and hard-surface contact equations are
authored once as SymPy and lowered through the repository process graph/SSA
pipeline; native and browser backends consume that same graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, reduce
import math
from typing import Any, Mapping

import sympy

from .ssa_c_backend import CFunctionArtifact, emit_ssa_function_to_c
from .symbolic_equation_compiler import (
    SymbolicEquationCompilation,
    SymbolicPublication,
    compile_sympy_equations,
)


@dataclass(frozen=True, slots=True)
class BalloonTireTopology:
    """Compile-static closed skin topology and its reference coordinates."""

    circumferential_segments: int
    section_segments: int
    rest_positions: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    face_zones: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    bead_rings: tuple[tuple[int, ...], tuple[int, ...]]
    face_rest_data: tuple[tuple[float, float, float, float, float,
                                float, float, float], ...]
    reference_volume_m3: float
    material_coordinates_uv: tuple[tuple[float, float], ...]
    natural_position_uv: tuple[tuple[float, float, float], ...]
    face_material_uv: tuple[tuple[float, float], ...]
    face_natural_jacobian_uv: tuple[tuple[float, float, float,
                                           float, float, float], ...]
    flexible_face_mask: tuple[bool, ...]
    rim_closure_face_mask: tuple[bool, ...]
    rest_surface_kind: str


def _signed_volume(
    positions: tuple[tuple[float, float, float], ...],
    faces: tuple[tuple[int, int, int], ...],
) -> float:
    volume6 = 0.0
    for ia, ib, ic in faces:
        a, b, c = positions[ia], positions[ib], positions[ic]
        volume6 += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
    return volume6 / 6.0


def _sample_molded_casing_uv_field(
    q: float,
    *,
    tire_radius_m: float,
    rim_radius_m: float,
    section_width_m: float,
) -> tuple[float, float]:
    """Sample the authored UV rest-position field from bead to bead.

    ``q`` is -1 at the inboard bead, zero at tread center, and +1 at the
    outboard bead. The tread is a cylindrical band: every tread row has the
    same wheel radius. Each sidewall is a cubic molded panel that leaves the
    shoulder smoothly, reaches the specified section width, then returns to
    the bead seat. This is the actual zero-bending reference, not a render
    profile layered over a toroidal solver surface. The sampled positions are
    differentiated below to produce the rotation-invariant natural Jacobian
    metric consumed by the membrane law.
    """

    tread_fraction = 0.42
    tread_half_width = 0.36 * section_width_m
    casing_half_width = 0.50 * section_width_m
    bead_half_width = 0.41 * section_width_m
    sign = -1.0 if q < 0.0 else 1.0
    magnitude = abs(float(q))
    if magnitude <= tread_fraction:
        axial = tread_half_width * magnitude / tread_fraction
        return tire_radius_m, sign * axial

    s = (magnitude - tread_fraction) / (1.0 - tread_fraction)
    one_minus = 1.0 - s
    # Cubic Bezier control points in the radial/positive-axial section plane.
    radial_control = (
        tire_radius_m,
        tire_radius_m,
        rim_radius_m + 0.045,
        rim_radius_m,
    )
    axial_control = (
        tread_half_width,
        casing_half_width,
        casing_half_width,
        bead_half_width,
    )
    weights = (
        one_minus ** 3,
        3.0 * one_minus ** 2 * s,
        3.0 * one_minus * s ** 2,
        s ** 3,
    )
    radial = sum(weight * value for weight, value in zip(
        weights, radial_control))
    axial = sum(weight * value for weight, value in zip(
        weights, axial_control))
    return radial, sign * axial


def build_balloon_tire_topology(
    *,
    major_radius_m: float,
    section_radius_m: float,
    circumferential_segments: int = 16,
    section_segments: int = 8,
    pneumatic_mode: str = "tube",
    rim_radius_m: float | None = None,
    section_width_m: float | None = None,
    mold_profile: str = "smooth-casing",
) -> BalloonTireTopology:
    """Build an outward-wound pressure skin and invariant rest surface.

    Both counts are compile-time choices.  Pressure, material, mass and bead
    properties remain runtime JSON parameters.  Two rows around the inner
    circumference form the mechanically attached left/right bead rings.
    """

    if major_radius_m <= section_radius_m or section_radius_m <= 0.0:
        raise ValueError("balloon tyre requires major_radius > section_radius > 0")
    if circumferential_segments < 6 or section_segments < 6:
        raise ValueError("balloon tyre requires at least 6 x 6 skin segments")
    if section_segments % 2:
        raise ValueError("section segment count must be even for paired beads")

    if pneumatic_mode not in {"tube", "tubeless"}:
        raise ValueError("pneumatic mode must be tube or tubeless")
    if mold_profile not in {"smooth-casing", "cheap-commercial-retread"}:
        raise ValueError(f"unknown tire mold profile {mold_profile!r}")
    if pneumatic_mode == "tubeless":
        if rim_radius_m is None or not (
                0.0 < rim_radius_m < major_radius_m + section_radius_m):
            raise ValueError("tubeless casing requires a physical rim radius")
        if section_width_m is None:
            section_width_m = 2.0 * section_radius_m
        if section_width_m <= 0.0:
            raise ValueError("tubeless casing requires a positive section width")
        section_rows = section_segments + 1
        section_values = tuple(
            -1.0 + 2.0 * iv / section_segments
            for iv in range(section_rows))
    else:
        section_rows = section_segments
        section_values = tuple(
            2.0 * math.pi * iv / section_segments
            for iv in range(section_rows))

    cross_section = []
    if pneumatic_mode == "tubeless":
        tire_radius_m = major_radius_m + section_radius_m
        cross_section = [
            _sample_molded_casing_uv_field(
                q,
                tire_radius_m=tire_radius_m,
                rim_radius_m=float(rim_radius_m),
                section_width_m=float(section_width_m),
            )
            for q in section_values
        ]
    else:
        cross_section = [
            (major_radius_m + section_radius_m * math.cos(v),
             section_radius_m * math.sin(v))
            for v in section_values
        ]
    section_arclength = [0.0]
    for (r0, z0), (r1, z1) in zip(cross_section, cross_section[1:]):
        section_arclength.append(
            section_arclength[-1] + math.hypot(r1 - r0, z1 - z0))

    positions: list[tuple[float, float, float]] = []
    material_uv: list[tuple[float, float]] = []
    for iu in range(circumferential_segments):
        u = 2.0 * math.pi * iu / circumferential_segments
        for iv, (radial, axial) in enumerate(cross_section):
            positions.append((
                radial * math.cos(u),
                radial * math.sin(u),
                axial,
            ))
            material_uv.append((
                major_radius_m * u,
                section_arclength[iv],
            ))

    def vertex(iu: int, iv: int) -> int:
        section_index = (iv if pneumatic_mode == "tubeless"
                         else iv % section_rows)
        return ((iu % circumferential_segments) * section_rows
                + section_index)

    faces: list[tuple[int, int, int]] = []
    face_zones: list[str] = []
    flexible_face_mask: list[bool] = []
    rim_closure_face_mask: list[bool] = []
    flexible_intervals = (section_segments if pneumatic_mode == "tubeless"
                          else section_rows)
    for iu in range(circumferential_segments):
        for iv in range(flexible_intervals):
            a = vertex(iu, iv)
            b = vertex(iu + 1, iv)
            c = vertex(iu + 1, iv + 1)
            d = vertex(iu, iv + 1)
            faces.extend(((a, b, c), (a, c, d)))
            section_midpoint = (
                (section_values[iv] + section_values[iv + 1]) * 0.5
                if pneumatic_mode == "tubeless" else
                2.0 * math.pi * (iv + 0.5) / section_segments)
            radial_fraction = (math.cos(section_midpoint)
                               if pneumatic_mode == "tube" else 0.0)
            if pneumatic_mode == "tubeless":
                q_mid = -1.0 + 2.0 * (iv + 0.5) / section_segments
                zone = ("bead" if (iv in {0, flexible_intervals - 1}
                                      or abs(q_mid) >= 0.90) else
                        "tread" if abs(q_mid) <= 0.42 else "sidewall")
            else:
                zone = ("bead" if pneumatic_mode == "tubeless" and
                        iv in {0, flexible_intervals - 1} else
                        "tread" if radial_fraction >= 0.5 else
                        "bead" if radial_fraction <= -0.5 else "sidewall")
            face_zones.extend((zone, zone))
            flexible_face_mask.extend((True, True))
            rim_closure_face_mask.extend((False, False))
    if pneumatic_mode == "tubeless":
        for iu in range(circumferential_segments):
            lower = vertex(iu, 0)
            lower_next = vertex(iu + 1, 0)
            upper = vertex(iu, section_segments)
            upper_next = vertex(iu + 1, section_segments)
            faces.extend(((lower, upper_next, lower_next),
                          (lower, upper, upper_next)))
            face_zones.extend(("rim-closure", "rim-closure"))
            flexible_face_mask.extend((False, False))
            rim_closure_face_mask.extend((True, True))

    edge_counts: dict[tuple[int, int], int] = {}
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (a, b) if a < b else (b, a)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    if any(count != 2 for count in edge_counts.values()):
        raise AssertionError("generated tyre skin is not a closed two-manifold")

    inner = section_segments // 2
    bead_rows = ((0, section_segments) if pneumatic_mode == "tubeless" else
                 ((inner - 1) % section_segments,
                  (inner + 1) % section_segments))
    bead_rings = tuple(
        tuple(vertex(iu, row) for iu in range(circumferential_segments))
        for row in bead_rows
    )
    rest = tuple(positions)
    face_tuple = tuple(faces)
    face_rest_data: list[tuple[float, float, float, float, float,
                               float, float, float]] = []
    face_material_uv: list[tuple[float, float]] = []
    face_natural_jacobian_uv: list[tuple[float, float, float,
                                         float, float, float]] = []
    u_period = 2.0 * math.pi * major_radius_m
    v_period = 2.0 * math.pi * section_radius_m
    for face_index, (ia, ib, ic) in enumerate(face_tuple):
        a, b, c = rest[ia], rest[ib], rest[ic]
        e1 = tuple(b[k] - a[k] for k in range(3))
        e2 = tuple(c[k] - a[k] for k in range(3))
        ua, va = material_uv[ia]

        def periodic_delta(value: float, origin: float) -> float:
            delta = value - origin
            if delta > 0.5 * u_period:
                delta -= u_period
            elif delta < -0.5 * u_period:
                delta += u_period
            return delta

        du1 = periodic_delta(material_uv[ib][0], ua)
        du2 = periodic_delta(material_uv[ic][0], ua)
        dv1 = material_uv[ib][1] - va
        dv2 = material_uv[ic][1] - va
        if pneumatic_mode == "tube":
            if dv1 > 0.5 * v_period:
                dv1 -= v_period
            elif dv1 < -0.5 * v_period:
                dv1 += v_period
            if dv2 > 0.5 * v_period:
                dv2 -= v_period
            elif dv2 < -0.5 * v_period:
                dv2 += v_period
        determinant = du1 * dv2 - du2 * dv1
        if abs(determinant) <= 1.0e-15:
            raise AssertionError("tire UV face has a singular material Jacobian")
        inv00, inv01 = dv2 / determinant, -du2 / determinant
        inv10, inv11 = -dv1 / determinant, du1 / determinant
        j_u = tuple(e1[k] * inv00 + e2[k] * inv10 for k in range(3))
        j_v = tuple(e1[k] * inv01 + e2[k] * inv11 for k in range(3))
        face_material_uv.append((
            ua + (du1 + du2) / 3.0,
            va + (dv1 + dv2) / 3.0,
        ))
        face_natural_jacobian_uv.append((*j_u, *j_v))
        g00 = sum(value * value for value in j_u)
        g01 = sum(j_u[k] * j_v[k] for k in range(3))
        g11 = sum(value * value for value in j_v)
        uv_area = abs(determinant) / 2.0
        area = uv_area * math.sqrt(max(1.0e-24, g00 * g11 - g01 * g01))
        face_rest_data.append((
            inv00, inv01, inv10, inv11, area, g00, g01, g11))
    volume = _signed_volume(rest, face_tuple)
    if volume <= 0.0:
        raise AssertionError("generated tyre skin winding is not outward")
    return BalloonTireTopology(
        circumferential_segments=circumferential_segments,
        section_segments=section_segments,
        rest_positions=rest,
        faces=face_tuple,
        face_zones=tuple(face_zones),
        edges=tuple(sorted(edge_counts)),
        bead_rings=(bead_rings[0], bead_rings[1]),
        face_rest_data=tuple(face_rest_data),
        reference_volume_m3=volume,
        material_coordinates_uv=tuple(material_uv),
        natural_position_uv=rest,
        face_material_uv=tuple(face_material_uv),
        face_natural_jacobian_uv=tuple(face_natural_jacobian_uv),
        flexible_face_mask=tuple(flexible_face_mask),
        rim_closure_face_mask=tuple(rim_closure_face_mask),
        rest_surface_kind=("open-uv-casing-plus-rigid-rim-volume-closure"
                           if pneumatic_mode == "tubeless" else
                           "closed-tube-membrane"),
    )


def balloon_tire_graph_abi(config: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the complete per-wheel graph state and runtime parameter ABI."""

    tires = config["tires"]
    skin = config["tire_skin"]
    topology = build_balloon_tire_topology(
        major_radius_m=float(tires["radius"]) - float(tires["toroid_section_radius_m"]),
        section_radius_m=float(tires["toroid_section_radius_m"]),
        circumferential_segments=int(skin["circumferential_segments"]),
        section_segments=int(skin["section_segments"]),
        pneumatic_mode=str(skin["pneumatic_mode"]),
        rim_radius_m=float(config["wheels"]["rim_radius"]),
        section_width_m=float(tires["width"]),
        mold_profile=("cheap-commercial-retread" if
                      str(skin.get("material_profile", "")) ==
                      "cheap-commercial-retread" else "smooth-casing"),
    )
    state = tuple(
        f"skin_{quantity}_{vertex}_{axis}"
        for quantity in ("position", "velocity")
        for vertex in range(len(topology.rest_positions))
        for axis in "xyz"
    )
    parameters = {
        "vertex_mass_kg": float(config["drivetrain"]["tire_mass_kg"])
        / len(topology.rest_positions),
        "reference_pressure_pa": float(tires["pressure_pa"]),
        "reference_volume_m3": topology.reference_volume_m3,
        "gas_polytropic_exponent": float(tires["gas_polytropic_exponent"]),
        "reference_temperature_k": float(tires["reference_temperature_k"]),
        "gas_molar_mass_kg_per_mol": float(tires["gas_molar_mass_kg_per_mol"]),
        "gas_specific_heat_ratio": float(tires["gas_specific_heat_ratio"]),
        "membrane_gas_permeability_mol_m_per_m2_s_pa": float(
            tires["membrane_gas_permeability_mol_m_per_m2_s_pa"]
        ),
        "gas_permeability_activation_energy_j_per_mol": float(
            tires["gas_permeability_activation_energy_j_per_mol"]
        ),
        **{
            name: float(value)
            for name, value in skin.items()
            if name not in {"model", "pneumatic_mode", "material_profile",
                            "circumferential_segments", "section_segments"}
        },
        "friction_coefficient": float(tires["static_friction"]),
    }
    layer_stacks = {
        zone: _tire_layer_stack(skin, zone)
        for zone in ("tread", "sidewall", "bead")
    }
    pneumatic_mode = str(skin["pneumatic_mode"])
    return {
        "identity": "compiled-balloon-skin-v1",
        "collision_authority": "deformed-skin-vertex-triangle-ccd",
        "rest_geometry_authority": "uv-natural-metric-independent-of-current-embedding",
        "shell_surface_authority": {
            "state_surface": "single-invariant-center-surface",
            "position_dofs": "one-position-per-vertex-shared-by-both-sides",
            "exterior_side": "positive-outward-winding-normal",
            "interior_side": "negative-outward-winding-normal",
            "thickness_rule": (
                "constitutive-thickness-is-centered-on-state-surface-and-does-"
                "not-create-independent-inner-or-outer-position-state"
            ),
            "stretch_reference": "per-face-uv-natural-metric",
            "rest_field": "uv-sampled-natural-position-and-jacobian",
            "constitutive_field": (
                "uv-sampled-directional-laminate-coefficients"),
            "material_axes": {
                "u": "circumferential", "v": "bead-to-bead"},
        },
        "state": state,
        "parameters": parameters,
        "vertex_count": len(topology.rest_positions),
        "face_count": len(topology.faces),
        "edge_count": len(topology.edges),
        "bead_vertex_count": sum(map(len, topology.bead_rings)),
        "topology": topology,
        "material_zones": tuple(sorted(set(topology.face_zones))),
        "pneumatic_mode": pneumatic_mode,
        "layer_stacks": layer_stacks,
        "thermal_state": {
            "field": "temperature_k",
            "location": "per-face-per-layer",
            "initial_temperature_k": float(skin["ambient_temperature_k"]),
            "sources": ("viscoelastic_loss", "contact_hysteresis", "bead_friction"),
            "couplings": ("gas_temperature", "pressure", "rubber_modulus", "friction"),
        },
        "rim_boundary": {
            "owner": "rim",
            "material": "steel",
            "rigid": True,
            "closes_pressure_volume": pneumatic_mode == "tubeless",
            "pressure_membrane": "inner-liner" if pneumatic_mode == "tubeless" else "tube",
            "bead_retention": "pressure-seated-non-bolted",
            "radius_m": float(config["wheels"]["rim_radius"]),
            "width_m": float(tires["width"]),
        },
    }


def _tire_layer_stack(skin: Mapping[str, Any], zone: str) -> tuple[dict[str, Any], ...]:
    """Describe one oriented thermo-mechanical laminate without flattening it."""

    thickness = float(skin["skin_thickness_m"]) * float(
        skin[f"{zone}_thickness_scale"])
    bias = math.radians(float(skin["composite_bias_angle_deg"]))

    def layer(identity: str, material: str, fraction: float, orientation: float,
              *, density: float, heat_capacity: float,
              conductivity: float, expansion: float) -> dict[str, Any]:
        return {
            "identity": identity,
            "material": material,
            "thickness_m": thickness * fraction,
            "orientation_rad": orientation,
            "density_kg_m3": density,
            "specific_heat_j_per_kg_k": heat_capacity,
            "thermal_conductivity_w_per_m_k": conductivity,
            "thermal_expansion_per_k": expansion,
        }

    rubber = {
        "density": float(skin["rubber_density_kg_m3"]),
        "heat_capacity": float(skin["rubber_specific_heat_j_per_kg_k"]),
        "conductivity": float(skin["rubber_thermal_conductivity_w_per_m_k"]),
        "expansion": float(skin["rubber_thermal_expansion_per_k"]),
    }
    composite = {
        "density": float(skin["composite_density_kg_m3"]),
        "heat_capacity": float(skin["composite_specific_heat_j_per_kg_k"]),
        "conductivity": float(skin["composite_thermal_conductivity_w_per_m_k"]),
        "expansion": float(skin["rubber_thermal_expansion_per_k"]) * 0.25,
    }
    layers = [
        layer("outer-rubber", "rubber", 0.48, 0.0, **rubber),
        layer("positive-bias-cord", "composite-cord", 0.16, bias, **composite),
        layer("negative-bias-cord", "composite-cord", 0.16, -bias, **composite),
    ]
    if zone == "bead":
        layers.append(layer(
            "steel-bead-reinforcement", "steel", 0.12, math.pi / 2.0,
            density=float(skin["steel_density_kg_m3"]),
            heat_capacity=float(skin["steel_specific_heat_j_per_kg_k"]),
            conductivity=float(skin["steel_thermal_conductivity_w_per_m_k"]),
            expansion=12.0e-6,
        ))
        inner_fraction = 0.08
    else:
        inner_fraction = 0.20
    if skin["pneumatic_mode"] == "tube":
        tube_fraction = min(0.35, float(skin["tube_thickness_m"]) / thickness)
        layers.append(layer(
            "inner-tube", "tube-rubber", tube_fraction, 0.0,
            density=float(skin["tube_density_kg_m3"]),
            heat_capacity=float(skin["tube_specific_heat_j_per_kg_k"]),
            conductivity=float(skin["tube_thermal_conductivity_w_per_m_k"]),
            expansion=float(skin["rubber_thermal_expansion_per_k"]),
        ))
    else:
        layers.append(layer("inner-liner", "low-permeability-rubber",
                            inner_fraction, 0.0, **rubber))
    return tuple(layers)


def _symbols(names: str) -> dict[str, sympy.Symbol]:
    return {name: sympy.Symbol(name, real=True) for name in names.split()}


def _vec(s: dict[str, sympy.Symbol], prefix: str) -> sympy.Matrix:
    return sympy.Matrix([s[f"{prefix}_{axis}"] for axis in "xyz"])


def symbolic_balloon_membrane_face_equations(
) -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """One triangle's StVK membrane, Kelvin loss, pressure, and volume terms.

    Mesh assembly scatters these face wrenches onto shared vertices.  That
    scatter is topology, not an alternative mechanical law.  The elastic and
    viscous terms are gradients of energy and dissipation potentials, so their
    internal resultant and moment vanish after closed-mesh assembly.
    """

    names = """
        x0_x x0_y x0_z x1_x x1_y x1_z x2_x x2_y x2_z
        v0_x v0_y v0_z v1_x v1_y v1_z v2_x v2_y v2_z
        rest_inverse_00 rest_inverse_01 rest_inverse_10 rest_inverse_11
        rest_area_m2 skin_thickness_m lame_lambda_pa lame_mu_pa
        membrane_damping_lambda_pa_s membrane_damping_mu_pa_s gas_pressure_pa
        natural_metric_00 natural_metric_01 natural_metric_11
        orthotropic_q11_pa orthotropic_q22_pa orthotropic_q12_pa
        orthotropic_q66_pa orthotropic_q16_pa orthotropic_q26_pa
        reference_pressure_pa
        r0_x r0_y r0_z r1_x r1_y r1_z r2_x r2_y r2_z
    """
    s = _symbols(names)
    x0, x1, x2 = (_vec(s, f"x{i}") for i in range(3))
    r0, r1, r2 = (_vec(s, f"r{i}") for i in range(3))
    v0, v1, v2 = (_vec(s, f"v{i}") for i in range(3))
    e1, e2 = x1 - x0, x2 - x0
    de1, de2 = v1 - v0, v2 - v0
    inv = sympy.Matrix([
        [s["rest_inverse_00"], s["rest_inverse_01"]],
        [s["rest_inverse_10"], s["rest_inverse_11"]],
    ])
    gram = sympy.Matrix([
        [e1.dot(e1), e1.dot(e2)],
        [e2.dot(e1), e2.dot(e2)],
    ])
    gram_rate = sympy.Matrix([
        [2 * e1.dot(de1), de1.dot(e2) + e1.dot(de2)],
        [de2.dot(e1) + e2.dot(de1), 2 * e2.dot(de2)],
    ])
    natural_metric = sympy.Matrix([
        [s["natural_metric_00"], s["natural_metric_01"]],
        [s["natural_metric_01"], s["natural_metric_11"]],
    ])
    strain = (inv.T * gram * inv - natural_metric) / 2
    strain_rate = inv.T * gram_rate * inv / 2
    tr_e = sympy.trace(strain)
    tr_edot = sympy.trace(strain_rate)
    scale = s["skin_thickness_m"] * s["rest_area_m2"]
    energy = scale * (
        s["lame_mu_pa"] * sympy.trace(strain * strain)
        + s["lame_lambda_pa"] * tr_e ** 2 / 2
    )
    engineering_shear = 2 * strain[0, 1]
    energy += scale * (
        s["orthotropic_q11_pa"] * strain[0, 0] ** 2
        + s["orthotropic_q22_pa"] * strain[1, 1] ** 2
        + 2 * s["orthotropic_q12_pa"] * strain[0, 0] * strain[1, 1]
        + s["orthotropic_q66_pa"] * engineering_shear ** 2
        + 2 * s["orthotropic_q16_pa"] * strain[0, 0] * engineering_shear
        + 2 * s["orthotropic_q26_pa"] * strain[1, 1] * engineering_shear
    ) / 2
    rayleigh = scale * (
        s["membrane_damping_mu_pa_s"] * sympy.trace(strain_rate * strain_rate)
        + s["membrane_damping_lambda_pa_s"] * tr_edot ** 2 / 2
    )
    cross = e1.cross(e2)
    reference_cross = (r1 - r0).cross(r2 - r0)
    pressure_vertex_force = s["gas_pressure_pa"] * cross / 6
    # The authored topology is the *inflated* reference shape.  Its carcass
    # therefore carries construction prestress at reference pressure; treating
    # it as an unstressed rubber sheet would inject a large startup impulse.
    # This conservative linearized prestress potential makes the discrete
    # pressure and construction loads cancel face-for-face at the reference
    # state, while StVK supplies deformation stiffness around that state.
    construction_vertex_force = -s["reference_pressure_pa"] * reference_cross / 6
    construction_potential = -construction_vertex_force.dot(x0 + x1 + x2)
    position_scalars = [component for vector in (x0, x1, x2) for component in vector]
    velocity_scalars = [component for vector in (v0, v1, v2) for component in vector]
    elastic = [-sympy.diff(energy, q) for q in position_scalars]
    damping = [-sympy.diff(rayleigh, qdot) for qdot in velocity_scalars]
    pressure = [pressure_vertex_force[axis] for _vertex in range(3) for axis in range(3)]
    construction = [
        construction_vertex_force[axis]
        for _vertex in range(3) for axis in range(3)
    ]
    # Do not ask SymPy to globally simplify these derivatives.  The compiler's
    # graph interning/CSE retains the shared strain terms; symbolic expansion
    # here makes a single face needlessly expensive to author and compile.
    total = [a + b + c + d for a, b, c, d in zip(
        elastic, damping, pressure, construction,
    )]
    damping_power = sum(f * v for f, v in zip(damping, velocity_scalars))
    outputs: dict[str, sympy.Expr] = {
        "strain_energy_j": energy,
        "dissipation_power_w": damping_power,
        "construction_prestress_potential_j": construction_potential,
        "signed_volume_contribution_m3": x0.dot(x1.cross(x2)) / 6,
        "double_area_m2": sympy.sqrt(cross.dot(cross)),
    }
    for vertex in range(3):
        for axis_index, axis in enumerate("xyz"):
            lane = 3 * vertex + axis_index
            outputs[f"force_{vertex}_{axis}_n"] = total[lane]
            outputs[f"elastic_force_{vertex}_{axis}_n"] = elastic[lane]
            outputs[f"damping_force_{vertex}_{axis}_n"] = damping[lane]
            outputs[f"pressure_force_{vertex}_{axis}_n"] = pressure[lane]
            outputs[f"construction_force_{vertex}_{axis}_n"] = construction[lane]
    equations = tuple(
        sympy.Eq(sympy.Symbol(name, real=True), value, evaluate=False)
        for name, value in outputs.items()
    )
    return equations, s


@lru_cache(maxsize=1)
def compile_balloon_membrane_face_ssa() -> SymbolicEquationCompilation:
    equations, _ = symbolic_balloon_membrane_face_equations()
    return compile_sympy_equations(
        equations,
        name="balloon_tire_membrane_face",
        publications=tuple(
            SymbolicPublication(str(eq.lhs), f"world.vehicle.tire.skin.{eq.lhs}")
            for eq in equations
        ),
        dtype="float64",
    )


def symbolic_balloon_gas_equations(
) -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    s = _symbols(
        "reference_pressure_pa reference_volume_m3 current_volume_m3 "
        "gas_polytropic_exponent minimum_volume_fraction reference_temperature_k"
    )
    minimum_volume = s["reference_volume_m3"] * s["minimum_volume_fraction"]
    safe_volume = sympy.Max(s["current_volume_m3"], minimum_volume)
    pressure = s["reference_pressure_pa"] * (
        s["reference_volume_m3"] / safe_volume
    ) ** s["gas_polytropic_exponent"]
    temperature = s["reference_temperature_k"] * (
        s["reference_volume_m3"] / safe_volume
    ) ** (s["gas_polytropic_exponent"] - 1)
    equations = (
        sympy.Eq(sympy.Symbol("gas_pressure_pa"), pressure, evaluate=False),
        sympy.Eq(
            sympy.Symbol("volume_ratio"),
            safe_volume / s["reference_volume_m3"],
            evaluate=False,
        ),
        sympy.Eq(sympy.Symbol("gas_temperature_k"), temperature, evaluate=False),
    )
    return equations, s


@lru_cache(maxsize=1)
def compile_balloon_gas_ssa() -> SymbolicEquationCompilation:
    equations, _ = symbolic_balloon_gas_equations()
    return compile_sympy_equations(equations, name="balloon_tire_gas", dtype="float64")


def symbolic_balloon_bead_constraint_equations(
) -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """A bead vertex/rim attachment with an explicit equal/opposite wrench."""

    s = _symbols(
        "vertex_x vertex_y vertex_z vertex_velocity_x vertex_velocity_y vertex_velocity_z "
        "target_x target_y target_z target_velocity_x target_velocity_y target_velocity_z "
        "rim_center_x rim_center_y rim_center_z bead_stiffness_n_per_m bead_damping_n_s_per_m"
    )
    vertex = _vec(s, "vertex")
    velocity = _vec(s, "vertex_velocity")
    target = _vec(s, "target")
    target_velocity = _vec(s, "target_velocity")
    rim_center = _vec(s, "rim_center")
    force = -s["bead_stiffness_n_per_m"] * (vertex - target) - s[
        "bead_damping_n_s_per_m"
    ] * (velocity - target_velocity)
    rim_force = -force
    rim_moment = (target - rim_center).cross(rim_force)
    outputs = {}
    for axis_index, axis in enumerate("xyz"):
        outputs[f"skin_force_{axis}_n"] = force[axis_index]
        outputs[f"rim_force_{axis}_n"] = rim_force[axis_index]
        outputs[f"rim_moment_{axis}_nm"] = rim_moment[axis_index]
    equations = tuple(
        sympy.Eq(sympy.Symbol(name), value, evaluate=False)
        for name, value in outputs.items()
    )
    return equations, s


@lru_cache(maxsize=1)
def compile_balloon_bead_constraint_ssa() -> SymbolicEquationCompilation:
    equations, _ = symbolic_balloon_bead_constraint_equations()
    return compile_sympy_equations(equations, name="balloon_tire_bead_constraint", dtype="float64")


def symbolic_balloon_bead_implicit_step_equations(
) -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """Backward-Euler bead attachment update and its equal/opposite wrench.

    The continuous law is exactly the bead Kelvin--Voigt law above.  Solving
    that linear junction analytically at the timestep boundary prevents the
    very stiff damping term from becoming an explicit numerical energy source.
    ``free_velocity`` already includes membrane, gas, gravity, and other
    external forces, so this operator only closes the rim/skin graph edge.
    """

    s = _symbols(
        "dt vertex_mass_kg vertex_x vertex_y vertex_z "
        "free_velocity_x free_velocity_y free_velocity_z "
        "target_x target_y target_z target_velocity_x target_velocity_y target_velocity_z "
        "rim_center_x rim_center_y rim_center_z bead_stiffness_n_per_m bead_damping_n_s_per_m"
    )
    vertex = _vec(s, "vertex")
    free_velocity = _vec(s, "free_velocity")
    target = _vec(s, "target")
    target_velocity = _vec(s, "target_velocity")
    rim_center = _vec(s, "rim_center")
    dt = s["dt"]
    mass = s["vertex_mass_kg"]
    stiffness = s["bead_stiffness_n_per_m"]
    damping = s["bead_damping_n_s_per_m"]
    displacement = vertex - target
    relative_free_velocity = free_velocity - target_velocity
    denominator = 1 + dt * damping / mass + dt ** 2 * stiffness / mass
    relative_velocity = (
        relative_free_velocity - dt * stiffness * displacement / mass
    ) / denominator
    velocity = target_velocity + relative_velocity
    position = vertex + dt * velocity
    # The rim wrench is -impulse/dt, but the impulse is itself O(dt):
    #   m*(velocity - free_velocity) = -(dt/D)[(c + dt*k)*v_rel + k*x]
    # so the dt cancels ANALYTICALLY and the wrench is
    #   [(c + dt*k)*v_rel + k*x] / D
    # -- finite at dt == 0, where it is exactly the continuous
    # Kelvin--Voigt force c*v_rel + k*x. The former -impulse/dt spelling
    # evaluated the removable singularity as 0/0 and reported NaN for a
    # zero-duration window; NaN is not a value.
    rim_force = (
        (damping + dt * stiffness) * relative_free_velocity
        + stiffness * displacement
    ) / denominator
    rim_moment = (target - rim_center).cross(rim_force)
    outputs: dict[str, sympy.Expr] = {}
    for axis_index, axis in enumerate("xyz"):
        outputs[f"position_{axis}_next"] = position[axis_index]
        outputs[f"velocity_{axis}_next"] = velocity[axis_index]
        outputs[f"rim_force_{axis}_n"] = rim_force[axis_index]
        outputs[f"rim_moment_{axis}_nm"] = rim_moment[axis_index]
    return tuple(
        sympy.Eq(sympy.Symbol(name), value, evaluate=False)
        for name, value in outputs.items()
    ), s


@lru_cache(maxsize=1)
def compile_balloon_bead_implicit_step_ssa() -> SymbolicEquationCompilation:
    equations, _ = symbolic_balloon_bead_implicit_step_equations()
    return compile_sympy_equations(
        equations, name="balloon_tire_bead_implicit_step", dtype="float64"
    )


@lru_cache(maxsize=1)
def balloon_tire_symbolic_compilations(
) -> Mapping[str, SymbolicEquationCompilation]:
    """Return the symbolic authorities linked by the authored tyre program.

    These exact compilation objects are the common boundary for eager Python
    execution and compiler lowering.  Python lambdifies their retained SymPy
    equations; native targets link their retained process graphs.
    """

    return {
        "balloon_tire_gas": compile_balloon_gas_ssa(),
        "balloon_tire_membrane_face": compile_balloon_membrane_face_ssa(),
        "balloon_tire_bead_implicit_step": compile_balloon_bead_implicit_step_ssa(),
    }


def balloon_tire_linked_process_graphs() -> dict[str, Any]:
    """Expose the canonical symbolic authorities to compiler lowering."""

    return {
        name: compilation.process_graph
        for name, compilation in balloon_tire_symbolic_compilations().items()
    }


def _elementwise_maximum(*values: Any) -> Any:
    def combine(left, right):
        if hasattr(left, "maximum"):
            return left.maximum(right)
        if hasattr(right, "maximum"):
            return right.maximum(left)
        return max(left, right)
    return reduce(combine, values)


def _elementwise_minimum(*values: Any) -> Any:
    def combine(left, right):
        if hasattr(left, "minimum"):
            return left.minimum(right)
        if hasattr(right, "minimum"):
            return right.minimum(left)
        return min(left, right)
    return reduce(combine, values)


def _python_unary(value: Any, method: str, scalar) -> Any:
    operation = getattr(value, method, None)
    return operation() if operation is not None else scalar(value)


@lru_cache(maxsize=1)
def _balloon_tire_python_bindings_cached() -> Mapping[str, Any]:
    bindings: dict[str, Any] = {}
    modules = [{
        "Max": _elementwise_maximum,
        "Min": _elementwise_minimum,
        "sqrt": lambda value: _python_unary(value, "sqrt", math.sqrt),
        "sin": lambda value: _python_unary(value, "sin", math.sin),
        "cos": lambda value: _python_unary(value, "cos", math.cos),
        "Abs": lambda value: _python_unary(value, "abs", abs),
    }]
    for name, compilation in balloon_tire_symbolic_compilations().items():
        metadata = compilation.function.metadata
        equations_by_name = {
            str(equation.lhs): equation.rhs
            for equation in compilation.equations
        }
        bindings[name] = sympy.lambdify(
            tuple(sympy.Symbol(argument) for argument in metadata["argument_names"]),
            tuple(equations_by_name[output] for output in metadata["output_names"]),
            modules=modules,
            cse=True,
        )
    return bindings


def balloon_tire_python_bindings() -> dict[str, Any]:
    """Lambdify the same equations and ABI that compiler targets consume."""

    return dict(_balloon_tire_python_bindings_cached())


def symbolic_balloon_vertex_triangle_contact_equations(
) -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """Unilateral CCD impulse for a deformed skin vertex and hard triangle.

    There is deliberately no position projection or penetration spring.  The
    selected terrain face is tested at the actual plane crossing; persistent
    resting contact uses the predicted inward velocity.  Contact impulses are
    returned to the graph for equal/opposite application to both bodies.
    """

    s = _symbols(
        "previous_x previous_y previous_z current_x current_y current_z "
        "velocity_x velocity_y velocity_z triangle_a_x triangle_a_y triangle_a_z "
        "triangle_b_x triangle_b_y triangle_b_z triangle_c_x triangle_c_y triangle_c_z "
        "skin_offset_m inverse_effective_mass_per_kg restitution friction_coefficient"
    )
    previous, current, velocity = _vec(s, "previous"), _vec(s, "current"), _vec(s, "velocity")
    a, b, c = _vec(s, "triangle_a"), _vec(s, "triangle_b"), _vec(s, "triangle_c")
    ab, ac = b - a, c - a
    raw_normal = ab.cross(ac)
    normal_length = sympy.sqrt(raw_normal.dot(raw_normal) + sympy.Float("1e-24"))
    normal = raw_normal / normal_length
    previous_distance = normal.dot(previous - a) - s["skin_offset_m"]
    current_distance = normal.dot(current - a) - s["skin_offset_m"]
    denominator = previous_distance - current_distance
    crossing_fraction = sympy.Min(sympy.Max(
        previous_distance / (denominator + sympy.Float("1e-18")), 0
    ), 1)
    crossing = previous + crossing_fraction * (current - previous) - s["skin_offset_m"] * normal
    ap = crossing - a
    d00, d01, d11 = ab.dot(ab), ab.dot(ac), ac.dot(ac)
    d20, d21 = ap.dot(ab), ap.dot(ac)
    bary_denom = d00 * d11 - d01 * d01 + sympy.Float("1e-24")
    bary_v = (d11 * d20 - d01 * d21) / bary_denom
    bary_w = (d00 * d21 - d01 * d20) / bary_denom
    bary_u = 1 - bary_v - bary_w
    inside = sympy.And(bary_u >= 0, bary_v >= 0, bary_w >= 0)
    touching = sympy.And(current_distance <= 0, inside)
    normal_velocity = normal.dot(velocity)
    closing = sympy.Min(normal_velocity, 0)
    normal_impulse_magnitude = sympy.Piecewise((
        -(1 + s["restitution"]) * closing
        / (s["inverse_effective_mass_per_kg"] + sympy.Float("1e-18")), touching
    ), (0, True))
    tangent_velocity = velocity - normal_velocity * normal
    tangent_speed = sympy.sqrt(tangent_velocity.dot(tangent_velocity) + sympy.Float("1e-24"))
    tangent_impulse_uncapped = tangent_speed / (
        s["inverse_effective_mass_per_kg"] + sympy.Float("1e-18")
    )
    tangent_impulse_magnitude = sympy.Min(
        tangent_impulse_uncapped,
        s["friction_coefficient"] * normal_impulse_magnitude,
    )
    impulse = normal_impulse_magnitude * normal - tangent_impulse_magnitude * tangent_velocity / tangent_speed
    outputs: dict[str, sympy.Expr] = {
        "contact_active": sympy.Piecewise((1, touching), (0, True)),
        "time_of_impact_fraction": crossing_fraction,
        "barycentric_u": bary_u,
        "barycentric_v": bary_v,
        "barycentric_w": bary_w,
        "normal_impulse_ns": normal_impulse_magnitude,
    }
    for axis_index, axis in enumerate("xyz"):
        outputs[f"skin_impulse_{axis}_ns"] = impulse[axis_index]
        outputs[f"terrain_impulse_{axis}_ns"] = -impulse[axis_index]
        outputs[f"contact_{axis}_m"] = crossing[axis_index]
    equations = tuple(
        sympy.Eq(sympy.Symbol(name), value, evaluate=False)
        for name, value in outputs.items()
    )
    return equations, s


def symbolic_balloon_contact_geometry_equations(
) -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """CCD branch geometry, separated only to retain compiler CSE boundaries."""

    s = _symbols(
        "previous_x previous_y previous_z current_x current_y current_z "
        "triangle_a_x triangle_a_y triangle_a_z triangle_b_x triangle_b_y triangle_b_z "
        "triangle_c_x triangle_c_y triangle_c_z skin_offset_m"
    )
    previous, current = _vec(s, "previous"), _vec(s, "current")
    a, b, c = _vec(s, "triangle_a"), _vec(s, "triangle_b"), _vec(s, "triangle_c")
    ab, ac = b - a, c - a
    raw_normal = ab.cross(ac)
    normal = raw_normal / sympy.sqrt(raw_normal.dot(raw_normal) + sympy.Float("1e-24"))
    d0 = normal.dot(previous - a) - s["skin_offset_m"]
    d1 = normal.dot(current - a) - s["skin_offset_m"]
    toi = sympy.Min(sympy.Max(d0 / (d0 - d1 + sympy.Float("1e-18")), 0), 1)
    point = previous + toi * (current - previous) - s["skin_offset_m"] * normal
    ap = point - a
    q00, q01, q11 = ab.dot(ab), ab.dot(ac), ac.dot(ac)
    q20, q21 = ap.dot(ab), ap.dot(ac)
    denominator = q00 * q11 - q01 * q01 + sympy.Float("1e-24")
    bv = (q11 * q20 - q01 * q21) / denominator
    bw = (q00 * q21 - q01 * q20) / denominator
    bu = 1 - bv - bw
    outputs: dict[str, sympy.Expr] = {
        "time_of_impact_fraction": toi,
        "previous_signed_distance_m": d0,
        "current_signed_distance_m": d1,
        "barycentric_u": bu,
        "barycentric_v": bv,
        "barycentric_w": bw,
    }
    for index, axis in enumerate("xyz"):
        outputs[f"normal_{axis}"] = normal[index]
        outputs[f"contact_{axis}_m"] = point[index]
    return tuple(
        sympy.Eq(sympy.Symbol(name), value, evaluate=False)
        for name, value in outputs.items()
    ), s


def symbolic_balloon_cylinder_contact_geometry_equations(
) -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """Exact swept point against an infinite hard roller cylinder.

    The cylinder axis is the wheel/roller Z axis.  The quadratic root is the
    first boundary crossing of the skin-offset circle; the returned contact
    point lies on the physical roller radius so the shared impulse integrator
    can restore the skin offset without penetration projection.
    """

    s = _symbols(
        "previous_x previous_y previous_z current_x current_y current_z "
        "cylinder_center_x cylinder_center_y cylinder_center_z "
        "cylinder_radius_m skin_offset_m"
    )
    previous, current = _vec(s, "previous"), _vec(s, "current")
    center = _vec(s, "cylinder_center")
    delta = current - previous
    qx, qy = previous[0] - center[0], previous[1] - center[1]
    radius = s["cylinder_radius_m"] + s["skin_offset_m"]
    qa = delta[0] ** 2 + delta[1] ** 2 + sympy.Float("1e-30")
    qb = 2 * (qx * delta[0] + qy * delta[1])
    qc = qx ** 2 + qy ** 2 - radius ** 2
    discriminant = sympy.Max(qb ** 2 - 4 * qa * qc, 0)
    toi = sympy.Min(sympy.Max(
        (-qb - sympy.sqrt(discriminant)) / (2 * qa), 0), 1)
    swept = previous + toi * delta
    radial_x, radial_y = swept[0] - center[0], swept[1] - center[1]
    radial_length = sympy.sqrt(radial_x ** 2 + radial_y ** 2 + sympy.Float("1e-30"))
    normal = sympy.Matrix((radial_x / radial_length, radial_y / radial_length, 0))
    previous_radius = sympy.sqrt(qx ** 2 + qy ** 2 + sympy.Float("1e-30"))
    current_radius = sympy.sqrt(
        (current[0] - center[0]) ** 2 + (current[1] - center[1]) ** 2
        + sympy.Float("1e-30"))
    contact = sympy.Matrix((
        center[0] + s["cylinder_radius_m"] * normal[0],
        center[1] + s["cylinder_radius_m"] * normal[1],
        swept[2],
    ))
    outputs: dict[str, sympy.Expr] = {
        "time_of_impact_fraction": toi,
        "previous_signed_distance_m": previous_radius - radius,
        "current_signed_distance_m": current_radius - radius,
        "barycentric_u": sympy.Integer(1),
        "barycentric_v": sympy.Integer(1),
        "barycentric_w": sympy.Integer(1),
    }
    for index, axis in enumerate("xyz"):
        outputs[f"normal_{axis}"] = normal[index]
        outputs[f"contact_{axis}_m"] = contact[index]
    return tuple(sympy.Eq(sympy.Symbol(name), value, evaluate=False)
                 for name, value in outputs.items()), s


def symbolic_balloon_contact_impulse_equations(
) -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """Frictional unilateral response consuming the CCD active-set bit.

    The active-set detector compares the compiled geometry outputs
    (distance <= 0 and all barycentrics >= 0).  It is graph control, not a
    second force law; keeping the bit at this ABI boundary lets straight-line
    scalar C and predicated GPU backends consume the exact same response.
    """

    s = _symbols(
        "contact_active normal_x normal_y normal_z velocity_x velocity_y velocity_z "
        "inverse_effective_mass_per_kg restitution friction_coefficient"
    )
    normal, velocity = _vec(s, "normal"), _vec(s, "velocity")
    vn = normal.dot(velocity)
    closing = sympy.Min(vn, 0)
    jn = s["contact_active"] * (-(1 + s["restitution"]) * closing) / (
        s["inverse_effective_mass_per_kg"] + sympy.Float("1e-18")
    )
    vt = velocity - vn * normal
    vt_length = sympy.sqrt(vt.dot(vt) + sympy.Float("1e-24"))
    jt = sympy.Min(
        vt_length / (s["inverse_effective_mass_per_kg"] + sympy.Float("1e-18")),
        s["friction_coefficient"] * jn,
    )
    impulse = jn * normal - jt * vt / vt_length
    outputs: dict[str, sympy.Expr] = {"normal_impulse_ns": jn}
    for index, axis in enumerate("xyz"):
        outputs[f"skin_impulse_{axis}_ns"] = impulse[index]
        outputs[f"terrain_impulse_{axis}_ns"] = -impulse[index]
    return tuple(
        sympy.Eq(sympy.Symbol(name), value, evaluate=False)
        for name, value in outputs.items()
    ), s


@lru_cache(maxsize=1)
def compile_balloon_contact_geometry_ssa() -> SymbolicEquationCompilation:
    equations, _ = symbolic_balloon_contact_geometry_equations()
    return compile_sympy_equations(
        equations, name="balloon_tire_contact_geometry", dtype="float64"
    )


@lru_cache(maxsize=1)
def compile_balloon_cylinder_contact_geometry_ssa() -> SymbolicEquationCompilation:
    equations, _ = symbolic_balloon_cylinder_contact_geometry_equations()
    return compile_sympy_equations(
        equations, name="balloon_tire_cylinder_contact_geometry", dtype="float64"
    )


@lru_cache(maxsize=1)
def compile_balloon_contact_impulse_ssa() -> SymbolicEquationCompilation:
    equations, _ = symbolic_balloon_contact_impulse_equations()
    return compile_sympy_equations(
        equations, name="balloon_tire_contact_impulse", dtype="float64"
    )


@lru_cache(maxsize=1)
def compile_balloon_vertex_triangle_contact_ssa() -> SymbolicEquationCompilation:
    equations, _ = symbolic_balloon_vertex_triangle_contact_equations()
    return compile_sympy_equations(
        equations, name="balloon_tire_vertex_triangle_contact", dtype="float64"
    )


def _compile_c(compilation: SymbolicEquationCompilation) -> CFunctionArtifact:
    artifact = emit_ssa_function_to_c(
        compilation.module, compilation.function.name, entry_name=compilation.function.name
    )
    if not artifact.complete:
        reasons = "; ".join(item.reason for item in artifact.shortfalls)
        raise RuntimeError(f"balloon tyre kernel does not lower to C: {reasons}")
    return artifact


@lru_cache(maxsize=1)
def compile_balloon_membrane_face_c() -> CFunctionArtifact:
    return _compile_c(compile_balloon_membrane_face_ssa())


@lru_cache(maxsize=1)
def compile_balloon_gas_c() -> CFunctionArtifact:
    return _compile_c(compile_balloon_gas_ssa())


@lru_cache(maxsize=1)
def compile_balloon_bead_constraint_c() -> CFunctionArtifact:
    return _compile_c(compile_balloon_bead_constraint_ssa())


@lru_cache(maxsize=1)
def compile_balloon_bead_implicit_step_c() -> CFunctionArtifact:
    return _compile_c(compile_balloon_bead_implicit_step_ssa())


@lru_cache(maxsize=1)
def compile_balloon_vertex_triangle_contact_c() -> CFunctionArtifact:
    return _compile_c(compile_balloon_vertex_triangle_contact_ssa())


@lru_cache(maxsize=1)
def compile_balloon_contact_geometry_c() -> CFunctionArtifact:
    return _compile_c(compile_balloon_contact_geometry_ssa())


@lru_cache(maxsize=1)
def compile_balloon_cylinder_contact_geometry_c() -> CFunctionArtifact:
    return _compile_c(compile_balloon_cylinder_contact_geometry_ssa())


@lru_cache(maxsize=1)
def compile_balloon_contact_impulse_c() -> CFunctionArtifact:
    return _compile_c(compile_balloon_contact_impulse_ssa())
