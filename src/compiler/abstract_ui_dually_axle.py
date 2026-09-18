"""Discoverable loose commercial dually rear-end for the living game world."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .abstract_ui_world import WorldObject
from .abstract_ui_pneumatic_wheel import pneumatic_wheel_assembly
from .mechanical_ports import (
    BearingRuleSet, bearing_interface_node, bearing_race_edge,
    generic_rotational_torque_port, rotating_hub_node,
)


DUALLY_AXLE_SCHEMA = "abstract-ui-commercial-dually-axle-v0"
DUALLY_WHEELS = (
    "left_inner", "left_outer", "right_inner", "right_outer",
)
DUALLY_GROUPS = {
    "left": ("left_inner", "left_outer"),
    "right": ("right_inner", "right_outer"),
}


def _bearing_rules(family: str) -> BearingRuleSet:
    """Concept specifications awaiting fine-model calibration, not catalog claims."""

    common = dict(
        radial_clearance_m=0.00008, axial_float_m=0.00012,
        radial_stiffness_n_m=4.0e8, axial_stiffness_n_m=2.6e8,
        tilt_stiffness_nm_rad=1.6e5, radial_damping_n_s_m=38_000.0,
        axial_damping_n_s_m=28_000.0, tilt_damping_nm_s_rad=1_800.0,
        rolling_friction_coefficient=0.0022, seal_drag_nm=2.5,
        viscous_drag_nm_s_rad=0.035, stribeck_speed_rad_s=3.0,
        temperature_limit_k=423.15,
    )
    variants = {
        "pinion": dict(
            bore_m=0.065, outer_diameter_m=0.140, width_m=0.042,
            contact_angle_deg=24.0, radial_static_limit_n=310_000.0,
            radial_dynamic_limit_n=245_000.0, axial_limit_n=185_000.0,
            speed_limit_rad_s=680.0),
        "carrier": dict(
            bore_m=0.105, outer_diameter_m=0.190, width_m=0.050,
            contact_angle_deg=22.0, radial_static_limit_n=520_000.0,
            radial_dynamic_limit_n=390_000.0, axial_limit_n=270_000.0,
            speed_limit_rad_s=360.0),
        "shaft-journal": dict(
            bore_m=0.062, outer_diameter_m=0.098, width_m=0.038,
            contact_angle_deg=0.0, radial_static_limit_n=180_000.0,
            radial_dynamic_limit_n=145_000.0, axial_limit_n=28_000.0,
            speed_limit_rad_s=420.0, axial_float_m=0.0008,
            seal_drag_nm=1.2),
        "hub-pair": dict(
            bore_m=0.120, outer_diameter_m=0.225, width_m=0.118,
            contact_angle_deg=26.0, radial_static_limit_n=980_000.0,
            radial_dynamic_limit_n=720_000.0, axial_limit_n=510_000.0,
            speed_limit_rad_s=230.0, radial_stiffness_n_m=6.8e8,
            axial_stiffness_n_m=5.2e8, seal_drag_nm=7.5,
            viscous_drag_nm_s_rad=0.065),
    }
    try:
        return BearingRuleSet(**(common | variants[family]))
    except KeyError as error:
        raise ValueError(f"unknown dually bearing family {family!r}") from error
DUALLY_VALIDATOR_STAGES = (
    "position-four-wheel-units-on-eventual-installation-pillars",
    "mount-four-tire-casings",
    "inflate-four-tires-through-wheel-ports",
    "balance-left-and-right-dual-groups",
    "clamp-axle-casing-at-inferred-structural-grasp-frame",
    "install-differential-and-solid-axle-shafts",
    "install-dual-hubs-and-hydraulic-brakes",
    "transfer-prepared-wheels-from-pillars-to-dual-hubs",
    "differential-and-hydraulic-brake-torque-proof",
)


@dataclass(frozen=True, slots=True)
class DuallyAxleAssembly:
    model: Mapping[str, Any]
    world_objects: tuple[WorldObject, ...]


def roadside_dually_axle_assembly(
    root: str, *, center_x: float, center_z: float,
) -> DuallyAxleAssembly:
    """Build one persistent, findable, serviceable commercial rear end."""

    identity = f"{root}/roadside-artifacts/commercial-dually-rear-end-001"
    wheel_radius = 0.535
    wheel_width = 0.285
    hub_z = {"left": -1.36, "right": 1.36}
    wheel_z = {
        "left_inner": -1.24, "left_outer": -1.52,
        "right_inner": 1.24, "right_outer": 1.52,
    }

    casing = {
        "identity": f"{identity}/axle-casing",
        "kind": "load-bearing-full-floating-banjo-axle-casing",
        "material": "ductile-iron-center-with-welded-dom-steel-tubes",
        "center_carrier_outer_diameter_m": 0.64,
        "tube_outer_diameter_m": 0.225,
        "tube_wall_m": 0.026,
        "flange_to_flange_width_m": 3.24,
        "mass_kg": 338.0,
        "lubricant_boundary": {
            "sealed_volume_l": 18.0,
            "service_fill_l": 15.0,
            "fluid": "sae-75w-140-gl-5",
            "removable_differential_cover": True,
            "cover_fasteners": 14,
            "fill_plug": "magnetic-side-fill-plug",
            "drain_plug": "magnetic-bottom-drain-plug",
            "breather": "high-mounted-sintered-axle-breather",
            "shaft_seals": ["left-inner", "right-inner"],
            "hub_seals": ["left-full-floating", "right-full-floating"],
        },
        "mounting": {
            "spring_pads_present": False,
            "torque_arm_brackets_present": False,
            "structural_tube": {
                "section": "welded-dom-steel-axle-tube",
                "load_role": "wheel-end-and-future-suspension-load-path",
                "weld_qualified_zones": [
                    "left-outboard-tube", "left-inboard-tube",
                    "right-inboard-tube", "right-outboard-tube",
                ],
                "keep_clear_zones": [
                    "hub-seal-land", "carrier-weld", "breather",
                    "hydraulic-line-clips", "differential-cover",
                ],
                "attachment_rule": (
                    "future-suspension-brackets-require-a-qualified-weld-"
                    "recipe-with-preheat-and-distortion-control"
                ),
            },
            "structural_grasp_regions": [
                {
                    "identity": f"{identity}/axle-casing/{side}-tube-saddle",
                    "owner": f"{identity}/axle-casing",
                    "center": [0.0, wheel_radius, z],
                    "fore_aft_span_m": 0.34,
                    "rated_vertical_load_n": 260_000.0,
                    "access_fraction": 0.94,
                    "preferred_pinch_axis": [0.0, 1.0, 0.0],
                    "material": "welded-dom-steel-axle-tube",
                    "section_outer_diameter_m": 0.225,
                    "section_wall_m": 0.026,
                    "mount_analogue": (
                        "front-and-rear-u-bolt-or-accessory-saddle-around-"
                        "structural-axle-tube"),
                    "keep_clear": ["hydraulic-line", "breather", "hub-seal"],
                }
                for side, z in (("left", -0.94), ("right", 0.94))
            ],
            "user_mount_surfaces": [
                "left-casing-tube", "right-casing-tube", "carrier-nose",
                "differential-cover-ring", "left-end-flange", "right-end-flange",
            ],
            "reuse_contract": "casing-is-the-structural-mount-and-lubricant-owner",
        },
    }

    wheels = []
    wheel_assemblies = {}
    for wheel in DUALLY_WHEELS:
        side = "left" if wheel.startswith("left") else "right"
        layer = "inner" if wheel.endswith("inner") else "outer"
        wheel_identity = f"{identity}/wheels/{wheel}"
        pneumatic_mode = "tubeless"
        material_profile = "cheap-commercial-retread"
        wheel_assembly = pneumatic_wheel_assembly(
            wheel_identity,
            center=(0.0, wheel_radius, wheel_z[wheel]),
            axis=(0.0, 0.0, 1.0),
            tire_radius_m=wheel_radius,
            section_width_m=wheel_width,
            rim_radius_m=0.28575,
            wheel_mass_kg=41.0,
            casing_mass_kg=66.0,
            rated_pressure_pa=760_000.0,
            pneumatic_mode=pneumatic_mode,
            material_profile=material_profile,
            pneumatic_feed="conventional-rim-valve-only",
        )
        wheel_assemblies[wheel] = wheel_assembly
        abstract_ui = wheel_assembly.as_mapping()
        wheels.append({
            "identity": wheel_identity,
            "name": wheel.replace("_", " ").title(),
            "side": side,
            "dual_layer": layer,
            "center": [0.0, wheel_radius, wheel_z[wheel]],
            "validator_initialization": {
                "custody": "gravity-parallel-articulated-pillar-hub",
                "position": [0.0, wheel_radius, wheel_z[wheel]],
                "orientation_axis": [0.0, 0.0, 1.0],
                "reason": "prepare-wheel-at-its-eventual-installation-location",
            },
            "eventual_installation": {
                "parent_hub": f"{identity}/hubs/{side}-dual",
                "position": [0.0, wheel_radius, wheel_z[wheel]],
                "orientation_axis": [0.0, 0.0, 1.0],
                "joint": "piloted-dual-wheel-stud-joint",
            },
            "radius_m": wheel_radius,
            "rim_radius_m": 0.28575,
            "section_width_m": wheel_width,
            "steel_wheel_mass_kg": 41.0,
            "tire_mass_kg": 66.0,
            "rated_pressure_pa": 760_000.0,
            "pneumatic_mode": pneumatic_mode,
            "material_profile": material_profile,
            "tube_compatible": True,
            "abstract_ui_assembly": abstract_ui,
            "wheel_center": abstract_ui["parts"]["wheel_center"],
            "rim": abstract_ui["parts"]["rim"],
            "tire_casing": {
                "bead_inboard": abstract_ui["parts"]["bead_inboard"],
                "bead_outboard": abstract_ui["parts"]["bead_outboard"],
                "sidewall": abstract_ui["parts"]["sidewall"],
                "tread": abstract_ui["parts"]["tread"],
            },
            "pneumatic_boundary": abstract_ui["parts"]["pneumatic_boundary"],
            "ports": {
                "outer_service_valve": abstract_ui["parts"]["outer_service_valve"] | {
                    "balance_first_moment": "valve-mass-minus-drilled-rim-material"},
                "rim_seat_ports": [],
                "bearing_rotary_union": None,
                "tube_stem_binding": None,
            },
        })

    brake_stations = tuple({
        "identity": f"{identity}/brakes/{side}",
        "side": side,
        "kind": "hydraulic-commercial-dual-wheel-drum-brake",
        "drum_diameter_m": 0.455,
        "shoe_width_m": 0.19,
        "drum_mass_kg": 43.0,
        "opposed_wheel_cylinders": 2,
        "working_pressure_pa": 17_500_000.0,
        "maximum_brake_torque_nm": 18_000.0,
        "thermal_mass_j_per_k": 35_000.0,
        "fluid": "dot-4-high-temperature",
        "circuit": f"{identity}/hydraulics/{side}-service-circuit",
        "acts_on": f"{identity}/hubs/{side}-dual",
    } for side in ("left", "right"))

    torque_input = generic_rotational_torque_port(
        f"{identity}/differential/pinion-input",
        f"{identity}/differential/carrier",
        axis=(1.0, 0.0, 0.0),
        rated_torque_nm=9_500.0,
        rated_speed_rad_s=210.0,
        flange={
            "kind": "commercial-companion-flange",
            "pilot_diameter_m": 0.114,
            "bolt_circle_m": 0.156,
            "fastener_count": 4,
            "open_side": "forward-driveshaft-connection",
        },
    )
    through_drive = generic_rotational_torque_port(
        f"{identity}/power-divider/rear-through-drive",
        f"{identity}/power-divider/carrier",
        axis=(1.0, 0.0, 0.0),
        rated_torque_nm=9_500.0,
        rated_speed_rad_s=210.0,
        role="output",
        flange={
            "kind": "commercial-inter-axle-companion-flange",
            "pilot_diameter_m": 0.114,
            "bolt_circle_m": 0.156,
            "fastener_count": 4,
            "open_side": "rear-inter-axle-shaft-connection",
        },
    )
    pinion_interface = {
        "identity": f"{identity}/axle-casing/pinion-interface",
        "kind": "interchangeable-pinion-cover-socket",
        "owner": casing["identity"],
        "active_part": "companion-flange-cover",
        "retention": "indexed-bolted-cover-with-radial-lip-seal",
        "oil_boundary": "installed-cover-must-close-casing-lubricant-volume",
        "parts": {
            "sealed-storage-cover": {
                "kind": "blind-bearing-and-oil-seal-cover",
                "rotational_port": None,
                "use": "weatherproof-storage-or-undriven-trailer-artifact",
            },
            "companion-flange-cover": {
                "kind": "serviceable-pinion-bearing-cartridge-and-driveshaft-flange",
                "rotational_port": torque_input,
                "use": "ordinary-driveshaft-or-generic-torque-source",
            },
            "validator-torque-cell-cover": {
                "kind": "instrumented-pinion-cartridge-with-reaction-torque-cell",
                "rotational_port": torque_input,
                "use": "validator-dynamometer-without-an-engine",
                "custody": "validator-fixture-temporary",
            },
            "motor-adapter-cover": {
                "kind": "pinion-cartridge-with-generic-motor-face",
                "rotational_port": torque_input,
                "use": "optional-electric-hydraulic-or-combustion-power-module",
            },
        },
        "change_sequence": [
            "drain-casing-below-pinion-aperture",
            "remove-input-torque-and-support-pinion-cartridge",
            "unbolt-cover-and-retain-bearing-shims",
            "install-selected-cover-with-new-lip-seal-and-gasket",
            "restore-preload-fill-oil-and-check-reaction-backlash",
        ],
    }
    casing["mounting"]["pinion_interface"] = pinion_interface["identity"]
    chain_interface = {
        "identity": f"{identity}/axle-casing/rear-chain-interface",
        "kind": "interchangeable-terminal-or-through-drive-cover-socket",
        "owner": casing["identity"],
        "active_part": "sealed-terminal-cap",
        "parts": {
            "sealed-terminal-cap": {
                "kind": "oil-tight-power-divider-end-cover",
                "rotational_port": None,
                "axle_role": "terminal",
            },
            "through-drive-cartridge": {
                "kind": "bearing-supported-inter-axle-output-cartridge",
                "rotational_port": through_drive,
                "axle_role": "intermediate",
            },
        },
        "conversion_sequence": [
            "remove-inter-axle-torque-and-drain-below-interface",
            "support-power-divider-shaft-and-remove-current-cover",
            "install-terminal-cap-or-through-drive-cartridge",
            "set-bearing-preload-and-close-lubricant-boundary",
            "connect-or-remove-separate-inter-axle-shaft",
            "refill-and-prove-reaction-torque-continuity",
        ],
        "chain_contract": {
            "compatible_next_owner": DUALLY_AXLE_SCHEMA,
            "inter_axle_shaft_is_separate_part": True,
            "ratio_compatibility": "upstream-and-downstream-final-drive-ratios-must-match",
            "capacity_limit": "every-port-and-shaft-rating-must-cover-transmitted-torque",
            "maximum_chain_axles": "configuration-and-validator-policy-not-renderer-limit",
        },
    }
    casing["mounting"]["rear_chain_interface"] = chain_interface["identity"]

    differential_detail = {
        "schema": "bakeable-differential-subassembly-v0",
        "simulation_scope": "differential-power-divider-bearings-and-shafts",
        "parts": [
            {"name": "input-pinion-shaft", "kind": "shaft", "bearings": ["pinion-head", "pinion-tail"], "reference_position": [-0.22, wheel_radius, 0.0]},
            {"name": "hypoid-ring-gear", "kind": "gear", "mates": ["input-pinion-shaft"], "reference_position": [0.0, wheel_radius, 0.0]},
            {"name": "differential-carrier", "kind": "rotating-carrier", "bearings": ["carrier-left", "carrier-right"], "reference_position": [0.0, wheel_radius, 0.0]},
            {"name": "left-side-gear", "kind": "splined-bevel-gear", "bearings": ["carrier-left"], "reference_position": [0.0, wheel_radius, -0.10]},
            {"name": "right-side-gear", "kind": "splined-bevel-gear", "bearings": ["carrier-right"], "reference_position": [0.0, wheel_radius, 0.10]},
            {"name": "spider-cross", "kind": "cross-shaft", "bearings": ["four-gear-thrust-bushings"], "reference_position": [0.0, wheel_radius, 0.0]},
            {"name": "left-axle-shaft", "kind": "full-floating-shaft", "bearings": ["left-inner", "left-hub-pair"], "reference_position": [0.0, wheel_radius, -0.72]},
            {"name": "right-axle-shaft", "kind": "full-floating-shaft", "bearings": ["right-inner", "right-hub-pair"], "reference_position": [0.0, wheel_radius, 0.72]},
        ],
        "bearings": [
            {"name": "pinion-head", "kind": "tapered-roller", "family": "pinion", "preloaded": True, "axis": [1.0, 0.0, 0.0], "reference_position": [-0.12, wheel_radius, 0.0],
             "structural_owner": casing["identity"], "rotating_owner": f"{identity}/differential/detail/input-pinion-shaft"},
            {"name": "pinion-tail", "kind": "tapered-roller", "family": "pinion", "preloaded": True, "axis": [1.0, 0.0, 0.0], "reference_position": [-0.29, wheel_radius, 0.0],
             "structural_owner": casing["identity"], "rotating_owner": f"{identity}/differential/detail/input-pinion-shaft"},
            {"name": "carrier-left", "kind": "tapered-roller", "family": "carrier", "preloaded": True, "axis": [0.0, 0.0, 1.0], "reference_position": [0.0, wheel_radius, -0.20],
             "structural_owner": casing["identity"], "rotating_owner": f"{identity}/differential/carrier"},
            {"name": "carrier-right", "kind": "tapered-roller", "family": "carrier", "preloaded": True, "axis": [0.0, 0.0, 1.0], "reference_position": [0.0, wheel_radius, 0.20],
             "structural_owner": casing["identity"], "rotating_owner": f"{identity}/differential/carrier"},
            {"name": "left-inner", "kind": "oil-lubricated-journal-support", "family": "shaft-journal", "preloaded": False, "axis": [0.0, 0.0, 1.0], "reference_position": [0.0, wheel_radius, -0.56],
             "structural_owner": casing["identity"], "rotating_owner": f"{identity}/differential/detail/left-axle-shaft"},
            {"name": "right-inner", "kind": "oil-lubricated-journal-support", "family": "shaft-journal", "preloaded": False, "axis": [0.0, 0.0, 1.0], "reference_position": [0.0, wheel_radius, 0.56],
             "structural_owner": casing["identity"], "rotating_owner": f"{identity}/differential/detail/right-axle-shaft"},
        ],
        "connectors": [
            {"name": "pinion-flange-spline", "kind": "splined-torque-connector", "bushing": "radial-lip-seal-compliance", "breakpoint": "spline-shear"},
            {"name": "ring-gear-bolts", "kind": "preloaded-bolted-ring", "bushing": "clamped-joint-contact", "breakpoint": "bolt-group-slip-or-fracture"},
            {"name": "left-side-spline", "kind": "floating-spline", "bushing": "oil-film-and-backlash", "breakpoint": "spline-strip"},
            {"name": "right-side-spline", "kind": "floating-spline", "bushing": "oil-film-and-backlash", "breakpoint": "spline-strip"},
            {"name": "validator-casing-cradles", "kind": "fixture-connector", "bushing": "replaceable-elastomer-saddle", "breakpoint": "fixture-release"},
        ],
        "analogue_bake": {
            "source": "fine-differential-only-simulation",
            "retains_named_parts": True,
            "runtime_reduction": "activity-weighted-per-part-damage-distribution",
            "damage_channels": [
                "tooth-contact-fatigue", "bearing-heat", "oil-starvation",
                "shaft-torsion", "spline-fretting", "seal-wear", "impact-overload",
            ],
            "coarse_runtime_rule": "damage-remains-addressable-by-part-after-reduction",
        },
    }
    detail_nodes = [
        {"identity": f"{identity}/differential/detail/{part['name']}",
         "kind": part["kind"], "mass_kg": 0.0,
         "reference_position": part["reference_position"],
         "detail_owner": f"{identity}/differential/carrier",
         "assembly_stage": "install-differential-and-solid-axle-shafts"}
        for part in differential_detail["parts"]
    ] + [
        bearing_interface_node(
            f"{identity}/differential/bearings/{bearing['name']}",
            structural_owner=bearing["structural_owner"],
            rotating_owner=bearing["rotating_owner"],
            axis=tuple(bearing["axis"]), bearing_kind=bearing["kind"],
            reference_position=tuple(bearing["reference_position"]),
            rule_set=_bearing_rules(bearing["family"]),
            preloaded=bool(bearing["preloaded"]),
            assembly_stage="install-differential-and-solid-axle-shafts",
        )
        for bearing in differential_detail["bearings"]
    ]
    detail_edges = [
        {"identity": f"{identity}/differential/connectors/{connector['name']}",
         "kind": connector["kind"], "bushing": connector["bushing"],
         "breakpoint": connector["breakpoint"],
         "nodes": [casing["identity"], f"{identity}/differential/carrier"]}
        for connector in differential_detail["connectors"]
    ]
    hub_bearings = {
        side: bearing_interface_node(
            f"{identity}/hubs/{side}-bearing-interface",
            structural_owner=casing["identity"],
            rotating_owner=f"{identity}/hubs/{side}-dual",
            axis=(0.0, 0.0, 1.0),
            bearing_kind="full-floating-tapered-roller-pair",
            reference_position=(0.0, wheel_radius, hub_z[side]),
            rule_set=_bearing_rules("hub-pair"),
            preloaded=True,
            assembly_stage="install-dual-hubs-and-hydraulic-brakes",
        ) for side in ("left", "right")
    }

    nodes = [
        {"identity": casing["identity"], "kind": casing["kind"],
         "mass_kg": casing["mass_kg"], "reference_position": [0.0, wheel_radius, 0.0],
         "geometry": {"primitive": "axial-structural-casing",
                      "axis": [0.0, 0.0, 1.0],
                      "length_m": casing["flange_to_flange_width_m"],
                      "tube_radius_m": casing["tube_outer_diameter_m"] * 0.5,
                      "center_radius_m": casing["center_carrier_outer_diameter_m"] * 0.5},
         "presentation": {"source": "abstract-ui-part-geometry",
                          "primitive": "axial-structural-casing"}},
        {"identity": f"{identity}/differential/carrier", "kind": "huge-hypoid-differential",
         "mass_kg": 184.0, "reference_position": [0.0, wheel_radius, 0.0],
         "assembly_stage": "install-differential-and-solid-axle-shafts"},
        {"identity": torque_input["identity"], "kind": torque_input["kind"],
         "mass_kg": 0.0, "reference_position": [-0.34, wheel_radius, 0.0],
         "assembly_stage": "install-differential-and-solid-axle-shafts"},
        {"identity": pinion_interface["identity"], "kind": pinion_interface["kind"],
         "mass_kg": 24.0, "reference_position": [-0.29, wheel_radius, 0.0],
         "assembly_stage": "install-differential-and-solid-axle-shafts"},
        {"identity": f"{identity}/power-divider/carrier", "kind": "inter-axle-power-divider",
         "mass_kg": 72.0, "reference_position": [-0.12, wheel_radius, 0.0],
         "assembly_stage": "install-differential-and-solid-axle-shafts"},
        {"identity": chain_interface["identity"], "kind": chain_interface["kind"],
         "mass_kg": 18.0, "reference_position": [0.31, wheel_radius, 0.0],
         "assembly_stage": "install-differential-and-solid-axle-shafts"},
        {"identity": through_drive["identity"], "kind": through_drive["kind"],
         "mass_kg": 0.0, "reference_position": [0.42, wheel_radius, 0.0],
         "assembly_stage": "install-differential-and-solid-axle-shafts"},
        *(rotating_hub_node(
            f"{identity}/hubs/{side}-dual",
            reference_position=(0.0, wheel_radius, hub_z[side]),
            axis=(0.0, 0.0, 1.0), mass_kg=58.0,
            flange_radius_m=0.205, barrel_radius_m=0.112, width_m=0.31,
            assembly_stage="install-dual-hubs-and-hydraulic-brakes",
        ) for side in ("left", "right")),
        *hub_bearings.values(),
        *(node for wheel in DUALLY_WHEELS
          for node in wheel_assemblies[wheel].graph_nodes),
        *({"identity": brake["identity"], "kind": brake["kind"],
           "mass_kg": brake["drum_mass_kg"],
           "reference_position": [0.0, wheel_radius, hub_z[brake["side"]]],
           "assembly_stage": "install-dual-hubs-and-hydraulic-brakes",
           "geometry": {"primitive": "brake-drum", "radius_m": 0.2275,
                        "width_m": 0.19, "axis": [0.0, 0.0, 1.0]},
           "presentation": {"source": "abstract-ui-part-geometry",
                            "primitive": "brake-drum"}}
          for brake in brake_stations),
        *detail_nodes,
    ]
    edges = [
        *(bearing_race_edge(
            f"{identity}/edges/casing-{side}-bearing-stator",
            bearing=hub_bearings[side], race="stationary",
            connected_node=casing["identity"],
            edge_class="load-bearing-structure",
            kind="bearing-stationary-race-seat",
        ) for side in ("left", "right")),
        *(bearing_race_edge(
            f"{identity}/edges/{side}-bearing-rotor-to-hub",
            bearing=hub_bearings[side], race="rotating",
            connected_node=f"{identity}/hubs/{side}-dual",
            edge_class="drivetrain",
            kind="bearing-rotating-race-constraint",
        ) for side in ("left", "right")),
        {"identity": f"{identity}/edges/differential-left-shaft",
         "kind": "solid-full-floating-axle-shaft", "diameter_m": 0.061,
         "nodes": [f"{identity}/differential/carrier", f"{identity}/hubs/left-dual"]},
        {"identity": f"{identity}/edges/differential-right-shaft",
         "kind": "solid-full-floating-axle-shaft", "diameter_m": 0.061,
         "nodes": [f"{identity}/differential/carrier", f"{identity}/hubs/right-dual"]},
        {"identity": f"{identity}/edges/pinion-to-power-divider",
         "kind": "generic-torque-port-to-power-divider",
         "ratio": 1.0,
         "nodes": [torque_input["identity"], f"{identity}/power-divider/carrier"]},
        {"identity": f"{identity}/edges/power-divider-to-hypoid-mesh",
         "kind": "power-divider-to-hypoid-mesh", "ratio": 4.88,
         "nodes": [f"{identity}/power-divider/carrier",
                   f"{identity}/differential/carrier"]},
        {"identity": f"{identity}/edges/power-divider-to-rear-interface",
         "kind": "selectable-through-drive-edge", "active": False,
         "nodes": [f"{identity}/power-divider/carrier", through_drive["identity"]]},
        *({"identity": f"{identity}/edges/hub-wheel-{wheel}",
           "kind": "piloted-dual-wheel-stud-joint",
           "assembly_stage": "transfer-prepared-wheels-from-pillars-to-dual-hubs",
           "edge_class": "drivetrain",
           "nodes": [f"{identity}/hubs/{'left' if wheel.startswith('left') else 'right'}-dual",
                     f"{identity}/wheels/{wheel}"]} for wheel in DUALLY_WHEELS),
        *({"identity": f"{identity}/edges/brake-{side}",
           "kind": "hydraulic-brake-torque-edge",
           "nodes": [f"{identity}/brakes/{side}", f"{identity}/hubs/{side}-dual"]}
          for side in ("left", "right")),
        *detail_edges,
        *(edge for wheel in DUALLY_WHEELS
          for edge in wheel_assemblies[wheel].graph_edges),
    ]

    model: dict[str, Any] = {
        "schema": DUALLY_AXLE_SCHEMA,
        "identity": identity,
        "name": "Roadside commercial dually rear end",
        "kind": "discoverable-heavy-mechanical-artifact",
        "transform": {"position": [center_x, 0.0, center_z], "yaw_degrees": 7.0},
        "spawn": {
            "placement": "highway-shoulder-or-service-yard-edge",
            "distribution": "seeded-sparse-roadside-artifact",
            "discoverable": True,
            "persistent_until": "salvaged-or-moved",
        },
        "axle_casing": casing,
        "pinion_interface": pinion_interface,
        "axle_chain_interface": chain_interface,
        "structure": {
            "type": "one-piece-full-floating-solid-drive-axle",
            "suspension": None,
            "suspension_invariant": "no-springs-dampers-arms-airbags-or-sprung-body",
            "support_frame_policy": (
                "infer-four-point-grasp-from-strong-accessible-structural-"
                "regions-never-from-wheel-count-or-wheel-location"),
            "future_platform_boundary": {
                "owner": "separate-n-axle-suspension-platform",
                "supported_families": [
                    "walking-beam-load-rocker", "air-spring-pivot-bogie",
                    "rail-like-equalized-wheelset",
                ],
                "axle_contract": (
                    "platform-attaches-to-qualified-casing-tube-zones-and-"
                    "does-not-become-part-of-the-differential-or-wheel-unit"
                ),
            },
        },
        "differential": {
            "kind": "massive-hypoid-open-commercial-differential",
            "final_drive_ratio": 4.88,
            "ring_gear_diameter_m": 0.525,
            "carrier_mass_kg": 184.0,
            "rated_input_torque_nm": 9_500.0,
            "rated_axle_torque_nm": 46_000.0,
            "external_input": torque_input,
            "power_divider": {
                "identity": f"{identity}/power-divider/carrier",
                "inter_axle_lock": "dog-clutch-with-zero-slip-engagement-gate",
                "rear_output": through_drive,
            },
            "detailed_subassembly": differential_detail,
        },
        "wheel_groups": {
            side: {
                "identity": f"{identity}/hubs/{side}-dual",
                "wheels": [f"{identity}/wheels/{wheel}" for wheel in group],
                "reference_position": [0.0, wheel_radius, hub_z[side]],
                "shared_rotation_coordinate": f"{side}_hub_angle_rad",
                "bearing": "full-floating-tapered-roller-pair",
            } for side, group in DUALLY_GROUPS.items()
        },
        "wheels": wheels,
        "brakes": {
            "kind": "split-dual-circuit-hydraulic-commercial-service-brake",
            "quick_connect": f"{identity}/hydraulics/frame-side-quick-coupler",
            "manual_test_port": f"{identity}/hydraulics/manual-test-pump-port",
            "stations": brake_stations,
            "parking_hold": "mechanical-cam-expander-on-both-drums",
        },
        "pneumatics": {
            "wheel_count": 4, "outer_service_valves": 4,
            "bearing_fed_paths": 0, "hub_passages": 0,
            "dual_group_manifolds": 0,
            "custody": "conventional-rotating-rim-valves-only",
        },
        "mechanical_graph": {"nodes": nodes, "edges": edges},
        "validator_program": {
            "mode": "complete-dually-axle", "tire_batch": 4,
            "stages": list(DUALLY_VALIDATOR_STAGES),
            "stop_after": DUALLY_VALIDATOR_STAGES[-1],
            "forbidden_stages": ["suspension", "chassis", "engine", "transmission"],
            "system_configuration": {
                "engine": {"required": False, "installed": False},
                "torque_producer": {
                    "required": True,
                    "selected": "validator-dynamometer",
                    "interface": torque_input["identity"],
                },
                "pinion_cover": "validator-torque-cell-cover",
            },
            "fixture_analogues": {
                "vehicle-pan-clamps": "three-casing-cradles-and-two-end-flange-pillars",
                "engine-or-transmission-drive": torque_input["identity"],
                "differential-brake-wrench-port": torque_input["identity"],
                "corner-wheel-pillars": [
                    f"{identity}/hubs/left-dual", f"{identity}/hubs/right-dual"
                ],
                "service-brake-controller": [
                    f"{identity}/hydraulics/left-service-circuit",
                    f"{identity}/hydraulics/right-service-circuit",
                ],
            },
            "torque_proof_sequence": [
                "attach-validator-dynamometer-to-open-pinion-flange",
                "drive-pinion-with-both-hydraulic-brakes-released",
                "apply-left-brake-and-observe-right-dual-group-differential-motion",
                "release-left-and-stop-at-bounded-slip",
                "apply-right-brake-and-observe-left-dual-group-differential-motion",
                "apply-both-brakes-and-measure-pinion-reaction-torque",
                "remove-test-torque-release-pressure-and-detach-dynamometer",
            ],
        },
        "interaction": {
            "inspect": True, "spin_pinion": True,
            "change_pinion_cover": True,
            "convert_terminal_or_middle_axle": True,
            "apply_hydraulic_test_pressure": True,
            "tow_or_winch": True, "hand_carry": False,
            "salvage": ["four-heavy-wheels", "differential", "axle-casing",
                        "axle-shafts", "brake-hardware"],
        },
    }

    root_object = WorldObject(
        identity, "commercial-dually-rear-end", root, model["name"],
        model["transform"], {
            "kind": "procedural-heavy-dually-axle", "wheel_count": 4,
            "dual_groups": 2, "wheel_radius_m": wheel_radius,
            "wheel_width_m": wheel_width,
        },
        material_bindings={
            "casing": "axle-rust", "differential": "drivetrain-black",
            "wheels": "rollbar-silver", "tires": "tire-rubber",
            "brakes": "brake-red",
        },
        capabilities=("discover", "inspect", "spin-pinion",
                      "apply-hydraulic-brake", "tow", "winch", "salvage",
                      "validate-complete-wheel-build"),
        semantic_parts=tuple({"identity": node["identity"], "kind": node["kind"]}
                             for node in nodes),
        physics={
            "body": "dynamic-heavy-rigid-assembly", "suspension": False,
            "mechanical_graph": f"{identity}/mechanical-graph",
            "mass_kg": sum(float(node.get("mass_kg", 0.0)) for node in nodes),
        },
        persistence={"scope": "world", "fields": ["transform", "damage", "salvage"]},
        extensions={"model": identity, "validator_program": model["validator_program"]},
    )
    return DuallyAxleAssembly(model, (root_object,))


def roadside_dually_axle_geometry_boxes(
    assembly: DuallyAxleAssembly,
) -> tuple[dict[str, Any], ...]:
    """Coarse world solids beneath the procedural axle presentation mesh."""

    model = assembly.model
    identity = str(model["identity"])
    x, _, z = (float(value) for value in model["transform"]["position"])

    def solid(name: str, center: list[float], half_extent: list[float],
              height: float, role: str, elevation: float) -> dict[str, Any]:
        return {
            "identity": f"{identity}/geometry/{name}", "kind": "dually-axle-fixture",
            "label": name.replace("-", " ").title(), "parent_identity": identity,
            "center": center, "half_extent": half_extent, "height": height,
            "floor_height": height, "wall_thickness": 0.02,
            "palette_role": role, "wall_palette_role": role,
            "geometry_mode": "solid", "openings": [],
            "placement": {"custody": "world-artifact", "elevation": elevation,
                          "yaw_degrees": float(model["transform"]["yaw_degrees"])},
            "physics": {"body": "dynamic-heavy-artifact", "collider": "solid"},
            "artifact": {"capabilities": ["inspect", "tow", "salvage"]},
        }

    boxes = [
        solid("axle-casing", [x, z], [0.16, 1.62], 0.22, "axle-rust", 0.47),
        solid("differential-pumpkin", [x, z], [0.38, 0.43], 0.62,
              "drivetrain-black", 0.30),
    ]
    for wheel in model["wheels"]:
        center = wheel["center"]
        boxes.append(solid(
            f"wheel-{wheel['side']}-{wheel['dual_layer']}",
            [x + float(center[0]), z + float(center[2])],
            [float(wheel["radius_m"]), float(wheel["section_width_m"]) / 2.0],
            2.0 * float(wheel["radius_m"]), "tire-rubber", 0.0,
        ))
    return tuple(boxes)


__all__ = [
    "DUALLY_AXLE_SCHEMA", "DUALLY_GROUPS", "DUALLY_VALIDATOR_STAGES",
    "DUALLY_WHEELS", "DuallyAxleAssembly", "roadside_dually_axle_assembly",
    "roadside_dually_axle_geometry_boxes",
]
