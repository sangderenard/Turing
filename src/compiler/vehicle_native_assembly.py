"""Runtime assembly programme for the canonical native vehicle graph.

The programme does not create a reduced vehicle.  It supplies staged live
parameters to the same compiled equation while the external rig owns clamps,
actuators, and reaction sensors.  A stage may advance only after its declared
equilibrium and conservation gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

import sympy

from .ssa_c_backend import CFunctionArtifact, emit_ssa_function_to_c
from .symbolic_equation_compiler import SymbolicPublication, compile_sympy_equations


@dataclass(frozen=True, slots=True)
class NativeAssemblyStage:
    identity: str
    operation: str
    component_patterns: tuple[str, ...]
    drivetrain_alpha: float
    corner_alphas: tuple[float, float, float, float]
    clamp_mode: str
    solver_metrics: tuple[str, ...]
    maximum_settle_seconds: float = 4.0
    consecutive_quiet_samples: int = 64
    required_systems: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PillarArmPlan:
    identity: str
    wheel_identity: str
    hub_identity: str
    gravity_parallel_axis: tuple[float, float, float]
    synchronized_wheels: tuple[str, ...]
    initialization_position: tuple[float, float, float]
    eventual_installation_position: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class RollerCoveragePlan:
    identity: str
    operation: str
    kind: str
    wheel_identities: tuple[str, ...]
    hub_identities: tuple[str, ...]
    articulation: str
    reason: str


@dataclass(frozen=True, slots=True)
class WheelFixturePlan:
    wheel_identities: tuple[str, ...]
    structural_support_identities: tuple[str, ...]
    wheel_to_structural_support: tuple[tuple[float, ...], ...]
    pillars: tuple[PillarArmPlan, ...]
    tire_mounting_rollers: tuple[RollerCoveragePlan, ...]
    articulated_dyno_rollers: tuple[RollerCoveragePlan, ...]
    rigid_axle_dyno_rollers: tuple[RollerCoveragePlan, ...]
    post_track_surface: str


@dataclass(frozen=True, slots=True)
class GraspArmPlan:
    identity: str
    support_identity: str
    contact_position: tuple[float, float, float]
    pinch_axis: tuple[float, float, float]
    initial_state: str = "touching-zero-reaction"


@dataclass(frozen=True, slots=True)
class StructuralGraspPlan:
    """Four load-bearing clamp points independent of wheel identities."""

    support_corners: tuple[Mapping[str, Any], ...]
    source: str
    selected_regions: tuple[str, ...]
    score: float
    arms: tuple[GraspArmPlan, ...]
    objective: Mapping[str, Any]
    regrasp_policy: Mapping[str, Any]


def _complete_grasp_plan(
    corners: tuple[Mapping[str, Any], ...], source: str,
    selected_regions: tuple[str, ...], score: float,
) -> StructuralGraspPlan:
    arms = tuple(GraspArmPlan(
        identity=f"validator:frame-grasper/arm-{index}",
        support_identity=str(corner["identity"]),
        contact_position=tuple(float(value) for value in corner["position"]),
        pinch_axis=tuple(float(value) for value in corner.get(
            "pinch_axis", (0.0, 1.0, 0.0))),
    ) for index, corner in enumerate(corners))
    return StructuralGraspPlan(
        corners, source, selected_regions, score, arms,
        {
            "primary": "minimum-predicted-structural-strain-energy",
            "constraints": [
                "six-axis-wrench-closure", "minimum-three-engaged-arms",
                "per-contact-capacity", "collision-free-articulation",
                "friction-cone-feasibility", "zero-commanded-engagement-motion",
            ],
            "tie_breakers": [
                "maximum-grasp-stiffness-condition", "maximum-access-margin",
                "maximum-minimum-contact-capacity",
            ],
        },
        {
            "searching_arms": 1,
            "minimum_engaged_arms": 3,
            "sequence": [
                "unload-selected-arm-with-three-arm-wrench-closure",
                "articulate-unloaded-arm-to-candidate",
                "touch-at-current-geometry-with-zero-commanded-displacement",
                "transfer-only-the-reaction-required-by-new-global-solution",
                "accept-only-if-capacity-and-strain-objective-do-not-regress",
                "release-superseded-contact-after-four-arm-equilibrium",
            ],
            "failure_rule": (
                "abort-swap-and-return-searching-arm-to-last-safe-contact"
            ),
        })


def infer_structural_grasp_frame(model: Mapping[str, Any]) -> StructuralGraspPlan:
    """Choose the strongest accessible approximately rectangular grasp frame.

    Explicit vehicle mount points win.  Otherwise paired structural clamp
    regions are scored by their weaker capacity, lateral baseline, fore/aft
    span, and access.  Each selected region contributes a forward and rearward
    contact, which is also the geometry used by ordinary U-bolts or accessory
    saddles.  Wheel locations never participate in this selection.
    """

    structure = model.get("structure", {})
    explicit = tuple(structure.get("structural_mount_points", ()))
    if not explicit:
        explicit = tuple(structure.get("support_corners", ()))
    if explicit:
        if len(explicit) != 4:
            raise ValueError("an explicit structural grasp frame requires four points")
        corners = tuple(dict(point) for point in explicit)
        return _complete_grasp_plan(
            corners, "explicit-vehicle-mounts",
            tuple(str(point["identity"]) for point in explicit), float("inf"))

    regions = tuple(structure.get("structural_grasp_regions", ()))
    if not regions:
        regions = tuple(model.get("axle_casing", {}).get("mounting", {}).get(
            "structural_grasp_regions", ()))
    candidates = []
    for region in regions:
        center = tuple(float(value) for value in region["center"])
        span = float(region["fore_aft_span_m"])
        capacity = float(region["rated_vertical_load_n"])
        access = float(region.get("access_fraction", 1.0))
        if span <= 0.0 or capacity <= 0.0 or not 0.0 < access <= 1.0:
            raise ValueError("structural grasp regions need positive span/load/access")
        candidates.append((region, center, span, capacity, access))
    best = None
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            lateral = abs(left[1][2] - right[1][2])
            if lateral <= 1.0e-6:
                continue
            score = (min(left[3], right[3]) * lateral
                     * min(left[2], right[2]) * left[4] * right[4])
            if best is None or score > best[0]:
                best = (score, left, right)
    if best is None:
        raise ValueError(
            "machine graph has no explicit four-point mounts or usable paired "
            "structural grasp regions")
    score, left, right = best
    selected = tuple(sorted((left, right), key=lambda item: item[1][2]))
    corners = []
    root = str(model.get("identity", "machine"))
    for region, center, span, capacity, access in selected:
        for fore_aft, sign in (("forward", -1.0), ("rearward", 1.0)):
            corners.append({
                "identity": f"{root}/structure/grasp/{region['identity']}/{fore_aft}",
                "position": [center[0] + sign * 0.5 * span,
                             center[1], center[2]],
                "owner": str(region["owner"]),
                "kind": "inferred-structural-shell-support-coordinate",
                "rated_vertical_load_n": capacity,
                "access_fraction": access,
                "pinch_axis": list(region.get(
                    "preferred_pinch_axis", (0.0, 1.0, 0.0))),
                "mount_analogue": region.get(
                    "mount_analogue", "general-structural-clamp"),
            })
    return _complete_grasp_plan(
        tuple(corners), "inferred-strong-accessible-region-pair",
        tuple(str(item[0]["identity"]) for item in selected), score)


def negotiate_wheel_fixture(model: Mapping[str, Any]) -> WheelFixturePlan:
    """Negotiate pillars and roller spans from one running-gear graph.

    Pillars are per wheel because assembly custody is per wheel even where
    rotation is coupled. Roller coverage is separately negotiated from hub,
    axle, articulation, and track relationships.
    """

    wheels = tuple(model.get("wheels", ()))
    wheel_ids = tuple(str(wheel["identity"]) for wheel in wheels)
    if len(set(wheel_ids)) != len(wheel_ids):
        raise ValueError("fixture negotiation requires unique wheel identities")

    wheel_groups = model.get("wheel_groups", {})
    group_for_wheel: dict[str, tuple[str, tuple[str, ...], tuple[float, ...]]] = {}
    for group_name, group in wheel_groups.items():
        members = tuple(str(identity) for identity in group.get("wheels", ()))
        hub = str(group.get("identity", group_name))
        hub_position = tuple(float(value) for value in group.get(
            "reference_position", (0.0, 0.0, 0.0)))
        for member in members:
            if member in group_for_wheel:
                raise ValueError(f"wheel belongs to more than one hub group: {member}")
            group_for_wheel[member] = (hub, members, hub_position)

    for wheel in wheels:
        identity = str(wheel["identity"])
        if identity not in group_for_wheel:
            hub = str(wheel.get("hub_identity", f"{identity}/independent-hub"))
            position = tuple(float(value) for value in wheel.get(
                "center", (0.0, 0.0, 0.0)))
            group_for_wheel[identity] = (hub, (identity,), position)

    has_structural_authority = bool(
        model.get("structure") or model.get("axle_casing"))
    support_corners = (infer_structural_grasp_frame(model).support_corners
                       if has_structural_authority else ())
    support_ids = tuple(str(corner["identity"]) for corner in support_corners)
    support_positions = tuple(tuple(float(value) for value in corner["position"])
                              for corner in support_corners)
    if support_positions:
        x_min = min(position[0] for position in support_positions)
        x_max = max(position[0] for position in support_positions)
        z_min = min(position[2] for position in support_positions)
        z_max = max(position[2] for position in support_positions)

        def support_weights(wheel: Mapping[str, Any]) -> tuple[float, ...]:
            position = group_for_wheel[str(wheel["identity"])][2]
            tx = (0.5 if x_max == x_min else
                  min(1.0, max(0.0, (position[0] - x_min) / (x_max - x_min))))
            tz = (0.5 if z_max == z_min else
                  min(1.0, max(0.0, (position[2] - z_min) / (z_max - z_min))))
            raw = tuple(
                ((1.0 - tx) if corner[0] == x_min else tx)
                * ((1.0 - tz) if corner[2] == z_min else tz)
                for corner in support_positions)
            total = sum(raw)
            return tuple(value / total for value in raw)
        wheel_to_support = tuple(support_weights(wheel) for wheel in wheels)
    else:
        support_ids = wheel_ids
        wheel_to_support = tuple(tuple(
            1.0 if row == column else 0.0
            for column in range(len(wheel_ids)))
            for row in range(len(wheel_ids)))

    pillars = tuple(PillarArmPlan(
        identity=f"validator:pillars/{identity}",
        wheel_identity=identity,
        hub_identity=group_for_wheel[identity][0],
        gravity_parallel_axis=(0.0, 1.0, 0.0),
        synchronized_wheels=group_for_wheel[identity][1],
        initialization_position=tuple(float(value) for value in wheel.get(
            "validator_initialization", {}).get(
                "position", wheel.get("center", (0.0, 0.0, 0.0)))),
        eventual_installation_position=tuple(float(value) for value in wheel.get(
            "eventual_installation", {}).get(
                "position", wheel.get("center", (0.0, 0.0, 0.0)))),
    ) for identity, wheel in zip(wheel_ids, wheels))

    mounting = tuple(RollerCoveragePlan(
        identity=f"validator:rollers/mount/{identity}",
        operation="tire-mounting", kind="per-wheel-pair",
        wheel_identities=(identity,),
        hub_identities=(group_for_wheel[identity][0],),
        articulation="vertical-mount-detent",
        reason="independent-rim-and-bead-seat-service",
    ) for identity in wheel_ids)

    ordered_groups: list[tuple[str, tuple[str, ...]]] = []
    seen_hubs: set[str] = set()
    for identity in wheel_ids:
        hub, members, _hub_position = group_for_wheel[identity]
        if hub not in seen_hubs:
            ordered_groups.append((hub, tuple(
                member for member in members if member in wheel_ids)))
            seen_hubs.add(hub)
    articulated = tuple(RollerCoveragePlan(
        identity=f"validator:rollers/dyno/{hub}",
        operation="articulated-dyno",
        kind=("long-pair" if len(members) > 1 else "per-wheel-pair"),
        wheel_identities=members, hub_identities=(hub,),
        articulation="horizontal-dyno-detent",
        reason=("shared-or-locked-hub-rotation" if len(members) > 1
                else "independent-wheel-rotation"),
    ) for hub, members in ordered_groups)

    structure = model.get("structure", {})
    solid_axle = "solid" in str(structure.get("type", "")).lower()
    rigid_axle = ((RollerCoveragePlan(
        identity="validator:rollers/dyno/rigid-axle-span",
        operation="non-articulating-rigid-axle-dyno",
        kind="axle-spanning-pair",
        wheel_identities=wheel_ids,
        hub_identities=tuple(hub for hub, _ in ordered_groups),
        articulation="locked-common-height",
        reason="solid-axle-allows-one-cross-axle-roller-system",
    ),) if solid_axle and wheel_ids else articulated)

    running_gear = model.get("running_gear", {})
    track_installed = bool(running_gear.get("track_installed", False))
    return WheelFixturePlan(
        wheel_identities=wheel_ids,
        structural_support_identities=support_ids,
        wheel_to_structural_support=wheel_to_support,
        pillars=pillars,
        tire_mounting_rollers=mounting,
        articulated_dyno_rollers=articulated,
        rigid_axle_dyno_rollers=rigid_axle,
        post_track_surface=("ground-projection-no-roller" if track_installed
                            else "negotiated-wheel-rollers"),
    )


QUALIFICATION_SPEC_PATH = (Path(__file__).resolve().parents[2]
                           / "configs" / "vehicles" / "qualification"
                           / "producer-neutral-v1.json")


def combine_c_function_artifacts(*artifacts: CFunctionArtifact) -> str:
    """Place compiler-emitted functions in one C unit with one shared prelude."""

    if not artifacts:
        return ""
    source = artifacts[0].source.rstrip()
    for artifact in artifacts[1:]:
        marker = f"TURING_EXPORT void {artifact.name}"
        offset = artifact.source.find(marker)
        if offset < 0:
            raise ValueError(f"compiler C artifact {artifact.name!r} has no exported definition")
        source += "\n\n" + artifact.source[offset:].strip()
    return source + "\n"


def load_vehicle_qualification_spec(path: str | Path | None = None) -> dict[str, Any]:
    """Load producer policy without changing the authoritative physics."""
    source = QUALIFICATION_SPEC_PATH if path is None else Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema") != "turing.vehicle-qualification-spec.v1":
        raise ValueError(f"unsupported vehicle qualification schema in {source}")
    return value


def qualification_stage_policy(spec: Mapping[str, Any], stage: str) -> dict[str, Any]:
    policy = dict(spec["observation"]["default_stage"])
    policy.update(spec["observation"].get("stage_overrides", {}).get(stage, {}))
    return policy


def native_vehicle_assembly_stages(
    *, enabled_systems: frozenset[str] | None = None,
) -> tuple[NativeAssemblyStage, ...]:
    """Return the dependency-ordered, runtime-selectable construction plan."""

    quiet = ("clamp_reaction_delta", "kinetic_energy", "energy_residual")
    stages = (
        NativeAssemblyStage("clamp-pan", "lock bare chassis datum in five pan clamps",
                            ("frame_cage_driver", "wrench_attachment_", "bumper_", "ballast_hanger_"),
                            0.0, (0.0, 0.0, 0.0, 0.0), "locked", quiet),
        NativeAssemblyStage("engine-pan", "fit engine-pan envelope and pan hardpoints",
                            ("engine_pan",), 0.0, (0.0, 0.0, 0.0, 0.0), "locked", quiet,
                            required_systems=("engine",)),
        NativeAssemblyStage("engine", "install engine as a measured rigid graph member",
                            ("engine",), 0.25, (0.0, 0.0, 0.0, 0.0), "locked", quiet,
                            required_systems=("engine",)),
        NativeAssemblyStage("transmission", "install clutch and transmission",
                            ("transmission",), 0.50, (0.0, 0.0, 0.0, 0.0), "locked", quiet),
        NativeAssemblyStage("transfer-and-differentials", "install transfer case and differentials",
                            ("transfer_case", "front_differential", "rear_differential",
                             "alternator", "alternator_cvt", "transmission_control_unit"),
                            1.0, (0.0, 0.0, 0.0, 0.0), "locked", quiet),
        NativeAssemblyStage("brace-on-balance", "solve and install density-sized corner ballast",
                            ("ballast_",), 1.0, (0.0, 0.0, 0.0, 0.0), "locked",
                            (*quiet, "center_of_mass_x", "center_of_mass_z", "ballast_fit")),
        NativeAssemblyStage("pillar-hubs", "lock four bare hubs on independent build pillars, lower the rollers clear, then bolt each wheel and rim to its hub",
                            ("wheel_",), 1.0, (0.0, 0.0, 0.0, 0.0),
                            "pillar-locked", (*quiet, "hub_pose_error", "pillar_wrench_delta", "rollers_down_clearance")),
        NativeAssemblyStage("mount-tire-casings", "with the rollers held down and clear, place each tyre casing or tube-and-casing over the wheel already bolted to the pillar hub",
                            ("tire_",), 1.0, (0.0, 0.0, 0.0, 0.0),
                            "pillar-locked", (*quiet, "bead_seat_error", "ambient_pressure_error", "rollers_down_clearance")),
        NativeAssemblyStage("inflate-tires-on-pillars", "admit nominal pressure, then articulate the existing pillar rollers from their lowered clearance to the measured roller-to-hub bead-capture distance",
                            (), 1.0, (0.0, 0.0, 0.0, 0.0),
                            "pillar-locked", (*quiet, "balloon_pressure_error", "contact_crossing_residual", "roller_to_hub_clamp_distance", "complete_bead_capture"), 8.0),
        NativeAssemblyStage("wheel-mesh-balance", "force-sense each supported wheel and fit density-sized rim ballast",
                            (), 1.0, (0.0, 0.0, 0.0, 0.0), "pillar-locked",
                            (*quiet, "wheel_radial_first_moment", "wheel_ballast_fit")),
        NativeAssemblyStage("set-suspension-rest-pose", "place four pillar hubs at the intended loaded assembly datum",
                            (), 1.0, (0.0, 0.0, 0.0, 0.0), "pillar-locked",
                            (*quiet, "hub_pose_error", "pillar_wrench_delta")),
        NativeAssemblyStage("front-linkages", "install front arms, bushings, uprights, bearings, halfshafts, rotors and calipers at rest pose",
                            ("knuckle_upright_front_", "brake_caliper_front_", "brake_rotor_front_",
                             "coilover_unsprung_front_", "coilover_sprung_front_"),
                            1.0, (1.0, 1.0, 0.0, 0.0),
                            "locked", quiet),
        NativeAssemblyStage("rear-linkages", "install rear arms, bushings, uprights, bearings, halfshafts, rotors and calipers at rest pose",
                            ("knuckle_upright_rear_", "brake_caliper_rear_", "brake_rotor_rear_",
                             "coilover_unsprung_rear_", "coilover_sprung_rear_"),
                            1.0, (1.0, 1.0, 1.0, 1.0),
                            "locked", quiet),
        NativeAssemblyStage(
            "suspension-load-transfer",
            "set all pedestals at ride height, center preload travel, verify the mandatory bump stop, apply ballast balance, scale the real spring rate until every chassis clamp wrench is zero, then release all clamps without motion",
            (), 1.0, (1.0, 1.0, 1.0, 1.0), "pedestal-load-transfer",
            (*quiet, "one-time-wheel-load-calibration", "ride-height-preload-center",
             "mandatory-bump-stop-range", "post-geometry-ballast-balance",
             "all-clamp-wrenches-zero", "four-live-wheel-loads",
             "zero-motion-clamp-release"), 24.0, 128),
        NativeAssemblyStage("armature-range-readiness", "verify sensed joints before any later solver-directed range sweep",
                            (), 1.0, (1.0, 1.0, 1.0, 1.0), "pillar-damped",
                            (*quiet, "joint_force_margin", "joint_motion_clearance")),
        NativeAssemblyStage("rolling-start", "drive the installed wheel graph until ignition catches, then select neutral and settle",
                            (), 1.0, (1.0, 1.0, 1.0, 1.0), "locked",
                            (*quiet, "engine_catch", "neutral_after_catch", "four_wheel_pressure_load"), 16.0,
                            required_systems=("engine",)),
        NativeAssemblyStage("equipment", "install selected body, electrical, hydraulic and pneumatic equipment",
                            ("body_shell_mounts", "starter_", "wiring_", "vehicle_computer",
                             "fusebox_", "lamp_", "steering_servo", "hydraulic_", "pneumatic_",
                             "brake_", "parking_brake_", "alignment_", "fuel_"),
                            1.0, (1.0, 1.0, 1.0, 1.0), "damped", quiet),
        NativeAssemblyStage(
            "accessory-installation",
            "mount selected rotating accessories at their authoritative graph wrench ports",
            (), 1.0, (1.0, 1.0, 1.0, 1.0), "locked",
            (*quiet, "accessory-mass-and-inertia-accounting", "port-wrench-continuity"),
        ),
        NativeAssemblyStage(
            "post-accessory-ballast-balance",
            "re-solve density-sized corner ballast after the installed accessory loadout",
            (), 1.0, (1.0, 1.0, 1.0, 1.0), "locked",
            (*quiet, "center_of_mass_x", "center_of_mass_z", "ballast_fit"),
        ),
        NativeAssemblyStage(
            "leveling-controller-program-capture",
            "solve individual wheel/body placement against measured opposing fixture forces and save the program",
            (), 1.0, (1.0, 1.0, 1.0, 1.0), "pillar-locked",
            (*quiet, "corner-pose-error", "opposing-force-response", "saved-program"), 24.0,
        ),
        NativeAssemblyStage(
            "differential-wrench-proof",
            "prove free and locked wheel-hub paths from both differential-brake rotor wrench ports",
            (), 1.0, (1.0, 1.0, 1.0, 1.0), "locked",
            (*quiet, "open_hub_isolation", "differential_shaft_spin",
             "zero_slip_hub_reconnect", "locked_hub_wheel_drive"),
            8.0,
        ),
        NativeAssemblyStage(
            "destructive-drivetrain-pull",
            "full-send engine against progressively immovable differential-port accessory",
            (), 1.0, (1.0, 1.0, 1.0, 1.0), "locked",
            (*quiet, "data-only", "stall", "clutch-thermal-rupture",
             "drivetrain-member-rupture"),
            60.0,
            required_systems=("engine",),
        ),
        NativeAssemblyStage("release", "release pan clamps progressively after final equilibrium",
                            (), 1.0, (1.0, 1.0, 1.0, 1.0), "progressive-release",
                            (*quiet, "ground_contact_count", "chassis_escape"), 12.0, 128),
    )
    if enabled_systems is None:
        return stages
    return tuple(
        stage for stage in stages
        if set(stage.required_systems) <= set(enabled_systems)
    )


@lru_cache(maxsize=1)
def compile_wheel_mesh_balance_c() -> CFunctionArtifact:
    """Compile a runtime wheel-mesh first-moment ballast remedy.

    A mesh integrator supplies mass properties about the live hub.  This
    equation places a density-sized rim slug opposite the measured radial
    first moment.  Nothing about the wheel design is frozen into this kernel.
    """

    names = (
        "mesh_mass mesh_first_moment_x mesh_first_moment_y mesh_polar_inertia "
        "ballast_radius ballast_density ballast_axial_width ballast_radial_depth "
        "maximum_ballast_thickness"
    )
    (mesh_mass, moment_x, moment_y, polar_inertia, radius, density,
     axial_width, radial_depth, maximum_thickness) = sympy.symbols(names, real=True)
    magnitude = sympy.sqrt(moment_x ** 2 + moment_y ** 2)
    ballast_mass = magnitude / radius
    safe_mass = sympy.Max(ballast_mass, sympy.Float("1e-30"))
    ballast_x = -moment_x / safe_mass
    ballast_y = -moment_y / safe_mass
    volume = ballast_mass / density
    thickness = volume / (axial_width * radial_depth)
    expressions = {
        "ballast_mass": ballast_mass,
        "ballast_local_x": ballast_x,
        "ballast_local_y": ballast_y,
        "ballast_volume": volume,
        "ballast_thickness": thickness,
        "corrected_mass": mesh_mass + ballast_mass,
        "corrected_polar_inertia": polar_inertia + ballast_mass * radius ** 2,
        "corrected_first_moment_x": moment_x + ballast_mass * ballast_x,
        "corrected_first_moment_y": moment_y + ballast_mass * ballast_y,
        "fit_margin": maximum_thickness - thickness,
    }
    equations = tuple(sympy.Eq(sympy.Symbol(name, real=True), expression, evaluate=False)
                      for name, expression in expressions.items())
    compiled = compile_sympy_equations(
        equations, name="vehicle_wheel_mesh_balance",
        publications=tuple(SymbolicPublication(name, f"rig.wheel_balance.{name}")
                           for name in expressions),
    )
    artifact = emit_ssa_function_to_c(compiled.module, compiled.function.name)
    if not artifact.complete:
        raise RuntimeError("wheel mesh balance model did not lower completely: " + "; ".join(
            item.reason for item in artifact.shortfalls))
    return artifact


@lru_cache(maxsize=1)
def compile_brace_on_balance_c() -> CFunctionArtifact:
    """Compile the four-corner static-moment cancellation model to C.

    Inputs are current first moments, not frozen coefficients.  The returned
    masses are additive, nonnegative brace-on weights; paired placement makes
    each axis correction independent of the other.
    """

    names = "moment_x moment_z half_length half_width density"
    moment_x, moment_z, half_length, half_width, density = sympy.symbols(names, real=True)
    positive = lambda value: (value + sympy.Abs(value)) / 2
    front = positive(-moment_x) / (2 * half_length)
    rear = positive(moment_x) / (2 * half_length)
    left = positive(moment_z) / (2 * half_width)
    right = positive(-moment_z) / (2 * half_width)
    masses = {
        "mass_front_left": front + left,
        "mass_front_right": front + right,
        "mass_rear_left": rear + left,
        "mass_rear_right": rear + right,
    }
    expressions = {
        **masses,
        **{name.replace("mass_", "volume_"): value / density
           for name, value in masses.items()},
        "corrected_moment_x": moment_x + half_length * (
            masses["mass_front_left"] + masses["mass_front_right"]
            - masses["mass_rear_left"] - masses["mass_rear_right"]),
        "corrected_moment_z": moment_z + half_width * (
            -masses["mass_front_left"] + masses["mass_front_right"]
            - masses["mass_rear_left"] + masses["mass_rear_right"]),
    }
    equations = tuple(sympy.Eq(sympy.Symbol(name, real=True), expression, evaluate=False)
                      for name, expression in expressions.items())
    compiled = compile_sympy_equations(
        equations, name="vehicle_brace_on_balance",
        publications=tuple(SymbolicPublication(name, f"rig.balance.{name}")
                           for name in expressions),
    )
    artifact = emit_ssa_function_to_c(compiled.module, compiled.function.name)
    if not artifact.complete:
        raise RuntimeError("brace-on balance model did not lower completely: " + "; ".join(
            item.reason for item in artifact.shortfalls))
    return artifact


@lru_cache(maxsize=1)
def compile_leveling_controller_c() -> CFunctionArtifact:
    """Compile a pressure/flow-aware four-corner coarse and trim controller.

    The Hadamard transform gives exact heave, roll, pitch and diagonal
    cross-weight coordinates.  Runtime calibration scales those modes before
    reconstruction into corner motion.  Coarse hydraulic state moves all
    corners within pressure/flow authority; a selected corner receives the
    short-stroke trim update for deterministic round-robin hunting.  Loss of
    support disables force/cross-weight hunting and selects an explicit
    airborne placement policy.
    """

    corners = ("front_left", "front_right", "rear_left", "rear_right")
    names = (
        "target_height target_roll target_pitch target_cross_weight_correction "
        "half_length half_width corner_stiffness maximum_offset dt "
        "pose_feedback_gain trim_feedback_gain calibrated_heave_gain "
        "calibrated_roll_gain calibrated_pitch_gain calibrated_cross_weight_gain "
        "hydraulic_pressure piston_area maximum_flow hydraulic_efficiency "
        "pressure_force_reserve_fraction coarse_rate trim_rate trim_stroke "
        "trim_entry_error support_fraction minimum_grounded_support_fraction "
        "chassis_vertical_velocity fall_velocity_threshold fall_velocity_blend "
        "fall_policy_selector landing_ready_corner_offset unloaded_placement_rate "
        "round_robin_corner "
        + " ".join(f"opposing_force_{corner}" for corner in corners) + " "
        + " ".join(f"measured_pose_error_{corner}" for corner in corners) + " "
        + " ".join(f"previous_correction_{corner}" for corner in corners) + " "
        + " ".join(f"previous_trim_{corner}" for corner in corners) + " "
        + " ".join(f"predicted_landing_offset_{corner}" for corner in corners)
    )
    values = dict(zip(names.split(), sympy.symbols(names, real=True)))

    def clamp(value: sympy.Expr, low: sympy.Expr, high: sympy.Expr) -> sympy.Expr:
        return sympy.Min(high, sympy.Max(low, value))

    def magnitude(value: sympy.Expr) -> sympy.Expr:
        # Keep real magnitude explicit.  Abs over the nested Min/Max command
        # graph is otherwise represented as conjugate by SymPy, which is both
        # unnecessary for declared-real controls and outside scalar-C SSA.
        return sympy.sqrt(value * value + sympy.Float("1e-24"))

    errors = [values[f"measured_pose_error_{corner}"] for corner in corners]
    heave_error = sum(errors) / 4
    roll_error = (-errors[0] + errors[1] - errors[2] + errors[3]) / 4
    pitch_error = (errors[0] + errors[1] - errors[2] - errors[3]) / 4
    cross_error = (errors[0] - errors[1] - errors[2] + errors[3]) / 4
    scaled_modes = (
        values["calibrated_heave_gain"] * heave_error,
        values["calibrated_roll_gain"] * roll_error,
        values["calibrated_pitch_gain"] * pitch_error,
        values["calibrated_cross_weight_gain"] * (
            cross_error - values["target_cross_weight_correction"]),
    )
    h, r, p, c = scaled_modes
    shaped_errors = (h - r + p + c, h + r + p - c,
                     h - r - p - c, h + r - p + c)
    maximum_error = sympy.Max(
        sympy.Max(magnitude(errors[0]), magnitude(errors[1])),
        sympy.Max(magnitude(errors[2]), magnitude(errors[3])),
    )
    trim_blend = 1 - clamp(
        maximum_error / sympy.Max(values["trim_entry_error"], sympy.Float("1e-9")), 0, 1)

    force_capacity = (values["hydraulic_pressure"] * values["piston_area"]
                      * values["hydraulic_efficiency"])
    force_reserve = sympy.Max(
        force_capacity * values["pressure_force_reserve_fraction"], sympy.Float("1e-9"))
    flow_rate = values["maximum_flow"] / (4 * values["piston_area"])
    coarse_rate_limit = sympy.Min(values["coarse_rate"], flow_rate)
    support_authority = clamp(
        values["support_fraction"]
        / sympy.Max(values["minimum_grounded_support_fraction"], sympy.Float("1e-9")), 0, 1)
    fall_speed = clamp(
        (-values["chassis_vertical_velocity"] - values["fall_velocity_threshold"])
        / sympy.Max(values["fall_velocity_blend"], sympy.Float("1e-9")), 0, 1)
    airborne_weight = 1 - support_authority
    falling_weight = airborne_weight * fall_speed
    policy_weights = tuple(
        1 - clamp(magnitude(values["fall_policy_selector"] - index), 0, 1)
        for index in range(3)
    )

    expressions: dict[str, sympy.Expr] = {
        "heave_error": heave_error,
        "roll_error": roll_error,
        "pitch_error": pitch_error,
        "cross_weight_error": cross_error,
        "support_authority": support_authority,
        "airborne_weight": airborne_weight,
        "falling_weight": falling_weight,
        "hydraulic_force_capacity": force_capacity,
        "hydraulic_flow_rate_limit": flow_rate,
    }
    for index, (corner, x_sign, z_sign, opposing_force, shaped_error) in enumerate(zip(
            corners, (1, 1, -1, -1), (-1, 1, -1, 1),
            (values[f"opposing_force_{corner}"] for corner in corners), shaped_errors)):
        previous_correction = values[f"previous_correction_{corner}"]
        previous_trim = values[f"previous_trim_{corner}"]
        force_authority = clamp(
            (force_capacity - magnitude(opposing_force)) / force_reserve, 0, 1)
        coarse_delta_limit = coarse_rate_limit * values["dt"] * force_authority
        coarse_delta = clamp(
            values["pose_feedback_gain"] * values["dt"] * shaped_error,
            -coarse_delta_limit, coarse_delta_limit)
        correction = clamp(
            previous_correction + support_authority * coarse_delta,
            -values["maximum_offset"], values["maximum_offset"])

        selected = 1 - clamp(magnitude(values["round_robin_corner"] - index), 0, 1)
        trim_delta_limit = values["trim_rate"] * values["dt"] * force_authority
        trim_delta = clamp(
            values["trim_feedback_gain"] * values["dt"] * shaped_error,
            -trim_delta_limit, trim_delta_limit)
        trim = clamp(
            previous_trim + support_authority * trim_blend * selected * trim_delta,
            -values["trim_stroke"], values["trim_stroke"])

        geometric = (values["target_height"] + values["target_pitch"] * x_sign
                     * values["half_length"] - values["target_roll"] * z_sign
                     * values["half_width"])
        elastic_compensation = opposing_force / values["corner_stiffness"]
        grounded_command = clamp(
            geometric + elastic_compensation + correction + trim,
            -values["maximum_offset"], values["maximum_offset"])
        hold_command = clamp(
            geometric + previous_correction + previous_trim,
            -values["maximum_offset"], values["maximum_offset"])
        fall_target = (policy_weights[0] * hold_command
                       + policy_weights[1] * values["landing_ready_corner_offset"]
                       + policy_weights[2] * values[f"predicted_landing_offset_{corner}"])
        fall_delta_limit = values["unloaded_placement_rate"] * values["dt"]
        fall_command = hold_command + clamp(
            fall_target - hold_command, -fall_delta_limit, fall_delta_limit)
        command = clamp(
            support_authority * grounded_command + airborne_weight * fall_command,
            -values["maximum_offset"], values["maximum_offset"])

        expressions[f"command_{corner}"] = command
        expressions[f"correction_{corner}_next"] = correction
        expressions[f"trim_{corner}_next"] = trim
        expressions[f"predicted_elastic_deflection_{corner}"] = elastic_compensation
        expressions[f"force_authority_{corner}"] = force_authority
        expressions[f"limit_margin_{corner}"] = values["maximum_offset"] - magnitude(command)
    equations = tuple(sympy.Eq(sympy.Symbol(name, real=True), expression, evaluate=False)
                      for name, expression in expressions.items())
    compiled = compile_sympy_equations(
        equations, name="vehicle_leveling_controller",
        publications=tuple(SymbolicPublication(name, f"rig.leveling.{name}")
                           for name in expressions),
    )
    artifact = emit_ssa_function_to_c(compiled.module, compiled.function.name)
    if not artifact.complete:
        call_details = [
            (instruction.attributes.get("callee"), len(instruction.args))
            for block in compiled.function.blocks.values()
            for instruction in block.instrs
            if str(instruction.op).casefold() == "call"
        ]
        raise RuntimeError("leveling controller did not lower completely: " + "; ".join(
            f"{item.operation}: {item.reason}" for item in artifact.shortfalls)
            + f"; calls={call_details}")
    return artifact


@lru_cache(maxsize=1)
def compile_leveling_sensor_bank_c() -> CFunctionArtifact:
    """Compile massless observations used by the leveling controller.

    These are deliberately implicit instruments rather than mechanical graph
    members: they contribute no mass, compliance, wiring, or wrench.  A cheap
    bounded first-order state prevents the controller from reading perfect,
    instantaneous solver truth while preserving truth as a separate validator
    channel.  Rich transducer and harness models belong to the later optics and
    electronics work, not to this vehicle-physics gate.
    """

    corners = ("front_left", "front_right", "rear_left", "rear_right")
    channels = (
        *(f"force_{corner}" for corner in corners),
        *(f"pose_{corner}" for corner in corners),
        *(f"pressure_{corner}" for corner in corners),
        "vertical_velocity",
    )
    names = (
        "dt force_bandwidth_hz pose_bandwidth_hz pressure_bandwidth_hz "
        "motion_bandwidth_hz force_range_n pose_range_m pressure_range_pa "
        "motion_range_m_s "
        + " ".join(f"truth_{channel}" for channel in channels) + " "
        + " ".join(f"previous_{channel}" for channel in channels)
    )
    values = dict(zip(names.split(), sympy.symbols(names, real=True)))

    def clamp(value: sympy.Expr, low: sympy.Expr, high: sympy.Expr) -> sympy.Expr:
        return sympy.Min(high, sympy.Max(low, value))

    expressions: dict[str, sympy.Expr] = {}
    maximum_normalized_residual: sympy.Expr = sympy.Float("0")
    for channel in channels:
        if channel.startswith("force_"):
            bandwidth, limit = values["force_bandwidth_hz"], values["force_range_n"]
        elif channel.startswith("pose_"):
            bandwidth, limit = values["pose_bandwidth_hz"], values["pose_range_m"]
        elif channel.startswith("pressure_"):
            bandwidth, limit = values["pressure_bandwidth_hz"], values["pressure_range_pa"]
        else:
            bandwidth, limit = values["motion_bandwidth_hz"], values["motion_range_m_s"]
        alpha = clamp(sympy.Float(str(2 * sympy.pi.evalf(18))) * bandwidth * values["dt"], 0, 1)
        truth = values[f"truth_{channel}"]
        raw = values[f"previous_{channel}"] + alpha * (
            truth - values[f"previous_{channel}"])
        observed = clamp(raw, -limit, limit)
        residual = truth - observed
        normalized = sympy.sqrt(residual * residual + sympy.Float("1e-24")) / sympy.Max(
            limit, sympy.Float("1e-12"))
        maximum_normalized_residual = sympy.Max(maximum_normalized_residual, normalized)
        expressions[f"observed_{channel}"] = observed
        expressions[f"residual_{channel}"] = residual
    expressions["maximum_normalized_residual"] = maximum_normalized_residual

    equations = tuple(sympy.Eq(sympy.Symbol(name, real=True), expression, evaluate=False)
                      for name, expression in expressions.items())
    compiled = compile_sympy_equations(
        equations, name="vehicle_leveling_sensor_bank",
        publications=tuple(SymbolicPublication(name, f"rig.leveling.sensor.{name}")
                           for name in expressions),
    )
    artifact = emit_ssa_function_to_c(compiled.module, compiled.function.name)
    if not artifact.complete:
        raise RuntimeError("leveling sensor bank did not lower completely: " + "; ".join(
            f"{item.operation}: {item.reason}" for item in artifact.shortfalls))
    return artifact


def assembly_manifest(config: Any) -> Mapping[str, Any]:
    mass = config.mass_properties()
    return {
        "schema": "turing.native-vehicle-assembly.v1",
        "vehicle_equation": "canonical-runtime-parameterized",
        "rig_ownership": "external-fixtures-and-stage-controller",
        "advance_policy": "all-stage-metrics-must-pass-for-consecutive-window",
        "mass_components": mass["components"],
        "stages": [stage.__dict__ if hasattr(stage, "__dict__") else {
            field: getattr(stage, field) for field in stage.__dataclass_fields__
        } for stage in native_vehicle_assembly_stages()],
        "balance_model": "vehicle_brace_on_balance",
        "balance_parameters_are_runtime_inputs": True,
        "loadout_sequence": {
            "equipment_before_accessories": True,
            "rotating_accessories": "mounted-at-authoritative-mechanical-graph-wrench-ports",
            "selected_body": "configuration-owned-physical-body-installed-as-final-accessory",
            "post_loadout_actions": ["mass-com-inertia-reduction", "corner-ballast-balance",
                                     "load-aware-leveling-program-capture"],
        },
        "leveling_program": {
            "kernel": "vehicle_leveling_controller",
            "inputs": ["target-body-height-roll-pitch", "four-opposing-corner-forces",
                       "live-geometry-stiffness-and-travel-limits",
                       "hydraulic-pressure-flow-area-and-efficiency",
                       "measured-corner-pose-errors", "support-and-fall-state",
                       "coarse-and-trim-state", "runtime-calibrated-four-mode-gains"],
            "controlled_coordinates": ["heave", "roll", "pitch", "cross-weight"],
            "saved_observations": ["commands", "fixture-reactions", "body-pose", "loadout",
                                   "hydraulic-authority", "coarse-state", "trim-state",
                                   "airborne-policy-state"],
            "startup_replay": False,
            "observations": {
                "kernel": "vehicle_leveling_sensor_bank",
                "kind": "implicit-massless-bounded-signal-state",
                "mechanical_mass_kg": 0.0,
                "mechanical_compliance": False,
                "harness_component": False,
                "controller_reads_truth_directly": False,
                "channels": ["four-hub-ground-response-forces",
                             "four-suspension-position-errors",
                             "four-tire-pressures", "body-vertical-velocity"],
            },
        },
        "wheel_mesh_balance": {
            "kernel": "vehicle_wheel_mesh_balance",
            "mesh_owned_inputs": ["mass", "radial-first-moment", "polar-inertia"],
            "remedy": "density-sized-rim-ballast-opposite-radial-first-moment",
            "runtime_parameters": ["radius", "density", "axial-width", "radial-depth",
                                   "maximum-thickness"],
            "required_gate": ["corrected-first-moment", "ballast-fit", "corrected-inertia"],
        },
        "periodic_gate": {
            "engine_harmonics_may_form_bounded_limit_cycle": True,
            "acceptance_metrics": ["dc-pose-drift", "cycle-energy-slope",
                                   "harmonic-envelope", "mean-clamp-wrench-balance"],
            "spectral_backend": "fftfree-C-ABI-observer",
            "spectral_force_feedback": False,
            "reason": "FFT bins classify expected vibration; causal bushing states own dissipation",
        },
        "engine_brace_candidate_families": {
            "tube_graph": "pan-to-frame triangulated elastic-plastic members",
            "evolved_structural_panel": {
                "design_representation": "material-removal graph over engine-pan envelope",
                "optimization_metrics": ["peak-mount-wrench", "strain-energy", "buckling-margin",
                                         "yield-margin", "mass", "service-access"],
                "mounting_boundary": "periodic-six-axis-clamp-ring",
                "field_repair_state": "individual-clamps-may-be-replaced-or-declared-missing",
                "baked_outputs": ["render-mesh", "material-and-strain-texture",
                                  "mass-com-inertia", "rim-boundary-response-operator",
                                  "yield-buckling-fracture-profile"],
                "runtime_rule": "reduced-rim-operator-is-derived-from-and-versioned-with-full-panel-solve",
            },
        },
    }


def stage_components(config: Any, stage: NativeAssemblyStage) -> tuple[Mapping[str, Any], ...]:
    """Select concrete JSON-owned mass records admitted by one stage."""

    return tuple(
        component for component in config.mass_properties()["components"]
        if any(component["identity"] == pattern or component["identity"].startswith(pattern)
               for pattern in stage.component_patterns)
    )


def assembled_point_mass_properties(
    components: tuple[Mapping[str, Any], ...], *, inertia_floor_kg_m2: float = 0.05,
) -> Mapping[str, Any]:
    """Reduce installed component records to live mass, COM, and principal inertia."""

    total = sum(float(component["mass_kg"]) for component in components)
    if total <= 0:
        raise ValueError("an assembled graph must contain positive mass")
    center = tuple(sum(float(component["mass_kg"]) * float(component["local_position"][axis])
                       for component in components) / total for axis in range(3))
    inertia = {
        "roll": inertia_floor_kg_m2 + sum(float(component["mass_kg"]) * (
            (float(component["local_position"][1]) - center[1]) ** 2
            + (float(component["local_position"][2]) - center[2]) ** 2)
            for component in components),
        "pitch": inertia_floor_kg_m2 + sum(float(component["mass_kg"]) * (
            (float(component["local_position"][0]) - center[0]) ** 2
            + (float(component["local_position"][1]) - center[1]) ** 2)
            for component in components),
        "yaw": inertia_floor_kg_m2 + sum(float(component["mass_kg"]) * (
            (float(component["local_position"][0]) - center[0]) ** 2
            + (float(component["local_position"][2]) - center[2]) ** 2)
            for component in components),
    }
    return {"mass_kg": total, "center_of_mass": center, "inertia_kg_m2": inertia}


__all__ = ["NativeAssemblyStage", "PillarArmPlan", "RollerCoveragePlan",
           "GraspArmPlan", "StructuralGraspPlan",
           "infer_structural_grasp_frame",
           "WheelFixturePlan", "negotiate_wheel_fixture",
           "assembled_point_mass_properties", "assembly_manifest",
           "combine_c_function_artifacts", "compile_brace_on_balance_c",
           "compile_leveling_controller_c",
           "compile_leveling_sensor_bank_c",
           "compile_wheel_mesh_balance_c", "native_vehicle_assembly_stages", "stage_components"]
