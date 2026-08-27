"""Vehicle slots, general contact surfaces, and baked WebGPU contact physics."""

import json
import base64
import math
import shutil
import subprocess

import pytest
import sympy

from src.compiler.abstract_ui_div_map import project_class_to_div_map
from src.compiler.abstract_ui_surfaces import (
    linear_gradient_solid,
    sampled_mud_oval_height_field,
    support_surface_model,
)
from src.compiler.abstract_ui_vehicles import (
    CONTACT_PATCH_OUTPUTS,
    CONTACT_PATCH_TENSOR_REDUCER_OUTPUTS,
    VEHICLE_STATE_OUTPUTS,
    WHEEL_NAMES,
    compile_symbolic_vehicle_physics,
    compile_symbolic_vehicle_physics_webgpu,
    compile_symbolic_vehicle_physics_wasm,
    compile_wheel_contact_abstract_tensor,
    compile_wheel_contact_ssa,
    compile_wheel_contact_wasm,
    compile_wheel_contact_webgpu,
    extra_precision_closure,
    load_default_car_configuration,
    symbolic_vehicle_equations,
    symbolic_wheel_contact_equations,
    vehicle_configuration_from_mapping,
    vehicle_webgpu_program_model,
)


class VehicleProbe:
    speed: float


def test_json_car_configuration_is_strict_parametric_and_content_addressed():
    config = load_default_car_configuration()
    assert config.source["tires"]["pressure_pa"] == 155000.0
    assert config.source["solid_contact"]["kinetic_friction"] == pytest.approx(.54)
    assert sum(config.source["mass_distribution"].values()) == pytest.approx(1.0)
    assert config.parameter_defaults()["inverse_mass"] == pytest.approx(1 / 620)
    assert config.source["powertrain"]["displacement_liters"] == 1.6
    assert config.source["powertrain"]["brake_mean_effective_pressure_pa"] == 800000.0
    assert config.parameter_defaults()["engine_rotating_inertia"] == pytest.approx(.22)
    mass = config.mass_properties()
    assert mass["total_mass_kg"] == pytest.approx(620)
    assert mass["allocated_component_mass_kg"] == pytest.approx(366)
    assert mass["residual_frame_cage_driver_mass_kg"] == pytest.approx(254)
    assert sum(item["mass_kg"] for item in mass["components"]) == pytest.approx(620)
    assert mass["center_of_mass"] == pytest.approx([-.02480645, .06838232, 0], abs=1e-8)
    assert mass["derived_axle_fractions"]["front"] == pytest.approx(.4799948)
    for axis in ("roll", "pitch", "yaw"):
        assert config.parameter_defaults()[f"inverse_inertia_{axis}"] == pytest.approx(
            1 / mass["inertia_kg_m2"][axis])
    assert config.source["powertrain"]["transmission_mass_kg"] == 58
    assert config.source["powertrain"]["transfer_case_mass_kg"] == 24
    assert config.source["drivetrain"]["wheel_mass_kg"] == 14
    assert config.source["drivetrain"]["tire_mass_kg"] == 12
    assert config.wheel_rotational_inertia() == pytest.approx(.84656)
    assert [config.parameter_defaults()[f"crank_axis_{axis}"] for axis in "xyz"] == pytest.approx([1, 0, 0])
    assert config.source["suspension"]["pneumatic_compression_damping"] == 3200.0
    assert config.source["suspension"]["pneumatic_rebound_damping"] == 4100.0
    assert config.source["suspension"]["pneumatic_efficiency"] == .96
    assert config.source["suspension"]["maximum_compression_speed"] == 1.25
    assert config.source["suspension"]["active_damping_minimum_scale"] == .88
    assert config.source["suspension"]["active_damping_maximum_scale"] == 1.18
    assert config.source["controls"]["angular_damping"] == 4.2
    assert config.source["traction_control"]["target_friction_utilization"] == .92
    assert config.source["traction_control"]["slip_sensor_damping_ratio"] > 1
    assert config.source["traction_control"]["utilization_sensor_damping_ratio"] > 1
    assert config.source["presentation"]["world_tile_size"] == .35
    assert config.source["transmission"]["mode_default"] == "automatic"
    assert config.source["transmission"]["starting_gear"] == 2
    assert config.source["transmission"]["forward_ratios"][0] == 6.4
    assert config.source["transmission"]["forward_ratios"][1] == 3.1
    assert config.source["transmission"]["ultra_low_range_ratio"] == 2.72
    assert config.source["drivetrain"]["differential_lock_maximum_torque_nm"] == 680.0
    assert config.source["wheels"]["track_half_width"] > (
        config.source["chassis"]["half_width"] + config.source["tires"]["width"] / 2)
    assert config.source["suspension"]["travel"] < 2 * config.source["tires"]["radius"]
    assert config.source["presentation"]["world_tile_strength"] == .78
    assert config.source["presentation"]["chase_camera_distance"] == 2.6
    assert "maximum_speed" not in config.source["controls"]
    assert len(config.digest) == 64

    invalid = json.loads(config.canonical_json)
    invalid["surprise_browser_policy"] = True
    with pytest.raises(ValueError, match="unknown"):
        vehicle_configuration_from_mapping(invalid)


def test_gradient_solid_has_no_ramp_specific_physics_category():
    contract = support_surface_model("world")
    assert contract["consumers"] == [
        "platformer-body", "projectile-body", "vehicle-contact-patch", "rigid-body",
    ]
    assert "contact-manifold" in contract["selection"]["rule"]
    assert contract["world_bottom"]["thickness"] == 8.0
    assert contract["world_bottom"]["sampled_surface_guard_depth"] == .75
    assert contract["world_bottom"]["role"].startswith("emergency-containment")
    slope = linear_gradient_solid(
        "world/slope", "world", center_x=2, center_z=3, half_width=1,
        half_run=2, low_height=.1, high_height=1.1,
    )
    assert slope["physics"]["collider"] == "solid-contact-surface"
    assert slope["physics"]["contact_mode"] == "normal-constraint-with-coulomb-friction"
    assert slope["surface"]["gradient"] == pytest.approx([.25, 0])
    assert slope["kind"] == "static-gradient-solid"
    assert slope["geometry_mode"] == "height-field-prism"


def test_sampled_mud_oval_has_crawler_relief_and_a_smooth_cut_course():
    terrain = sampled_mud_oval_height_field("terrain", "world", center_x=0, center_z=0)
    surface = terrain["surface"]
    columns, rows = surface["resolution"]
    assert (columns, rows) == (49, 33)
    assert len(surface["heights"]) == columns * rows
    assert min(surface["heights"]) <= .13
    assert max(surface["heights"]) - min(surface["heights"]) >= .75
    assert surface["features"] == {
        "landscape": "procedural-crawler-hills", "course": "smooth-cut-oval",
        "authority": "outer-courtyard-depth-map",
    }
    assert terrain["geometry_mode"] == "sampled-height-field-prism"


def test_contact_patch_equations_publish_pneumatic_load_and_coulomb_force_pair():
    equations, _ = symbolic_wheel_contact_equations()
    ordered = sorted(set().union(*(equation.rhs.free_symbols for equation in equations)), key=str)
    evaluate = sympy.lambdify(ordered, [equation.rhs for equation in equations], "math")
    values = {str(symbol): 0.0 for symbol in ordered}
    values.update({
        "dt": 1 / 120, "support": 1, "hub_height": .25, "geometric_compression": .21,
        "previous_compression": .21, "surface_height": 0,
        "normal_y": 1, "forward_x": 1, "right_z": 1,
        "slip_longitudinal": .01, "attachment_x": .54, "attachment_y": -.22,
        "corner_weight": 620 * 9.81 * .27, "suspension_rest_length": .24,
        "chassis_clearance": .22, "suspension_travel": .34,
            "spring_stiffness": 7200, "pneumatic_compression_damping": 1350,
            "linkage_motion_ratio": 1.0,
        "pneumatic_rebound_damping": 1850, "pneumatic_efficiency": .96,
        "maximum_compression_speed": 1.8,
        "active_damping_minimum_scale": .88, "active_damping_maximum_scale": 1.18,
        "active_damping_body_velocity_gain_s_per_m": .22,
        "active_damping_rebound_release_gain_s_per_m": .08,
        "wheelbase_half_length": .62, "track_half_width": .56,
        "corner_front_sign": 1, "corner_side_sign": -1,
        "tire_pressure": 185000, "minimum_contact_area": .006,
        "maximum_contact_area": .045, "mu_static": 1.18, "mu_kinetic": .92,
        "load_sensitivity": .075, "longitudinal_stiffness": 9800,
        "lateral_stiffness": 11200, "slip_transition_speed": .38,
    })
    result = dict(zip(CONTACT_PATCH_OUTPUTS,
                      evaluate(*(values[str(symbol)] for symbol in ordered))))
    assert .006 <= result["contact_area"] <= .045
    assert result["chassis_force_y"] > 0
    assert result["chassis_force_x"] < 0
    assert result["chassis_torque_x"] == pytest.approx(
        values["attachment_y"] * result["chassis_force_z"]
        - values["attachment_z"] * result["chassis_force_y"])
    assert result["chassis_torque_y"] == pytest.approx(
        values["attachment_z"] * result["chassis_force_x"]
        - values["attachment_x"] * result["chassis_force_z"])
    assert result["chassis_torque_z"] == pytest.approx(
        values["attachment_x"] * result["chassis_force_y"]
        - values["attachment_y"] * result["chassis_force_x"])
    values["slip_longitudinal"] = 2.0
    sliding = dict(zip(CONTACT_PATCH_OUTPUTS,
                       evaluate(*(values[str(symbol)] for symbol in ordered))))
    assert abs(sliding["chassis_force_x"]) <= (
        values["mu_static"] * 1.18 * sliding["chassis_force_y"]
    )
    # A ravine/drop impact can traverse the whole suspension range in one
    # fixed step. The same compiled law must stay finite and Coulomb-bounded.
    for compression in (-.08, 0, .12, .34, .7):
        for previous in (0, .17, .34):
            for vertical_speed in (-18, 0, 12):
                values.update({
                    "geometric_compression": compression,
                    "previous_compression": previous,
                    "chassis_velocity_y": vertical_speed,
                    "hub_velocity_y": vertical_speed,
                    "roll_velocity": 4.5,
                    "pitch_velocity": -5.5,
                    "slip_longitudinal": 7.0,
                    "slip_lateral": -5.0,
                })
                impact = dict(zip(CONTACT_PATCH_OUTPUTS,
                                  evaluate(*(values[str(symbol)] for symbol in ordered))))
                assert all(math.isfinite(float(value)) for value in impact.values())
                normal_load = impact["chassis_force_y"]
                tangent_load = math.hypot(impact["chassis_force_x"],
                                          impact["chassis_force_z"])
                assert normal_load >= 0
                assert tangent_load <= values["mu_static"] * 1.18 * normal_load + 1e-6


def test_contact_patch_compiles_through_repository_ssa_to_four_lane_webgpu():
    compiled = compile_wheel_contact_ssa()
    assert compiled.function.metadata["symbolic_dtype"] == "float32"
    assert compiled.process_graph.G.graph["sympy_translation_fallbacks"] == ()
    emitted = compile_wheel_contact_webgpu()
    assert emitted.complete
    assert emitted.launch_plan.workgroup_size == (4, 1, 1)
    assert emitted.launch_plan.groups == (1, 1, 1)
    assert "@compute @workgroup_size(4, 1, 1)" in emitted.source
    assert len(CONTACT_PATCH_OUTPUTS) == 7
    vehicle_equations, _ = symbolic_vehicle_equations()
    arguments = {
        str(symbol)
        for equation in vehicle_equations
        for symbol in equation.rhs.free_symbols
    }
    assert {f"contact_wrench_force_{axis}" for axis in "xyz"} <= arguments
    assert {f"contact_wrench_torque_{axis}" for axis in "xyz"} <= arguments


def test_scalar_contact_wasm_fallback_puts_spring_load_into_the_chassis():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute WebAssembly")
    artifact = compile_wheel_contact_wasm()
    values = {name: 0.0 for name in artifact.input_names}
    values.update({
        "dt": 1 / 120, "support": 1, "geometric_compression": .2,
        "previous_compression": .2, "normal_y": 1, "forward_x": 1, "right_z": 1,
        "corner_weight": 620 * 9.81 / 4, "suspension_travel": .34,
        "spring_stiffness": 7200, "linkage_motion_ratio": 1,
        "pneumatic_compression_damping": 3200, "pneumatic_rebound_damping": 4100,
        "pneumatic_efficiency": .96, "maximum_compression_speed": 1.25,
        "active_damping_minimum_scale": .88, "active_damping_maximum_scale": 1.18,
        "active_damping_body_velocity_gain_s_per_m": .22,
        "active_damping_rebound_release_gain_s_per_m": .08,
        "tire_pressure": 155000, "minimum_contact_area": .008,
        "maximum_contact_area": .06, "mu_static": 1.18, "mu_kinetic": .92,
        "load_sensitivity": .075, "longitudinal_stiffness": 9200,
        "lateral_stiffness": 14000, "slip_transition_speed": .38,
    })
    script = r"""
const payload=JSON.parse(require("fs").readFileSync(0,"utf8")),bytes=Buffer.from(payload.bytes,"base64");
(async()=>{const {instance}=await WebAssembly.instantiate(bytes,{}),view=new DataView(instance.exports.memory.buffer);
payload.inputs.forEach((value,index)=>view.setFloat64(payload.inputOffsets[index],value,true));
instance.exports.abstract_ui_wheel_contact(0);console.log(JSON.stringify(payload.outputOffsets.map(
  offset=>view.getFloat64(offset,true))));})().catch(error=>{console.error(error);process.exit(1)});
"""
    completed = subprocess.run([node, "-e", script], input=json.dumps({
        "bytes": base64.b64encode(artifact.binary).decode("ascii"),
        "inputs": [values[name] for name in artifact.input_names],
        "inputOffsets": list(artifact.input_offsets), "outputOffsets": list(artifact.output_offsets),
    }), capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    result = dict(zip(artifact.output_names, json.loads(completed.stdout)))
    assert result["chassis_force_y"] > 1000
    assert result["contact_area"] > 0


def test_sympy_contact_precompile_is_one_opt_in_packed_tensor_dispatch():
    compiled = compile_wheel_contact_abstract_tensor(packed_outputs=True)
    regions = [name for name in compiled.module.functions
               if name.startswith("numerical_region_")]
    artifact = compiled.artifacts[0]
    metadata = artifact.api.to_mapping()["metadata"]

    assert regions == ["numerical_region_0"]
    assert compiled.packed_outputs is True
    assert artifact.complete
    assert artifact.launch_plan.workgroup_size == (4, 1, 1)
    assert metadata["packed_outputs"] is True
    assert compiled.output_names == CONTACT_PATCH_OUTPUTS
    assert metadata["output_span"][-1] < metadata["output_span"][0]
    assert f"outputs[0u + linear_index] = v_{metadata['output_span'][0]};" in artifact.source
    assert len(metadata["io_layout"]["outputs"]) == 1
    assert len(metadata["output_span"]) == len(CONTACT_PATCH_OUTPUTS)
    assert "tensor_sqrt" not in compiled.source
    assert ".sqrt()" in compiled.source


def test_packed_contact_wrench_rows_translate_past_tensor_contact_area():
    assert CONTACT_PATCH_OUTPUTS[:6] == (
        "chassis_force_x", "chassis_force_y", "chassis_force_z",
        "chassis_torque_x", "chassis_torque_y", "chassis_torque_z",
    )
    wheel_wrenches = sympy.Matrix(6, 4, tuple(range(1, 25)))
    reduced = wheel_wrenches * sympy.ones(4, 1)
    assert list(reduced) == [sum(range(start, start + 4))
                             for start in range(1, 25, 4)]
    assert CONTACT_PATCH_TENSOR_REDUCER_OUTPUTS[0] == "contact_area"


def test_vehicle_tensor_precision_closure_is_explicit_and_collapses_at_boundary():
    from src.common.tensors.abstraction import AbstractTensor

    ordinary = extra_precision_closure(lambda value: value + 1, limbs=1)
    widened = extra_precision_closure(
        lambda value: (value * value + 1).sqrt(), limbs=2,
    )
    values = AbstractTensor.get_tensor([0.0, 3.0])

    assert ordinary(values).tolist() == [1.0, 4.0]
    assert widened(values).tolist() == pytest.approx([1.0, 10.0 ** .5])


def test_living_map_has_a_vehicle_slot_not_a_car_specific_control_mode():
    projection = project_class_to_div_map(VehicleProbe, seed="vehicle-probe")
    slot = projection.model["vehicle_slot"]
    vehicle = slot["vehicles"][0]
    terrain = next(box for box in projection.model["document_geometry"]["boxes"]
                   if box["identity"] == vehicle["offroad_terrain"])
    courtyard = next(box for box in projection.model["document_geometry"]["boxes"]
                     if box["kind"] == "courtyard")
    assert terrain["parent_identity"] == courtyard["identity"]
    assert terrain["center"] == courtyard["center"]
    assert terrain["half_extent"] == pytest.approx(
        [value * .96 for value in courtyard["half_extent"]])
    assert max(terrain["surface"]["cell_size"]) <= .51
    inventory_item = next(item for item in projection.model["inventory"]["items"] if item["slot"] == 9)
    depth_item = next(item for item in projection.model["inventory"]["items"] if item["slot"] == 10)
    assert inventory_item["properties"]["operation"] == "mount-vehicle-slot"
    assert depth_item["name"] == "Depth map"
    assert vehicle["physics"]["parallel_spring_lanes"] == list(WHEEL_NAMES)
    webgpu_program = next(
        program for program in slot["programs"]
        if program["identity"] == vehicle["physics"]["webgpu_program"]
    )
    wrench = webgpu_program["wrench_reduction"]
    integration = webgpu_program["vehicle_integration"]
    tensor_contact = webgpu_program["tensor_contact_precompile"]
    assert tensor_contact["packed_outputs"] is True
    assert len(tensor_contact["kernel"]["io"]["outputs"]) == 1
    assert len(tensor_contact["kernel"]["output_span"]) == len(CONTACT_PATCH_OUTPUTS)
    assert tensor_contact["storage_rows"] == {
        name: index for index, name in enumerate(CONTACT_PATCH_OUTPUTS)
    }
    assert tensor_contact["reducer_output_order"] == list(CONTACT_PATCH_TENSOR_REDUCER_OUTPUTS)
    assert tensor_contact["precompile_output_translation"] == "reducer-publications-to-authored-contact-abi"
    assert tensor_contact["wrench_row_start"] == 0
    assert tensor_contact["wrench_row_count"] == 6
    assert wrench["input_shapes"] == [[6, 4], [4, 1]]
    assert wrench["output_shape"] == [6, 1]
    assert ".matmul(" in wrench["abstract_tensor_source"]
    assert wrench["kernel"]["backend_variant"] == "webgpu_tiled_gemm"
    assert wrench["kernel"]["problem_shape"] == {"m": 6, "n": 1, "k": 4}
    assert wrench["kernel"]["scalars"] == {"alpha": 1.0, "beta": 0.0}
    assert integration["outputs"] == list(VEHICLE_STATE_OUTPUTS)
    assert integration["state_residency"] == "gpu-persistent-with-passive-presentation-snapshots"
    assert len(integration["kernel"]["io"]["feeds"]) == 1
    assert len(integration["kernel"]["io"]["outputs"]) == 1
    assert "var<workgroup> tile_A" in wrench["kernel"]["source"]
    assert projection.model["motion_cues"]["world_tiling"]["physics_effect"] == "none"
    assert projection.model["motion_cues"]["vehicle_camera"]["writes"] == ["presentation-camera"]
    assert projection.model["motion_cues"]["camera_depth"]["format"] == "DEPTH_COMPONENT24"
    assert projection.model["motion_cues"]["camera_depth"]["resolution"] == "half-viewport"
    script = projection.javascript
    assert "function updateVehicleChaseCamera" in script
    assert "function initializeVehicleFirstExperience" in script
    assert "function armBrowserFullscreenOnFirstGesture" in script
    assert "initializeVehicleFirstExperience();" in script
    assert 'setShaderOnlyMode(true)' in script
    assert "function renderCameraDepthPass" in script
    assert 'box.geometry_mode==="vehicle-wheel"' in script
    assert "wheelAngles[index]+omega" in script
    assert "vWorldPosition.xz/tileSize" in script
    assert terrain["surface"]["kind"] == "sampled-height-field"
    assert "gradient_test_solid" not in vehicle
    assert "function sampleDeclaredSurface" in script
    assert "function applyDepthMapTool" in script
    assert 'recover.textContent="RIGHT CAR"' in script
    assert 'respawn.textContent="RESPAWN"' in script
    assert "function respawnActiveVehicleAtOrigin" in script
    assert 'box.geometry_mode==="sampled-height-field-prism"' in script
    assert "...vehicleRuntime.rollCageBoxes" in script
    assert "...vehicleRuntime.powertrainBoxes" in script
    assert 'kind:"vehicle-frame-member"' in script
    assert 'kind:"vehicle-mechanical-edge"' in script
    assert 'geometry_mode:"vehicle-link"' in script
    assert "solveVehicleMechanicalGraph" in script
    assert "controlVehicleTransmission" in script
    assert '[["ULTRA LOW","lowRange"],["DIFF LOCK","diffLock"]]' in script
    assert 'button.textContent=label' in script
    assert "rebuildPortableSceneMesh({dynamicOnly:true})" in script
    assert "removeVehicleInventoryItem" in script
    assert 'event.code==="KeyV"' in script
    assert "viewportControls.yaw=state.yaw" not in script
    assert 'reportRuntimeFault("vehicle-step"' in script
    assert 'reportRuntimeFault(vehicleRuntime.active?"mounted-frame":"world-frame"' in script
    assert 'disabledPresentationStages.add("wheel-shader")' in script
    assert "function synchronizeVehicleLookYaw" in script
    assert 'primitive:"balloon-tire-sidewall-and-tread"' in script
    assert 'primitive:"heavy-six-spoke-wheel-hub-and-brake"' in script
    assert "Math.cos(spin),ss=Math.sin(spin)" in script
    assert "design_supported_mass_kg" in script
    assert 'details.textContent="STATS"' in script
    assert 'TC 0%' in script and 'ABS 0%' in script
    assert 'tractionIntervention=(1-tractionScale)*100' in script
    assert 'brakeIntervention=(1-brakeScale)*100' in script
    assert 'indicator.setAttribute("aria-valuenow",intervention.toFixed(1))' in script
    assert "const VEHICLE_HUD_VERTEX_SHADER" in script
    assert "function drawVehicleShaderHud" in script
    assert "drawVehicleShaderHud(gl,width,height)" in script
    assert "function drawVehicleCanvasHud" in script
    assert 'if(!liveDom)return' in script
    assert 'if(vehicleRuntime.contactMonitor?.classList.contains("expanded"))updateVehicleContactMonitor();' in script
    assert 'if(vehicleRuntime.contactMonitor)vehicleRuntime.contactMonitor.hidden=false;' in script
    assert 'if(vehicleRuntime.contactMonitor)vehicleRuntime.contactMonitor.hidden=true;' in script
    assert 'kind:"vehicle-driver-seat"' not in script
    assert 'shaderViewer.geometry.push(...vehicleRuntime.frameBoxes' in script
    assert 'const mountedVehicleGeometry=vehicleRuntime.active?[' in script
    assert 'const geometry = [...baseGeometry,...mountedVehicleGeometry]' in script
    assert '["transmission-shaft","vehicle-transmission"' in script
    assert '["front-differential","vehicle-differential"' in script
    assert 'steer:index<2?' in script
    assert "basePitch+viewportControls.pitch" in script
    assert "const config=vehicle.configuration,chassis=config.chassis" in script
    assert slot["terrain_sampling"]["service"] == projection.model["contact_surfaces"]["identity"]
    assert projection.model["contact_surfaces"]["consumers"][0] == "platformer-body"
    assert vehicle["configuration"]["tires"]["pressure_pa"] == 155000.0
    assert vehicle["configuration"]["presentation"]["wheel_palette_role"] == "drivetrain-black"
    worker = projection.model["loop_deployment"]["workers"][0]["source"]
    assert "dispatchVehicleContacts" in worker
    assert 'mode:"packed-contact-gemm"' in worker
    assert "contactPass.dispatchWorkgroups" in worker
    assert "wrenchMatrix" not in worker
    assert "reductionPass.dispatchWorkgroups" in worker
    assert "result.chassis_wrench" in worker
    assert "contacts.chassis_wrench" in worker
    assert "dispatchVehicleContactWasm" in worker
    assert "lockstep-worker-contact-wasm+vehicle-wasm" in worker
    assert "function vehicleDynoRecords" in worker
    assert 'm.type==="vehicle-dyno"' in worker
    assert 'type:"vehicle-dyno-result"' in worker
    assert 'dynoButton.textContent="DYNO"' in script
    assert "function vehicleTerrainRestPose" in script
    assert "wrenchMatrix" not in script
    fallback = vehicle["physics"]["contact_fallback"]
    fallback_plugin = next(plugin for plugin in projection.model["world"]["plugins"]
                           if plugin["identity"] == fallback["plugin"])
    assert fallback_plugin["capability"] == "vehicle-contact-fallback"
    assert fallback_plugin["abi"]["authority"] == "same-symbolic-law-as-primary-webgpu-contact-kernel"
    assert "if(!Number.isFinite(state.shiftAge))" in worker
    assert "ultra_low_range_ratio" in worker
    assert "differential_lock:body.transmission?.diffLock?1:0" in worker
    assert '"chassis_torque_x","chassis_torque_y","chassis_torque_z"' in worker
    assert "attachment[1]*force[2]-attachment[2]*force[1]" not in worker
    assert "geometric_compression:geometricCompression" in worker
    assert "c2Unit((alignment-.18)/.34)" in worker
    assert "alignment>.18" not in worker
    assert "target_compression_${name}" in worker
    assert "distanceAlongSuspension>0" in worker
    assert "resolveVehicleSolidContact" in worker
    assert "vehicleCageContactWrench" in worker
    assert "resolveVehicleCagePenetration" in worker
    assert 'node.kind==="roll-cage-node"' in worker
    assert "cage_contact_stiffness" in worker
    assert "resolveWorldBottom" in worker
    assert "swept=previousY>=boundary&&nextY<bottom" in worker
    assert "contactSurfaceAt" in worker
    assert "function sampleSurface(s,x,z)" in worker
    assert 's?.kind==="sampled-height-field"' in worker
    assert "function radialTireContact" in worker
    assert "radialAngles=[-.95,-.48,0,.48,.95]" in worker
    assert "lateralFractions=[-.38,0,.38]" in worker
    assert "rollingRaw=[axle[1]*radialOut[2]" in worker
    assert "cage_static_friction" in worker and "cage_kinetic_friction" in worker
    assert "function installVehiclePresentationMesh" in script
    assert "buildExtrudedBoxMesh(dynamicGeometry" in script
    assert "drawSceneMeshes(gl)" in script
    assert "targetSpeed" not in worker
    assert "wheelOmegas" in worker
    assert "total_torque_y" in worker
    assert "rotateBodyVector" in worker
    assert "centerOfMass=c.mechanical_graph?.load_audit?.center_of_mass" in worker
    assert "steer=i<2?steeringAngle:0" in worker
    assert "steeringAngle=-(body.appliedSteering||0)" in worker
    assert "steeringAngle=-(body.controls?.steering||0)" not in worker
    assert "previous_slip_longitudinal_" in worker
    assert "friction_utilization_" in worker
    assert "body.tractionScales" in worker
    assert "frontDifferentialTorque:out.front_differential_torque" in worker
    assert 'm.type==="vehicle-recover"' in worker
    assert 'm.type==="vehicle-respawn"' in worker
    assert "body.damperScales" in worker
    assert "resolveVehicleSuspensionTravelStop(body)" in worker
    assert "support.height+s.rest_length-s.travel-attachment[1]" in worker
    channel = projection.model["loop_deployment"]["channels"][0]
    assert channel["record_layout"][8] == "vehicle.roll"
    structure = vehicle["physics"]["chassis_structure"]
    assert len(structure["nodes"]) == 4
    assert len(structure["members"]) == 6
    assert structure["pose_reduction"].startswith("sum-node-forces")
    assert structure["material"]["solver_interpretation"].startswith("rigid-limit")
    graph = vehicle["physics"]["mechanical_graph"]
    assert graph["state_law"].startswith("node-force-and-node-moment")
    assert all(node["wrench"].keys() == {"force", "moment"} for node in graph["nodes"])
    mass_nodes = {node["identity"]: node["mass_kg"] for node in graph["nodes"]
                  if node.get("mass_in_total")}
    assert mass_nodes["powertrain.engine"] == 142
    assert mass_nodes["powertrain.transmission"] == 58
    assert mass_nodes["powertrain.transfer_case"] == 24
    assert mass_nodes["powertrain.front_differential"] == 18
    assert mass_nodes["powertrain.rear_differential"] == 20
    assert all(mass_nodes[f"suspension.{corner}.wheel_rim"] == 17 for corner in WHEEL_NAMES)
    assert all(mass_nodes[f"suspension.{corner}.tire_carcass"] == 18 for corner in WHEEL_NAMES)
    assert '"front-half-shaft","vehicle-half-shaft"' not in script
    load_audit = graph["load_audit"]
    assert load_audit["spring_load_sum_n"] == pytest.approx(620 * 9.81)
    assert abs(load_audit["configured_vs_derived_front_fraction_error"]) < .001
    assert load_audit["corners"]["rear_left"]["design_supported_mass_kg"] > (
        load_audit["corners"]["front_left"]["design_supported_mass_kg"])
    constraints = {edge["constraint"] for edge in graph["edges"]}
    assert {"rigid-distance", "spring-damper", "constant-velocity-torque-shaft",
            "six-axis-compliant-mount"} <= constraints
    for corner in WHEEL_NAMES:
        corner_edges = [edge for edge in graph["edges"]
                        if edge["identity"].startswith(f"suspension.{corner}.")]
        assert sum("_arm_" in edge["identity"] for edge in corner_edges) == 4
        assert any(edge["identity"].endswith(".upright") for edge in corner_edges)
        assert any(edge["identity"].endswith(".coilover") for edge in corner_edges)
        assert sum("_pickup_mount_" in edge["identity"] for edge in corner_edges) == 4
        assert any(edge["identity"].endswith(".coilover_tower") for edge in corner_edges)
        assert any(edge["identity"].endswith(".hub_to_wheel") for edge in corner_edges)
        assert any(edge["identity"].endswith(".wheel_to_tire") for edge in corner_edges)
        assert any(edge["identity"].endswith(".wheel_bearing") for edge in corner_edges)
        assert any(edge["identity"].endswith(".service_brake") for edge in corner_edges)
    assert any(edge["identity"] == "steering.rack_and_pinion" for edge in graph["edges"])
    assert sum(edge["identity"].startswith("steering.wheel.rim_")
               for edge in graph["edges"]) == 8
    assert vehicle["physics"]["contact_patch_shape"]["shape"] == "wide-short-balloon-tire-footprint"
    assert vehicle["physics"]["cage_contact"]["samples"] == [
        "roll-cage-nodes", "cage-member-midpoints",
    ]
    initial = slot["initial_state"]
    assert initial == {
        "mounted_vehicle": vehicle["identity"],
        "placement": "at-player-spawn",
        "presentation": "full-viewport-driving",
        "browser_fullscreen": "request-on-first-user-gesture",
        "dismount_enabled": True,
    }
    torque_graph = vehicle["physics"]["torque_graph"]
    assert [node["identity"] for node in torque_graph["nodes"][:7]] == [
        "engine", "clutch", "transmission", "transfer_case", "final_drive",
        "front_differential", "rear_differential",
    ]
    transmission = torque_graph["nodes"][2]
    assert transmission["starting_gear"] == 2
    assert transmission["crawler_gear"] == 1
    assert transmission["forward_ratios"][0] > transmission["forward_ratios"][1]
    transfer_case = torque_graph["nodes"][3]
    assert transfer_case["kind"] == "two-range-transfer-case"
    assert transfer_case["ultra_low_range_ratio"] == 2.72
    assert any(edge.get("loss_model") == "smooth-drag-plus-parametric-efficiency"
               for edge in torque_graph["edges"])
    assert vehicle["physics"]["transmission_policy"]["authority"] == "lockstep-worker-state"
    assert any(edge["channel"] == "powertrain_reaction_torque_xyz"
               for edge in torque_graph["edges"])
    assert "if(!instance||tickInFlight)return" in worker
    assert "main contact bridge scheduler attempted a concurrent GPU dispatch" in projection.script
    assert "await step(body,dt)" in worker
    assert "if(vehicleRuntime.active)return;" in projection.script
    assert "vehicle-contact-monitor" in projection.script
    assert "updateVehicleContactMonitor" in projection.script
    assert 'geometry_mode === "ramp"' not in projection.script
    assert "function ramp(" not in projection.script
    assert "worldBottom:model.contact_surfaces?.world_bottom" in projection.script


def test_chassis_wasm_contract_is_force_driven_and_persists_full_pose_and_wheel_spin():
    compiled = compile_symbolic_vehicle_physics()
    arguments = set(compiled.function.metadata["argument_names"])
    publications = {item.output for item in compiled.publications}
    assert {"total_force_x", "total_force_y", "total_force_z",
            "total_torque_x", "total_torque_y", "total_torque_z"} <= arguments
    assert {"roll_next", "pitch_next", "yaw_next",
            "roll_velocity_next", "pitch_velocity_next", "yaw_velocity_next"} <= publications
    assert {f"wheel_omega_{wheel}_next" for wheel in WHEEL_NAMES} <= publications
    assert {f"traction_scale_{wheel}" for wheel in WHEEL_NAMES} <= publications
    assert {f"brake_scale_{wheel}" for wheel in WHEEL_NAMES} <= publications
    assert {f"damper_scale_{wheel}" for wheel in WHEEL_NAMES} <= publications
    assert {"differential_lock", "differential_lock_stiffness",
            "differential_lock_maximum_torque"} <= arguments
    assert {"active_damping_minimum_scale", "active_damping_maximum_scale",
            "active_damping_body_velocity_gain_s_per_m"} <= arguments
    assert {"engine_torque", "clutch_torque", "transmission_output_torque",
            "driveline_torque", "front_differential_torque",
            "rear_differential_torque", "engine_angular_acceleration",
            "powertrain_reaction_torque_x",
            "wheel_gyroscopic_reaction_torque_x",
            "wheel_gyroscopic_reaction_torque_y"} <= publications
    assert "maximum_speed" not in arguments


def test_complete_vehicle_transition_compiles_to_one_packed_webgpu_kernel():
    artifact = compile_symbolic_vehicle_physics_webgpu()
    assert artifact.complete
    assert artifact.shortfalls == ()
    assert "@compute" in artifact.source
    assert "var<storage, read> feeds" in artifact.source
    assert "var<storage, read_write> outputs" in artifact.source


def test_gpu_terrain_contact_is_spatially_and_temporally_top_skin_crossing_aware():
    program = vehicle_webgpu_program_model(load_default_car_configuration())
    source = program["terrain_contact_geometry"]["kernel"]["source"]
    assert "spatial_crossing" in source
    assert "temporal_crossing" in source
    assert "mix(world_hub, probe, fraction)" in source
    assert "support_position = evaluation_position" in source
    assert "wall_colliders: array<f32>" in source
    assert "let wall_count = u32(terrain_parameters[11u])" in source
    assert "let slice_hub = world_hub + axle" in source
    assert "tire_radial_compression" in source
    assert program["terrain_contact_geometry"]["terrain_parameter_abi"][-1] == "wall_count"
    assembly = program["graph_adapters"]["assembly"]["source"]
    assert "normal_load * normal_load" not in assembly
    assert "normal_load_0 * normal_load_0" in assembly
    assert program["terrain_contact_geometry"]["terrain_upload_policy"] == (
        "initialization-or-authored-height-edit-only"
    )
    assert program["graph_adapters"]["host_awaits_between_nodes"] == 0


def test_abs_and_traction_sensors_are_compiled_persistent_second_order_states():
    equations, symbols = symbolic_vehicle_equations()
    publications = {str(equation.lhs) for equation in equations}
    assert "slip_sensor_frequency_hz" in symbols
    assert "utilization_sensor_frequency_hz" in symbols
    for wheel in WHEEL_NAMES:
        assert f"slip_sensor_velocity_{wheel}" in symbols
        assert f"measured_friction_utilization_{wheel}" in symbols
        assert f"slip_sensor_velocity_{wheel}_next" in publications
        assert f"friction_utilization_{wheel}_next" in publications
        assert f"friction_utilization_sensor_velocity_{wheel}_next" in publications


def test_symbolic_diff_lock_exchanges_equalizing_torque_across_each_axle():
    equations, symbols = symbolic_vehicle_equations()
    outputs = {str(equation.lhs): equation.rhs for equation in equations}
    config = load_default_car_configuration()
    values = {name: 0.0 for name in symbols}
    values.update(config.parameter_defaults())
    values.update({"dt": 1 / 120, "yaw_cos": 1.0, "transfer_case_ratio": 1.0,
                   "wheel_omega_front_left": 12.0, "wheel_omega_front_right": 0.0})

    def wheel_pair(locked: float) -> tuple[float, float]:
        substitutions = {symbols[name]: value for name, value in values.items() if name in symbols}
        substitutions[symbols["differential_lock"]] = locked
        return tuple(float(outputs[f"wheel_omega_{name}_next"].evalf(subs=substitutions))
                     for name in ("front_left", "front_right"))

    open_left, open_right = wheel_pair(0.0)
    locked_left, locked_right = wheel_pair(1.0)
    assert locked_left < open_left
    assert locked_right > open_right


def test_emitted_wasm_turns_pedal_into_wheel_spin_not_commanded_chassis_speed():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute WebAssembly")
    artifact = compile_symbolic_vehicle_physics_wasm()
    config = load_default_car_configuration()
    values = {name: 0.0 for name in artifact.input_names}
    values.update(config.parameter_defaults())
    values.update({"dt": 1 / 120, "throttle": 1.0, "yaw_cos": 1.0,
                   "transfer_case_ratio": 1.0,
                   "roll_velocity": .5, "pitch_velocity": 1.0,
                   "drive_fraction_front_left": .21, "drive_fraction_front_right": .21,
                   "drive_fraction_rear_left": .29, "drive_fraction_rear_right": .29,
                   "friction_utilization_front_left": 1.3,
                   "slip_longitudinal_front_left": 1.5,
                   "previous_slip_longitudinal_front_left": .1})
    script = r"""
    const payload=JSON.parse(require("fs").readFileSync(0,"utf8")),bytes=Buffer.from(payload.bytes,"base64");
    (async()=>{const {instance}=await WebAssembly.instantiate(bytes,{}),memory=new Float64Array(instance.exports.memory.buffer);
    memory.set(payload.values,Number(process.argv[1])/8);instance.exports.abstract_ui_vehicle_step(0);
    const start=Number(process.argv[2])/8,count=Number(process.argv[3]);
console.log(JSON.stringify(Array.from(memory.slice(start,start+count))));})().catch(error=>{console.error(error);process.exit(1)});
"""
    completed = subprocess.run([
        node, "-e", script, str(artifact.input_offsets[0]), str(artifact.output_offsets[0]),
        str(len(artifact.output_names)),
    ], input=json.dumps({"bytes": base64.b64encode(artifact.binary).decode("ascii"),
                         "values": [values[name] for name in artifact.input_names]}),
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    result = dict(zip(artifact.output_names, json.loads(completed.stdout)))
    assert result["velocity_x_next"] == pytest.approx(0.0)
    assert result["wheel_omega_front_left_next"] > 0
    assert result["wheel_omega_rear_left_next"] > result["wheel_omega_front_left_next"]
    assert 0.08 <= result["traction_scale_front_left"] < 1.0
    assert .08 <= result["brake_scale_front_left"] < 1.0
    assert result["brake_scale_front_left"] <= result["traction_scale_front_left"]
    assert .9 < result["traction_scale_front_right"] <= 1.0
    assert 1.0 < result["damper_scale_front_left"] <= 1.18
    assert result["damper_scale_front_left"] > result["damper_scale_front_right"]
    assert result["engine_torque"] > 0
    assert result["clutch_torque"] < result["engine_torque"]
    assert result["transmission_output_torque"] > result["clutch_torque"]
    assert result["driveline_torque"] > result["transmission_output_torque"]
    assert result["front_differential_torque"] == pytest.approx(result["driveline_torque"] * .42)
    assert result["rear_differential_torque"] == pytest.approx(result["driveline_torque"] * .58)
    assert result["engine_angular_acceleration"] > 0
    assert result["powertrain_reaction_torque_x"] == pytest.approx(
        -result["engine_acceleration_torque"])
    assert result["powertrain_reaction_torque_y"] == pytest.approx(0)
    assert result["powertrain_reaction_torque_z"] == pytest.approx(0)
