import pytest

from src.compiler.state_loop_deployment import StateLoop, identify_state_loops, living_map_loop_deployment, plan_state_loops, state_loop


def test_living_map_separates_fixed_physics_from_presenting_graphics():
    model = living_map_loop_deployment("world", "physics")
    hosts = {p["loop"]["clock"]: p["execution_host"] for p in model["placements"]}
    assert hosts["fixed-step"].startswith("worker:")
    assert hosts["animation-frame"] == "main"
    assert model["channels"][0]["policy"] == "latest-complete-snapshot"
    source = model["workers"][0]["source"]
    assert "setInterval(tick" in source
    assert 'm.type==="recycle"' in source
    assert 'm.type==="impulse"' in source
    assert 'm.type==="player-jump"' in source
    assert 'armEngine("player-jump-force")' in source
    assert 'm.type==="support"' in source
    assert "function topSupport(previous,next,r,excluded)" in source
    assert "function sampledTerrainSupport(previous,next,r,excluded)" in source
    assert "previous[1]-r>=top-.012&&next[1]-r<=top+.006" in source
    assert "WebAssembly.instantiate" in source
    assert "new ArrayBuffer(snapshotCapacity*SNAPSHOT_STRIDE*8)" in source
    assert "const SNAPSHOT_STRIDE=162" in source
    assert "vehicle.radial-probe-penetration[4][15]" in str(model["channels"][0]["record_layout"])
    assert "FIXED_DT=" in source
    assert "PHYSICS_SUBSTEPS=3" in source
    assert "body.outriggerWrench={force:totalForce,torque:totalTorque}" in source
    assert "body.position[axis]+=correction[axis]" not in source
    assert '"locking-and-leveling"' in source
    assert "energy.ignitionOn=false;runtime.phase=\"deploying\"" in source
    assert 'm.type==="vehicle-outrigger-hand-pump"' in source
    assert "SUBSTEP_DT=FIXED_DT/PHYSICS_SUBSTEPS" in source
    assert 'dt=engineStage==="kinematic-coast"?1/30:FIXED_DT' in source
    assert "snapshot.buffer.mapAsync(GPUMapMode.READ).then" in source
    assert "snapshot.epoch=Number(body.gpuEpoch||0)" in source
    assert "bodies.get(body.identity)!==body" in source
    assert "async function tick" not in source
    assert "await step(body,dt)" not in source
    assert "resident-webgpu-graph" in source
    assert "resident-webgpu-fault" in source
    assert "checkedShaderModule" in source
    assert "wallColliders" in source
    assert "const fields=allColliders.filter" in source
    assert "parameters=new Float32Array([fields.length,walls.length,...descriptors])" in source
    assert "vehicleGpu?.residentGraph&&vehicleGpu.terrainReady" in source
    assert "function residentVehicleWasmStep(body,dt)" in source
    assert "function terrainSegmentCrossing(start,finish,body,reach)" in source
    assert "function solidSegmentCrossing(start,finish,body)" in source
    assert "function analyticTorusTireContact" in source
    assert "radialContact=analyticTorusTireContact" in source
    assert "arcAngle=2*(Math.PI-arcStart)" in source
    assert "function radialTireContact" not in source
    assert "terrainSegmentCrossing(ringPoint,nextRing,body,reach)||solidSegmentCrossing(ringPoint,nextRing,body)" in source
    assert "for(let subdivision=1;subdivision<=8;subdivision++)" in source
    assert "for(let iteration=0;iteration<8;iteration++)" in source
    assert "penetrationMax=Math.max(penetrationMax,penetration)" not in source
    assert "integralCompression=penetrationMax" not in source
    assert "function tireHalfSpaceOverlap" not in source
    assert "runScalarWasm(contactInstance,contactAbi" in source
    assert "runScalarWasm(vehicleInstance,vehicleAbi" in source
    assert "else if(vehicleInstance&&contactInstance)for(let substep=0;substep<PHYSICS_SUBSTEPS;substep++)" in source
    assert 'type:"vehicle-wasm-fallback"' in source
    assert '"resident-wasm-fallback"' in source
    assert "vehicle-contact-bridge" not in source
    assert "dispatchVehicleContacts" not in source
    channel = next(item for item in model["channels"] if "world.body-pose" in item["fields"])
    assert "vehicle.front-differential-torque" in channel["record_layout"]
    assert "vehicle.powertrain-reaction-torque[3]" in channel["record_layout"]
    assert channel["transport"] == "transferable-array-buffer-pool"
    assert channel["synchronization"] == "ownership-transfer-no-locks"
    assert model["scheduler"]["membership"] == {
        "policy": "dynamic-awake-set",
        "drop": "host-observed-supported-low-speed-body",
        "restore": ["collision-touch", "physics-field-change"],
        "identity": "retained-outside-solver",
    }
    assert model["scheduler"]["engine_gears"] == {
        "full_dynamics": "force-integration-at-120hz",
        "kinematic_coast": {
            "condition": "moving-with-constant-velocity",
            "acceleration_epsilon": 0.001,
            "confirmation_ticks": 12,
            "action": "skip-force-wasm-and-advance-at-30hz",
            "guards": ["bounds", "static-collision"],
        },
        "asleep": {
            "condition": "all-members-quiescent",
            "delay_ticks": 180,
            "velocity_epsilon": 0.012,
            "action": "disarm-fixed-step-timer",
        },
            "wake_events": [
                "body-upsert", "control", "impulse", "wrench-change",
            "collider-field-change", "physics-field-change",
        ],
        "telemetry": "full-dynamics-kinematic-coast-or-asleep",
        }
    assert model["wrench_abi"]["applies_to"] == "every-physics-body"
    assert model["wrench_abi"]["message"] == "wrench"
    assert 'm.type==="wrench"' in source
    assert 'm.type==="vehicle-transmission"' in source
    assert "function updateVehicleTransmission" in source
    assert 'state.reason=gear===2?"crawler-demand-integrated":"downshift-demand-integrated"' in source
    assert "if(!Number.isFinite(state.shiftAge))" in source
    assert 'command.lowRange?"driver-ultra-low":"driver-high-range"' in source
    assert 'm.type==="vehicle-dyno"' in source
    assert 'type:"vehicle-dyno-result"' in source
    assert "function applyPendingVehicleCommands(body)" in source
    assert "applyPendingVehicleCommands(body);prepareVehicleEnergy(body,dt);prepareVehicleControls(body,dt);" in source
    assert "function prepareVehicleControls(body,dt)" in source
    assert 'state.reason=command.smoothLaunch?"driver-smooth-launch":"driver-direct-launch"' in source
    assert "body.brakeLocks" in source
    assert "brake_lock_${name}" in source
    assert "function ensureVehicleDamage(body)" in source
    assert 'body.damage={mode:"parametric-damage"' in source
    assert "function ensureVehicleEnergy(body)" in source
    assert 'm.type==="vehicle-fuel-ignition"' in source
    assert 'm.type==="vehicle-driver-assistance"' in source
    assert "function ensureVehicleDriverAssistance(body)" in source
    assert "driver.cruiseTargetSpeedMps-roadSpeed" in source
    assert "governor_angular_speed:governorOmega" in source
    assert "tippingAngle=Math.atan2(rearLever,cgHeight)" in source
    assert "driver.rearDifferentialBrakeCommand" in source
    assert "body.rearDifferentialBrakeCommand" in source
    assert "state.requestedIgnitionProfileIdentity" in source
    assert 'requestedIgnitionProfile.dispatch==="ecu-electronic"' in source
    assert "state.brakeLightsOn=" in source
    assert 'm.type==="vehicle-pneumatics"' in source
    assert 'policy.mode==="manual-wheel"' in source
    assert "state.pneumaticCompressorPowerW" in source
    assert "state.hydraulicPumpPowerW" in source
    assert "tire_reference_volume:2*Math.PI*Math.PI" in source
    assert "sidewall_deformation_lateral:Number(body.tireDeformation" in source
    assert "slip_lateral_${name}" in source
    assert "bump_stop_progressive_stiffness:s.bump_stop_progressive_stiffness_n_per_m2" in source
    assert "omega>12||starterAllowed" in source
    assert "engine_angular_speed:body.energy?.ignitionOn===false?0" in source
    assert "body.frontKnuckleSteerAngle||0" in source
    assert "const solveCorner=(corner,rackTravel,connected)" in source
    assert "neutralLength2=" in source
    assert "body.wheelSteerAngles" in source
    assert "frontRackTravel:frontTravel" in source
    assert "ecuAvailable=energy.ecuOnline&&!ecuFeedFailed" in source
    assert "speedBlend=Math.pow(c2Unit(roadSpeed/referenceSpeed),rateCurve)" in source
    assert "allowedRate=ecuSteeringActive?ecuRate:servoPowered?assistedRate:manualRate" in source
    assert '"manual-human-force"' in source
    assert "humanTorque-resistingTorque" in source
    assert "steeringServoPowerW" in source
    assert "steeringCommand=Number(body.appliedSteering" in source
    assert "ecuVelocityRateControl" in source
    assert "reversingAgainstMotion" in source
    assert "directionChangeBrake" in source
    assert "body.featheredThrottleVelocity=0" in source
    assert "includeSolidTops=true" in source
    assert "body.identity,false" in source
    assert "function vehicleDriveFractions(body)" in source
    assert "function updateVehicleDamage(body,dt)" in source
    assert 'damage.mode="parametric-damage"' in source
    assert "requestParametricVehiclePipelines(body,`damage · ${reason}`)" in source
    assert 'm.type==="vehicle-chassis-profile"' in source
    assert 'm.type==="vehicle-chassis-leveling"' in source
    assert 'm.type==="vehicle-body-wrenches"' in source
    assert 'm.type==="vehicle-outriggers"' in source
    assert "function updateVehicleOutriggers(body,dt)" in source
    assert "terrainSegmentCrossing(previous,foot,body,2.4)" in source
    assert "runtime.brakeInterlock" in source
    assert "body.brakeLocks={front_left:true,front_right:true,rear_left:true,rear_right:true}" in source
    assert "body.linkLengthModifiers[edge.identity]" in source
    assert "b.bodyAssemblyWrenches.push" in source
    assert 'm.type==="vehicle-steering-system"' in source
    assert 'm.type==="vehicle-parameters"' in source
    assert "function applyVehicleChassisProfile(body,profile)" in source
    assert "energy.dryMassKg=Math.max(1,Number(geometry.mass)-initialFuel);energy.lastMassKg=-1" in source
    assert "function updateVehicleChassisLeveling(body,dt)" in source
    assert "function updateVehicleSteeringWrench(body,dt)" in source
    assert 'requestParametricVehiclePipelines(body,"shock parameter control")' in source
    assert "Number.POSITIVE_INFINITY" in source
    assert "body.supportSurfaceLatch" in source
    assert "maximum_sink_depth_m" in source
    assert "if(resolveWorldBottom(body,previousPosition))" in source
    assert "function resetVehicleDrivetrainState(body,reason)" in source
    assert 'if(body.kind==="vehicle"){body.supportSurfaceLatch=null;body.bottomRejected=false;return false;}' in source
    assert "No post-step positional rejection" in source
    assert "Object.values(output).every(Number.isFinite)" in source
    assert "vehicle entered the world-bottom guard before integration" in source
    assert "resident GPU crossed the world-bottom guard" in source
    assert "halfshaftHealth" in source
    assert "springPlasticSet" in source
    assert 'm.type==="vehicle-power-unit"' in source
    assert "b.pendingControls={" in source
    assert 'throw new Error("vehicle graph scheduler attempted a concurrent GPU dispatch")' in source
    assert "function armEngine(reason)" in source
    assert "function coastStep(body,dt)" in source
    assert 'setEngineStage("kinematic-coast"' in source
    assert 'setEngineStage("asleep"' in source


def test_state_fields_have_one_writer():
    loops = [StateLoop("a", "x", "event", ("pose",)), StateLoop("b", "x", "event", ("pose",))]
    with pytest.raises(ValueError, match="two writers"):
        plan_state_loops(loops)


def test_worker_isolation_rejects_main_thread_effects():
    with pytest.raises(ValueError, match="presentation effects"):
        plan_state_loops((StateLoop("paint", "ui", "event", (),
                                    effects=("dom",), isolation="worker"),))


def test_source_annotation_is_recognized_without_executing_user_code():
    loops = identify_state_loops('''
@state_loop(domain="world.physics", clock="fixed-step", frequency_hz=120, isolation="worker")
def integrate(self, dt):
    self.position = self.position + self.velocity * dt
''')
    assert len(loops) == 1
    assert loops[0].clock == "fixed-step"
    assert loops[0].frequency_hz == 120
    assert loops[0].writes == ("position",)
    assert plan_state_loops(loops)["placements"][0]["execution_host"].startswith("worker:")


def test_runtime_annotation_preserves_python_function():
    @state_loop(domain="physics", clock="fixed-step", frequency_hz=60, isolation="worker")
    def tick(value):
        return value + 1

    assert tick(2) == 3
    assert tick.__abstract_ui_state_loop__["frequency_hz"] == 60
