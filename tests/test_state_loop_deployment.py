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
    assert 'm.type==="support"' in source
    assert "function topSupport(previous,next,r,excluded)" in source
    assert "function sampledTerrainSupport(previous,next,r,excluded)" in source
    assert "previous[1]-r>=top-.012&&next[1]-r<=top+.006" in source
    assert "WebAssembly.instantiate" in source
    assert "new ArrayBuffer(snapshotCapacity*SNAPSHOT_STRIDE*8)" in source
    assert "const SNAPSHOT_STRIDE=71" in source
    assert "FIXED_DT=" in source
    assert 'dt=engineStage==="kinematic-coast"?1/30:FIXED_DT' in source
    assert "snapshot.buffer.mapAsync(GPUMapMode.READ).then" in source
    assert "async function tick" not in source
    assert "await step(body,dt)" not in source
    assert "resident-webgpu-graph" in source
    assert "resident-webgpu-fault" in source
    assert "checkedShaderModule" in source
    assert "wallColliders" in source
    assert "if(!vehicleGpu.terrainReady)" in source
    assert "vehicleWasm" not in source
    assert "contactWasm" not in source
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
    assert "applyPendingVehicleCommands(body);" in source
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
