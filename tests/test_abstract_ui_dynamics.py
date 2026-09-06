"""Device telemetry and the explicit user/world dynamics boundary."""

from src.compiler.abstract_ui_dynamics import DeviceMonitor, viewport_dynamics_space
from src.compiler.abstract_ui_viewports import ViewportControlPolicy


def test_device_monitor_is_derived_from_ordered_control_bindings():
    policy = ViewportControlPolicy("controls", "actor")
    monitor = DeviceMonitor.from_bindings("monitor", "actor", policy.bindings)
    data = monitor.to_data()

    assert [group["device"] for group in data["groups"]] == [
        "pointer", "keyboard", "gamepad",
    ]
    keyboard = next(group for group in data["groups"] if group["device"] == "keyboard")
    assert [(signal["label"], signal["action"]) for signal in keyboard["signals"]] == [
        ("W", "move-forward"), ("S", "move-backward"),
        ("A", "strafe-left"), ("D", "strafe-right"),
        ("Shift", "run"), ("Shift", "run"), ("Space", "jump"),
    ]
    assert len({signal["source"] for group in data["groups"] for signal in group["signals"]}) == 17


def test_dynamics_space_keeps_user_integration_separate_from_world_physics():
    data = viewport_dynamics_space("world", "actor").to_data()
    user, physics = data["lanes"]

    assert (user["kind"], user["phase"]) == ("user-dynamics", "integrate-user")
    assert all(channel["status"] == "bound" for channel in user["channels"])
    assert (physics["kind"], physics["phase"]) == ("world-physics", "solve-world")
    assert physics["channels"] == [
        {"name": "geometry", "status": "bound"},
        {"name": "contacts", "status": "unbound"},
        {"name": "collision", "status": "unbound"},
        {"name": "gravity", "status": "unbound"},
    ]
    assert [stage["operation"] for stage in physics["stages"]] == [
        "specialize-world-identities", "weld-static-collider-batches",
        "broad-phase-player-world", "narrow-phase-contacts",
        "resolve-player-contacts", "publish-physics-pose",
    ]
    assert all(stage["status"] == "selected-unbound" for stage in physics["stages"])
    assert physics["dispatch_policy"]["backend_candidates"] == [
        "webgpu-compute", "wasm-simd", "cpu",
    ]
    assert physics["dispatch_policy"]["welded_world"] is True
    assert physics["equation_program"] == {
        "source_language": "sympy-equation-set",
        "selection": "stage-selection-defines-active-equations",
        "lowering": [
            "sympy-expressions", "canonical-process-graph",
            "compiler-ssa", "webassembly",
        ],
        "state_layout": "dense-runtime-identities-plus-typed-arrays",
        "semantic_identity_authority": "world.identity_specialization",
        "status": "contract-only-unbound",
    }
    assert data["dependencies"] == [
        {"relationship": "integrates", "target": "actor"},
        {"relationship": "solves", "target": "world"},
        {"relationship": "clocked-by", "target": "world/timer"},
    ]
