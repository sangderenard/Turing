"""SymPy-authored parametric world physics and direct WASM lowering."""

import json
import shutil
import subprocess

import pytest
import sympy

from src.compiler.abstract_ui_physics import (
    PHYSICS_STATE_OUTPUTS,
    compile_symbolic_world_physics,
    compile_symbolic_world_physics_wasm,
    symbolic_world_physics_wasm_plugin,
    symbolic_world_physics_equations,
)


def _defaults():
    model = symbolic_world_physics_equations()
    values = {parameter.name: parameter.default for parameter in model.parameters}
    values.update({
        "position_x": 0.0, "position_y": 0.2875, "position_z": 0.0,
        "velocity_x": 0.0, "velocity_y": 0.0, "velocity_z": 0.0,
        "dt": 0.01,
    })
    return values


def _evaluate(values):
    model = symbolic_world_physics_equations()
    ordered = sorted(
        set().union(*(equation.rhs.free_symbols for equation in model.equations)),
        key=str,
    )
    function = sympy.lambdify(
        ordered, [equation.rhs for equation in model.equations], "math",
    )
    return dict(zip(
        (str(equation.lhs) for equation in model.equations),
        function(*(values[str(symbol)] for symbol in ordered)),
    ))


def test_gravity_floor_contact_and_drag_are_one_simultaneous_transition():
    values = _defaults()
    values.update({"position_y": 3.0, "velocity_y": 0.0})
    falling = _evaluate(values)
    assert falling["velocity_y_next"] < 0
    assert falling["position_y_next"] < values["position_y"]

    values.update({"position_y": 0.2875, "velocity_y": 0.0})
    floor = _evaluate(values)
    assert floor["position_y_next"] == pytest.approx(0.2875)
    assert floor["velocity_y_next"] == pytest.approx(0.0)
    assert floor["contact_penetration"] > 0


def test_portal_transposes_position_and_velocity_about_authored_anchors():
    values = _defaults()
    values.update({
        "gravity_y": 0.0, "linear_drag": 0.0, "portal_active": 1.0,
        "position_x": 2.0, "position_y": 1.0, "position_z": 3.0,
        "velocity_x": 4.0, "velocity_y": 0.0, "velocity_z": 0.0,
        "portal_source_x": 1.0, "portal_source_y": 1.0, "portal_source_z": 1.0,
        "portal_target_x": 10.0, "portal_target_y": 2.0, "portal_target_z": 20.0,
        "portal_cos": 0.0, "portal_sin": 1.0,
        "minimum_x": -100.0, "minimum_y": -100.0, "minimum_z": -100.0,
        "maximum_x": 100.0, "maximum_y": 100.0, "maximum_z": 100.0,
    })
    result = _evaluate(values)
    # Integrate first, then rotate the source-relative x/z vector by +90°.
    assert result["position_x_next"] == pytest.approx(8.0)
    assert result["position_y_next"] == pytest.approx(2.0)
    assert result["position_z_next"] == pytest.approx(21.04)
    assert result["velocity_x_next"] == pytest.approx(0.0)
    assert result["velocity_z_next"] == pytest.approx(4.0)


def test_selected_solid_wall_plane_rejects_the_player_on_contact():
    values = _defaults()
    values.update({
        "gravity_y": 0.0, "linear_drag": 0.0,
        "position_x": 0.95, "velocity_x": 10.0,
        "minimum_x": -100.0, "minimum_y": -100.0, "minimum_z": -100.0,
        "maximum_x": 100.0, "maximum_y": 100.0, "maximum_z": 100.0,
        # The player approaches the x=1 wall from its left side. The host
        # publishes +x as the violating normal and the face coordinate as plane.
        "obstacle_active": 1.0, "obstacle_normal_x": 1.0,
        "obstacle_normal_z": 0.0, "obstacle_plane": 1.0,
    })
    result = _evaluate(values)
    assert result["position_x_next"] == pytest.approx(1.0 - values["radius"])
    assert result["velocity_x_next"] == pytest.approx(-1.25)
    assert result["contact_penetration"] == pytest.approx(0.1125)


def test_equations_lower_through_repository_ssa_without_fallbacks():
    model = symbolic_world_physics_equations()
    compiled = compile_symbolic_world_physics()
    assert len(model.equations) == 8
    assert tuple(compiled.output_ids) == (*PHYSICS_STATE_OUTPUTS,
                                          "contact_penetration",
                                          "specific_kinetic_energy")
    assert compiled.process_graph.G.graph["sympy_translation_fallbacks"] == ()
    assert compiled.function.metadata["symbolic_source"] == "sympy"


def test_parametric_equations_emit_and_execute_as_webassembly():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute WebAssembly")
    artifact = compile_symbolic_world_physics_wasm()
    values = _defaults()
    values["position_y"] = 3.0
    ordered = [values[name] for name in artifact.input_names]
    script = r"""
const bytes = Buffer.from(process.argv[1], "base64");
(async () => {
  const {instance} = await WebAssembly.instantiate(bytes, {});
  const memory = new Float64Array(instance.exports.memory.buffer);
  memory.set(JSON.parse(process.argv[2]), Number(process.argv[3]) / 8);
  instance.exports.abstract_ui_world_physics_step(0);
  const start = Number(process.argv[4]) / 8;
  console.log(JSON.stringify(Array.from(memory.slice(start, start + 8))));
})().catch(error => { console.error(error); process.exit(1); });
"""
    import base64
    completed = subprocess.run([
        node, "-e", script, base64.b64encode(artifact.binary).decode("ascii"),
        json.dumps(ordered), str(artifact.input_offsets[0]),
        str(artifact.output_offsets[0]),
    ], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    results = dict(zip(artifact.output_names, json.loads(completed.stdout)))
    assert results["velocity_y_next"] < 0
    assert results["position_y_next"] < 3.0


def test_physics_plugin_publishes_the_direct_scalar_arena_abi():
    plugin = symbolic_world_physics_wasm_plugin().to_data(include_binary=False)
    assert plugin["source_language"] == "sympy"
    assert plugin["capability"] == "physics"
    assert plugin["host_contract"]["invocation"] == "arena-base-pointer"
    assert plugin["abi"]["kind"] == "ssa-scalar-arena-v0"
    assert plugin["abi"]["output_names"][-2:] == [
        "contact_penetration", "specific_kinetic_energy",
    ]
