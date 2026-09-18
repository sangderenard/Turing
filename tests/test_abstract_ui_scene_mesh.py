"""Contracts for identity-preserving AbstractUI scene mesh construction."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from src.compiler.abstract_ui_scene_mesh import compile_scene_mesh_wasm


def test_scene_mesh_retains_topology_identity_form_and_round_trip_contracts():
    model = compile_scene_mesh_wasm().to_model()
    assert model["source_language"] == "python"
    assert model["source"].startswith("def instantiate_box_vertex(")
    assert model["binary_base64"]
    assert model["topology"]["vertices_per_instance"] == 36
    assert sum(len(face["corners"]) for face in model["topology"]["faces"]) == 36
    assert model["identity_spans"] == {
        "source": "document_geometry.boxes[].identity",
        "ordering": "document-order",
        "unit": "vertex",
        "span_size": "composed-primitive-count-times-36",
        "primitive_vertices": 36,
    }
    form = model["context_menu"]["items"][0]
    assert form["label"] == "Form"
    assert {item["parameter"] for item in form["instructions"]} == {
        "height", "half_extent.x", "half_extent.z", "*",
    }
    assert model["round_trip"]["document_sink"] == "dom[data-node-id=identity]"
    assert model["boundary_contract"] == {
        "source": "dom-border",
        "meaning": "wall",
        "identity": "document-object-identity",
        "height_parameter": "height",
        "thickness_parameter": "boundary.thickness",
        "opening_order": "document-order",
        "opening_kinds": ["door", "window", "portal"],
        "composition": "identity-preserving-boundary-union",
        "floor": "mandatory-slab",
        "interior": "hollow",
        "ceiling_rule": "height-at-absolute-maximum",
        "absolute_maximum_height": 4.0,
        "opening_operation": "boundary-minus-ordered-openings",
        "bevel_parameter": "radius",
        "bevel_realization": "unimplemented",
    }


def test_compiled_scene_mesh_executes_with_published_abi(tmp_path: Path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    artifact = compile_scene_mesh_wasm()
    wasm_path = tmp_path / "scene-mesh.wasm"
    wasm_path.write_bytes(artifact.binary)
    parameters = json.dumps([parameter["name"] for parameter in artifact.parameters])
    script = f"""
const {{readFileSync}} = require("fs");
(async () => {{
  const {{instance}} = await WebAssembly.instantiate(readFileSync(process.argv[1]), {{}});
  const names = {parameters};
  const values = {{center_x:2, center_z:3, half_x:1, half_z:0.5, height:2,
    unit_x:-1, unit_y:1, unit_z:1}};
  const offsets = {{}}; let cursor = 0;
  names.slice(1).forEach(name => {{ offsets[name] = cursor; cursor += 8; }});
  Object.entries(values).forEach(([name, value]) =>
    new Float64Array(instance.exports.memory.buffer, offsets[name], 1)[0] = value);
  instance.exports.instantiate_scene_mesh(...names.map(name => name === "count" ? 1 : offsets[name]));
  console.log(JSON.stringify(names.slice(-3).map(name =>
    new Float64Array(instance.exports.memory.buffer, offsets[name], 1)[0])));
}})();
"""
    completed = subprocess.run(
        [node, "-e", script, str(wasm_path)], check=True, capture_output=True, text=True,
    )
    assert json.loads(completed.stdout) == pytest.approx([1.0, 2.0, 3.5])
