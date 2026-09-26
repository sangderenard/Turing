"""World-object, Pluck adapter, and WebAssembly plugin contracts."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from src.compiler.abstract_ui_world import (
    WorldObject,
    compile_world_transform_wasm,
    document_world_objects,
    pluck_placed_object,
    world_graph_model,
)


def _geometry():
    return {
        "coordinate_space": "data-world",
        "boxes": [
            {
                "identity": "world/courtyard",
                "kind": "courtyard",
                "parent_identity": "world",
                "center": [2.0, 3.0],
                "half_extent": [1.0, 1.5],
                "height": 0.6,
                "floor_height": 0.03,
                "wall_thickness": 0.05,
                "radius": 8.0,
                "palette_role": "courtyard-face",
                "wall_palette_role": "courtyard-wall",
                "openings": [{
                    "identity": "world/courtyard/opening:gate",
                    "kind": "gate", "side": "south", "offset": 0.0,
                    "width": 0.8, "height": 0.6,
                }],
            },
        ],
    }


def test_document_geometry_promotes_to_game_ready_world_objects():
    (item,) = document_world_objects("world", _geometry())
    data = item.to_data()
    assert data["parent"] == "world"
    assert data["form"]["recipe"] == "boundary-floor-with-openings"
    assert data["material_bindings"] == {
        "floor": "courtyard-face", "walls": "courtyard-wall",
    }
    assert data["physics"] == {
        "body": "static",
        "collider": "boundary-shell-plus-floor",
        "collision_authority": "world-physics",
        "enabled": True,
    }
    roles = [part["role"] for part in data["semantic_parts"]]
    assert roles == ["floor", "boundary-wall", "boundary-wall",
                     "boundary-wall", "boundary-wall", "opening"]
    assert data["extensions"]["abstract_ui.document_geometry"]["openings"]
    assert data["extensions"]["pluck.compatibility"]["triangle_group_actions"]


def test_pluck_placed_object_adapter_retains_unknown_game_metadata_losslessly():
    payload = {
        "id": "camera_lab",
        "type": "camera",
        "label": "Lab camera",
        "pos": [1.0, 2.0, 3.0],
        "yaw_deg": 30.0,
        "mesh_id": "camera_video",
        "sensor_name": "full_frame_35mm",
        "lens_name": "zoom_24_70",
        "material_bindings": {"body": "matte_black"},
        "future_game_field": {"kept": [1, 2, 3]},
    }
    data = pluck_placed_object(payload, parent="room-a").to_data()
    assert data["form"]["mesh_preset"] == "camera_video"
    assert {"enter-camera", "aim", "focus"} <= set(data["capabilities"])
    assert data["extensions"]["pluck.placed_object"] == payload


def test_world_graph_publishes_identity_and_semantic_mesh_packet_contract():
    plugin = compile_world_transform_wasm()
    model = world_graph_model("world", _geometry(), plugins=(plugin,))
    assert model["object_order"] == ["world/courtyard"]
    assert model["mesh_packet"] == {
        "schema": "abstract-ui-world-mesh-packet-v0",
        "topology": "triangle-list",
        "vertex_layout": ["position.xyz", "normal.xyz", "color.rgb"],
        "identity_table": "variable-length-object-spans",
        "semantic_part_table": "variable-length-part-spans",
        "material_binding_table": "world-object-material-bindings",
        "revision_source": "living-document-edit-revision",
        "authority": "world-object-recipes-not-renderer-buffers",
    }
    assert model["plugins"][0]["operation"] == "transform-position-yaw"
    assert "binary_base64" not in model["plugins"][0]
    assert model["plugins"][0]["module"] == model["wasm_modules"][0]["content_key"]
    assert model["wasm_modules"][0]["binary_base64"]
    specialization = model["identity_specialization"]
    assert specialization["authority"] == "authored-string-identity"
    assert specialization["missing_runtime_id"] == 0
    assert specialization["objects"] == [{
        "runtime_id": 1, "identity": "world/courtyard",
    }]
    assert [part["runtime_id"] for part in specialization["semantic_parts"]] == list(
        range(1, 7)
    )
    assert all(part["object_runtime_id"] == 1 for part in specialization["semantic_parts"])


def test_conceptual_world_object_can_own_separate_geometry_realization():
    owner = WorldObject(
        "world/rig", "rig", "world", "Rig", {}, {"kind": "rig"},
    )
    geometry = _geometry()
    geometry["boxes"][0]["parent_identity"] = owner.identity
    model = world_graph_model(
        "world",
        geometry,
        conceptual_objects=(owner,),
        properties={"validator_rig": owner.identity},
    )
    assert model["properties"] == {"validator_rig": owner.identity}
    assert model["conceptual_object_order"] == [owner.identity]
    child = next(item for item in model["objects"] if item["identity"] == "world/courtyard")
    assert child["parent"] == owner.identity


def test_world_transform_plugin_executes_through_published_wasm_abi(tmp_path: Path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    plugin = compile_world_transform_wasm()
    wasm_path = tmp_path / "world-transform.wasm"
    wasm_path.write_bytes(plugin.binary)
    parameters = json.dumps([item["name"] for item in plugin.parameters])
    entrypoint = json.dumps(plugin.entrypoint)
    script = f"""
const {{readFileSync}} = require("fs");
(async () => {{
  const {{instance}} = await WebAssembly.instantiate(readFileSync(process.argv[1]), {{}});
  const names = {parameters};
  const values = {{local_x:1, local_y:2, local_z:0,
    translate_x:10, translate_y:20, translate_z:30, yaw_cos:0, yaw_sin:1}};
  const offsets = {{}}; let cursor = 0;
  names.slice(1).forEach(name => {{ offsets[name] = cursor; cursor += 8; }});
  Object.entries(values).forEach(([name, value]) =>
    new Float64Array(instance.exports.memory.buffer, offsets[name], 1)[0] = value);
  instance.exports[{entrypoint}](...names.map(name => name === "count" ? 1 : offsets[name]));
  console.log(JSON.stringify(names.slice(-3).map(name =>
    new Float64Array(instance.exports.memory.buffer, offsets[name], 1)[0])));
}})();
"""
    completed = subprocess.run(
        [node, "-e", script, str(wasm_path)], check=True,
        capture_output=True, text=True,
    )
    assert json.loads(completed.stdout) == pytest.approx([10.0, 22.0, 31.0])
