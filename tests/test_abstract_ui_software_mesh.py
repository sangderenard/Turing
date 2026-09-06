"""Focused contracts for the Python-authored Canvas perspective kernel."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from src.common.tensors.fused_ir import canonical_elementwise_op
from src.compiler.abstract_ui_software_mesh import compile_software_mesh_wasm


def test_frontend_operator_aliases_normalize_after_case_folding():
    assert canonical_elementwise_op("Div") == ("truediv", False)
    assert canonical_elementwise_op("RSub") == ("sub", True)


def test_python_projection_compiles_to_complete_browser_wasm():
    artifact = compile_software_mesh_wasm()
    assert artifact.binary[:4] == b"\x00asm"
    assert artifact.entrypoint == "project_mesh"
    assert artifact.operation_count == 28
    assert [parameter["role"] for parameter in artifact.parameters].count("output") == 3


def test_compiled_projection_executes_with_its_published_abi(tmp_path: Path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    artifact = compile_software_mesh_wasm()
    wasm_path = tmp_path / "mesh.wasm"
    wasm_path.write_bytes(artifact.binary)
    parameters = json.dumps([parameter["name"] for parameter in artifact.parameters])
    script = f"""
const {{readFileSync}} = require("fs");
(async () => {{
  const {{instance}} = await WebAssembly.instantiate(readFileSync(process.argv[1]), {{}});
  const names = {parameters};
  const values = {{vertex_x:1, vertex_y:2, vertex_z:5, camera_x:0, camera_y:1,
    camera_z:0, forward_x:0, forward_y:0, forward_z:1, right_x:1, right_y:0,
    right_z:0, up_x:0, up_y:1, up_z:0, width:800, height:400}};
  const offsets = {{}}; let cursor = 0;
  names.slice(1).forEach(name => {{ offsets[name] = cursor; cursor += 8; }});
  Object.entries(values).forEach(([name, value]) =>
    new Float64Array(instance.exports.memory.buffer, offsets[name], 1)[0] = value);
  instance.exports.project_mesh(...names.map(name => name === "count" ? 1 : offsets[name]));
  console.log(JSON.stringify(names.slice(-3).map(name =>
    new Float64Array(instance.exports.memory.buffer, offsets[name], 1)[0])));
}})();
"""
    completed = subprocess.run(
        [node, "-e", script, str(wasm_path)], check=True, capture_output=True, text=True,
    )
    screen_x, screen_y, depth = json.loads(completed.stdout)
    assert screen_x == pytest.approx(457.14285714285717)
    assert screen_y == pytest.approx(142.85714285714283)
    assert depth == pytest.approx(5.0)
