import ast
import json
from pathlib import Path
import shutil
import subprocess

from src.compiler.javascript_runtime_utilities import WORLD_REGISTRY_SOURCE


def test_game_upload_preserves_world_vertices_and_replaces_empty_frame(tmp_path):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "src/compiler/abstract_ui_div_map.py").read_text(encoding="utf-8"))
    source = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                  and target.id == "DIV_MAP_JAVASCRIPT" for target in node.targets))
    start = source.index("function installAuthoredWorldSurfacePacket(")
    end = source.index("function rebuildPortableSceneMesh(", start)
    packet = dict(schema="abstract-ui-world-mesh-packet-v0", topology="triangle-list",
                  vertex_layout=["position.xyz", "normal.xyz", "color.rgb"],
                  vertices=[[0, 0, 0, 0, 0, 1, 1, 0, 0],
                            [1, 0, 0, 0, 0, 1, 1, 0, 0],
                            [0, 1, 0, 0, 0, 1, 1, 0, 0]],
                  object_spans=[dict(identity="roller", first_vertex=0, vertex_count=3)],
                  semantic_part_spans=[dict(identity="roller/surface", object_identity="roller",
                                            first_vertex=0, vertex_count=3)])
    script = WORLD_REGISTRY_SOURCE + "\n" + source[start:end] + "\n" + r'''
const assert = require('node:assert/strict');
const uploads = [], draws = [], formats = [];
const gl = {ARRAY_BUFFER:1, FLOAT:2, DYNAMIC_DRAW:3, TRIANGLES:4,
  createVertexArray:()=>({}), createBuffer:()=>({}), bindVertexArray:()=>{}, bindBuffer:()=>{},
  enableVertexAttribArray:()=>{}, vertexAttribPointer:(...args)=>formats.push(args),
  bufferData:(kind,mesh)=>uploads.push(Array.from(mesh)), drawArrays:(...args)=>draws.push(args)};
const shaderViewer = {gl,program:{},vao:{},vertexCount:0};
''' + "const packet = " + json.dumps(packet) + ";\n" + r'''
installAuthoredWorldSurfacePacket(packet);
assert.deepEqual(uploads[0], packet.vertices.flat());
assert.deepEqual(formats.map(row=>row.slice(4)), [[36,0],[36,12],[36,24]]);
assert.equal(shaderViewer.authoredWorldSurface.partSpans[0].objectIdentity,'roller');
drawSceneMeshes(gl); assert.deepEqual(draws.at(-1),[4,0,3]);
const saved = shaderViewer.authoredWorldSurface;
assert.throws(()=>installAuthoredWorldSurfacePacket({...packet,object_spans:[]}),/unowned/);
assert.equal(shaderViewer.authoredWorldSurface,saved); assert.equal(uploads.length,1);
installAuthoredWorldSurfacePacket({...packet,vertices:[],object_spans:[],semantic_part_spans:[]});
drawSceneMeshes(gl); assert.deepEqual(uploads.at(-1),[]); assert.deepEqual(draws.at(-1),[4,0,0]);
'''
    path = tmp_path / "game-world-upload.js"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run([shutil.which("node") or "node", str(path)], capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
