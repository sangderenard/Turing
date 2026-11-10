"""Layer-aware existing spawn support, without changing camera clearance defaults."""
from pathlib import Path
import json
import shutil
from src.compiler.closed_solid_geometry import emit_closed_solid_javascript
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]

def test_existing_ground_sampler_selects_below_reference_layer():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js required')
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('src/compiler/abstract_ui_div_map.py','utf8');
const code=source.match(/^function cameraGroundHeight\b[^]*?^}/m)[0];
const colliders=[{identity:'ground',surface:{domain:{minimum_x:-3,maximum_x:3,minimum_z:-3,maximum_z:3},height:.2}},
 {identity:'bridge',minimum:[-2,3.42,-2],maximum:[2,4,2]}];
const ctx={shaderViewer:{colliders},model:{},sampleDeclaredSurface:s=>({height:s.height})};vm.createContext(ctx);vm.runInContext(__HELPERS__,ctx);vm.runInContext(code,ctx);
assert.equal(ctx.cameraGroundHeight(0,0),4); // unchanged chase-camera ceiling clearance
assert.equal(ctx.cameraGroundHeight(0,0,1),.2); // player below bridge
assert.equal(ctx.cameraGroundHeight(0,0,4.2),4); // player on bridge
assert.equal(ctx.cameraGroundHeight(2.5,0,1),.2);
assert.equal(ctx.cameraGroundHeight(8,8,1),0);
assert.equal(ctx.cameraGroundHeight(0,0,-1),.2); // sampled terrain keeps one-sided recovery
assert(source.includes('position[2]+forward*sy+right*cy,position[1])'));
assert(source.includes('ground=cameraGroundHeight(position[0]+offset[0],position[2]+offset[2],position[1])'));
console.log('PASS layer-aware existing sampler');
'''
    script = script.replace('__HELPERS__', json.dumps(emit_closed_solid_javascript()))
    result = subprocess.run([node, '-e', script], cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
