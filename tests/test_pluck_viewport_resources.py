"""Execute the actual renderer resource setup without a GPU or generated build."""
from pathlib import Path
import subprocess


def test_pluck_resource_contract(tmp_path):
    source = Path('src/compiler/abstract_ui_div_map.py').read_text(encoding='utf-8')
    body = source[source.index('function createScalarTexture('):source.index('function activateViewportShader(')]
    script = r'''
const assert=require('node:assert/strict');
const calls=[];let unit=0;
const gl=new Proxy({TEXTURE0:100,createTexture:()=>({}),activeTexture:u=>unit=u-100,
 texImage2D:(...args)=>calls.push({unit,data:Array.from(args[8]),width:args[3]}),
 getUniformLocation:(_,name)=>name,uniform1i:(name,value)=>calls.push({name,value}),
 uniform1f:(name,value)=>calls.push({name,value})},
 {get:(o,k)=>k in o?o[k]:(k.toUpperCase()===k?1:()=>{})});
const model={appearance:{colors:{'portal-in':'a','portal-out':'b','body-shell-glass':'c'}}};
const colorVector=x=>({a:[1,0,0],b:[0,1,0],c:[0,0,1]})[x];
const shaderViewer={shaderResources:new Map()};
''' + body + r'''
const names=['pbr','phong','enamel','texstack','thermal_k'];
const choice={identity:'pluck',resource_bindings:names.map((member_name,i)=>({member_name,sampler:member_name,recommended_texture_unit:16+i}))};
configurePluckPhongResources(gl,{},choice);
const uploads=calls.filter(x=>x.data);
assert.deepEqual(uploads.map(x=>x.unit),[5,6,7,8,9]);
assert.deepEqual(uploads.map(x=>x.width),[48,24,24,48,1]);
assert.deepEqual(uploads[4].data,[0]);
assert(calls.some(x=>x.name==='uThermalEmission'&&x.value===0));
assert(calls.some(x=>x.name==='uThermalGain'&&x.value===0));
assert.equal(shaderViewer.shaderResources.get('pluck').length,10);
const before=calls.length;
assert.throws(()=>configurePluckPhongResources(gl,{}, {resource_bindings:[{member_name:'new_unknown_field'}]}),/Unsupported Pluck resource new_unknown_field/);
assert.equal(calls.length,before);
for(const v of [undefined,[],[NaN],[Infinity]])assert.throws(()=>createScalarTexture(gl,0,v),/Invalid scalar texture data/);
console.log('PASS actual Pluck resource setup: thermal disabled, finite texture contracts, noncolliding units, unknown bindings rejected');
'''
    path=tmp_path/'check.cjs'
    path.write_text(script,encoding='utf-8')
    subprocess.run(['node',str(path)],check=True)
