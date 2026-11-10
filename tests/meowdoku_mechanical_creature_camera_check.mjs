import assert from 'node:assert/strict';
import fs from 'node:fs';
import {mechanicalCreatureStudioCamera} from '../examples/meowdoku/web/mechanical_creature_studio_camera.js';
const source=fs.readFileSync('src/compiler/abstract_ui_div_map.py','utf8');
for(const literal of ['float tangentHalfFov=0.70;','float nearPlane=0.04,farPlane=128.0;','cross(forward,vec3(0,1,0))'])assert.ok(source.includes(literal),'Original camera projection changed: '+literal);
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),apply=(m,v)=>[0,1,2,3].map(r=>v.reduce((s,x,c)=>s+m[c*4+r]*x,0));
for(const cameraFacing of [[0,0,-1],[1,.4,2],[-.2,.99,.1]])for(const point of [[0,0,0],[2,-.1,7],[-3,1.2,-4]]){
  const eye=[1,.2875,-2],camera=mechanicalCreatureStudioCamera({cameraPosition:eye,cameraFacing,width:1234,height:731}),
    relative=point.map((v,i)=>v-eye[i]),positiveView=[dot(relative,camera.right),dot(relative,camera.up),dot(relative,camera.forward)],
    expected=[positiveView[0]/(.70*(1234/731)),positiveView[1]/.70,
      ((128+.04)/(128-.04))*positiveView[2]-(2*128*.04)/(128-.04),positiveView[2]],
    actual=apply(camera.projection,apply(camera.view,[...point,1]));
  actual.forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<2e-6,`Existing game clip mismatch ${i}`));
}
assert.throws(()=>mechanicalCreatureStudioCamera({cameraPosition:[0,0,0],cameraFacing:[0,1,0],width:3,height:3}),/degenerate/);
console.log('Original game camera clip/depth parity passed; no control or pose mutation.');
