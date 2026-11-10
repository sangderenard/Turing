import assert from 'node:assert/strict';
import fs from 'node:fs';
import {cameraMountGeometry} from '../examples/meowdoku/web/marble_camera_mount.js';
const camera={eye_m:[.3,.4,.1],target_m:[0,0,.18],aperture_radius_m:.005,aperture_blades:8,aperture_rotation_radians:.2};
const input={camera,unitM:.3048,originM:[0,0,.18],floorY:-5,
  bounds:{leftX:-24,rightX:24,floorY:-5,ceilingY:23.1,backZ:-18.6,frontZ:24},material:38};
assert.equal(cameraMountGeometry(input).attached,false);
assert.match(cameraMountGeometry(input).reason,/unavailable/);
assert.throws(()=>cameraMountGeometry({...input,material:-1}),/shared material/);
const path=process.argv[process.argv.indexOf('--native-source')+1];
if(process.argv.includes('--native-source')){
  const housingSource=JSON.parse(fs.readFileSync(path,'utf8')),before=JSON.stringify(housingSource);
  const result=cameraMountGeometry({...input,camera:housingSource.camera,housingSource});
  assert.equal(result.attached,true,result.reason);assert.equal(result.faces.length,240);assert.equal(result.components.length,5);
  assert(result.faces.every(f=>f[3]===38));assert.equal(JSON.stringify(housingSource),before);
  const minimum=housingSource.housingVerticesM.reduce((n,p)=>Math.min(n,p[1]),Infinity);
  assert.equal(result.attachmentM[1],minimum);
  for(const face of result.faces)for(const p of face.slice(0,3)){
    assert(p[1]>=input.floorY);assert(p[1]*input.unitM+input.originM[1]<=minimum+1e-12);
  }
  const changed=structuredClone(housingSource.camera);changed.eye_m[0]+=.1;
  assert.equal(cameraMountGeometry({...input,camera:changed,housingSource}).attached,false,'Stale native pose cannot silently mount the current camera');
  const floor=result.attachment[1];
  assert.equal(cameraMountGeometry({...input,camera:housingSource.camera,housingSource,floorY:floor,bounds:{...input.bounds,floorY:floor}}).attached,false);
  assert.equal(cameraMountGeometry({...input,camera:housingSource.camera,housingSource,bounds:{...input.bounds,rightX:-20}}).attached,false);
  for(const component of result.components){const edges=new Map();
    for(const [a,b,c]of result.faces.slice(component.firstTriangle,component.firstTriangle+component.triangleCount))
      for(const [p,q]of [[a,b],[b,c],[c,a]]){const x=p.map(Math.fround).join(','),y=q.map(Math.fround).join(','),key=[x,y].sort().join('|');const uses=edges.get(key)??[];uses.push(x<y?1:-1);edges.set(key,uses);}
    assert([...edges.values()].every(v=>v.length===2&&v[0]===-v[1]));
  }
  console.log('PASS physical camera mount: exact native housing/pose source, underside contact, floor/room bounds, shared material, closed240tri geometry, immutable pose, stale/missing source and impossible floor remain unattached.');
}else console.log('PASS camera mount absence/validation contract; native housing proof runs through the Python wrapper.');
