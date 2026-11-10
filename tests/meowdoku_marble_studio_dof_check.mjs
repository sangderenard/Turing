import assert from 'node:assert/strict';
import {dofParameters,dofAxialDepth,dofCircleRadius,dofCameraFromView} from '../examples/meowdoku/web/marble_studio_dof.js';
const camera={focus_distance_m:.1,aperture_radius_m:.002,fov_degrees:50};
const p=dofParameters(camera,{unitM:.01,near:.01,far:30,height:1080});
assert.equal(dofAxialDepth(0,.01,30),.01);
assert.ok(Math.abs(dofAxialDepth(1,.01,30)-30)<1e-10);
for(const z of [.01,.1,1,15,30]){
  const depth=(30-30*.01/z)/(30-.01);
  assert.ok(Math.abs(dofAxialDepth(depth,.01,30)-z)<1e-9);
}
assert.equal(dofCircleRadius(.1,p),0);
assert.ok(dofCircleRadius(.05,p)<0);assert.ok(dofCircleRadius(.2,p)>0);
assert.equal(dofCircleRadius(.2,{...p,aperture:.001}),dofCircleRadius(.2,p)/2);
assert.equal(dofCircleRadius(.2,{...p,aperture:0}),0);
assert.ok(Math.abs(dofCircleRadius(.2,p)-.002*(1/.1-1/.2)*1080/(2*Math.tan(25*Math.PI/180)))<1e-10);
const view=[1,0,0,0,0,1,0,0,0,0,1,0,-2,-3,-4,1];
const pose=dofCameraFromView(view,{unitM:.01,targetDistance:5,fovDegrees:50});
assert.deepEqual(pose.eye_m,[.02,.03,.04]);assert.deepEqual(pose.target_m,[.02,.03,.04-.05]);
const turned=[0,0,1,0,0,1,0,0,-1,0,0,0,4,-3,-2,1];
const rotated=dofCameraFromView(turned,{unitM:.01,targetDistance:5,fovDegrees:50});
assert.deepEqual(rotated.eye_m,[.02,.03,.04]);assert.deepEqual(rotated.target_m,[.02-.05,.03,.04]);
for(const bad of [{focus_distance_m:0},{aperture_radius_m:-1},{fov_degrees:180}])assert.throws(()=>dofParameters({...camera,...bad},{unitM:.01,near:.01,far:30,height:1080}));
assert.throws(()=>dofParameters(camera,{unitM:0,near:.01,far:30,height:1080}));
assert.throws(()=>dofAxialDepth(1.1,.01,30));
console.log('Studio DOF geometry: depth inversion, axial focus, aperture response, current view pose, invalid inputs passed.');
