import assert from 'node:assert/strict';
import {cameraMovements,shiftedProjection,studioCameraPose} from '../examples/meowdoku/web/marble_camera_movements.js';
import {dofFocusAt} from '../examples/meowdoku/web/marble_studio_dof.js';
import {chaseViewMatrix} from '../examples/meowdoku/web/marble_chase_camera.js';
const near=(a,b)=>assert(Math.abs(a-b)<1e-6,`${a} != ${b}`);
const c=cameraMovements({fov_degrees:40},{shift:[.25,-.1],tiltDegrees:[12,-8]});
near(c.lens_tilt[0],12*Math.PI/180);
const projection=new Float32Array(16);projection[0]=2;projection[5]=3;projection[11]=-1;
const shifted=shiftedProjection(projection,c);near(shifted[8],.25);near(shifted[9],-.1);assert.equal(projection[8],0);
// Native primary ray at a given NDC projects back to the same pixel.
for(const [x,y]of [[0,0],[-.7,.8],[.5,-.5]]){const z=-4,px=(x+.25)*4/2,py=(y-.1)*4/3;
 near((2*px+shifted[8]*z)/-z,x);near((3*py+shifted[9]*z)/-z,y);}
const p={focus:2},uv=[.7,.2],t=Math.tan(40*Math.PI/360),aspect=1.5;
near(dofFocusAt(uv,p,c,aspect),2/(1-(.4+.25)*t*aspect*Math.tan(c.lens_tilt[1])-(-.6-.1)*t*Math.tan(c.lens_tilt[0])));
near(dofFocusAt([.5,.5],p,{fov_degrees:40}),2);
const base={yaw:.2,pitch:.3,distance:5,target:[1,2,3]},orbit=studioCameraPose(base);
const free=studioCameraPose({...base,yaw:1,freeEye:orbit.eye});assert.deepEqual(free.eye,orbit.eye);assert.notDeepEqual(free.target,orbit.target);
const raised=studioCameraPose({...base,height:6,floor:-5});assert.equal(raised.eye[1],1);assert.deepEqual(raised.target,base.target);
const v=chaseViewMatrix(free);for(let i=0;i<3;i++)near(v[i]*free.eye[0]+v[4+i]*free.eye[1]+v[8+i]*free.eye[2]+v[12+i],0);
assert.throws(()=>cameraMovements({}, {tiltDegrees:[31,0]}));
console.log('PASS shared camera movements: native ray/projection parity, tilted focal plane, fixed-eye pointing, floor-relative height, unchanged orbit, bounds. CPU only.');

const {rangefinderRay}=await import('../examples/meowdoku/web/marble_rangefinder.js');
const view=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const found=rangefinderRay({state:{schema:'turing.studio-visible-scene.v1',channel:'studio',camera:{view,fovDegrees:40,aspect:1.5,tilt_shift:c.tilt_shift},frame:{unitM:1,rayOriginM:[0,0,0]}}},.5,.5);
near(found.ray[4]/-found.ray[6],.25*Math.tan(40*Math.PI/360)*1.5);
near(found.ray[5]/-found.ray[6],-.1*Math.tan(40*Math.PI/360));
