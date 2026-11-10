import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {MarbleChaseCamera,chaseViewMatrix} from "../examples/meowdoku/web/marble_chase_camera.js";
import {cameraNudgeGravity} from "../examples/meowdoku/web/marble_sim_preview.js";

const close=(a,b,epsilon=1e-8)=>assert.ok(Math.abs(a-b)<epsilon,`${a} != ${b}`);
const vectorClose=(a,b,epsilon=1e-8)=>a.forEach((v,i)=>close(v,b[i],epsilon));
const norm=v=>Math.hypot(...v),sub=(a,b)=>a.map((v,i)=>v-b[i]);
const sample=(camera,t,p=[t,1,0],extras={})=>camera.update({position:p,radius:.1,sampleSeconds:t,nowSeconds:t,...extras});
const camera=new MarbleChaseCamera({yaw:0});
let pose=sample(camera,0,[0,1,0]);
close(norm(sub(pose.eye,pose.target)),1.2);
assert.ok(pose.eye[1]>pose.target[1]);assert.ok(pose.eye[2]>pose.target[2]);
// View basis is orthonormal with no roll. Real drive forward/right match it.
for(const yaw of [0,.5,Math.PI/2,Math.PI,-2.4]){
  const c=new MarbleChaseCamera({yaw}),p=sample(c,0,[2,1,-3]),m=chaseViewMatrix(p);
  const transform=v=>[0,1,2].map(row=>m[row]*v[0]+m[4+row]*v[1]+m[8+row]*v[2]+m[12+row]);
  vectorClose(transform(p.eye),[0,0,0],1e-6);
  vectorClose(transform(p.target),[0,0,-p.distance],1e-6);
  const forward=cameraNudgeGravity(0,1,yaw),right=cameraNudgeGravity(1,0,yaw);
  // Toy law x/y maps to world x/-z; camera's horizontal forward and right.
  const f=[forward[0],0,-forward[1]],r=[right[0],0,-right[1]];
  const eyeForward=sub(p.target,p.eye);eyeForward[1]=0;
  vectorClose(f.map(v=>v/.72),eyeForward.map(v=>v/norm(eyeForward)),1e-8);
  vectorClose(r.map(v=>v/.72),[m[0],m[4],m[8]],1e-6);
}
// Never auto-steer the control coordinate system while the user holds a key.
for(let i=1;i<=180;i++)pose=sample(camera,i/60,[i/60,1,0],{steering:true});
close(pose.yaw,0);
for(let i=181;i<=360;i++)pose=sample(camera,i/60,[i/60,1,0]);
assert.ok(pose.yaw>1.4&&pose.yaw<Math.PI/2);
assert.ok(norm(sub(camera.focus,[6,1,0]))<=.300001);
assert.ok(norm(camera.lead)<=.300001);
// Fixed heading, stopping, and repeated display frames cannot spin the view.
const fixedYaw=camera.yaw;
for(let i=361;i<=480;i++)pose=sample(camera,i/60,[6,1,-(i-360)/60],{autoHeading:false});
close(pose.yaw,fixedYaw);
for(let i=1;i<=120;i++)pose=sample(camera,8,[6,1,-2],{nowSeconds:8+i/60,autoHeading:false});
assert.ok(norm(camera.lead)<1e-4);
// Manual orbit has bounded elevation and suppresses auto-turn for two seconds.
camera.orbit(1,100,10);close(camera.pitch,1.25);
camera.orbit(0,-100,10);close(camera.pitch,.16);
const orbitYaw=camera.yaw;
for(let i=1;i<=100;i++)pose=sample(camera,10+i/60,[6,1,-2-i/60]);
close(pose.yaw,orbitYaw);
camera.drag(true,12);
for(let i=1;i<=180;i++)pose=sample(camera,12+i/60,[6,1,-4-i/60]);
close(pose.yaw,orbitYaw);
camera.drag(false,15);
camera.zoomBy(100);camera.zoomBy(100);close(camera.zoom,80);
for(let i=0;i<4;i++)camera.zoomBy(-100);close(camera.zoom,6);
camera.recenter(16);close(camera.pitch,.55);close(camera.zoom,12);close(camera.yaw,0);
// Replay seeks/epoch changes/teleports/tab interruptions are not velocity.
pose=sample(camera,1,[100,2,100],{epoch:2,nowSeconds:16});
vectorClose(camera.velocity,[0,0,0]);vectorClose(camera.focus,[100,2,100]);
pose=sample(camera,2,[101,2,100],{epoch:2,nowSeconds:17});
vectorClose(camera.velocity,[0,0,0]);vectorClose(camera.focus,[101,2,100]);
pose=sample(camera,1.9,[100,2,100],{epoch:2,nowSeconds:17.01});
vectorClose(camera.velocity,[0,0,0]);
// Body size alone sets chase distance. Authored position and input remain intact.
const pos=Object.freeze([0,0,0]);
const tiny=new MarbleChaseCamera(),large=new MarbleChaseCamera();
close(sample(tiny,0,pos,{radius:1e-4}).distance,.0012);
close(sample(large,0,pos,{radius:2}).distance,24);
// Equivalent sampled motion at display rates 30/60/120 Hz stays close.
const simulate=fps=>{const c=new MarbleChaseCamera({yaw:0});let p;
  for(let i=0;i<=3*fps;i++)p=sample(c,i/fps,[i/fps,1,0]);return p;};
for(const fps of [30,120]){const a=simulate(fps),b=simulate(60);
  assert.ok(norm(sub(a.eye,b.eye))<.03);assert.ok(Math.abs(a.yaw-b.yaw)<.025);}
for(const bad of [NaN,Infinity,0,-1])assert.throws(()=>sample(new MarbleChaseCamera(),0,[0,0,0],{radius:bad}));
assert.throws(()=>sample(new MarbleChaseCamera(),0,[NaN,0,0]));
assert.throws(()=>chaseViewMatrix({eye:[0,0,0],target:[0,0,0]}));
assert.throws(()=>chaseViewMatrix({eye:[0,1,0],target:[0,0,0]}));
// Source hook gate: camera follows the displayed native frame, not latest
// queued steps. Static/photo camera helper remains a distinct orbit pose.
const source=readFileSync(new URL("../examples/meowdoku/web/marble_studio.js",import.meta.url),"utf8");
assert.match(source,/sampleSeconds:sim.runtime.shown\*sim.runtime.snap\*sim.runtime.manifest.law.dt/);
assert.match(source,/steering:drive.active,autoHeading:\$\("preview-camera"\).value==="marble"/);
const photoCamera=source.slice(source.indexOf('function studioPhotoCamera('),source.indexOf('function tableFaces('));
assert.match(photoCamera,/linkedPhotoOrbit\(\{yaw,pitch,distance,photoTransform\}\)/);
assert.doesNotMatch(photoCamera,/chaseCamera|driveViewYaw/);
assert.doesNotMatch(source,/pitch\+\(follow\?Math.PI\/2/);
assert.match(source,/lostpointercapture/);
const html=readFileSync(new URL("../examples/meowdoku/web/marble_studio.html",import.meta.url),"utf8");
assert.match(html,/value="marble">Auto chase marble/);assert.match(html,/id="chase-recenter"/);
console.log("PASS stable marble chase: control axes, steering lock, follow/lead bounds, manual orbit/zoom, sample lifecycle, frame rates and source hooks");
