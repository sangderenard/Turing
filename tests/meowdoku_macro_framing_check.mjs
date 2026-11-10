import assert from 'node:assert/strict';
import {studioScaleTransform,studioFrameDistance} from '../examples/meowdoku/web/marble_studio_scale.js';
import {cameraWithLens,fovFromFocal} from '../examples/meowdoku/web/marble_lens_controls.js';
import {photoCameraFromStudio} from '../examples/meowdoku/web/marble_photo.js';
for(const unitMm of [.1,1,10,200])for(const radiusMm of [.1,1,2,10,200])for(const aspect of [.5,1,2]){
 const subject=studioScaleTransform({version:1,mode:'fixed',unitMm},radiusMm,false);
 const framed={...subject,fovDegrees:fovFromFocal(.05)};
 const d=studioFrameDistance(framed,aspect),m=d*subject.unitM;
 assert(m>=.001&&m<=10);
 const half=Math.atan(Math.tan(framed.fovDegrees*Math.PI/360)*Math.min(1,aspect));
 assert(subject.radiusM*1.2<=m*Math.sin(half)+1e-12,'Whole sphere and margin fit selected FOV');
 const scene=photoCameraFromStudio(-.3,.2,d,subject.unitM,framed.fovDegrees,subject.subjectCenter);
 const camera=cameraWithLens(scene.camera,{key:'custom',focusMm:m*1000,focalMm:50,fNumber:16});
 assert(Math.abs(Math.hypot(...camera.eye_m.map((v,i)=>v-camera.target_m[i]))-camera.focus_distance_m)<1e-12);
 if(radiusMm===2&&aspect>=1)assert(m<.02,'2mm radius macro fits below20mm without scale-linked floor');
 const conservative=studioFrameDistance(framed,aspect,.23)*subject.unitM;
 assert(conservative>=.23,'Source minimum remains enforced');
}
const tiny={...studioScaleTransform({version:1,mode:'fixed',unitMm:10},.1,false),fovDegrees:90};
assert.equal(studioFrameDistance(tiny)*tiny.unitM,.001);
assert.throws(()=>studioFrameDistance(tiny,1,.0009));
assert.throws(()=>studioFrameDistance({...tiny,subjectRadius:100000}));
console.log('PASS macro framing: selected FOV, physical unit invariance, target focus,1mm numerical domain and source minima');
