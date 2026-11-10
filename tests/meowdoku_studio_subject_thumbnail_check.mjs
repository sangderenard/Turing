import assert from 'node:assert/strict';
import {studioSubjectThumbnailFrame} from '../examples/meowdoku/web/marble_studio_subject_thumbnail.js';
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],faces=[];
for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1])faces.push([[x,y,z],[x,y,z],[x,y,z]]);
const packet={scene:{subject:{faces,model:I}},render:{referenceCamera:{view:I,fovDegrees:50}}};
const project=(m,p)=>[0,1,2,3].map(i=>m[i]*p[0]+m[i+4]*p[1]+m[i+8]*p[2]+m[i+12]);
for(const [width,height]of [[128,128],[64,192],[192,64]]) {
 const frame=studioSubjectThumbnailFrame({packet,width,height});
 assert.equal(frame.width,width);assert.equal(frame.height,height);assert(frame.near>0&&frame.far>frame.near);
 for(const face of faces)for(const p of face) {
  const clip=project(frame.projection,project(frame.view,p));
  assert(Math.abs(clip[0]/clip[3])<1&&Math.abs(clip[1]/clip[3])<1&&Math.abs(clip[2]/clip[3])<1,'Every authored vertex fits');
 }
}
const model=[3,0,0,0,0,3,0,0,0,0,3,0,7,8,9,1],scaled=studioSubjectThumbnailFrame({packet,model,lightTransform:model});
assert.deepEqual(scaled.bounds.center,[7,8,9]);assert.equal(scaled.bounds.radius,Math.sqrt(27));
for(const options of [{width:0},{width:513},{height:1.5},{model:Array(16).fill(0)},
 {lightTransform:[-1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]}])assert.throws(()=>studioSubjectThumbnailFrame({packet,...options}));
console.log(JSON.stringify({pass:true,allBoundVerticesFit:true,portraitLandscapeSquare:true,invalidDimensionsAndRigRejected:true,browserRendering:false}));
