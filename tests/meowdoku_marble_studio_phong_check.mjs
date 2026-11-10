import assert from 'node:assert/strict';
import {studioPhongFaces} from '../examples/meowdoku/web/marble_studio_phong.js';
const normal=[0,0,1],face=id=>[[0,0,0],[1,0,0],[0,1,0],id,[normal,normal,normal]];
const faces=[face(1),face(0),face(0)],pbr=new Array(32).fill(0);pbr[5]=1;pbr[7]=1;pbr[23]=1;
const capture={revision:3,state:{channel:'studio',subject:{id:'subject',faces,
  phong:{interiorCount:2,glassCount:1,bubbles:[{start:1,count:1}]}},
  ray:{pbr},camera:{view:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]},
  parts:[{id:'table/slab',faces:[face(0),face(1)]},{id:'gobo/7',faces:[face(0)]},{id:'room/floor',faces:[face(0)]}]}};
const dynamic={schema:'turing.studio-dynamic-geometry.v1',revision:3,step:6,
  subject:{id:'subject',faces:structuredClone(faces)},staticRay:structuredClone(capture.state.ray),camera:structuredClone(capture.state.camera)};
const before=structuredClone({capture,dynamic}),plan=studioPhongFaces(capture,dynamic);
assert.equal(plan.insideOpaque.length,1);assert.equal(plan.outsideOpaque.length,0);
assert.equal(plan.transparent.length,3,'Outer shell, table and gobo share exact world-space glass sort; analytic bubble stays omitted only for Phong');
assert.equal(dynamic.subject.faces.length,3,'Full ray geometry was not filtered');
assert.deepEqual({capture,dynamic},before);
for(const mutate of [v=>v.revision++,v=>v.step=0,v=>v.subject.id='other',v=>v.subject.faces[0][3]=0,
  v=>v.staticRay.pbr[0]=1,v=>v.camera.view[12]=1]) {
  const changed=structuredClone(dynamic);mutate(changed);assert.throws(()=>studioPhongFaces(capture,changed));
}
console.log('PASS hidden Phong partition: exact identity, bubble filtering, combined furniture glass, no source mutation');
