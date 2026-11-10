import assert from 'node:assert/strict';
import {studioDisplayedSubjectCenter,linkedPhotoOrbit,PhotographicCameraBinding,mountPhotographicCameraBinding} from '../examples/meowdoku/web/marble_camera_binding.js';
import {studioScaleTransform} from '../examples/meowdoku/web/marble_studio_scale.js';
import {OBJECT_CENTER,photoCameraFromStudio} from '../examples/meowdoku/web/marble_photo.js';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-12,`${a} != ${b}`);
for(const unitMm of [.1,10,304.8])for(const radiusMm of [.1,1,25,200])for(const table of [false,true]){
 const transform=studioScaleTransform({version:1,mode:'fixed',unitMm},radiusMm,table);
 const actual=studioDisplayedSubjectCenter(transform,[.2,9,-.3]);
 near(actual[0],transform.subjectCenter[0]+.2/transform.unitM);
 near(actual[1],transform.subjectCenter[1]);
 near(actual[2],transform.subjectCenter[2]-.3/transform.unitM);
 const pose=linkedPhotoOrbit({yaw:-.5,pitch:.2,distance:3.45,photoTransform:transform});
 assert.deepEqual(pose,photoCameraFromStudio(-.5,.2,3.45,transform.unitM,transform.fovDegrees,transform.subjectCenter));
 assert.deepEqual(pose.camera.target_m,OBJECT_CENTER.map((v,i)=>v+transform.subjectCenter[i]*transform.unitM));
 near(Math.hypot(...pose.camera.eye_m.map((v,i)=>v-pose.camera.target_m[i])),3.45*transform.unitM);
}
assert.throws(()=>studioDisplayedSubjectCenter({subjectCenter:[0,0,0],unitM:0}));
assert.throws(()=>studioDisplayedSubjectCenter({subjectCenter:[0,0,0],unitM:1},[NaN,0,0]));
const first=photoCameraFromStudio(-.5,.2,3.45),second=photoCameraFromStudio(1,.4,5);
const binding=new PhotographicCameraBinding();assert.deepEqual(binding.resolve(first),first);
binding.hold(first);first.camera.eye_m[0]=90;
const held=binding.resolve(second);assert.notEqual(held.camera.eye_m[0],90);
held.camera.eye_m[0]=100;assert.notEqual(binding.resolve(second).camera.eye_m[0],100);
assert.throws(()=>binding.hold({camera:{eye_m:[0,0,0],target_m:[0,0,0],fov_degrees:20}}));
assert.equal(binding.followUser,false);binding.follow();assert.deepEqual(binding.resolve(second),second);

// Real control handlers against a small DOM adapter, not browser QA.
class Element {
 constructor(){this.children=[];this.style={};this.handlers=new Map();}
 append(...items){this.children.push(...items);items.forEach(x=>x.parent=this);}
 remove(){this.parent.children=this.parent.children.filter(x=>x!==this);}
 setAttribute(k,v){this[k]=v;}
 addEventListener(k,fn){const a=this.handlers.get(k)||[];a.push(fn);this.handlers.set(k,a);}
 removeEventListener(k,fn){this.handlers.set(k,(this.handlers.get(k)||[]).filter(x=>x!==fn));}
 dispatch(type){for(const fn of this.handlers.get(type)||[])fn({target:this});}
}
const host=new Element(),find=(el,id)=>el.id===id?el:el.children.map(e=>find(e,id)).find(Boolean);
host.id='photo-controls';const doc={createElement:()=>new Element(),getElementById:id=>find(host,id)};
let changes=0,bad=false,captures=0;
const ui=mountPhotographicCameraBinding({doc,binding,captureLinked:()=>{captures++;if(bad)throw Error('incomplete input');return second;},framePhoto:()=>first,onChange:()=>changes++});
const follow=doc.getElementById('photo-camera-follow-user'),copy=doc.getElementById('photo-camera-copy-user'),frame=doc.getElementById('photo-camera-frame-subject');
assert.equal(captures,0,'Mounting controls cannot capture an uninitialized scene');
follow.checked=false;follow.dispatch('change');assert.equal(binding.followUser,false);assert.equal(changes,1);
follow.checked=true;follow.dispatch('change');assert.equal(binding.followUser,true);
bad=true;follow.checked=false;follow.dispatch('change');assert.equal(binding.followUser,true);assert.equal(follow.checked,true);assert.match(doc.getElementById('photo-camera-binding-status').textContent,/incomplete input/);
bad=false;copy.dispatch('click');assert.equal(binding.followUser,false);assert.deepEqual(binding.resolve(first),second);
frame.dispatch('click');assert.deepEqual(binding.resolve(second),first);
const before=changes;ui.dispose();copy.dispatch('click');follow.dispatch('change');assert.equal(changes,before);assert.equal(host.children.length,0);
console.log('PASS camera binding: actual centre across sizes/units, linked Exposure isolation, held pose immutability, real UI handlers, validation and teardown');
