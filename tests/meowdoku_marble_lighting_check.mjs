import assert from 'node:assert/strict';
import {createLightingPoseState, validateLightingPoseState, resolveLightingPoses, createLightingPoseControls} from '../examples/meowdoku/web/marble_lighting.js';
const ids = ['key-left','key-right','background','kicker-left','kicker-right','fill-lower','fill-upper'];
const initial = ids.map((id,i) => ({id,label:id,pos:[i-3, i+1, -2],dir:[1, i-3, -2],aimAtSubject:false}));
const copy = value => JSON.parse(JSON.stringify(value));
const near = (a,b) => assert.ok(Math.abs(a-b) < 1e-12, `${a} != ${b}`);
const nearVector = (a,b) => a.forEach((v,i) => near(v,b[i]));
const state = createLightingPoseState(initial);
assert.deepEqual(validateLightingPoseState(copy(state),ids),state);
const manual = resolveLightingPoses(state,ids);
for (let i=0;i<ids.length;i++) {
  assert.equal(manual[i].id, ids[i]);
  assert.deepEqual(manual[i].pos,initial[i].pos);
  nearVector(manual[i].dir,initial[i].dir.map(v => v/Math.hypot(...initial[i].dir)));
  const changed = copy(state); changed.poses[i].pos=[-8,5,7]; changed.poses[i].yaw=45; changed.poses[i].pitch=-20;
  const resolved = resolveLightingPoses(changed,ids);
  for (let j=0;j<ids.length;j++) if (j!==i) assert.deepEqual(resolved[j],manual[j]);
  near(Math.hypot(...resolved[i].dir),1);
  changed.poses[i].aimAtSubject=true;
  const locked=resolveLightingPoses(changed,ids)[i];
  nearVector(locked.dir,[8,-5,-7].map(v=>v/Math.hypot(8,5,7)));
  changed.poses[i].pos=[0,0,0];
  nearVector(resolveLightingPoses(changed,ids)[i].dir,resolved[i].dir);
  changed.poses[i].pos=[Number.MIN_VALUE,0,0];
  nearVector(resolveLightingPoses(changed,ids)[i].dir,[-1,0,0]);
  changed.poses[i].aimAtSubject=false;
  nearVector(resolveLightingPoses(changed,ids)[i].dir,resolved[i].dir);
}
for (const dir of [[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[Number.MAX_VALUE,Number.MAX_VALUE,0],[Number.MIN_VALUE,0,0]]) {
  const s = createLightingPoseState([{id:'one',pos:[0,0,0],dir,aimAtSubject:false}]);
  near(Math.hypot(...resolveLightingPoses(s,['one'])[0].dir),1);
}
for (const mutate of [s=>s.version=2,s=>s.poses.pop(),s=>s.poses.push(s.poses[0]),s=>s.poses.reverse(),s=>s.poses[0].id='unexpected',s=>s.poses[0].id=s.poses[1].id,s=>s.poses[0].pos=[1,2],s=>s.poses[0].pos[0]=51,s=>s.poses[0].pos[1]=NaN,s=>s.poses[0].pos[2]=Infinity,s=>s.poses[0].yaw=181,s=>s.poses[0].pitch=-91,s=>s.poses[0].yaw='0',s=>s.poses[0].aimAtSubject=1]) {
  const bad=copy(state); mutate(bad); assert.throws(()=>validateLightingPoseState(bad,ids));
}
assert.throws(()=>createLightingPoseState([{...initial[0],dir:[0,0,0]}]));
assert.throws(()=>createLightingPoseState([{...initial[0],dir:[NaN,0,1]}]));
assert.throws(()=>validateLightingPoseState(state,['a','a']));
assert.throws(()=>validateLightingPoseState(null,ids));
const clone=validateLightingPoseState(state,ids); clone.poses[0].pos[0]=40;
assert.equal(state.poses[0].pos[0],-3);
const resolvedClone=resolveLightingPoses(state,ids); resolvedClone[0].pos[0]=40;
assert.equal(state.poses[0].pos[0],-3);

// Minimal DOM event harness exercises the production editor without WebGL.
class Element {
  constructor(tag,doc) { this.tagName=tag;this.ownerDocument=doc;this.children=[];this.dataset={};this.events={};this.value='';this.disabled=false; }
  appendChild(child) { this.children.push(child); return child; }
  setAttribute(name,value) { this[name]=value; }
  addEventListener(name,fn) { this.events[name]=fn; }
  fire(name) { this.events[name]?.(); }
}
const doc={createElement:tag=>new Element(tag,doc)};
const container=new Element('div',doc);
let changes=0;
const editor=createLightingPoseControls(container,initial,()=>changes++);
assert.equal(changes,0);
const all=element=>[element,...element.children.flatMap(all)];
const nodes=all(container);
const field=(name,type)=>nodes.find(n=>n.dataset.lightingPose===name && (!type||n.type===type));
const select=field('fixture'); assert.equal(select.children.length,7);
for (let i=0;i<ids.length;i++) {
  select.value=ids[i]; select.fire('change');
  const x=field('X','number');x.value=String(i+10);x.fire('input');
  assert.equal(editor.getPoses()[i].pos[0],i+10);
}
const yaw=field('yaw','number');yaw.value='35';yaw.fire('input');
const beforeLock=editor.getPoses()[6].dir;
const lock=field('lock');lock.checked=true;lock.fire('change');
assert.equal(yaw.disabled,true);assert.equal(field('yaw','range').disabled,true);
assert.equal(editor.getState().poses[6].yaw,35);
lock.checked=false;lock.fire('change');assert.equal(yaw.disabled,false);
nearVector(editor.getPoses()[6].dir,beforeLock);
const before=editor.getState(), count=changes;
const x=field('X','number');x.value='';x.fire('input');x.value='999';x.fire('input');
assert.deepEqual(editor.getState(),before);assert.equal(changes,count);
assert.ok(nodes.find(n=>n.role==='status').textContent.includes('between'));
x.fire('change');assert.equal(x.value,String(before.poses[6].pos[0]));
const bad=copy(before);bad.poses[5].pitch=Infinity;
assert.throws(()=>editor.setState(bad));assert.deepEqual(editor.getState(),before);assert.equal(changes,count);
editor.setState(state);assert.deepEqual(editor.getState(),state);assert.equal(changes,count+1);
const external=editor.getState();external.poses[0].pos[0]=42;
assert.deepEqual(editor.getState(),state);
const supplied=copy(state);editor.setState(supplied);supplied.poses[1].pos[0]=42;
assert.deepEqual(editor.getState(),state);
console.log('marble lighting: seven independent poses, directions, lock singularities, validation and editor events passed');
// Moving subject is supplied by the preview, not persisted in a profile.
const moving=copy(state);moving.poses.forEach(p=>p.aimAtSubject=true);
const target=[12,-4,7];
const aimed=resolveLightingPoses(moving,ids,target);
for (let i=0;i<ids.length;i++) {
  const delta=target.map((v,j)=>v-moving.poses[i].pos[j]);
  nearVector(aimed[i].dir,delta.map(v=>v/Math.hypot(...delta)));
  assert.deepEqual(aimed[i].pos,moving.poses[i].pos);
}
nearVector(resolveLightingPoses(moving,ids,moving.poses[0].pos)[0].dir,manual[0].dir);
assert.deepEqual(resolveLightingPoses(state,ids,target),manual);
for (const invalid of [[0,0],[0,NaN,0],[Infinity,0,0],['0',0,0],null]) {
  assert.throws(()=>resolveLightingPoses(state,ids,invalid));
}
for (const extreme of [[Number.MAX_VALUE,Number.MAX_VALUE,0],[0,-Number.MAX_VALUE,Number.MAX_VALUE]]) {
  for (const pose of resolveLightingPoses(moving,ids,extreme)) near(Math.hypot(...pose.dir),1);
}
editor.setState(moving);
assert.deepEqual(editor.getPoses(target),aimed);
assert.deepEqual(editor.getState(),moving);
console.log('marble lighting: moving target, coincident fallback, finite target validation and target-free state passed');
