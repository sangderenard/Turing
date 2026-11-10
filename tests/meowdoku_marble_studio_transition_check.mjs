// Coordinator contract tests, not physics or renderer substitutes. Native GPU
// preparation and production renderer admission require their own real tests.
import assert from 'node:assert/strict';
import {createStudioPhysicsTransition, sameStudioTransitionState, StudioSceneChangedError}
  from '../examples/meowdoku/web/marble_studio_transition.js';
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const candidate=()=>({runtime:{ready:true,running:false,handoffPaused:true,shown:1,steps:6},
  checkpoint:{frame:1,step:6,seconds:.016,start:{state_f32:[1,2,3]}}});
const receipts=({capture,candidate})=>Object.fromEntries(['phong','ray'].map(kind=>
  [kind,{ready:true,revision:capture.revision,step:candidate.checkpoint.step}]));
function fixture(overrides={}) {
  let scene={revision:0,state:{mesh:new Float64Array([0,1,2]),camera:[0,2,3],material:{strength:1}}};
  const released=[],visible={scene:'unchanged',camera:'unchanged'},events=[];
  const controller=createStudioPhysicsTransition({captureScene:()=>scene,prepareCandidate:async()=>candidate(),
    stagePresentation:async args=>receipts(args),releaseCandidate:async args=>{released.push(args);},
    onStatus:s=>events.push(s.phase),...overrides});
  return {controller,released,visible,events,get scene(){return scene;},set scene(v){scene=v;}};
}

assert(sameStudioTransitionState({b:new Float32Array([1,-0]),a:[2]}, {a:[2],b:new Float32Array([1,-0])}));
assert(!sameStudioTransitionState([0],[-0]));
assert(!sameStudioTransitionState([],new Array(2)));
assert(!sameStudioTransitionState(new Uint8Array([1]),new Int8Array([1])));
assert.throws(()=>sameStudioTransitionState(new Map(),new Map()),/plain data/);
{
  const f=fixture(),p=f.controller.prepare();assert.equal(f.controller.prepare(),p);
  await p;assert.equal(f.controller.state().phase,'ready-paused');
  assert.deepEqual(f.visible,{scene:'unchanged',camera:'unchanged'});
  const ready=f.controller.takeReady();assert.equal(ready.candidate.runtime.running,false);
  assert.equal(ready.revision,0);assert.equal(f.controller.takeReady(),null);
  await f.controller.cancel();await f.controller.settled();assert.equal(f.released.length,0,'Transferred runtime is no longer owned');
}
for(const phase of ['prepare','stage']) {
  const barrier=deferred();let entered=false;
  const f=fixture(phase==='prepare'?{prepareCandidate:async()=>{entered=true;await barrier.promise;return candidate();}}:
    {stagePresentation:async args=>{entered=true;await barrier.promise;return receipts(args);}});
  const p=f.controller.prepare(),rejected=assert.rejects(p,StudioSceneChangedError);await tick();assert(entered);
  f.scene.state.mesh[0]=100;barrier.resolve();await rejected;await f.controller.settled();
  assert.equal(f.released.length,1);assert.equal(f.controller.takeReady(),null);
}
{
  const f=fixture();await f.controller.prepare();f.scene.revision++;
  assert.throws(()=>f.controller.takeReady(),StudioSceneChangedError);await f.controller.settled();
  assert.equal(f.released.length,1,'Change-and-revert invalidated by authoring revision');
}
{
  const f=fixture();await f.controller.prepare();f.scene.state.camera[0]=1;
  assert.throws(()=>f.controller.takeReady(),StudioSceneChangedError);await f.controller.settled();
  assert.equal(f.released.length,1,'Exact comparison catches unversioned mutation');
}
for(const phase of ['prepare','stage']) {
  const barrier=deferred();let using=true;
  const f=fixture({...(phase==='prepare'?{prepareCandidate:async()=>{await barrier.promise;using=false;return candidate();}}:
    {stagePresentation:async args=>{await barrier.promise;using=false;return receipts(args);}}),
    releaseCandidate:async()=>{assert.equal(using,false,'Must not dispose an in-flight candidate');f.released.push('released');}});
  const p=f.controller.prepare(),rejected=assert.rejects(p,StudioSceneChangedError);await tick();
  const cancel=f.controller.cancel();await tick();assert.equal(f.released.length,0);
  barrier.resolve();await Promise.all([rejected,cancel]);await f.controller.settled();assert.equal(f.released.length,1);
}
{
  const barrier=deferred();let calls=0;
  const f=fixture({prepareCandidate:async()=>{calls++;if(calls===1)await barrier.promise;return candidate();}});
  const first=f.controller.prepare(),rejected=assert.rejects(first,StudioSceneChangedError);await tick();
  f.scene.revision++;const second=f.controller.prepare();await tick();assert.equal(calls,1,'No overlapping old/new candidates');
  barrier.resolve();await rejected;await second;assert.equal(calls,2);assert.equal(f.released.length,1);
  assert.equal(f.controller.takeReady().revision,1);await f.controller.settled();
}
for(const bad of ['phong','ray']) {
  const f=fixture({stagePresentation:async args=>{const r=receipts(args);r[bad].step++;return r;}});
  await assert.rejects(f.controller.prepare(),new RegExp(`${bad} must acknowledge`));
  assert.equal(f.released.length,1);assert.equal(f.controller.takeReady(),null);
}
{
  let staged;const f=fixture({stagePresentation:async args=>(staged=receipts(args))});
  await f.controller.prepare();staged.phong.ready=false;
  assert.throws(()=>f.controller.takeReady(),/phong must acknowledge/);await f.controller.settled();
  assert.equal(f.released.length,1,'Lost/released renderer readiness is rechecked before reveal');
}
{
  const f=fixture({prepareCandidate:async()=>{const c=candidate();c.runtime.running=true;return c;}});
  await assert.rejects(f.controller.prepare(),/paused/);assert.equal(f.released.length,1);
}
{
  const f=fixture({prepareCandidate:async({capture})=>{capture.state.mesh[0]=100;return candidate();},
    stagePresentation:async args=>{assert.equal(args.capture.state.mesh[0],0);return receipts(args);}});
  await f.controller.prepare();assert.equal(f.controller.takeReady().state.mesh[0],0);
}
{
  const f=fixture({releaseCandidate:async()=>{throw new Error('queue settlement failed');}});
  await f.controller.prepare();f.scene.revision++;
  assert.throws(()=>f.controller.takeReady(),StudioSceneChangedError);
  await assert.rejects(f.controller.settled(),/release hidden physics/);
  assert.equal(f.controller.state().cleanupFailures,1);
}
{
  let c,coalesced;const f=fixture({onStatus:s=>{if(s.phase==='preparing')coalesced=c.prepare();}});c=f.controller;
  const p=c.prepare();assert.equal(coalesced,p);await p;await c.cancel();
}
{
  let c,cancellation;const f=fixture({onStatus:s=>{if(s.phase==='preparing')cancellation=c.cancel();}});c=f.controller;
  await assert.rejects(c.prepare(),StudioSceneChangedError);await cancellation;assert.equal(f.released.length,0);
}
console.log('PASS hidden Studio transition: exact state, coalescing, stale/cancel races, dual-consumer gate, one-time transfer, safe cleanup');
