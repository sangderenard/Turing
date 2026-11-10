// Bounded manager/render-boundary state machine. Only hydration, GPU creation
// and camera imports are explicit doubles; the room module itself is unchanged.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
if(!vm.SourceTextModule)throw Error('Use node --experimental-vm-modules');
const moduleSource=fs.readFileSync(new URL('../examples/meowdoku/web/mechanical_creature_studio_room.js',import.meta.url),'utf8');
const sourceIdentity='python:test.StudioManager',I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],created=[],disposed=[],draws=[],hydrations=[],slow=new Map();
const asset=(id,revision=1)=>({schema:'turing.studio-scene-asset.v1',identity:'studio-scene:'+id,revision,source:{classIdentity:sourceIdentity}});
const building=a=>({assetIdentity:a.identity,sourceRevision:a.revision,placement:I,entrance:null,proxyIds:['studio-room','studio-table']});
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const hydrated=a=>({identity:a.identity,packet:{classIdentity:sourceIdentity,revision:a.revision,scene:{subject:{id:'subject'}}},samplers:{}});
const fakeHydrate=async a=>{hydrations.push(a.identity);if(a.bad)throw Error('test checksum failure');if(slow.has(a.identity))await slow.get(a.identity).promise;return hydrated(a);};
let failedStage=null,failedDraw=null;
const fakeRenderer=({asset,placement})=>{
  if(asset.identity===failedStage)throw Error('test GPU stage failure');
  const r={identity:asset.identity,placement,ready:true,counts:{parts:37},bounds:{},draw(c){if(r.identity===failedDraw)throw Error('test GPU draw failure');assert(r.ready);draws.push({id:r.identity,phase:c.phase});},
    dispose(){if(r.ready){r.ready=false;disposed.push(r.identity);}}};created.push(r);return r;
};
const sandbox=vm.createContext({structuredClone,console:{warn(){}}}),module=new vm.SourceTextModule(moduleSource,{context:sandbox});
const verifications=[];
const imports={'./marble_studio_scene_asset.js':{hydrateStudioSceneAsset:fakeHydrate,
  verifyStudioSceneAssetIdentity:async a=>{verifications.push(a.identity);if(a.bad)throw Error('test checksum failure');return a.identity;},createStudioSceneRenderer:fakeRenderer},
  './mechanical_creature_studio_camera.js':{mechanicalCreatureStudioCamera:c=>({view:I,projection:I,gameSnapshotSequence:c.stateLoopRuntime.latestSequence})}};
await module.link(name=>{const values=imports[name];assert(values);return new vm.SyntheticModule(Object.keys(values),function(){for(const [k,v]of Object.entries(values))this.setExport(k,v);},{context:sandbox});});
await module.evaluate();const {installStudioRoom}=module.namespace;
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const context=a=>({gl:{},sceneGeometry:['studio-room','studio-table'].map(identity=>({identity,appearance:{studio_asset_identity:a.identity}})),stateLoopRuntime:{latestSequence:41}});
function host(){const originalSet=new Set(['projectile']),originalDraws=[];const previous={identities:()=>originalSet,draw:c=>{originalDraws.push(c.phase);return true;}};
  return {mechanicalCreatureImportedAppearance:previous,previous,originalSet,originalDraws};}
const frame=(host,c)=>{const adapter=host.mechanicalCreatureImportedAppearance,identities=adapter.identities(c);
  adapter.draw({...c,identities,phase:'opaque'});adapter.draw({...c,identities,phase:'transparent'});return identities;};
const a=asset('a'),b=asset('b',2),owned=host(),statuses=[];
const room=await installStudioRoom({asset:a,building:building(a),host:owned,onStatus:s=>statuses.push(s)});
let visible=false;room.ready.then(()=>visible=true);await flush();assert(!visible);assert.equal(created.length,0);
const c=context(a),adapter=owned.mechanicalCreatureImportedAppearance,ids=adapter.identities(c);
assert.deepEqual([...ids],['projectile','studio-room','studio-table']);assert.deepEqual([...owned.originalSet],['projectile']);
assert(!room.readStatus().ready);adapter.draw({...c,identities:ids,phase:'opaque'});await flush();assert(!visible);
adapter.draw({...c,identities:ids,phase:'transparent'});const receipt=await room.ready;
assert(visible);assert.equal(receipt.sequence,0);assert.equal(receipt.gameSnapshotSequence,41);assert.equal(receipt.environment.colliderPublication,'pre-existing-native-environment');
assert(room.readStatus().ready);assert.equal(created.length,1);frame(owned,c);assert.equal(created.length,1);
// A hydrated candidate cannot hide old proxies until its matching environment exists.
const pending=room.update({asset:b,building:building(b),sequence:1});let bVisible=false;pending.then(()=>bVisible=true);await flush();
frame(owned,c);assert.equal(room.readStatus().identity,a.identity);assert(!bVisible);assert(!disposed.includes(a.identity));
const cb=context(b),ib=adapter.identities(cb);adapter.draw({...cb,identities:ib,phase:'opaque'});
assert(!disposed.includes(a.identity));adapter.draw({...cb,identities:ib,phase:'transparent'});await pending;
assert.equal(room.readStatus().identity,b.identity);assert(disposed.includes(a.identity));
// Replaying the same owned input object avoids rehydration, not the visible ack.
const beforeHydrations=hydrations.length,same=room.update({asset:b,building:building(b),sequence:2});await flush();frame(owned,cb);await same;assert.equal(hydrations.length,beforeHydrations);assert(verifications.includes(b.identity));
await assert.rejects(room.update({asset:b,building:building(b),sequence:2}),e=>e.code==='STALE_STUDIO_UPDATE');
await assert.rejects(room.update({asset:a,building:building(a),sequence:3}),e=>e.code==='STALE_STUDIO_UPDATE');
await assert.rejects(room.update({asset:b,building:{...building(b),assetIdentity:'wrong'},sequence:3}));
const bad=asset('bad',3);bad.bad=true;const failed=room.update({asset:bad,building:building(bad),sequence:3});await assert.rejects(failed,/checksum/);frame(owned,cb);
assert.equal(room.readStatus().identity,b.identity);assert(room.readStatus().ready);
const gpuBad=asset('gpu-bad',3);failedStage=gpuBad.identity;const failedGpu=room.update({asset:gpuBad,building:building(gpuBad),sequence:4});await flush();frame(owned,cb);
await assert.rejects(failedGpu,/GPU stage/);assert.equal(room.readStatus().identity,b.identity);failedStage=null;
room.dispose();assert.equal(owned.mechanicalCreatureImportedAppearance,owned.previous);await assert.rejects(room.update({asset:b,building:building(b),sequence:5}),e=>e.code==='CLOSED_STUDIO_ROOM');
// Native host transaction path: stage in one original frame, commit at the
// next existing controls boundary, then acknowledge both actual visible passes.
const dynamicHost=host(),queues=[],queue=p=>{const wait=deferred();queues.push({...p,...wait});return wait.promise;};
const d0=asset('dynamic-0',7),dynamic=await installStudioRoom({asset:d0,building:building(d0),sequence:10,host:dynamicHost,beforeCommit:queue,onStatus(){throw Error('observer failure');}});
const dc=context(d0);frame(dynamicHost,dc);assert.equal(queues.length,1);assert.equal(dynamic.readStatus().identity,null);
const apply=(q,c)=>{assert(q.isCurrent());c.sceneGeometry=context({identity:q.identity}).sceneGeometry;
  const receipt={assetIdentity:q.identity,revision:q.sourceRevision,sequence:q.sequence,geometryRevision:q.sequence+100,
    colliderPublication:'posted-to-original-worker'};assert(q.commit(receipt));q.resolve(receipt);return receipt;};
const firstNativeReceipt=apply(queues[0],dc);assert.throws(()=>queues[0].commit(firstNativeReceipt),/only once/);
assert(!dynamic.readStatus().ready);frame(dynamicHost,dc);const d0Receipt=await dynamic.ready;
assert.equal(d0Receipt.environment.colliderPublication,'posted-to-original-worker');assert.equal(d0Receipt.sequence,10);
const d1=asset('dynamic-1',8),u1=dynamic.update({asset:d1,building:building(d1),sequence:11});await flush();frame(dynamicHost,dc);
assert.equal(dynamic.readStatus().identity,d0.identity);assert.equal(queues.length,2);
const d2=asset('dynamic-2',8),u2=dynamic.update({asset:d2,building:building(d2),sequence:12});await assert.rejects(u1,e=>e.code==='SUPERSEDED_STUDIO_UPDATE');
assert(!queues[1].isCurrent());assert.equal(queues[1].commit({}),false);queues[1].reject(Error('native cancellation'));await flush();frame(dynamicHost,dc);
assert.equal(queues.length,3);apply(queues[2],dc);frame(dynamicHost,dc);assert.equal((await u2).identity,d2.identity);
// While one hydration finishes, only the latest queued complete batch starts.
const d3=asset('slow',9),d4=asset('never-hydrate',9),d5=asset('latest',9),wait=deferred();slow.set(d3.identity,wait);
const u3=dynamic.update({asset:d3,building:building(d3),sequence:13});
const u4=dynamic.update({asset:d4,building:building(d4),sequence:14});
const u5=dynamic.update({asset:d5,building:building(d5),sequence:15});
await assert.rejects(u3,e=>e.code==='SUPERSEDED_STUDIO_UPDATE');await assert.rejects(u4,e=>e.code==='SUPERSEDED_STUDIO_UPDATE');
assert(!hydrations.includes(d4.identity));assert(!hydrations.includes(d5.identity));wait.resolve();await flush();assert(hydrations.includes(d5.identity));
frame(dynamicHost,dc);apply(queues.at(-1),dc);frame(dynamicHost,dc);await u5;
// A bad host receipt cannot publish the candidate or retire working resources.
const d6=asset('bad-receipt',10),u6=dynamic.update({asset:d6,building:building(d6),sequence:16});await flush();frame(dynamicHost,dc);
const q6=queues.at(-1);assert.throws(()=>q6.commit({assetIdentity:d6.identity,revision:9,sequence:16}),/does not match/);
q6.reject(Error('native receipt rejected'));await assert.rejects(u6,/receipt rejected/);assert.equal(dynamic.readStatus().identity,d5.identity);
// A draw failure is not acknowledged; original hook can show native proxies.
const d7=asset('draw-failure',10),u7=dynamic.update({asset:d7,building:building(d7),sequence:17});await flush();frame(dynamicHost,dc);apply(queues.at(-1),dc);
failedDraw=d7.identity;assert.throws(()=>frame(dynamicHost,dc),/GPU draw/);await assert.rejects(u7,/GPU draw/);assert(!dynamic.readStatus().ready);failedDraw=null;
const refused=dynamicHost.mechanicalCreatureImportedAppearance.identities(dc);assert.deepEqual([...refused],['projectile']);
// Closing cancels staged commits and releases both old and new GPU resources.
const d8=asset('closing',11),u8=dynamic.update({asset:d8,building:building(d8),sequence:18});await flush();frame(dynamicHost,dc);
const q8=queues.at(-1);dynamic.dispose();await assert.rejects(u8,e=>e.code==='CLOSED_STUDIO_ROOM');assert(!q8.isCurrent());assert.equal(q8.commit({}),false);
assert.equal(dynamicHost.mechanicalCreatureImportedAppearance,dynamicHost.previous);assert(created.every(x=>!x.ready));
// Cache hits verify content again, even for a changed input claiming the same ID.
const cacheHost=host(),cacheAsset=asset('cache'),cacheRoom=await installStudioRoom({asset:cacheAsset,building:building(cacheAsset),host:cacheHost});
frame(cacheHost,context(cacheAsset));await cacheRoom.ready;const beforeCache=hydrations.length;
const tampered={...cacheAsset,bad:true};await assert.rejects(cacheRoom.update({asset:tampered,building:building(tampered),sequence:1}),/checksum/);
assert.equal(hydrations.length,beforeCache);assert(cacheRoom.readStatus().ready);cacheRoom.dispose();assert(created.every(x=>!x.ready));
assert(!/requestAnimationFrame|setInterval|setTimeout/.test(moduleSource),'Room must not add a loop or polling clock');
console.log(JSON.stringify({pass:true,initialVisibleAck:true,atomicEnvironmentCallback:true,staleManagerRejection:true,
  latestQueuedHydrationOnly:true,oldRendererRetainedUntilVisible:true,collectionPreserved:true,observerFailuresIsolated:true,
  stageAndDrawFailuresCovered:true,disposalAndCanceledCommits:true,newLoop:false,gpuRenderingTested:false}));
