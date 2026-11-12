import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
import {receiveSceneResponses} from '../examples/meowdoku/web/marble_scene_responses.js';
globalThis.crypto??=webcrypto;
const payload=new ArrayBuffer(4*80+3*512+1536),v=new DataView(payload);
for(let i=0;i<4;i++) {
  const offset=i*80;
  for(const [word,value] of [[0,i+1],[1,1],[2,0],[3,i===2?0:i===3?2:i],[4,1],[5,41],[7,1],[11,1],[15,i<2?1:2]])v.setUint32(offset+word*4,value,true);
}
for(let i=0;i<4;i++)for(let j=16;j<20;j++)v.setUint32(i*80+j*4,0xffffffff,true);
v.setUint32(320+2*512+11*4,1,true);
v.setUint32(320,0x7fc01234,true);v.setUint32(320+512,0x80000000,true);
const sha=async bytes=>Buffer.from(await crypto.subtle.digest('SHA-256',bytes)).toString('hex');
const manifest={schema:'studio-scene-responses-v1',scene_revision:'a'.repeat(64),publication_sequence:4,
 byte_order:'little',payload_bytes:payload.byteLength,payload_sha256:await sha(payload),
 families:Object.fromEntries([[1,'forward','evaluated-forward-responses-v1'],[2,'reactive','evaluated-forward-responses-v1'],
  [3,'camera_rhs_summands','native-camera-response-pairs-v1'],[4,'camera_complete_sample','native-camera-scene-sample-v1']].map(([id,name,codec])=>[id,{name,codec}])),
 contexts:[{id:0,scene_revision:'a'.repeat(64),physical_scene_identity:'a'.repeat(64),stream_id:'stream',sequence:0,semantics:'native copied evidence',dispatch:{},source_sha256:'b'.repeat(64)}],
 sections:[{name:'scene_index',offset:0,stride:80,count:4},{name:'native_rows',offset:320,stride:512,count:3},
  {name:'camera_rhs_rows',offset:1856,stride:1536,count:1},{name:'scalar_cells',offset:3392,stride:24,count:0}],scalar_snapshot:null,
 indexing:{semantics:'lookup buckets only; exact positions/directions retained',stencil:'icosahedron12-v1',bounds:null,dimensions:null},
 retention:{mode:'bounded-fifo',capacity:4,retained_entries:4,evicted_entries:2,duplicate_entries:1,invalidated_entries:0,incomplete_dispatches:1,unavailable_on_eviction:true},
 complete_retained_snapshot:true,physical_completeness_claim:false,application_ready:false};
const options={sceneRevision:'a'.repeat(64),information:'evaluated-responses-v1'};
const result=await receiveSceneResponses(manifest,payload,options);
assert.equal(result.nativeRows[0],0x7fc01234);
assert.equal(result.nativeRows[128],0x80000000);
assert.equal(result.index[40],3);
assert.equal(result.cameraRhsRows.byteLength,1536);
assert.equal(result.memory.gpuBytes,0);assert.equal(result.applicationReady,false);
assert.deepEqual(result.metadata.retention,manifest.retention,'eviction and incompleteness preserved, not made zero');
await assert.rejects(()=>receiveSceneResponses(manifest,payload,{sceneRevision:'a'.repeat(64)}),/opt-in/);
await assert.rejects(()=>receiveSceneResponses(manifest,payload,{...options,sceneRevision:'scene-b'}),/Stale/);
await assert.rejects(()=>receiveSceneResponses(manifest,payload,{...options,maxPayloadBytes:1}),/budget/);
for(const edit of [m=>m.sections[1].offset++,m=>m.sections[1].count++,m=>m.payload_sha256='0'.repeat(64),
 m=>m.contexts.push({...m.contexts[0]}),m=>m.contexts[0].physical_scene_identity='other',
 m=>m.families[1].codec='unknown',m=>m.application_ready=true,m=>m.physical_completeness_claim=true,
 m=>m.retention.unavailable_on_eviction=false]) {
 const bad=structuredClone(manifest);edit(bad);await assert.rejects(()=>receiveSceneResponses(bad,payload,options));
}
for(const [word,value] of [[0,5],[2,99],[3,99],[4,7],[7,2],[15,0]]) {
 const bad=payload.slice(0);new DataView(bad).setUint32(word*4,value,true);
 const payload_sha256=await sha(bad);await assert.rejects(()=>receiveSceneResponses({...manifest,payload_sha256},bad,options));
}
for(let i=0;i<4;i++)for(let j=16;j<20;j++)v.setUint32(i*80+j*4,0xffffffff,true);
v.setUint32(320+2*512+11*4,1,true);
v.setUint32(320,0,true);assert.equal(result.nativeRows[0],0x7fc01234,'transport memory mutation cannot rewrite retained copy');
console.log('PASS scene-attached native response transport: exact bits, typed families, bounded memory, closure, hash, opt-in and stale identity');
{
 const m=structuredClone(manifest),bytes=new ArrayBuffer(payload.byteLength+24+48),w=new DataView(bytes);
 new Uint8Array(bytes).set(new Uint8Array(result.nativeRows.buffer),320);
 new Uint8Array(bytes).set(new Uint8Array(result.index.buffer),0);
 new Uint8Array(bytes).set(new Uint8Array(result.cameraRhsRows.buffer),1856);
 m.sections[3].count=1;m.sections.push({name:'application_rows',offset:3416,stride:48,count:1});
 m.indexing={...m.indexing,stencil:'axes6-v1',bounds:[[0,0,0],[1,1,1]],dimensions:[32,32,32]};
 m.scalar_snapshot={metadata:{identity:[1,2],total_cells:32**3,bands:3,scanned_cells:32**3,omitted_unvisited:32**3*3-1,
  evicted_entries:0,complete_snapshot:true,metadata:{}},semantics:'original cumulative scalar deposit columns; no directional or material-radiance claim'};
 for(const [j,x]of [4,2,3,1,123,16].entries())w.setUint32(3392+j*4,x,true);
 m.application={version:'native-evaluated-rgb-v1',ready:false,section:'application_rows',support:'recipient-cell-stencil-v1',
  rgb:'linear-srgb',alpha:'available-original-camera-samples',component_stage:'native-film-filtered-before-linear-profile',
  rgb_premultiplied:false,physical_completeness_claim:false,specular_override_supported:false,
  query_contexts:[{id:7,origin_m:[0,0,1],lens:{},film:{},layer_specs:[{},{},{}],
   transport:{scene_revision:m.scene_revision,source_sha256:'b'.repeat(64),transport_scene_sha256:'c'.repeat(64),shared_band0_path:true}}],current_query_context_ids:[7],
   current_transport_scene_sha256:'c'.repeat(64),current_camera_revision:'camera-a',current_live_revision:0};
 for(const [j,x]of [0x10000001,1,2,3,5,7,2,1,4].entries())w.setUint32(3416+j*4,x,true);
 for(const [j,x]of [-.25,.5,2].entries())w.setFloat32(3416+(9+j)*4,x,true);
 m.payload_bytes=bytes.byteLength;m.payload_sha256=await sha(bytes);
 const decoded=await receiveSceneResponses(m,bytes,options);
 assert.deepEqual([...decoded.scalarCells],[4,2,3,1,123,16]);
 assert.equal(new Float32Array(decoded.applicationRows.buffer)[9],-.25,'signed native RGB is not clamped or reconstructed');
 assert.equal(decoded.applicationRows[7],1);assert.equal(decoded.applicationRows[8],4);assert.equal(decoded.applicationReady,false);
 for(const [offset,value]of [[3392,32**3],[3396,3],[3416,0],[3416+4*4,6],[3416+5*4,99],[3416+6*4,4],[3416+7*4,5],[3416+8*4,0]]) {
  const bad=bytes.slice(0);new DataView(bad).setUint32(offset,value,true);const h=await sha(bad);
  await assert.rejects(()=>receiveSceneResponses({...m,payload_sha256:h},bad,options));
 }
 for(const mutate of [a=>a.ready=true,a=>a.rgb_premultiplied=true,a=>a.current_query_context_ids=[99],a=>a.query_contexts[0].transport.scene_revision='bad']) {
  const bad=structuredClone(m);mutate(bad.application);await assert.rejects(()=>receiveSceneResponses(bad,bytes,options));
 }
}
console.log('PASS scalar24 and optional application48 exact copies, six-bin bounds, native signed RGB, counts and query ownership');
