// CPU-only actual-store v3 roundtrip and adversarial contract checks.
import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
import {receiveSceneResponses} from '../examples/meowdoku/web/marble_scene_responses.js';
import {validateSceneResponseReceipt} from '../examples/meowdoku/web/marble_light_field.js';
import {packMarblePhoto} from '../examples/meowdoku/web/marble_photo.js';
globalThis.crypto??=webcrypto;
let input='';for await(const chunk of process.stdin)input+=chunk;
const {manifest,base64,words}=JSON.parse(input),buf=Buffer.from(base64,'base64');
const payload=buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength);
const opts={information:'evaluated-responses-v1',sceneRevision:manifest.scene_revision};
const result=await receiveSceneResponses(manifest,payload,opts);
assert.equal(result.metadata.schema,'studio-scene-responses-v3');
assert.deepEqual(Array.from(result.nativeRows),words);
assert.equal(result.applicationReady,false);assert.equal(result.memory.gpuBytes,0);
for(const edit of [m=>delete m.forward_ancestry_contract,m=>m.forward_ancestry_contract.enqueue_causes.extra=3,
  m=>m.forward_ancestry_contract.application_ready=0,m=>m.families[2].codec='evaluated-forward-responses-v1',
  m=>m.contexts[0].dispatch.native_queue.parent_dispatch.seed=-1,
  m=>m.contexts[0].dispatch.native_queue.generation=9,m=>m.schema='studio-scene-responses-v2']) {
  const m=structuredClone(manifest);edit(m);await assert.rejects(()=>receiveSceneResponses(m,payload,opts));
}
for(const [offset,value] of [[2,4],[3,2],[101,99],[102,13],[103,3],[125,0x7fc00000],[126,0],[127,0xffffffff]]) {
  const p=payload.slice(0),w=new Uint32Array(p);w[manifest.sections[1].offset/4+128+offset]=value;
  const m={...manifest,payload_sha256:Buffer.from(await crypto.subtle.digest('SHA-256',p)).toString('hex')};
  await assert.rejects(()=>receiveSceneResponses(m,p,opts));
}
// A one-bounce parent has no authentic enqueue opportunity. Keep its forward
// record while rejecting any reactive row against zero queue capacity.
const forwardPayload=new ArrayBuffer(80+512),fw=new Uint32Array(forwardPayload);
fw.set(new Uint32Array(payload,0,20));fw.set(result.nativeRows.subarray(0,128),20);
const forwardManifest=structuredClone(manifest);
forwardManifest.payload_bytes=forwardPayload.byteLength;
forwardManifest.payload_sha256=Buffer.from(await crypto.subtle.digest('SHA-256',forwardPayload)).toString('hex');
forwardManifest.sections=[{name:'scene_index',offset:0,stride:80,count:1},
  {name:'native_rows',offset:80,stride:512,count:1},
  {name:'camera_rhs_rows',offset:592,stride:1024,count:0},{name:'scalar_cells',offset:592,stride:24,count:0}];
forwardManifest.retention.retained_entries=1;
forwardManifest.retention.partitions.native_only_retained_entries=1;
const queue=forwardManifest.contexts[0].dispatch.native_queue;
queue.capacity=0;queue.parent_dispatch.bounce_limit=1;
assert.equal((await receiveSceneResponses(forwardManifest,forwardPayload,opts)).families.forward,1);
for(const edit of [q=>q.parent_dispatch.ray_count=0,q=>q.parent_dispatch.bounce_limit=0,
  q=>q.parent_dispatch.total_rays=12,q=>{q.logical_ray_base=0xffffffff;q.capacity=1;}]) {
  const m=structuredClone(forwardManifest);edit(m.contexts[0].dispatch.native_queue);
  await assert.rejects(()=>receiveSceneResponses(m,forwardPayload,opts));
}
const zeroReactive=structuredClone(manifest);zeroReactive.contexts[0].dispatch.native_queue.capacity=0;
await assert.rejects(()=>receiveSceneResponses(zeroReactive,payload,opts));
const receipt={schema:manifest.schema,digest:manifest.scene_revision,live_revision:0,sequence:1,
  name:'light-scene-response-'+'1'.repeat(32)+'.bin',byte_length:payload.byteLength,
  sha256:manifest.payload_sha256,metadata:manifest,application_ready:false};
assert.equal(validateSceneResponseReceipt(receipt,manifest.scene_revision,0).schema,manifest.schema);
const geometry={interior:[[[0,0,0],[1,0,0],[0,1,0],0,[[0,0,1],[0,0,1],[0,0,1]]]],glass:[]};
const unpack=b=>JSON.parse(new TextDecoder().decode(new Uint8Array(b,16,new DataView(b).getUint32(4,true))));
const requested={receiveLightField:true,lightFieldInformation:'evaluated-responses-v1'};
const ordinary=unpack(packMarblePhoto(geometry,manifest.scene_revision,requested));
assert.ok(!Object.hasOwn(ordinary,'light_field_reactive_queue'));
assert.deepEqual(unpack(packMarblePhoto(geometry,manifest.scene_revision,{...requested,lightFieldReactiveQueue:'native-ancestry-v1'})),
  {...ordinary,light_field_reactive_queue:'native-ancestry-v1'});
for(const mode of [null,true,0,{},'bad'])assert.throws(()=>packMarblePhoto(geometry,manifest.scene_revision,{...requested,lightFieldReactiveQueue:mode}));
assert.throws(()=>packMarblePhoto(geometry,manifest.scene_revision,{lightFieldReactiveQueue:'native-ancestry-v1'}));
console.log('PASS actual v3 ancestry roundtrip: exact 256 words, strict parent/codec/contract, explicit packer opt-in, no GPU');
