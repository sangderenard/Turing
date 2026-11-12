// CPU-only version/ABI checks; --stdin accepts a real producer snapshot.
import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
import {receiveSceneResponses,CAMERA_PAIR_CONTRACT} from '../examples/meowdoku/web/marble_scene_responses.js';
import {validateSceneResponseReceipt} from '../examples/meowdoku/web/marble_light_field.js';
import {packMarblePhoto} from '../examples/meowdoku/web/marble_photo.js';
globalThis.crypto??=webcrypto;
const UINT=0xffffffff,A='a'.repeat(64),B='b'.repeat(64);
const sha=async bytes=>Buffer.from(await crypto.subtle.digest('SHA-256',bytes)).toString('hex');
const options={sceneRevision:A,information:'evaluated-responses-v1'};

if(process.argv.includes('--stdin')) {
  let input='';for await(const chunk of process.stdin)input+=chunk;
  const fixture=JSON.parse(input),bytes=Buffer.from(fixture.payload,'base64');
  const payload=bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);
  const result=await receiveSceneResponses(fixture.manifest,payload,{...options,sceneRevision:fixture.manifest.scene_revision,...(fixture.queryIndex?{recordIndexBytes:1048576,cameraQuerySegments:true}:{})});
  assert.equal(result.metadata.schema,'studio-scene-responses-v2');
  if(fixture.queryIndex){assert.ok(result.recordIndex);assert.equal(result.recordIndex.energyStatements,false);
    if(fixture.expectedQueryReferences!==undefined)assert.equal(result.recordIndex.stats.references,fixture.expectedQueryReferences);}
  if(fixture.cameraRhsWords)assert.deepEqual(Array.from(result.cameraRhsRows),fixture.cameraRhsWords);
  assert.equal(result.applicationReady,false);assert.equal(result.memory.gpuBytes,0);
  console.log(JSON.stringify({result:'PASS',pairs:result.families.camera_rhs_summands,
    cameraWords:result.cameraRhsRows.length,gpuBytes:0,applicationReady:false}));
} else {
  const geometry={interior:[[[0,0,0],[1,0,0],[0,1,0],0,[[0,0,1],[0,0,1],[0,0,1]]]],glass:[]};
  const unpack=buffer=>JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,16,new DataView(buffer).getUint32(4,true))));
  const requested={receiveLightField:true,lightFieldInformation:'evaluated-responses-v1'};
  const ordinary=unpack(packMarblePhoto(geometry,A,requested));
  assert.ok(!Object.hasOwn(ordinary,'light_field_camera_records'));
  assert.deepEqual(unpack(packMarblePhoto(geometry,A,{...requested,lightFieldCameraRecords:'interval-responses-v2'})),
    {...ordinary,light_field_camera_records:'interval-responses-v2'});
  for(const lightFieldCameraRecords of [null,true,1,{},[],'interval-responses-v1'])
    assert.throws(()=>packMarblePhoto(geometry,A,{...requested,lightFieldCameraRecords}),/Interval camera records/);
  assert.throws(()=>packMarblePhoto(geometry,A,{receiveLightField:true,lightFieldCameraRecords:'interval-responses-v2'}),/explicit/);
  const payload=new ArrayBuffer(160+2048),words=new Uint32Array(payload),floats=new Float32Array(payload);
  for(let row=0;row<2;row++) {
    const i=row*20,r=40+row*256,primitive=row?UINT:2;
    words.set([3,1,0,row,0,UINT,primitive,1],i);
    words[i+15]=2;words.fill(UINT,i+16,i+20);
    words[r+4]=row?UINT:6;words[r+5]=4;words[r+6]=3;words[r+40]=7;
    words[r+115]=17;words[r+124]=0;words[r+125]=UINT;words[r+126]=primitive;words[r+127]=UINT;
    floats[r+8]=.25;floats[r+9]=.5;floats[r+10]=.75;
    words[r+11]=row?0x7fc12345:0x3f800000;
    floats[r+12]=1;words.set(words.subarray(r+12,r+15),i+12);
    words[r+128]=0x7fc01234;words[r+129]=0x80000000;
    // A zero and unknown source state are retained, not combined as coverage.
    words[r+128+96]=row?7:0;words[r+128+108]=row?0x15:0;
  }
  const manifest={schema:'studio-scene-responses-v2',scene_revision:A,publication_sequence:2,
    byte_order:'little',payload_bytes:payload.byteLength,payload_sha256:await sha(payload),
    camera_pair_contract:{...CAMERA_PAIR_CONTRACT},
    families:Object.fromEntries([[1,'forward','evaluated-forward-responses-v1'],[2,'reactive','evaluated-forward-responses-v1'],
      [3,'camera_rhs_summands','native-camera-response-pairs-v2'],[4,'camera_complete_sample','native-camera-scene-sample-v1']].map(([id,name,codec])=>[id,{name,codec}])),
    contexts:[{id:0,scene_revision:A,physical_scene_identity:A,stream_id:'actual-camera',sequence:0,
      semantics:'actual native intervals and aligned response copies; not coverage',dispatch:{},source_sha256:B}],
    sections:[{name:'scene_index',offset:0,stride:80,count:2},{name:'native_rows',offset:160,stride:512,count:0},
      {name:'camera_rhs_rows',offset:160,stride:1024,count:2},{name:'scalar_cells',offset:2208,stride:24,count:0}],scalar_snapshot:null,
    indexing:{semantics:'lookup buckets only; exact positions/directions retained',stencil:'icosahedron12-v1',bounds:null,dimensions:null},
    retention:{mode:'bounded-fifo',capacity:4,retained_entries:2,evicted_entries:0,duplicate_entries:0,invalidated_entries:0,
      incomplete_dispatches:0,unavailable_on_eviction:true},complete_retained_snapshot:true,physical_completeness_claim:false,application_ready:false};
  const result=await receiveSceneResponses(manifest,payload,options);
  assert.equal(result.cameraRhsRows.length,512);
  assert.deepEqual(Array.from(result.cameraRhsRows),Array.from(words.subarray(40)));
  assert.equal(result.cameraRhsRows[128],0x7fc01234);
  assert.equal(result.cameraRhsRows[129],0x80000000);
  assert.equal(result.families.camera_rhs_summands,2);
  assert.equal(result.applicationReady,false);assert.equal(result.memory.gpuBytes,0);
  const indexedOptions={...options,recordIndexBytes:1048576,cameraQuerySegments:true};
  const grid={...manifest,indexing:{...manifest.indexing,bounds:[[0,0,0],[4,1,1]],dimensions:[4,1,1]}};
  async function query(edit=()=>{},opts=indexedOptions) {
    const copy=payload.slice(0),w=new Uint32Array(copy),f=new Float32Array(copy);
    f[48]=-1;f[49]=.5;f[50]=.5;f[51]=3;f[52]=2;f[12]=2;
    // First hit is deliberately nowhere near this current query.
    f[40+116]=999;w[40+119]=1;edit(w,f);
    return receiveSceneResponses({...grid,payload_sha256:await sha(copy)},copy,opts);
  }
  const q=await query();
  assert.equal(q.recordIndex.stats.cameraQueryRows,1);
  assert.equal(q.recordIndex.stats.cameraQueryUnavailableRows,1);
  assert.equal(q.recordIndex.stats.pointRows,0);
  for(let x=0;x<4;x++)assert.deepEqual([...q.recordIndex.records([x,0,0])],[{row:0,support:4}]);
  assert.equal(q.recordIndex.energyStatements,false);
  assert.equal((await query(()=>{},{...options,recordIndexBytes:1048576})).recordIndex.stats.references,0);
  assert.equal((await query(()=>{},options)).recordIndex,null);
  for(const cameraQuerySegments of [null,0,1,'true',{}])
    await assert.rejects(()=>receiveSceneResponses(grid,payload,{...indexedOptions,cameraQuerySegments}),/boolean opt-in/);
  await assert.rejects(()=>receiveSceneResponses(grid,payload,{...options,cameraQuerySegments:true}),/positive index budget/);
  await assert.rejects(()=>receiveSceneResponses({...grid,schema:'studio-scene-responses-v1'},payload,indexedOptions),/v2 envelope/);
  await assert.rejects(()=>query(()=>{},{...indexedOptions,recordIndexBytes:65536+31}),/budget/);
  for(const edit of [w=>w[44]=UINT, (w,f)=>f[51]=NaN,(w,f)=>f[51]=Infinity,
    (w,f)=>f[51]=0,(w,f)=>f[51]=-1,(w,f)=>f[48]=Infinity,
    (w,f)=>{f[52]=0;f[12]=0;}]) {
    const invalid=await query(edit);assert.equal(invalid.recordIndex.stats.references,0);
    assert.equal(invalid.recordIndex.stats.cameraQueryUnavailableRows,2);
  }
  for(const status of [0,1,2,4,5,6,7])for(const medium of [0,4,UINT,UINT-1]) {
    let expected;
    const retained=await query(w=>{w[40+32]=status;w[40+5]=medium;w[40+48]=0xfedcba98;
      w[40+80]=0x12345678;expected=Array.from(w.subarray(40));});
    assert.equal(retained.recordIndex.stats.references,4);
    assert.deepEqual(Array.from(retained.cameraRhsRows),expected);
    assert.equal(retained.index[11],0);assert.equal(retained.applicationReady,false);
  }
  const inherited=Object.create(CAMERA_PAIR_CONTRACT);
  Object.keys(CAMERA_PAIR_CONTRACT).forEach((_,i)=>inherited['unrelated'+i]=0);
  await assert.rejects(()=>receiveSceneResponses({...manifest,camera_pair_contract:inherited},payload,options),/camera-pair contract/);
  await assert.rejects(()=>receiveSceneResponses(manifest,payload,{sceneRevision:A}),/opt-in/);
  await assert.rejects(()=>receiveSceneResponses(manifest,payload,{...options,sceneRevision:B}),/Stale/);
  for(const edit of [m=>m.schema='studio-scene-responses-v3',m=>delete m.camera_pair_contract,
    m=>m.camera_pair_contract.response_offset_words=0,m=>m.camera_pair_contract.row_words=384,
    m=>m.camera_pair_contract.extra=true,m=>m.camera_pair_contract.application_ready=true,
    m=>m.families[3].codec='native-camera-response-pairs-v1',m=>m.sections[2].stride=1536,
    m=>m.application_ready=true,m=>m.physical_completeness_claim=true,m=>m.payload_sha256='0'.repeat(64)]) {
    const bad=structuredClone(manifest);edit(bad);await assert.rejects(()=>receiveSceneResponses(bad,payload,options));
  }
  for(const [offset,value]of [[1,2],[4,1],[5,0],[6,0],[7,0],[8,1],[11,1],[12,0],[15,1],[16,0],
    [40+6,0],[40+6,9],[40+40,0],[40+40,8],[40+126,1]]) {
    const changed=payload.slice(0);new Uint32Array(changed)[offset]=value;
    const payload_sha256=await sha(changed);
    await assert.rejects(()=>receiveSceneResponses({...manifest,payload_sha256},changed,options));
  }
  const receipt={schema:manifest.schema,digest:A,live_revision:0,sequence:1,
    name:'light-scene-response-'+'1'.repeat(32)+'.bin',byte_length:payload.byteLength,
    sha256:manifest.payload_sha256,metadata:manifest,application_ready:false};
  assert.equal(validateSceneResponseReceipt(receipt,A,0).schema,manifest.schema);
  for(const schema of ['studio-scene-responses-v1','studio-scene-responses-v3'])
    assert.throws(()=>validateSceneResponseReceipt({...receipt,schema},A,0));
  const badReceipt=structuredClone(receipt);
  badReceipt.metadata.camera_pair_contract.application_ready=0;
  assert.throws(()=>validateSceneResponseReceipt(badReceipt,A,0),/camera-pair contract/);
  // A legacy envelope cannot smuggle v2 metadata even if it has no pair rows.
  const legacy=structuredClone(manifest);legacy.schema='studio-scene-responses-v1';
  await assert.rejects(()=>receiveSceneResponses(legacy,payload,options),/v2 envelope/);
  console.log('PASS v2 camera pairs: exact two-row ABI, native-only attachment, bands, version/receipt closure, opt-in, no application');
}
