import assert from 'node:assert/strict';
import test from 'node:test';
import {webcrypto} from 'node:crypto';
import {receiveSceneResponses} from '../examples/meowdoku/web/marble_scene_responses.js';

globalThis.crypto??=webcrypto;
const U=0xffffffff,scene='a'.repeat(64),options={sceneRevision:scene,information:'evaluated-responses-v1',recordIndexBytes:1024*1024};
const bits=values=>new Uint32Array(new Float32Array(values).buffer);
const families=Object.fromEntries([[1,'forward','evaluated-forward-responses-v1'],[2,'reactive','evaluated-forward-responses-v1'],
  [3,'camera_rhs_summands','native-camera-response-pairs-v1'],[4,'camera_complete_sample','native-camera-scene-sample-v1']]
  .map(([id,name,codec])=>[id,{name,codec}]));
async function fixture(specs,{bounds=[[0,0,0],[4,4,4]],dimensions=[4,4,4]}={}) {
  const index=new Uint32Array(specs.length*20),native=[],rhs=[];
  for(const [i,s]of specs.entries()) {
    const family=s.family??1,point=s.end??[.5,.5,.5],direction=s.direction??[0,1,0],present=s.present??true,
      row=new Uint32Array(family===3?384:128),idx=index.subarray(i*20,(i+1)*20),ref=family===3?rhs.length:native.length;
    idx.set([family,family===4?1:5,s.context??0,ref,s.namespace??1,s.recipient??7,s.primitive??U,1]);
    idx.set(bits(point),8);idx[11]=Number(present);idx.set(bits(direction),12);idx[15]=family<=2?1:2;
    idx.fill(U,16,20);idx[19]=s.bin??2;
    if(family===4) {row[3]=idx[6];row.set(bits(point),8);row[11]=idx[11];row.set(bits(direction),12);}
    else if(family<=2) {
      row.set([5,family,11,13,2,present?9:U]);row[30]=1;row[112]=s.primitive??U;
      row.set(bits(point),21);row.set(bits(direction),27);
      row.set(bits(s.entryDirection??[1,0,0]),114);row.set(bits(s.start??[-1,.5,.5]),117);
      row[124]=s.medium??3;row[32]=0x1234;row[64]=0x5678;row[120]=0x7fc01234;
    }
    s.edit?.(idx,row);(family===3?rhs:native).push(row);
  }
  const sections=[['scene_index',80,specs.length],['native_rows',512,native.length],['camera_rhs_rows',1536,rhs.length],['scalar_cells',24,0]];
  let end=0;const layout=sections.map(([name,stride,count])=>{const section={name,stride,count,offset:end};end+=stride*count;return section;});
  const payload=new ArrayBuffer(end),bytes=new Uint8Array(payload);
  bytes.set(new Uint8Array(index.buffer),0);
  native.forEach((row,i)=>bytes.set(new Uint8Array(row.buffer),layout[1].offset+i*512));
  rhs.forEach((row,i)=>bytes.set(new Uint8Array(row.buffer),layout[2].offset+i*1536));
  const contexts=[...new Set(specs.map(s=>s.context??0))].map(id=>({id,scene_revision:scene,physical_scene_identity:scene,
    stream_id:'original-stream-'+id,sequence:1,semantics:'fixture exact native observations',dispatch:{},source_sha256:'b'.repeat(64)}));
  const manifest={schema:'studio-scene-responses-v1',scene_revision:scene,publication_sequence:1,byte_order:'little',
    payload_bytes:end,payload_sha256:Buffer.from(await crypto.subtle.digest('SHA-256',payload)).toString('hex'),
    families,contexts,sections:layout,scalar_snapshot:null,
    indexing:{semantics:'lookup buckets only; exact positions/directions retained',stencil:'axes6-v1',bounds,dimensions},
    retention:{mode:'bounded-fifo',capacity:Math.max(1,specs.length),retained_entries:specs.length,evicted_entries:0,
      duplicate_entries:0,invalidated_entries:0,incomplete_dispatches:0,unavailable_on_eviction:true},
    complete_retained_snapshot:true,physical_completeness_claim:false,application_ready:false};
  return {manifest,payload};
}
async function receive(specs,geometry,overrides) {
  const f=await fixture(specs,geometry);return receiveSceneResponses(f.manifest,f.payload,{...options,...overrides});
}
const lookup=(result,cell,filters)=>[...result.recordIndex.records(cell,filters)];

test('crossing with both endpoints outside indexes every crossed cell, not deposition',async()=>{
  const result=await receive([{start:[-1,.5,.5],end:[5,.5,.5]}]);
  for(let x=0;x<4;x++)assert.deepEqual(lookup(result,[x,0,0]),[{row:0,support:2}]);
  assert.deepEqual(lookup(result,[0,1,0]),[]);
  assert.equal(result.recordIndex.stats.segmentRows,1);assert.equal(result.recordIndex.stats.pointRows,0);
  assert.equal(result.recordIndex.energyStatements,false);assert.equal(result.applicationReady,false);
  assert.equal(result.nativeRows[120],0x7fc01234);assert.equal(result.nativeRows[124],3);
});
test('point plus segment is one reference with explicit support bits',async()=>{
  const result=await receive([{start:[.25,.5,.5],end:[2.5,.5,.5]}]);
  assert.deepEqual(lookup(result,[0,0,0]),[{row:0,support:2}]);
  assert.deepEqual(lookup(result,[2,0,0]),[{row:0,support:3}]);
  assert.equal(result.recordIndex.stats.references,3);
});
test('negative travel, clipping and internal-plane endpoint obey half-open cells',async()=>{
  const result=await receive([{start:[5,.5,.5],end:[1,.5,.5],entryDirection:[-1,0,0]}]);
  assert.deepEqual(lookup(result,[0,0,0]),[]);
  assert.deepEqual(lookup(result,[1,0,0]),[{row:0,support:3}]);
  assert.deepEqual(lookup(result,[3,0,0]),[{row:0,support:2}]);
  const exact=await receive([{start:[2,.5,.5],end:[1,.5,.5],entryDirection:[-1,0,0]}]);
  assert.deepEqual(lookup(exact,[2,0,0]),[]);assert.deepEqual(lookup(exact,[1,0,0]),[{row:0,support:3}]);
});
test('upper-face coplanar segment is outside; lower face is inside',async()=>{
  const result=await receive([{start:[-1,4,.5],end:[5,4,.5]},{start:[-1,0,.5],end:[5,0,.5]}]);
  assert.equal(result.recordIndex.stats.unindexedRows,1);
  for(let x=0;x<4;x++)assert.deepEqual(lookup(result,[x,0,0]),[{row:1,support:2}]);
});
test('diagonal traversal advances tied planes without duplicate or corner-only support',async()=>{
  const result=await receive([{start:[-1,-1,-1],end:[5,5,5],entryDirection:[1,1,1]}]);
  for(let x=0;x<4;x++)assert.deepEqual(lookup(result,[x,x,x]),[{row:0,support:2}]);
  assert.deepEqual(lookup(result,[1,0,0]),[]);assert.equal(result.recordIndex.stats.references,4);
});
test('all record families remain distinct; exact filters do not widen semantics',async()=>{
  const result=await receive([{family:1},{family:2,context:1},{family:3},{family:4}]);
  assert.equal(lookup(result,[0,0,0]).length,4);
  for(let family=1;family<=4;family++)assert.equal(lookup(result,[0,0,0],{family}).length,1);
  assert.deepEqual(lookup(result,[0,0,0],{family:2,context:0}),[]);
  assert.deepEqual(lookup(result,[0,0,0],{directionBin:1}),[]);
  assert.equal(result.recordIndex.stats.segmentUnavailableRows,2);
});
test('unlocated and unmatched interior records stay raw; genuine segments can be spatially indexed',async()=>{
  const result=await receive([{namespace:0,recipient:U,primitive:2,start:[-1,.5,.5],end:[5,.5,.5]},
    {namespace:0,recipient:U,present:false,end:[2,.5,.5]},{family:4,present:false}]);
  assert.deepEqual(lookup(result,[1,0,0]),[{row:0,support:2}]);
  assert.equal(result.recordIndex.stats.unindexedRows,2);
  assert.equal(result.families.forward,2);assert.equal(result.nativeRows.length,3*128);
});
test('missing entry direction, zero-length and nonfinite intervals give endpoint-only support',async()=>{
  for(const spec of [{entryDirection:[0,0,0]},{start:[.5,.5,.5]},
    {edit:(idx,row)=>row[117]=0x7fc00000}]) {
    const result=await receive([spec]);assert.deepEqual(lookup(result,[0,0,0]),[{row:0,support:1}]);
    assert.equal(result.recordIndex.stats.segmentUnavailableRows,1);
  }
});
test('disagreeing attachment cannot manufacture a segment',async()=>{
  for(const edit of [(idx,row)=>row[21]=bits([3])[0],(idx,row)=>row[1]=2,
    (idx,row)=>row[27]=bits([1])[0],(idx,row)=>row[30]=0,
    (idx,row)=>{idx[6]=1;row[112]=2;}]) {
    const result=await receive([{edit}]);assert.deepEqual(lookup(result,[0,0,0]),[{row:0,support:1}]);
    assert.equal(result.recordIndex.stats.segmentRows,0);
  }
});
test('null domain explicitly retains but does not spatially classify records',async()=>{
  const result=await receive([{}, {family:4}],{bounds:null,dimensions:null});
  assert.equal(result.recordIndex.stats.unindexedRows,2);assert.equal(result.recordIndex.bytes,0);
  assert.throws(()=>lookup(result,[0,0,0]),/outside/);
});
test('explicit opt-in and budget remain mandatory; default has no index allocation',async()=>{
  const f=await fixture([{}]);
  await assert.rejects(()=>receiveSceneResponses(f.manifest,f.payload,{...options,information:undefined}),/opt-in/);
  const plain=await receiveSceneResponses(f.manifest,f.payload,{...options,recordIndexBytes:0});
  assert.equal(plain.recordIndex,null);assert.equal(plain.memory.recordIndexBytes,0);
  for(const recordIndexBytes of [-1,1.5,Infinity,64*1024*1024+1,1,65536])
    await assert.rejects(()=>receiveSceneResponses(f.manifest,f.payload,{...options,recordIndexBytes}),/budget|scratch/);
  const exact=await receiveSceneResponses(f.manifest,f.payload,{...options,recordIndexBytes:65544});
  assert.equal(exact.memory.recordIndexBytes+exact.memory.recordIndexScratchReserve,65544);
});
test('unrepresentable grid spans refuse indexing instead of inventing support',async()=>{
  for(const bounds of [[[-1e308,0,0],[1e308,4,4]],[[0,0,0],[1e-320,4,4]]])
    await assert.rejects(()=>receive([{}],{bounds,dimensions:[4,4,4]}),/representable/);
});
test('lookup rejects unsupported queries and retains exact signed zeros and payload bits',async()=>{
  const result=await receive([{edit:(idx,row)=>{row[10]=0x80000000;row[31]=0;}}]);
  assert.throws(()=>lookup(result,[4,0,0]),/outside/);assert.throws(()=>lookup(result,[0,0,0],{power:5}),/filter/);
  for(const key of ['constructor','toString','__proto__'])
    assert.throws(()=>lookup(result,[0,0,0],Object.fromEntries([[key,0]])),/filter/);
  assert.equal(result.nativeRows[10],0x80000000);assert.equal(result.nativeRows[120],0x7fc01234);
  assert.equal(result.recordIndex.applicationReady,false);assert.equal(result.memory.gpuBytes,0);
});
test('index never modifies decoded records or conflates repeated distinct observations',async()=>{
  const f=await fixture([{},{}]),before=new Uint8Array(f.payload).slice();
  const plain=await receiveSceneResponses(f.manifest,f.payload,{...options,recordIndexBytes:0});
  const indexed=await receiveSceneResponses(f.manifest,f.payload,options);
  assert.deepEqual(new Uint8Array(f.payload),before);
  for(const key of ['index','nativeRows','cameraRhsRows','scalarCells','applicationRows'])assert.deepEqual(indexed[key],plain[key]);
  assert.deepEqual(lookup(indexed,[0,0,0]),[{row:0,support:3},{row:1,support:3}]);
});
test('deterministic segments agree with independent cell-by-cell slab classification',async()=>{
  let seed=527;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
  const specs=Array.from({length:150},()=>({start:Array.from({length:3},()=>Math.fround(random()*8-2)),
    end:Array.from({length:3},()=>Math.fround(random()*8-2))}));
  const result=await receive(specs);
  function expected(s,cell) {
    const point=s.end.every((v,j)=>v>=cell[j]&&v<cell[j]+1);let enter=0,leave=1,segment=true;
    for(let j=0;j<3;j++) {
      const delta=s.end[j]-s.start[j];
      if(delta===0) {if(s.start[j]<cell[j]||s.start[j]>=cell[j]+1)segment=false;}
      else {
        const a=(cell[j]-s.start[j])/delta,b=(cell[j]+1-s.start[j])/delta;
        enter=Math.max(enter,Math.min(a,b));leave=Math.min(leave,Math.max(a,b));
      }
    }
    return Number(point)|(segment&&enter<leave?2:0);
  }
  for(let z=0;z<4;z++)for(let y=0;y<4;y++)for(let x=0;x<4;x++) {
    const cell=[x,y,z],want=specs.map((s,row)=>({row,support:expected(s,cell)})).filter(r=>r.support);
    assert.deepEqual(lookup(result,cell),want);
  }
});
test('crossing-reference budget failure does not publish a truncated partial index',async()=>{
  await assert.rejects(()=>receive([{start:[-1,.5,.5],end:[5,.5,.5]}],undefined,{recordIndexBytes:65536+3*8}),/no partial/);
});
test('empty snapshot is a zero-reference index and still makes no availability claim',async()=>{
  const result=await receive([]);
  assert.deepEqual(lookup(result,[0,0,0]),[]);assert.equal(result.recordIndex.stats.references,0);
  assert.equal(result.recordIndex.applicationReady,false);assert.equal(result.recordIndex.energyStatements,false);
});
