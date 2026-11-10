import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {StudioRangefinder,rangefinderKey,rangefinderRay,rangefinderScene} from '../examples/meowdoku/web/marble_rangefinder.js';
import {packStudioScene} from '../examples/meowdoku/web/marble_studio_scene.js';

const identity=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const triangle=(z,mat)=>[[-.1,-.1,z],[.1,-.1,z],[0,.1,z],mat,[[0,0,1],[0,0,1],[0,0,1]]];
const glass=[.5,.5,.5,.1,0,1,1.5,0,0,0,0,0,0,0,0,0],opaque=[.5,.5,.5,.8,0,0,1,1,0,0,0,0,0,0,0,0];
function capture(){return {revision:8,state:packStudioScene({channel:'studio',unitM:1,rayOriginM:[0,0,.18],
  pbr:{data:[...glass,...opaque],stride:16},parts:[
    {id:'glass',bodyId:'glass',faces:[triangle(-.02,0)]},
    {id:'wall',bodyId:'wall',faces:[triangle(-.04,1)]},
    {id:'coincident',bodyId:'separate',faces:[triangle(-.04,0)]},
    {id:'behind',bodyId:'behind',faces:[triangle(-.06,0)]}],
  camera:{view:identity,fovDegrees:50,aspect:1},authoring:{},room:null})};}
const raw=capture(),ray=rangefinderRay(raw,.5,.5);
assert.equal(ray.ray[6],-1);assert(Math.abs(ray.ray[2]-.18)<1e-7);
assert.equal(rangefinderScene(raw).count,4);
assert.notEqual(rangefinderKey(raw),rangefinderKey({...raw,revision:9}));
assert.throws(()=>rangefinderRay(raw,1.1,.5),/Invalid/);
assert.throws(()=>rangefinderScene({...raw,state:{...raw.state,triangleSources:[]}}),/identity/);

// Real prebuilt original-law Wasm, fetched through the same digest verifier.
// No mocked intersection or JavaScript geometry kernel.
const oldFetch=globalThis.fetch;
globalThis.fetch=async url=>{try{const data=await readFile(url);return new Response(data);}catch{return new Response('',{status:404});}};
let q;
try {
  q=await StudioRangefinder.create(new URL('../build/dot-client-ray-20261109/studio-compute/',import.meta.url));
  const result=await q.query(raw,{u:.5,v:.5});
  assert.equal(result.hits.length,4,'all depths, coincident owners, and opaque-hidden hits retained');
  assert.deepEqual(result.hits.map(h=>h.partId),['glass','wall','coincident','behind']);
  assert.deepEqual(result.hits.map(h=>h.behindOpaque),[false,false,false,true]);
  for(const [i,z]of [.02,.04,.04,.06].entries())assert(Math.abs(result.hits[i].axialDistanceM-z)<1e-7);
  assert.equal(result.hits[0].material,0);assert.equal(result.hits[1].material,1);
  const off=await q.query(raw,{u:.7,v:.5});
  assert.equal(off.hits.length,4);assert(off.hits[0].distanceM>off.hits[0].axialDistanceM);
  assert(Math.abs(off.hits[0].axialDistanceM-.02)<1e-7);
  const canceled=new AbortController();canceled.abort();await assert.rejects(()=>q.query(raw,{u:.5,v:.5,signal:canceled.signal}),/canceled/);
  await assert.rejects(()=>q.query(raw,{u:.5,v:.5,spread:.1}),/one selected/);
  const changed=capture();changed.state.ray.vertices[0]=NaN;await assert.rejects(()=>q.query(changed,{u:.5,v:.5}),/Invalid captured/);
  const large=capture(),n=20000,one=large.state.ray.vertices.slice(0,9);
  large.state.ray.vertices=new Float32Array(n*9);for(let i=0;i<n;i++)large.state.ray.vertices.set(one,i*9);
  large.state.ray.material_ids=Array(n).fill(0);large.state.triangleSources=Array.from({length:n},(_,i)=>({partId:'coincident-test',bodyId:'test',triangle:i}));
  const multi=await q.query(large,{u:.5,v:.5});assert.equal(multi.hits.length,n,'all hits retained across compiled batch boundary');
  const mid=new AbortController(),work=q.query(large,{u:.5,v:.5,signal:mid.signal});mid.abort();await assert.rejects(()=>work,/canceled/);
  assert.equal(q.busy,false,'cancellation releases query ownership');
  console.log(JSON.stringify({passed:true,backend:'actual compiled original-law Wasm',hits:result.hits.map(h=>({part:h.partId,axialM:h.axialDistanceM,behindOpaque:h.behindOpaque})),scope:result.scope,elapsedMs:result.elapsedMs}));
} finally {q?.close();globalThis.fetch=oldFetch;}
