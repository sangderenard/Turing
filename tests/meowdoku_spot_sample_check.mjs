import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {SPOT_LUMA,MAX_SPOT_PIXELS,MAX_SPOT_HDR_BYTES,readSpotHdr,sampleSpotHdr,
  proposeSpotAdjustment,createCompletedHdrSampler} from '../examples/meowdoku/web/marble_spot_sample.js';

const array=(width,height,fn)=>({shape:[height,width,3],dtype:'<f8',data:Float64Array.from(
  Array.from({length:height},(_,y)=>Array.from({length:width},(_,x)=>fn(x,y))).flat(2))});
const constant=array(9,5,()=>[2,4,8]),before=constant.data.slice();
for(const [u,v,spread]of [[0,0,0],[1,1,0],[.5,.5,.1],[0,1,.5]]) {
  const s=sampleSpotHdr(constant,{u,v,spread});s.rgb.forEach((x,i)=>assert.ok(Math.abs(x-[2,4,8][i])<1e-12));
  assert.ok(Math.abs(s.luminance-SPOT_LUMA.reduce((s,x,i)=>s+x*[2,4,8][i],0))<1e-12);
}
assert.deepEqual(constant.data,before,'raw input never changes');
const grid=array(5,3,(x,y)=>[y*10+x,y*10+x,y*10+x]);
assert.deepEqual(sampleSpotHdr(grid,{u:0,v:0}).rgb,[0,0,0]);
assert.deepEqual(sampleSpotHdr(grid,{u:1,v:1}).rgb,[24,24,24],'same top-to-bottom array orientation as PNG');
assert.equal(sampleSpotHdr(grid,{u:.5,v:.5}).pixelCount,1);
const impulse=array(7,7,(x,y)=>[x===3&&y===3?10:0,0,0]);
const gaussian=sampleSpotHdr(impulse,{u:.5,v:.5,spread:1/7});
const exactWeights=Array.from({length:7},(_,y)=>Array.from({length:7},(_,x)=>Math.exp(-.5*((x-3)**2+(y-3)**2)))).flat().reduce((a,b)=>a+b,0);
assert.ok(Math.abs(gaussian.rgb[0]-10/exactWeights)<1e-12);
assert.equal(gaussian.pixelCount,49);assert.equal(gaussian.sigmaPixels,1);
assert.match(gaussian.sampling,/accumulated pixels/);
assert.match(sampleSpotHdr(constant,{u:.5,v:.5}).sampling,/One accumulated pixel/);
assert.throws(()=>sampleSpotHdr(constant,{u:-.1,v:.5}),/horizontal/);
assert.throws(()=>sampleSpotHdr(constant,{u:0,v:0,spread:NaN}),/spread/);
assert.throws(()=>sampleSpotHdr({...constant,dtype:'|u1'},{u:0,v:0}),/floating/);
assert.throws(()=>sampleSpotHdr({shape:[MAX_SPOT_PIXELS+1,1,3],dtype:'<f4',data:[]},{u:0,v:0}),/bounded/);
assert.throws(()=>sampleSpotHdr(array(1,1,()=>[NaN,1,1]),{u:0,v:0}),/nonfinite/);
const negative=sampleSpotHdr(array(1,1,()=>[-1,2,3]),{u:0,v:0});assert.equal(negative.negativePixels,1);assert.equal(negative.rgb[0],-1);
const aborted=new AbortController();aborted.abort();assert.throws(()=>sampleSpotHdr(constant,{u:0,v:0,signal:aborted.signal}),/canceled/);

const neutral=proposeSpotAdjustment({rgb:[1,2,3]},{kind:'white'});
const corrected=[neutral.params.red,neutral.params.green,neutral.params.blue].map((x,i)=>x*[1,2,3][i]);
assert.ok(Math.max(...corrected)-Math.min(...corrected)<1e-12);
assert.equal(neutral.params.whitebalance,0);assert.equal(neutral.params.tint,0);
assert.equal(neutral.params.exposure,undefined);
assert.throws(()=>proposeSpotAdjustment(negative,{kind:'white'}),/positive/);
assert.throws(()=>proposeSpotAdjustment({rgb:[.00001,1,1]},{kind:'white'}),/range/);
const proposal=proposeSpotAdjustment({rgb:[1,1,1]},{kind:'exposure',whiteReference:10,params:{red:2,green:2,blue:2}});
assert.ok(Math.abs(proposal.params.exposure-Math.log2(.9))<1e-12);
assert.throws(()=>proposeSpotAdjustment({rgb:[1,1,1]},{kind:'exposure'}),/display white/);
assert.throws(()=>proposeSpotAdjustment({rgb:[0,0,0]},{kind:'exposure',whiteReference:1}),/positive/);
assert.throws(()=>proposeSpotAdjustment({rgb:[1e-12,1e-12,1e-12]},{kind:'exposure',whiteReference:1}),/travel/);

// Actual NumPy compressed archive with Unicode profile metadata; the selective
// reader must load the profiled channels and leave raw sensor/metadata alone.
const npz=execFileSync('python',['-c',`import io,sys,numpy as np
b=io.BytesIO();r=np.array([[[1.,2.,3.],[4.,5.,6.]],[[7.,8.,9.],[10.,11.,12.]]])
np.savez_compressed(b,linear_rgb_exposure=r*100,linear_srgb_exposure=r,color_profile_json=np.asarray('{"id":"fixture"}'))
sys.stdout.buffer.write(b.getvalue())`],{timeout:20000});
const buffer=npz.buffer.slice(npz.byteOffset,npz.byteOffset+npz.byteLength);
const parsed=await readSpotHdr(buffer);assert.deepEqual(parsed.shape,[2,2,3]);assert.deepEqual([...parsed.data],[1,2,3,4,5,6,7,8,9,10,11,12]);
let source={id:'job:exposure-000001.npz',url:'/fixture.npz',revision:1,complete:true,profiled:true},fetches=0;
let fetchHook=null;
const sampler=createCompletedHdrSampler({getSource:()=>source,fetchImpl:async(url,options)=>{
  fetches++;if(fetchHook)return fetchHook(url,options);
  return {ok:true,headers:new Headers(),arrayBuffer:async()=>buffer.slice(0)};
}});
const result=await sampler.sample({u:1,v:1});assert.deepEqual(result.rgb,[10,11,12]);assert.equal(result.sourceId,source.id);
await sampler.sample({u:0,v:0,spread:.25});assert.equal(fetches,1,'one immutable cached archive');
source={...source,complete:false};await assert.rejects(()=>sampler.sample({u:0,v:0}),/completed final/);
source={...source,complete:true,revision:2};
fetchHook=async()=>({ok:true,headers:new Headers(),arrayBuffer:async()=>{source={...source,revision:3};return buffer;}});
await assert.rejects(()=>sampler.sample({u:0,v:0}),/changed/);
fetchHook=async()=>({ok:false,status:404});await assert.rejects(()=>sampler.sample({u:0,v:0}),/404/);
fetchHook=async()=>({ok:true,headers:new Headers({'Content-Length':MAX_SPOT_HDR_BYTES+1})});
await assert.rejects(()=>sampler.sample({u:0,v:0}),/byte limit/);
fetchHook=null;sampler.clear();await sampler.sample({u:0,v:0});
source={...source,width:3,height:2};await assert.rejects(()=>sampler.sample({u:0,v:0}),/dimensions differ/);
source={...source,width:2,height:2};assert.deepEqual((await sampler.sample({u:0,v:0})).rgb,[1,2,3]);
await assert.rejects(()=>sampler.sample({u:0,v:0,signal:aborted.signal}),/canceled/);
console.log('Spot HDR: actual selective NPZ, Gaussian weights/edges, pixel orientation, proposals, identity/cancel and failure tests PASS');
