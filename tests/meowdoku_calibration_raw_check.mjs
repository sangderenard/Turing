import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {createHash} from "node:crypto";
import {parseNpz} from "../examples/meowdoku/web/npz.js";
import {captureLinearCalibration,summarizeLinearArray,MAX_CALIBRATION_HDR_BYTES} from "../examples/meowdoku/web/marble_calibration.js";

// Exercise NumPy's real compressed format, including current Unicode metadata.
const npz=execFileSync("python",["-c",`import io,sys,numpy as np
b=io.BytesIO()
r=np.array([[[2.,4.,8.],[0.,1.,3.]],[[5.,6.,7.],[9.,10.,11.]]],dtype=np.float32)
np.savez_compressed(b,linear_rgb_exposure=r,linear_srgb_exposure=r.astype(np.float64)-1,
 color_profile_json=np.asarray('{"id":"test-profile"}'))
sys.stdout.buffer.write(b.getvalue())`]);
const buffer=npz.buffer.slice(npz.byteOffset,npz.byteOffset+npz.byteLength);
await assert.rejects(()=>parseNpz(buffer),/unsupported .npy dtype/);
const arrays=await parseNpz(buffer,["linear_rgb_exposure","linear_srgb_exposure"]);
assert.deepEqual([...arrays.keys()],["linear_rgb_exposure","linear_srgb_exposure"]);
assert.deepEqual([...arrays.get("linear_rgb_exposure").data],[2,4,8,0,1,3,5,6,7,9,10,11]);
assert.equal((await parseNpz(buffer,[])).size,0);
await assert.rejects(()=>parseNpz(buffer,"raw"),/selection/);
const sum=summarizeLinearArray(arrays.get("linear_rgb_exposure"));
assert.deepEqual(sum.sum,[16,21,29]);assert.deepEqual(sum.mean,[4,5.25,7.25]);
assert.deepEqual(sum.minimum,[0,1,3]);assert.deepEqual(sum.maximum,[9,10,11]);
assert.equal(sum.pixelCount,4);assert.equal(sum.normalization,"none");
assert.deepEqual(summarizeLinearArray(arrays.get("linear_srgb_exposure")).negativeSamples,[1,0,0]);
assert.throws(()=>summarizeLinearArray({shape:[1,1,3],dtype:"<f4",data:[NaN,1,1]}),/nonfinite/);
assert.throws(()=>summarizeLinearArray({shape:[1,1,3],dtype:"|u1",data:[1,1,1]}),/floating/);

const originalFetch=globalThis.fetch,requests=[];
const status={digest:"a".repeat(64),hdr:"exposure-000123.npz",transport:"fixed:3",film_response:{version:1},
 color_profile:{id:"test-profile"},display_transform:{normalization:"auto99"},display_white:123,
 elapsed_seconds:2,forward_ray_jobs:100,sensor_sites:4,epoch:123};
try{
 globalThis.fetch=async(url,options)=>{requests.push({url,options});return {ok:true,headers:new Headers({"Content-Length":String(npz.byteLength)}),arrayBuffer:async()=>buffer.slice(0)};};
 const budget={remaining:MAX_CALIBRATION_HDR_BYTES};
 const result=await captureLinearCalibration(status,budget);
 assert.equal(result.status,"captured");assert.deepEqual(result.sensor.sum,[16,21,29]);
 assert.deepEqual(Buffer.from(result.archive.data,"base64"),npz,"original raw archive is retained exactly");
 assert.equal(result.archive.sha256,createHash("sha256").update(npz).digest("hex"));
 assert.equal(result.provenance.displayWhite,123);assert.deepEqual(result.provenance.colorProfile,{id:"test-profile"});
 assert.equal(budget.remaining,MAX_CALIBRATION_HDR_BYTES-npz.byteLength);
 assert.match(requests[0].url,/\/api\/marble_photo\/hdr\?digest=a{64}&name=exposure-000123.npz/);
 assert.equal(requests[0].options.cache,"no-store");
 // Display white and tone choice cannot change retained measurements.
 const second=await captureLinearCalibration({...status,display_white:999,display_transform:{normalization:"manual"}});
 assert.deepEqual(second.sensor,result.sensor);assert.equal(second.archive.data,result.archive.data);
 assert.equal(second.provenance.displayWhite,999);
 const small={remaining:1};await assert.rejects(()=>captureLinearCalibration(status,small),/limit/);assert.equal(small.remaining,1);
 await assert.rejects(()=>captureLinearCalibration({...status,hdr:null}),/no valid raw/);
 globalThis.fetch=async()=>({ok:false,status:404});await assert.rejects(()=>captureLinearCalibration(status),/HTTP 404/);
 // Even an omitted/unreliable Content-Length cannot bypass the byte budget.
 globalThis.fetch=async()=>({ok:true,headers:new Headers(),arrayBuffer:async()=>buffer.slice(0)});
 await assert.rejects(()=>captureLinearCalibration(status,{remaining:1}),/limit/);
}finally{globalThis.fetch=originalFetch;}
console.log("Raw calibration: exact NPZ retention, numeric selection, no PNG normalization, metadata, limits and failures PASS");
