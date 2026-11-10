// Contract tests only: synthetic observation records exercise exact remapping.
// This is not a visibility render, real ID-pass result or coverage certificate.
import assert from 'node:assert/strict';
import {packStudioScene} from '../examples/meowdoku/web/marble_studio_scene.js';
import {capturePhongTriangleSource,bindPhongVisibilityViews,publishSampledPhongTriangles,
  phongPublicationDigest} from '../examples/meowdoku/web/marble_phong_visibility.js';

const identity=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const row=[.5,.4,.3,.1,0,1,1.5,0,0,0,0,0,0,0,0,0];
const face=(z,material,group)=>[[-1,-1,z],[1,-1,z],[0,1,z],material,
  [[0,0,1],[0,0,1],[0,0,1]],group,{originalMetadata:[z,material]}];
const faces=[face(-1,0,3),face(-2,1,4),face(-3,0,5),face(-4,1,6)];
const capture=()=>({revision:17,state:packStudioScene({channel:'studio',unitM:.008,
  rayOriginM:[0,0,.18],parts:[],pbr:{stride:16,data:[...row,...row]},
  subject:{id:'marble-exact',faces,model:identity,radius:1},
  camera:{view:identity,fovDegrees:50,aspect:1},authoring:{},room:null})});
const views=[{index:2,direction:[1,0,0],camera:{eye_m:[1,0,0],target_m:[0,0,0],fov:50},condition_key:'light:day'},
  {index:7,direction:[0,1,0],camera:{eye_m:[0,1,0],target_m:[0,0,0],fov:50},condition_key:'light:day'},
  {index:10,direction:[0,0,1],camera:{eye_m:[0,0,1],target_m:[0,0,0],fov:50},condition_key:'light:night'}];
const raw=capture(),before=structuredClone(raw),source=await capturePhongTriangleSource(raw);
assert.deepEqual(raw,before,'capture is read-only');
assert.deepEqual(source.faces,faces,'all positions, normals, group IDs and metadata remain exact');
assert.deepEqual(source.materialIds,[0,1,0,1]);
assert.deepEqual(source.triangleSources[3],{partId:'marble-exact',bodyId:'marble-exact',triangle:3});
assert(Object.isFrozen(source.faces[0][0]));
assert.throws(()=>source.faces[0][0][0]=500,TypeError);
const bound=await bindPhongVisibilityViews(views);
assert.deepEqual(await bindPhongVisibilityViews(bound),bound,'already-bound descriptors retain their digest');
const observation=(viewIndex,triangleIds)=>({schema:'turing.phong-primary-visibility-sample.v1',
  revision:source.revision,meshSha256:source.meshSha256,materialSha256:source.materialSha256,
  viewIndex,cameraSha256:bound.find(v=>v.index===viewIndex).cameraSha256,triangleIds,
  sampleCount:128,producer:{kind:'primary-raster-triangle-id',id:'contract-fixture-not-a-render',readbackComplete:true}});
const samples=[observation(2,[3,1,3]),observation(7,[2,1])],saved=structuredClone(samples);
const result=await publishSampledPhongTriangles(source,views,samples);
assert.deepEqual(samples,saved,'input records remain unchanged');
assert.deepEqual(result.sourceIndices,[1,2,3],'union is deduplicated in original source order');
assert.deepEqual(result.sourceToDraw,[-1,0,1,2]);
assert.deepEqual(result.faces,[faces[1],faces[2],faces[3]],'no geometry/material simplification');
assert.deepEqual(result.materialIds,[1,0,1]);
assert.deepEqual(result.triangleSources.map(x=>x.triangle),[1,2,3]);
assert.deepEqual(result.sampledViewIndices,[2,7]);
assert.deepEqual(result.unsampledViewIndices,[10]);
assert.deepEqual(result.fullSource,source,'unpruned source is retained');
assert.equal(result.consumerPolicy.presentationOnly,true);
for(const key of ['universalVisibility','physicsPruning','shadowPruning','refractionPruning',
  'transparentLayerCompleteness','geodesicExpressionMinimization'])assert.equal(result.consumerPolicy[key],false);
assert.match(result.producerStatus,/does not implement or simulate/);
assert(Object.isFrozen(result.sourceToDraw));
samples[0].triangleIds.push(0);assert.deepEqual(result.sourceIndices,[1,2,3]);
raw.state.subject.faces[0][0][0]=900;assert.deepEqual(source.faces,faces,'capture is detached');
const empty=await publishSampledPhongTriangles(source,views,[observation(2,[])]);
assert.equal(empty.faces.length,0);assert.equal(empty.fullSource.faces.length,4);
await assert.rejects(()=>publishSampledPhongTriangles(source,views,[]),/Missing actual/);
for(const [key,value]of [['revision',18],['meshSha256','bad'],['materialSha256','bad'],
  ['cameraSha256','bad'],['viewIndex',999],['triangleIds',[4]],['triangleIds',[-1]],
  ['triangleIds',[1.5]],['sampleCount',0]]) {
  const sample=observation(2,[1]);sample[key]=value;
  await assert.rejects(()=>publishSampledPhongTriangles(source,views,[sample]),/observations/);
}
for(const producer of [{kind:'estimated-frustum',id:'not-real',readbackComplete:true},
  {kind:'primary-raster-triangle-id',id:'incomplete',readbackComplete:false}]) {
  const sample=observation(2,[1]);sample.producer=producer;
  await assert.rejects(()=>publishSampledPhongTriangles(source,views,[sample]),/observations/);
}
const changedViews=structuredClone(views);changedViews[0].camera.eye_m[0]=2;
await assert.rejects(()=>publishSampledPhongTriangles(source,changedViews,[observation(2,[1])]),/observations/);
const changedCondition=structuredClone(views);changedCondition[0].condition_key='changed-light';
await assert.rejects(()=>publishSampledPhongTriangles(source,changedCondition,[observation(2,[1])]),/observations/);
for(const edit of [s=>{s.faces[0][0][0]=200;},s=>{s.pbr[0]=.9;},s=>{s.model[12]=5;},
  s=>{s.triangleSources[0].triangle=2;},s=>{s.materialIds[0]=1;}]) {
  const changed=structuredClone(source);edit(changed);
  await assert.rejects(()=>publishSampledPhongTriangles(changed,views,[observation(2,[1])]),/changed/);
}
const bad=capture();bad.state.subject.ray.material_ids[0]=1;
await assert.rejects(()=>capturePhongTriangleSource(bad),/disagree/);
await assert.rejects(()=>bindPhongVisibilityViews([views[0],views[0]]),/unique/);
assert.equal(await phongPublicationDigest({b:2,a:[1]}),await phongPublicationDigest({a:[1],b:2}));
await assert.rejects(()=>phongPublicationDigest({a:NaN}),/Nonfinite/);
console.log(JSON.stringify({passed:true,scope:'exact subset/remap publication contracts only',
  actualVisibilityPass:false,sourceTriangles:source.faces.length,publishedTriangles:result.faces.length,
  preservedFullSource:true,physicsOrRefractionPruning:false}));
