import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createStudioSceneAsset,hydrateStudioSceneAsset,studioSceneDrawPlan,studioSceneMatrix,
  studioSceneEntranceParts} from '../examples/meowdoku/web/marble_studio_scene_asset.js';
import {studioFlatSamplerPacket} from '../examples/meowdoku/web/marble_studio_subject_renderer.js';
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],samplers=studioFlatSamplerPacket(),uniforms=[];
const scalar=(name,value,type=5124)=>uniforms.push({name,value,type,size:1});
for(const n of ['uFieldGain','uShadowEnabled','uShadowLightCount','uThermalEmission','uThermalGain','uRenderPass','uEnableEmissionDirect','uNumLights'])scalar(n,0);
for(const s of samplers.textures)scalar(s.name,s.unit,s.target==='TEXTURE_2D'?35678:s.target==='TEXTURE_3D'?35679:36289);
const materials={};for(const [i,[name,stride]]of [['pbr',16],['phong',8],['enamel',8],['texstack',16]].entries()) {
  materials[name]={layout:{floats_per_record:stride},data:Array(2*stride).fill(0)};scalar('turing_ssbo_'+name,i,35678);
}
materials.pbr.data[7]=1;materials.pbr.data[21]=1;materials.pbr.data[23]=.5;
const face=(a,b,c,id=0)=>[a,b,c,id,[[0,0,-1],[0,0,-1],[0,0,-1]],19];
const parts=[{id:'room/front',bodyId:'room',faces:[face([-2,0,1],[2,4,1],[2,0,1]),face([-2,0,1],[-2,4,1],[2,4,1])]},
  {id:'table/slab',bodyId:'table',faces:[face([0,0,0],[1,0,0],[0,1,0],1)]},
  {id:'fixture/test',bodyId:'fixture',faces:[face([0,2,0],[1,2,0],[0,3,0])]}];
let offset=0;for(const p of parts){p.firstTriangle=offset;p.triangleCount=p.faces.length;offset+=p.faces.length;}
const packet={schema:'turing.studio-class-world-import.v1',revision:42,scene:{schema:'turing.studio-visible-scene.v1',channel:'studio',
  frame:{unitM:.5,rayOriginM:[0,0,0]},camera:{view:Object.fromEntries(I.map((v,i)=>[i,v])),fovDegrees:50,aspect:1},parts,
  ray:{pbr:[...materials.pbr.data]},room:{leftX:-2,rightX:2,backZ:-2,frontZ:1,floorY:0,ceilingY:4},
  subject:{id:'subject',faces:[face([Math.PI,0,0],[1,0,0],[0,1,0]),face([0,0,0],[1,0,0],[0,1,0],1)],
    model:[.5,0,0,0,0,.5,0,0,0,0,.5,0,0,1,0,1],phong:{interiorCount:1,glassCount:1,bubbles:[]}}},
  render:{uniforms,materials,referenceCamera:{view:Object.fromEntries(I.map((v,i)=>[i,v]))},
    shaders:{vertex:'#version 300 es\nvoid main(){}',fragment:'#version 300 es\n#define PLUCK_TRANSMISSION 1\n#define PLUCK_PREMULTIPLIED_GLASS 1\nvoid main(){}'}}};
packet.scene.triangleSources=parts.flatMap(p=>p.faces.map((_,triangle)=>({partId:p.id,bodyId:p.bodyId,triangle})));
const before=JSON.stringify(packet),asset=await createStudioSceneAsset(packet),hydrated=await hydrateStudioSceneAsset(asset);
const mutable=structuredClone(asset),inFlight=hydrateStudioSceneAsset(mutable);mutable.render.materials.pbr.data[0]=999;
assert.equal((await inFlight).packet.render.materials.pbr.data[0],asset.render.materials.pbr.data[0]);
assert.equal(JSON.stringify(packet),before);assert.deepEqual(hydrated.packet.scene.parts,parts);
assert.deepEqual(hydrated.packet.scene.subject.faces,packet.scene.subject.faces);
assert.deepEqual(hydrated.packet.scene.triangleSources,packet.scene.triangleSources);
assert.deepEqual(hydrated.packet.render.referenceCamera.view,I);
assert(!('ray' in asset.scene));assert(!('ray' in asset.scene.subject));assert(!('faces' in asset.scene.subject));
assert.deepEqual(studioSceneMatrix(new Float32Array(I)),I);
const placement=[...I];placement[12]=32;placement[13]=1.544;placement[14]=-18;
const plan=studioSceneDrawPlan(hydrated.packet,{placement});
assert.deepEqual(plan.groups.map(g=>g.faces.length),[2,1,1,2]);assert.equal(plan.groups[1].emissionDirect,true);
assert.equal(plan.groups[3].emissionDirect,false);assert.deepEqual(plan.groups[3].faces[0],parts[1].faces[0]);
assert.deepEqual(plan.groups[3].faces[1][0],[0,1,0]);assert.equal(plan.subjectModel[0],.25);
assert.equal(plan.counts.staticTriangles,4);assert.equal(plan.counts.subjectTriangles,2);
const entrance={partId:'room/front',widthM:1,heightM:1},opened=studioSceneEntranceParts(hydrated.packet.scene,entrance);
const signedArea=f=>Math.abs((f[1][0]-f[0][0])*(f[2][1]-f[0][1])-(f[1][1]-f[0][1])*(f[2][0]-f[0][0]))/2;
assert.equal(opened.parts[0].faces.reduce((n,f)=>n+signedArea(f),0),12);
assert.equal(opened.entrance.sourceTriangles,2);assert.equal(opened.entrance.changedSourceTriangles,2);
assert.equal(opened.entrance.sources.length,opened.parts[0].faces.length);assert.equal(JSON.stringify(packet),before);
for(const f of opened.parts[0].faces)assert(f.slice(0,3).every(p=>p[0]<=-1)||f.slice(0,3).every(p=>p[0]>=1)||f.slice(0,3).every(p=>p[1]>=2));
let negative=0;
for(const mutate of [p=>delete p.scene.camera.view[15],p=>p.scene.camera.view.extra=0,p=>p.scene.parts[1].id='room/front',
  p=>p.scene.parts[0].firstTriangle=1,p=>p.scene.parts[0].bodyId='',p=>p.scene.parts[1].faces[0][3]=99,
  p=>p.scene.parts[1].faces[0][4][0][1]=NaN,p=>p.scene.triangleSources[0].triangle=1,p=>p.scene.frame.unitM=0,
  p=>p.scene.subject.model[15]=0,p=>p.render.materials.pbr.data[0]=1,p=>p.render.uniforms.find(u=>u.name==='uFieldGain').value=1]) {
  const bad=structuredClone(packet);mutate(bad);await assert.rejects(()=>createStudioSceneAsset(bad));negative++;
}
for(const mutate of [a=>a.name='changed',a=>a.geometry.bytesBase64=a.geometry.bytesBase64.slice(0,-4),a=>a.scene.parts[0].id='other']) {
  const bad=structuredClone(asset);mutate(bad);await assert.rejects(()=>hydrateStudioSceneAsset(bad));negative++;
}
assert.throws(()=>studioSceneDrawPlan(hydrated.packet,{placement:[2,0,0,0,0,2,0,0,0,0,2,0,0,0,0,1]}));negative++;
assert.throws(()=>studioSceneDrawPlan(hydrated.packet,{entrance:{...entrance,widthM:20}}));negative++;
assert.throws(()=>studioSceneDrawPlan(hydrated.packet,{entrance:{...entrance,partId:'table/slab'}}));negative++;
let actual=null;
if(process.argv[2]) {
  const input=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),a=await createStudioSceneAsset(input,{name:'Complete Marble Studio'}),h=await hydrateStudioSceneAsset(a);
  // Compare bounded tuples. A whole-capture assertion failure can format tens
  // of millions of diff characters and overwhelm the shared development host.
  const equalFaces=(a,b)=>{assert.equal(a.length,b.length);for(let i=0;i<a.length;i++)assert.deepEqual(a[i],b[i],`Source face ${i}`);};
  equalFaces(h.packet.scene.subject.faces,input.scene.subject.faces);
  for(let i=0;i<input.scene.parts.length;i++){
    const a=h.packet.scene.parts[i],b=input.scene.parts[i];equalFaces(a.faces,b.faces);
    assert.deepEqual({...a,faces:null},{...b,faces:null});
  }
  assert.deepEqual(h.packet.scene.triangleSources,input.scene.triangleSources);
  const p=studioSceneDrawPlan(h.packet,{placement:[1,0,0,0,0,1,0,0,0,0,1,0,32,1.544,-18.00788329297247,1],
    entrance:{partId:'room/front',widthM:2,heightM:2.4}});
  const wall=p.entrance,source=input.scene.parts.find(x=>x.id==='room/front'),derived=studioSceneEntranceParts(h.packet.scene,{partId:'room/front',widthM:2,heightM:2.4});
  const sourceArea=source.faces.reduce((n,f)=>n+signedArea(f),0),derivedArea=derived.parts.find(x=>x.id==='room/front').faces.reduce((n,f)=>n+signedArea(f),0);
  assert(Math.abs(sourceArea-derivedArea-2*2.4/(input.scene.frame.unitM**2))<1e-10);
  if(process.argv[3])fs.writeFileSync(process.argv[3],JSON.stringify(a));
  actual={identity:a.identity,bytes:JSON.stringify(a).length,counts:p.counts,bounds:p.bounds,subjectPose:Array.from(p.subjectModel),
    entrance:{sourceTriangles:wall.sourceTriangles,derivedTriangles:wall.derivedTriangles,changedSourceTriangles:wall.changedSourceTriangles},
    exactSourceRoundTrip:true,sharedTransparent:p.groups[3].faces.length};
}
console.log(JSON.stringify({pass:true,negativeContracts:negative,losslessSourceDoubles:true,noDuplicatedRayGeometry:true,actual,browserRendering:false}));
