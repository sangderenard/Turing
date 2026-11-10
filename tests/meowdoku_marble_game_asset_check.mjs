// Pure exact appearance serialization/admission. No WebGL or physics runner.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import {createStudioGameAsset,hydrateStudioGameAsset,prepareStudioAmmunition} from '../examples/meowdoku/web/marble_game_asset.js';
import {studioFlatSamplerPacket} from '../examples/meowdoku/web/marble_studio_subject_renderer.js';
import {geometrySha256} from '../examples/meowdoku/web/marble_object.js';
import {adaptMechanicalCreatureProjectiles} from '../examples/meowdoku/web/mechanical_creature_instances.js';

const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const uniforms=[],scalar=(name,value,type=5124)=>uniforms.push({name,value,type,size:1});
for(const name of ['uFieldGain','uShadowEnabled','uShadowLightCount','uThermalEmission','uThermalGain',
  'uRenderPass','uEnableEmissionDirect','uNumLights'])scalar(name,0);
for(const sampler of studioFlatSamplerPacket().textures)
  scalar(sampler.name,sampler.unit,sampler.target==='TEXTURE_2D'?35678:sampler.target==='TEXTURE_3D'?35679:36289);
const materials={};
for(const [unit,[name,stride]]of [['pbr',16],['phong',8],['enamel',8],['texstack',16]].entries()){
  materials[name]={layout:{floats_per_record:stride},data:Array.from({length:stride*2},(_,i)=>(i+unit)/37)};
  scalar(`turing_ssbo_${name}`,unit,35678);
}
materials.pbr.data[5]=0;materials.pbr.data[7]=1;
materials.pbr.data[21]=1;materials.pbr.data[23]=.55;
const normals=[[0,0,1],[0,0,1],[0,0,1]];
const interior=[[-0,1e-30,.1],[.2,.3,0],[.4,0,.123456789],0,normals,-2147483648];
const undecorated=[[.01,.02,.03],[.04,.05,.06],[.07,.08,.09],0,normals];
const shell=[[1,0,0],[0,1,0],[0,0,1],1,[[1,0,0],[0,1,0],[0,0,1]],2147483647];
const packet={schema:'turing.studio-class-world-import.v1',revision:17,
  render:{uniforms,materials,referenceCamera:{view:I},shaders:{
    vertex:'#version 300 es\nvoid main(){}',
    fragment:'#version 300 es\n#define PLUCK_TRANSMISSION 1\n#define PLUCK_PREMULTIPLIED_GLASS 1\nvoid main(){}'}},
  scene:{channel:'studio',frame:{unitM:.3048,rayOriginM:[0,0,.18]},camera:{view:I,fovDegrees:50,aspect:1},
    ray:{pbr:[...materials.pbr.data]},subject:{id:'subject',model:I,faces:[interior,undecorated,shell],
      phong:{interiorCount:2,glassCount:1,bubbles:[]}},
    authoring:{object:{v:1,generator:'meowdoku-marble-object-v1',design:{'base-shape':'marble',coverage:0,'band-count':0},
      physical:{'radius-mm':10,'shell-density':2500},materials:{index:{shell:1,interior:0}}}}}};
const before=structuredClone(packet),asset=await createStudioGameAsset(packet,{name:'  Test marble  '});
const optional=await prepareStudioAmmunition(packet,{name:'Test marble'});
assert.equal(optional.supported,true);assert.equal(optional.reason,null);assert.equal(optional.asset.identity,asset.identity);
for(const change of [p=>p.scene.authoring.object.design['base-shape']='d6',p=>p.scene.authoring.object.design['base-shape']='bunny',
  p=>p.scene.authoring.object.design['base-shape']='obj',p=>p.scene.authoring.object.design.coverage=1,
  p=>p.scene.authoring.object.design['band-count']=1,p=>p.scene.subject.faces[2][0][0]=.9,
  p=>{p.scene.subject.phong.interiorCount=3;p.scene.subject.phong.glassCount=0;}]) {
  const candidate=structuredClone(packet);change(candidate);const result=await prepareStudioAmmunition(candidate);
  assert.equal(result.supported,false);assert.equal(result.asset,null);assert.equal(result.code,'UNSUPPORTED_STUDIO_AMMUNITION');assert(result.reason);
}
for(const change of [p=>p.render.shaders.vertex='wrong',p=>p.scene.subject.faces[0][3]=999,
  p=>p.scene.subject.faces[0][0][0]=NaN,p=>p.scene.subject.faces[0][0][0]=1e40,
  p=>p.render.uniforms.find(u=>u.name==='uFieldGain').value=1]) {
  const candidate=structuredClone(packet);candidate.scene.authoring.object.design['base-shape']='bunny';change(candidate);
  await assert.rejects(()=>prepareStudioAmmunition(candidate),e=>e.code!=='UNSUPPORTED_STUDIO_AMMUNITION');
}
assert.deepEqual(packet,before);
assert.equal(asset.name,'Test marble');
assert.match(asset.identity,/^studio-marble:[a-f0-9]{64}$/);
assert.equal(asset.identity,(await createStudioGameAsset(packet,{name:'Test marble'})).identity);
assert.equal(asset.mesh.sha256,await geometrySha256({interior:[interior,undecorated],glass:[shell]}));
assert.equal(Buffer.from(asset.mesh.bytesBase64,'base64').length,3*76);
assert.deepEqual(asset.mesh.groups,[-2147483648,0,2147483647]);
assert.deepEqual(asset.render.materials,packet.render.materials);
assert.equal(asset.contract.physics,'existing-game-sphere-unchanged');
assert.equal(asset.contract.radius,'existing-projectile-archetype');
assert.equal(asset.contract.rayTracedBake,false);
const assetBefore=structuredClone(asset),hydrated=await hydrateStudioGameAsset(asset);
assert.deepEqual(asset,assetBefore);
assert.deepEqual(hydrated.packet.render.materials,packet.render.materials);
assert.equal(hydrated.source.materials,hydrated.packet.render.materials);
assert.equal(hydrated.source.geometry.faces,hydrated.packet.scene.subject.faces);
for(let i=0;i<packet.scene.subject.faces.length;i++){
  const original=packet.scene.subject.faces[i],restored=hydrated.packet.scene.subject.faces[i];
  assert.equal(restored[3],original[3]);
  assert.equal(restored[5],original[5]??0);
  for(const [a,b]of [[original.slice(0,3),restored.slice(0,3)],[original[4],restored[4]]])
    for(let j=0;j<3;j++)for(let k=0;k<3;k++)assert(Object.is(b[j][k],Math.fround(a[j][k])),`exact float32 ${i}/${j}/${k}`);
}
packet.render.materials.phong.data[0]=999;
assert.notEqual(hydrated.packet.render.materials.phong.data[0],999);
packet.render.materials.phong.data[0]=before.render.materials.phong.data[0];
hydrated.packet.render.materials.phong.data[0]=888;
assert.notEqual(asset.render.materials.phong.data[0],888);

// Asset metadata's 10mm authored size never overwrites the original .11m body.
const state={identity:'existing-body',position:[2,3,4],velocity:[5,6,7],box:{identity:'existing-body',radius:.11},
  record:{identity:'existing-body',status:'active',geometry:{kind:'sphere',radius:.11},physics:{mass:.12,restitution:.78}}};
const runtimeBody={identity:state.identity,slot:0,generation:4};
const runtime={ready:true,bodies:new Map([[state.identity,runtimeBody]]),slotRecords:[runtimeBody],latestSequence:10};
const stateBefore=structuredClone(state),runtimeBefore=structuredClone(runtime);
const instances=adaptMechanicalCreatureProjectiles({projectileState:new Map([[state.identity,state]]),stateLoopRuntime:runtime,
  source:hydrated.source,fixedQuaternionXYZW:[0,0,0,1]});
assert.equal(instances.instances[0].radiusM,.11);
assert.deepEqual(state,stateBefore);assert.deepEqual(runtime,runtimeBefore);

// Identity is a content hash, not an authorization signature. Re-signing below
// exercises semantic admission independently of the corruption guard.
const sign=asset=>{const {identity,...body}=asset;return {...body,identity:'studio-marble:'+createHash('sha256').update(JSON.stringify(body)).digest('hex')};};
const failures=[];let negativeContracts=0;
async function rejects(label,run){negativeContracts++;try{await run();failures.push(label);}catch{}}
for(const [label,mutate]of [
  ['dice creation',p=>p.scene.authoring.object.design['base-shape']='d4'],
  ['coated shell creation',p=>p.scene.authoring.object.design.coverage=1],
  ['banded shell creation',p=>p.scene.authoring.object.design['band-count']=1],
  ['non-unit exterior creation',p=>p.scene.subject.faces[2][0][0]=.9],
  ['missing shell creation',p=>{p.scene.subject.phong.interiorCount=3;p.scene.subject.phong.glassCount=0;}],
  ['float32 overflow creation',p=>p.scene.subject.faces[0][0][0]=1e40],
  ['invalid face material creation',p=>p.scene.subject.faces[0][3]=2],
  ['invalid group creation',p=>p.scene.subject.faces[0][5]=2147483648],
]){const p=structuredClone(before);mutate(p);await rejects(label,()=>createStudioGameAsset(p));}
await rejects('empty name',()=>createStudioGameAsset(before,{name:' '}));
await rejects('long name',()=>createStudioGameAsset(before,{name:'x'.repeat(121)}));
for(const [label,mutate]of [
  ['identity tamper',a=>a.name='changed'],
  ['material tamper',a=>a.render.materials.pbr.data[0]=9],
  ['group tamper',a=>a.mesh.groups[0]=0],
  ['byte tamper',a=>{const bytes=Buffer.from(a.mesh.bytesBase64,'base64');bytes[0]^=1;a.mesh.bytesBase64=bytes.toString('base64');}],
]){const a=structuredClone(asset);mutate(a);await rejects(label,()=>hydrateStudioGameAsset(a));}
for(const [label,mutate]of [
  ['rehash dice hydrate',a=>a.source.object.design['base-shape']='d4'],
  ['rehash coated shell hydrate',a=>a.source.object.design.coverage=1],
  ['rehash banded shell hydrate',a=>a.source.object.design['band-count']=1],
  ['rehash missing shell hydrate',a=>{a.mesh.phong.interiorCount=3;a.mesh.phong.glassCount=0;}],
  ['rehash group count hydrate',a=>a.mesh.groups.pop()],
  ['rehash invalid group hydrate',a=>a.mesh.groups[0]=.5],
  ['rehash truncated bytes hydrate',a=>a.mesh.bytesBase64=a.mesh.bytesBase64.slice(0,-4)],
  ['rehash invalid format hydrate',a=>a.mesh.format='different'],
  ['rehash wrong mesh digest hydrate',a=>a.mesh.sha256='0'.repeat(64)],
  ['rehash physics contract hydrate',a=>a.contract.physics='different'],
  ['rehash excess triangle budget hydrate',a=>a.mesh.triangles=200001],
]){const a=structuredClone(asset);mutate(a);await rejects(label,()=>hydrateStudioGameAsset(sign(a)));}
for(const [label,value,offset]of [['rehash non-unit exterior hydrate',.9,2*76],
  ['rehash nonfinite geometry hydrate',NaN,0]]){
  const a=structuredClone(asset),bytes=Buffer.from(a.mesh.bytesBase64,'base64');bytes.writeFloatLE(value,offset);
  a.mesh.bytesBase64=bytes.toString('base64');a.mesh.sha256=createHash('sha256').update(bytes).digest('hex');
  await rejects(label,()=>hydrateStudioGameAsset(sign(a)));
}
let actual=null;
if(process.argv[2]){
  const raw=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),read=await hydrateStudioGameAsset(raw);
  assert.equal(raw.mesh.sha256,await geometrySha256({interior:read.packet.scene.subject.faces,glass:[]}));
  assert.deepEqual(read.packet.render.materials,raw.render.materials);
  actual={identity:read.identity,triangles:read.packet.scene.subject.faces.length,meshSha256:read.meshSha256};
  if(process.argv[3]){
    const capture=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
    assert.deepEqual(read.packet.render.materials,capture.render.materials);
    assert.deepEqual(read.packet.render.shaders,capture.render.shaders);
    assert.deepEqual(read.packet.render.uniforms,capture.render.uniforms);
    assert.equal(read.packet.scene.subject.faces.length,capture.scene.subject.faces.length);
    let comparisons=0;
    for(let i=0;i<read.packet.scene.subject.faces.length;i++){
      const a=capture.scene.subject.faces[i],b=read.packet.scene.subject.faces[i];
      assert.equal(a[3],b[3]);assert.equal(a[5]??0,b[5]);
      for(const [av,bv]of [[a.slice(0,3),b.slice(0,3)],[a[4],b[4]]])
        for(let j=0;j<3;j++)for(let k=0;k<3;k++){
          assert(Object.is(Math.fround(av[j][k]),bv[j][k]),`capture GPU mismatch ${i}/${j}/${k}`);comparisons++;
        }
    }
    actual.capturedGpuFloatComparisons=comparisons;
    actual.capturedGpuAttributeMismatches=0;
  }
}
console.log(JSON.stringify({passed:failures.length===0,negativeContracts,acceptedInvalidCases:failures,
  exactFloat32AndMaterialIds:true,originalPhysicsUnchanged:true,actual,browserRenderingTested:false}));
assert.deepEqual(failures,[],'Invalid asset cases must be rejected');
