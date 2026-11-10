// Run with --experimental-vm-modules. The collection module body is unchanged;
// only GPU rendering/icon imports are explicit test doubles. Asset hydration,
// sphere admission, body adapter, camera and light-transform validation are real.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import * as assetsModule from '../examples/meowdoku/web/marble_game_asset.js';
import * as bodiesModule from '../examples/meowdoku/web/mechanical_creature_instances.js';
import * as cameraModule from '../examples/meowdoku/web/mechanical_creature_studio_camera.js';
import {studioFlatSamplerPacket,validateStudioLightTransform} from '../examples/meowdoku/web/marble_studio_subject_renderer.js';

if(!vm.SourceTextModule)throw Error('Run node with --experimental-vm-modules');
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],drawCalls=[],created=[],disposed=[];
const hooks={rendererFailure:false,iconFailure:false,failSubject:null};
const createStudioSubjectRenderer=({gl,packet,samplers})=>{
  if(hooks.rendererFailure||packet.scene.subject.id===hooks.failSubject)throw Error('test GPU allocation failure');
  const record={gl,packet,samplers,closed:false};created.push(record);
  return {draw:args=>{assert(!record.closed);drawCalls.push({record,args});return true;},
    dispose:()=>{if(!record.closed){record.closed=true;disposed.push(record);}}};
};
const renderStudioSubjectThumbnail=()=>{if(hooks.iconFailure)throw Error('test thumbnail failure');return {dataUrl:'data:image/png;base64,AAAA'};};
const source=fs.readFileSync(new URL('../examples/meowdoku/web/mechanical_creature_studio_collection.js',import.meta.url),'utf8');
const context=vm.createContext({});
const module=new vm.SourceTextModule(source,{context});
const dependencies={
  './marble_game_asset.js':assetsModule,
  './mechanical_creature_instances.js':bodiesModule,
  './mechanical_creature_studio_camera.js':cameraModule,
  './marble_studio_subject_renderer.js':{createStudioSubjectRenderer,validateStudioLightTransform},
  './marble_studio_subject_thumbnail.js':{renderStudioSubjectThumbnail},
};
await module.link(specifier=>{
  const exports=dependencies[specifier];assert(exports,`unexpected dependency ${specifier}`);
  return new vm.SyntheticModule(Object.keys(exports),function(){for(const [key,value]of Object.entries(exports))this.setExport(key,value);},{context});
});
await module.evaluate();
const {installStudioCollection,studioRigToGame}=module.namespace;
const pose=(scale,angle,position)=>{const c=Math.cos(angle)*scale,s=Math.sin(angle)*scale;return [c,0,-s,0,0,scale,0,0,s,0,c,0,...position,1];};
const point=(m,p)=>[0,1,2].map(r=>m[r]*p[0]+m[4+r]*p[1]+m[8+r]*p[2]+m[12+r]);
const near=(a,b,e=2e-12)=>assert(Math.abs(a-b)<=e,`${a} != ${b}`);
const sourcePose=pose(.037,.73,[1,-2,3]),targetPose=pose(.11,-.91,[7,5,-9]);
const sourceBefore=sourcePose.slice(),targetBefore=targetPose.slice(),rig=studioRigToGame(sourcePose,targetPose);
for(const p of [[0,0,0],[1,2,3],[-4,.5,9]])
  point(rig,point(sourcePose,p)).forEach((value,axis)=>near(value,point(targetPose,p)[axis]));
assert.deepEqual(sourcePose,sourceBefore);assert.deepEqual(targetPose,targetBefore);
near(validateStudioLightTransform(rig).areaScale,(.11/.037)**2);
for(const bad of [pose(0,0,[0,0,0]),pose(-1,0,[0,0,0]),[...I.slice(0,15),0]])
  assert.throws(()=>studioRigToGame(bad,I));
const anisotropic=I.slice();anisotropic[0]=2;
assert.throws(()=>studioRigToGame(anisotropic,I));
const shear=I.slice();shear[1]=.1;
assert.throws(()=>studioRigToGame(I,shear));

function packet(subject='sphere'){
  const uniforms=[],uniform=(name,value,type=5124)=>uniforms.push({name,value,type,size:1});
  for(const name of ['uFieldGain','uShadowEnabled','uShadowLightCount','uThermalEmission','uThermalGain',
    'uRenderPass','uEnableEmissionDirect','uNumLights'])uniform(name,0);
  for(const sampler of studioFlatSamplerPacket().textures)
    uniform(sampler.name,sampler.unit,sampler.target==='TEXTURE_2D'?35678:sampler.target==='TEXTURE_3D'?35679:36289);
  const materials={};
  for(const [unit,[name,stride]]of [['pbr',16],['phong',8],['enamel',8],['texstack',16]].entries()){
    materials[name]={layout:{floats_per_record:stride},data:Array(stride).fill(0)};uniform(`turing_ssbo_${name}`,unit,35678);
  }
  materials.pbr.data[7]=1;
  return {schema:'turing.studio-class-world-import.v1',revision:1,
    render:{uniforms,materials,referenceCamera:{view:I},shaders:{vertex:'#version 300 es\nvoid main(){}',
      fragment:'#version 300 es\n#define PLUCK_TRANSMISSION 1\n#define PLUCK_PREMULTIPLIED_GLASS 1\nvoid main(){}'}},
    scene:{channel:'studio',frame:{unitM:.3048},camera:{view:I},ray:{pbr:[...materials.pbr.data]},
      subject:{id:subject,model:sourcePose,faces:[[[1,0,0],[0,1,0],[0,0,1],0,[[1,0,0],[0,1,0],[0,0,1]],7]],
        phong:{interiorCount:0,glassCount:1,bubbles:[]}},
      authoring:{object:{design:{'base-shape':'marble',coverage:0,'band-count':0}}}}};
}
const asset=await assetsModule.createStudioGameAsset(packet(),{name:'Chosen sphere'});
const secondAsset=await assetsModule.createStudioGameAsset(packet('other'),{name:'Other sphere'});
function host(){
  const original={identity:'ordinary-ammo',quantity:9},items=[original],admissions=[];
  return {original,items,admissions,mechanicalCreatureAmmo:{
    addAsset(row){admissions.push(row);const item={identity:`ammo:${row.assetIdentity}`,quantity:row.quantity,
      properties:{appearance_asset_identity:row.assetIdentity}};items.push(item);return item;},
    items:()=>items,selected:()=>original,select:()=>{throw Error('installation must not select ammunition');},
  }};
}
function body(identity,appearanceAssetIdentity,slot,position,pickup=false){
  return {identity,appearanceAssetIdentity,ammunitionItemIdentity:appearanceAssetIdentity?`ammo:${appearanceAssetIdentity}`:undefined,
    position,velocity:[1,2,3],sleeping:false,box:{identity,radius:.11},
    record:{identity,status:pickup?'pickup':'active',geometry:{kind:'sphere',radius:.11},physics:{mass:.12}},slot};
}
const ordinary=body('ordinary',undefined,0,[0,2,-2]);
const chosenNear=body('chosen-near',asset.identity,1,[0,2,-3]);
const chosenFar=body('chosen-far',asset.identity,2,[0,2,-8]);
const other=body('other-source',secondAsset.identity,3,[0,2,-5]);
const pickup=body('chosen-pickup',asset.identity,null,[0,2,-4],true);
const active=[ordinary,chosenNear,chosenFar,other],snapshotBodies=active.map(b=>({identity:b.identity,slot:b.slot,generation:1}));
const input={gl:{},projectileState:new Map(active.map(b=>[b.identity,b])),projectilePickupState:new Map([[pickup.identity,pickup]]),
  stateLoopRuntime:{ready:true,bodies:new Map(snapshotBodies.map(b=>[b.identity,b])),slotRecords:snapshotBodies,latestSequence:12},
  cameraPosition:[0,2,0],cameraFacing:[0,0,-1],width:640,height:360};
const inputBefore=structuredClone(input),ownedHost=host(),statuses=[];
const collection=await installStudioCollection({assets:[{asset,quantity:2}],host:ownedHost,onStatus:s=>statuses.push(s)});
assert.equal(ownedHost.admissions.length,0,'admission waits for the actual render callback');
const adapter=ownedHost.mechanicalCreatureImportedAppearance;
assert.equal(statuses.at(-1).state,'waiting-for-original-render');
const suppress=adapter.identities(input);
assert.deepEqual([...suppress],['chosen-near','chosen-far','chosen-pickup']);
assert(!suppress.has('ordinary')&&!suppress.has('other-source'));
assert.equal(ownedHost.admissions.length,1);
assert.equal(ownedHost.admissions[0].quantity,2);
assert.equal(ownedHost.items[0],ownedHost.original);
assert.equal(ownedHost.original.quantity,9);
assert.equal(ownedHost.mechanicalCreatureAmmo.selected(),ownedHost.original);
assert.equal(collection.readStatus().bodyCount,3);
assert.equal(collection.readStatus().snapshotSequence,12);
assert.equal(statuses.at(-1).state,'ready');
const createdCount=created.length;adapter.identities(input);
assert.equal(created.length,createdCount,'repeated frames reuse uploaded resources');
assert.equal(ownedHost.admissions.length,1,'repeated frames never refill inventory');
// The original inventory owns shot decrements. An ordinary shot changes no
// imported body count, but its picker quantity must still trigger a report.
const statusBeforeDefaultShot=statuses.at(-1),statusCount=statuses.length;
ownedHost.original.quantity=8;
adapter.identities(input);
assert.equal(statuses.length,statusCount+1,'default ammo quantity change reaches the picker');
assert.equal(statuses.at(-1).inventory.find(item=>item.identity==='ordinary-ammo').quantity,8);
assert.equal(statusBeforeDefaultShot.inventory.find(item=>item.identity==='ordinary-ammo').quantity,9,
  'reports snapshot quantities without retaining mutable inventory rows');
ownedHost.items[1].quantity=1;
adapter.identities(input);
assert.equal(statuses.at(-1).inventory.find(item=>item.identity===ownedHost.items[1].identity).quantity,1);
assert.equal(ownedHost.admissions.length,1,'reporting decrements never replenishes a chosen stack');
adapter.draw({...input,identities:suppress,phase:'opaque'});
adapter.draw({...input,identities:suppress,phase:'transparent'});
assert.deepEqual(drawCalls.slice(-3).map(d=>d.args.model[14]),[-8,-4,-3],'transparent subjects are back-to-front');
for(const draw of drawCalls){
  const m=draw.args.model;
  point(draw.args.lightTransform,point(sourcePose,[0,0,0])).forEach((value,axis)=>near(value,m[12+axis]));
  assert.equal(m[0],.11);assert.equal(m[5],.11);assert.equal(m[10],.11);
}
assert.deepEqual(input,inputBefore,'rendering reads but does not alter original lifecycle inputs');
await assert.rejects(()=>installStudioCollection({assets:[asset],host:ownedHost}),/already has/);
const disposedBefore=disposed.length;collection.dispose();collection.dispose();
assert.equal(disposed.length,disposedBefore+1);
assert.equal(ownedHost.mechanicalCreatureImportedAppearance,null);
assert.equal(adapter.identities(input).size,0);assert.equal(adapter.draw({...input,identities:suppress,phase:'opaque'}),false);

for(const [kind,set]of [['renderer',()=>hooks.rendererFailure=true],['thumbnail',()=>hooks.iconFailure=true]]){
  const failedHost=host();set();const failed=await installStudioCollection({assets:[asset],host:failedHost});
  assert.equal(failedHost.mechanicalCreatureImportedAppearance.identities(input).size,0,`${kind} failure must preserve original rendering`);
  assert.equal(failedHost.admissions.length,0);assert.equal(failed.readStatus().ready,0);
  failed.dispose();hooks.rendererFailure=false;hooks.iconFailure=false;
}
const partialHost=host();hooks.failSubject='other';
const partial=await installStudioCollection({assets:[asset,secondAsset],host:partialHost});
assert.deepEqual([...partialHost.mechanicalCreatureImportedAppearance.identities(input)],['chosen-near','chosen-far','chosen-pickup']);
assert.equal(partial.readStatus().ready,1);partial.dispose();hooks.failSubject=null;
// A nonspherical Studio scene still needs the original empty appearance holder
// for its room. It must not create a projectile renderer, icon or inventory.
const emptyHost=host(),emptyCreated=created.length,emptyDraws=drawCalls.length,emptyStatuses=[];
const empty=await installStudioCollection({assets:[],host:emptyHost,onStatus:value=>emptyStatuses.push(value)});
const emptyAdapter=emptyHost.mechanicalCreatureImportedAppearance;
for(let i=0;i<3;i++){
  const ids=emptyAdapter.identities(input);assert.equal(ids.size,0);
  assert.equal(emptyAdapter.draw({...input,identities:ids,phase:'opaque'}),true);
  assert.equal(emptyAdapter.draw({...input,identities:ids,phase:'transparent'}),true);
}
assert.equal(emptyStatuses.at(-1).state,'ready');assert.equal(emptyStatuses.at(-1).assets.length,0);
assert.equal(created.length,emptyCreated);assert.equal(drawCalls.length,emptyDraws);
assert.equal(emptyHost.admissions.length,0);assert.equal(emptyHost.original.quantity,9);
assert.equal(emptyHost.mechanicalCreatureAmmo.selected(),emptyHost.original);
empty.dispose();empty.dispose();assert.equal(emptyHost.mechanicalCreatureImportedAppearance,null);
for(const rows of [null,undefined,{},Array(9).fill(asset),[asset,asset],[{asset,quantity:0}],[{asset,quantity:129}]]){
  const h=host();await assert.rejects(()=>installStudioCollection({assets:rows,host:h}));
  assert.equal(h.mechanicalCreatureImportedAppearance,undefined);assert.equal(h.admissions.length,0);
}
const bad=structuredClone(asset);bad.name='tampered';const rejectedHost=host();
await assert.rejects(()=>installStudioCollection({assets:[bad],host:rejectedHost}));
assert.equal(rejectedHost.mechanicalCreatureImportedAppearance,undefined);

// Async ownership/interruption tests. Preserve every result for one compact
// report so a single admission race does not hide a second cleanup issue.
const failures=[];
const racedHost=host();
const racing=await Promise.allSettled([installStudioCollection({assets:[asset],host:racedHost}),
  installStudioCollection({assets:[secondAsset],host:racedHost})]);
if(racing.filter(r=>r.status==='fulfilled').length!==1)failures.push('concurrent installs both acquired one host');
for(const result of racing)if(result.status==='fulfilled')result.value.dispose();
const interruptedHost=host();
let interrupted=null,observerRejected=false;
try{interrupted=await installStudioCollection({assets:[asset],host:interruptedHost,onStatus:()=>{throw Error('observer interrupted');}});}catch{observerRejected=true;}
if(observerRejected&&interruptedHost.mechanicalCreatureImportedAppearance)
  failures.push('throwing status observer leaves installed hook after rejected install');
if(interrupted){
  assert.equal(interruptedHost.mechanicalCreatureImportedAppearance.identities(input).size,3,
    'observer failures do not interrupt ordinary rendering');
  interrupted.dispose();
}

let realRig=null;
if(process.argv[2]){
  const realAsset=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),hydrated=await assetsModule.hydrateStudioGameAsset(realAsset);
  const mapped=studioRigToGame(hydrated.packet.scene.subject.model,targetPose);
  for(const p of [[0,0,0],[1,2,3]])point(mapped,point(hydrated.packet.scene.subject.model,p)).forEach((value,axis)=>near(value,point(targetPose,p)[axis]));
  realRig={asset:hydrated.identity,areaScale:validateStudioLightTransform(mapped).areaScale};
}
console.log(JSON.stringify({passed:failures.length===0,rigInverse:true,selectedTagsOnly:true,lifecycleInputsUnchanged:true,
  realHydrationAndAdmission:true,gpuBoundaryMocked:true,issues:failures,realRig}));
assert.deepEqual(failures,[]);
