// Exact scene construction in Node, without a page, controls, fetch or WebGL.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {createHash} from 'node:crypto';
import {buildStudioScene,buildStudioPhotoFixtures,studioSceneMaterials,computeStudioFixtureRoles,createStudioSceneGeometry} from '../examples/meowdoku/web/studio_scene_builder.js';
import {prepareObjectSnapshot} from '../examples/meowdoku/web/marble_object_snapshot.js';
import {OBJECT_CENTER,photoCameraFromStudio} from '../examples/meowdoku/web/marble_photo.js';
import {studioScaleTransform} from '../examples/meowdoku/web/marble_studio_scale.js';
import {defaultLightingRacks} from '../examples/meowdoku/web/marble_lighting_racks.js';
import {STUDIO_LIGHTING_FIELDS,studioLightingFields} from '../examples/meowdoku/web/studio_lighting_fields.js';
import {LIGHTING_PROFILE_FIELDS} from '../examples/meowdoku/web/marble_studio_lighting.js';
import {tagPartResponseFaces} from '../examples/meowdoku/web/marble_response_lineage.js';

assert.equal(typeof document,'undefined');assert.equal(typeof window,'undefined');
const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/studio_scene_builder_authoring.json',import.meta.url),'utf8'));
const {assets,authoring}=fixture,prepared=await prepareObjectSnapshot(authoring.object);
const materialSource=JSON.stringify(assets),objectSource=JSON.stringify(prepared.value);
const originalFetch=globalThis.fetch;
globalThis.fetch=()=>{throw Error('Scene construction must not fetch');};
const hash=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const page=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const html=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.html',import.meta.url),'utf8');
const boot=page.slice(page.indexOf('    for (const [select,initial] of '),page.indexOf('    studioLighting=setupStudioLighting'));
const materialDefaults=Object.fromEntries([...boot.matchAll(/\["([^"]+)","([^"]+)"\]/g)].map(m=>[m[1],m[2]]));
const attributes=text=>Object.fromEntries([...text.matchAll(/([\w-]+)="([^"]*)"/g)].map(m=>[m[1],m[2]]));
const metadata={};
for(const id of LIGHTING_PROFILE_FIELDS) {
  const input=new RegExp(`<input\\b([^>]*\\bid="${id}"[^>]*)>`).exec(html);
  if(input) {
    const a=attributes(input[1]),type=a.type??'text';
    metadata[id]={type,...(a.min===undefined?{}:{min:a.min}),...(a.max===undefined?{}:{max:a.max}),
      defaultValue:type==='checkbox'?/\bchecked\b/.test(input[1]):type==='range'?Number(a.value):a.value??''};
  } else {
    const select=new RegExp(`<select\\b[^>]*id="${id}"[^>]*>([\\s\\S]*?)<\\/select>`).exec(html);assert(select,`Missing HTML control ${id}`);
    const items=[...select[1].matchAll(/<option\b([^>]*)>(.*?)<\/option>/g)],options=items.map(m=>attributes(m[1]).value??m[2]);
    const selected=items.findIndex(m=>/\bselected\b/.test(m[1]));
    metadata[id]={type:'select-one',...(Object.hasOwn(materialDefaults,id)?{materials:true}:{}),...(options.length?{options}:{}),
      defaultValue:materialDefaults[id]??options[Math.max(0,selected)]??''};
  }
}
assert.deepEqual(STUDIO_LIGHTING_FIELDS,metadata,'Detached validation schema/defaults must exactly match HTML and page material initialization');
const descriptors=studioLightingFields(Object.keys(assets.material_index));
assert.deepEqual(descriptors['backdrop-material'].options.map(v=>v.value),['calibration',...Object.keys(assets.material_index)]);
assert(descriptors['floor-material-a'].options.some(v=>v.value==='white_plaster'));

// Optional extraction-time oracle: execute the actual pre-refactor pure
// functions against the identical captured values, not a recreated scene.
// No document or control stand-ins are provided to this context.
async function legacyScene(options) {
  const source=fs.readFileSync(process.env.STUDIO_SCENE_BASELINE_SOURCE,'utf8'),deps={};
  for(const name of ['marble_lighting_racks','marble_light_placement','marble_light_tools','marble_fixture_sizes','marble_lighting',
    'marble_light_strengths','marble_table_lights','marble_room','marble_studio_table','marble_studio_gobos','marble_light_stand_settings',
    'marble_backdrop_supports','marble_backdrop','marble_shop_lights','marble_materials','marble_scene_materials','marble_studio_scene'])
    Object.assign(deps,await import(`../examples/meowdoku/web/${name}.js`));
  const {lighting,materials,subject,geometrySubject=subject,solo,shopEmissionProfiles,rayOriginM}=options;
  const context=vm.createContext({...deps,structuredClone,Float32Array,ids:materials.ids,FIXTURE_MAT_BASE:materials.fixtureMaterialBase,
    FIXTURE_HOUSING_MAT:materials.fixtureMaterialBase+deps.STUDIO_LIGHTS.length,
    tables:{pbr:{table:{floats_per_record:materials.pbr.stride},data:materials.pbr.data}},
    studioControlValue:id=>lighting.controls[id],studioRenderState:()=>lighting,
    studioLighting:{strengths:{getSolo:()=>solo}},studioTableState:()=>lighting.table,
    studioSubjectTransform:saved=>saved?subject:geometrySubject,studioPhotoScale:()=>subject.unitM,
    studioRoomScale:()=>Number(lighting.controls['room-scale']||3),studioFloorDrop:()=>deps.tableFloorDrop(lighting.table),
    shopEmissionProfiles,OBJECT_CENTER:rayOriginM,opticalControls:{sceneRecord:options.getOpticalRecord},
    rgb:value=>[1,3,5].map(k=>parseInt(value.slice(k,k+2),16)/255)});
  vm.runInContext(source.slice(source.indexOf('const ROOM_FIXTURES ='),source.indexOf('// Keep the ray-traced photo')),context);
  vm.runInContext(source.slice(source.indexOf('function quadFace('),source.indexOf('function digitFaces(')),context);
  for(const name of ['surfaceMaterial','tableFaces','goboFaces','lightStandFacesForStudio','backdropSupportFacesForStudio',
    'shopLightState','shopLightGeometryForStudio','shopLightFacesForStudio','checkerboardFaces','backdropRollFaces','computeFixtureRoles','buildFixtureFaces','sceneForPhoto']) {
    const start=source.indexOf(`function ${name}(`),lineEnd=source.indexOf('\n',start);
    const end=source.slice(start,lineEnd).endsWith('}')?lineEnd:source.indexOf('\n}',start)+2;
    vm.runInContext(source.slice(start,end),context);
  }
  return structuredClone(context.sceneForPhoto(true));
}

const cases=[];
const captured=structuredClone(authoring.lighting);cases.push(['captured',captured]);
const furnished=structuredClone(captured);
furnished.controls['backdrop-layers']=2;furnished.controls['backdrop-sweep']=true;furnished.controls['window-glass']=true;
furnished.gobos={version:1,gobos:[{id:1,enabled:true,shape:'ellipse',width:2,height:3,x:2,y:1,z:0,
  pitch:23,yaw:-67,roll:121,front:'black_felt',back:'white_plaster'}]};
furnished.shopLights.fixtures.forEach(f=>{f.exposureMode='continuous';f.dimmer=.7;});
furnished.lightingRacks=defaultLightingRacks();furnished.lightingRacks.buses[0].enabled=true;
cases.push(['furnished',furnished]);
const dark=structuredClone(furnished);dark.shopLights.fixtures.forEach(f=>f.enabled=false);
cases.push(['solo',dark,'key-left']);
const legacy=structuredClone(captured);legacy.scale={version:1,mode:'legacy'};legacy.table.enabled=false;
legacy.lightStands.enabled=false;legacy.shopLights.fixtures=[];cases.push(['legacy-scale',legacy]);

// These hashes are measured against the pre-extraction page source. They pin
// every vertex, material row, group range and emitter record, not just counts.
const expected={
  captured:'be5f1adbb4998704bfe24f81f4683c0066d1ed48682adc9e03442077e0696216',
  furnished:'2aa1f71be96f486cacbb7fe6e4243391947750697bf438352fcdce695e356954',
  solo:'0700dba4c91cc71da929f80c7b4b05e599c9e2e0720070b120f0927496e6f6c8',
  'legacy-scale':'754074c64e0480ebd2296d3086c34e219486f59603fb721de2b2ec3e39396cb3',
};
for(const [name,lighting,solo]of cases) {
  const before=JSON.stringify(lighting),materials=studioSceneMaterials(assets,prepared.value,lighting);
  const subject=studioScaleTransform(lighting.scale,prepared.value.physical['radius-mm'],lighting.table.enabled);
  const camera=photoCameraFromStudio(-.5,.2,3.45,subject.unitM,subject.fovDegrees,subject.subjectCenter).camera;
  const options={lighting,materials,subject,rayOriginM:OBJECT_CENTER,shopEmissionProfiles:assets.emitterProfiles,solo};
  const pbrBefore=Array.from(materials.pbr.data),blob=buildStudioPhotoFixtures(options);
  const {response_lineage,...copiedBlob}=buildStudioPhotoFixtures({...options,lightFieldInformation:'evaluated-responses-v1'});
  assert.deepEqual(copiedBlob,blob,'opt-in adds identity rows only, without changing native input geometry/materials');
  let identityCursor=0;const partIds=new Set();
  for(const part of response_lineage.parts) {
    assert.equal(part.start,identityCursor);assert(part.count>0);assert(!partIds.has(part.source_id));
    partIds.add(part.source_id);identityCursor+=part.count;
  }
  assert.equal(identityCursor,blob.material_ids.length,'original part identities cover exact fixture upload order');
  if(process.env.STUDIO_SCENE_BASELINE_SOURCE)assert.deepEqual(blob,await legacyScene(options),`${name}: byte-for-byte original page scene`);
  const shopCount=blob.emitter_profiles.records.length-10;
  const previous={...blob,emitter_profiles:{...blob.emitter_profiles,
    records:[...blob.emitter_profiles.records.slice(0,shopCount),...blob.emitter_profiles.records.slice(shopCount+10)]}};
  if(expected[name])assert.equal(hash(previous),expected[name],`${name}: captured original scene geometry and legacy emitter records`);
  const baseEmitters=blob.emitter_profiles.records.slice(shopCount,shopCount+10);
  assert.equal(baseEmitters.length,10,'each base light uses a native material emission profile');
  assert(baseEmitters.every(record=>record.profile==='led_cool_white'));
  assert.equal(blob.vertices.length,blob.material_ids.length*9);
  assert(blob.vertices.every(Number.isFinite));assert(blob.pbr.every(Number.isFinite));
  assert(blob.material_ids.every(id=>Number.isInteger(id)&&id>=0&&id<blob.pbr.length/16));
  assert.deepEqual(Array.from(materials.pbr.data),pbrBefore,'Capture does not mutate shared material rows');
  const scene=buildStudioScene({...options,camera,objectSnapshot:prepared.value});
  assert.deepEqual(scene.fixtures,blob);assert.deepEqual(scene.camera,camera);assert.notEqual(scene.camera,camera);
  assert.equal(scene.objectSnapshot,prepared.value);assert.deepEqual(scene.object_center_m,OBJECT_CENTER.map((v,i)=>v+subject.subjectCenter[i]*subject.unitM));
  assert.equal(JSON.stringify(lighting),before);
  // Execute the actual page wrapper against the same captured values. The
  // wrapper delegates to the imported production builder, with no fake page.
  const wrapperStart=page.indexOf('export function sceneForPhoto('),wrapperEnd=page.indexOf('\n}',wrapperStart)+2;
  const bridge=vm.createContext({buildStudioPhotoFixtures,studioBuilderOptions:()=>options,OBJECT_CENTER,shopEmissionProfiles:assets.emitterProfiles,
    studioLighting:{strengths:{getSolo:()=>solo}},opticalControls:null,lightField:null,
    studioSceneRevision:0,photoSceneCacheKey:null,photoSceneCacheValue:null});
  vm.runInContext(page.slice(wrapperStart,wrapperEnd).replace('export function','function'),bridge);
  assert.deepEqual(bridge.sceneForPhoto(true),blob,`${name}: actual page wrapper delegates unchanged`);
  const geometry=createStudioSceneGeometry(options),roles=computeStudioFixtureRoles({...options,ids:materials.ids,fixtureMaterialBase:materials.fixtureMaterialBase,subjectPosition:subject.subjectCenter}).fixtureRoles;
  for(const [label,build]of [['room',geometry.roomFaces],['table',geometry.tableFaces],['gobo',geometry.goboFaces],['backdrop',geometry.backdropRollFaces],
    ['holders',geometry.backdropHolderFaces],['stands',cb=>geometry.lightStandFaces(roles,cb)],['shop/racks',geometry.shopLightFaces]]) {
    const parts=[],faces=build(p=>{parts.push(p);tagPartResponseFaces(p,`${label}/${p.id}`,true);});assert.equal(parts.reduce((n,p)=>n+p.faces.length,0),faces.length,`${label} part coverage`);
    assert(faces.every(f=>f.responseSource),'original builder callbacks retain identities on the actual returned faces');
    if(name==='furnished')assert(faces.length>0,`${label} actually exercised`);
  }
  console.log(`PASS ${name}: ${blob.material_ids.length} triangles, ${blob.pbr.length/16} material rows, SHA256 ${hash(blob)}`);
}
assert.equal(JSON.stringify(assets),materialSource);assert.equal(JSON.stringify(prepared.value),objectSource);
assert.throws(()=>buildStudioScene({}),/complete resolved/);
assert.throws(()=>buildStudioScene({lighting:captured,objectSnapshot:authoring.object}),/prepared object/);
const materials=studioSceneMaterials(assets,prepared.value,captured),subject=studioScaleTransform(captured.scale,10,captured.table.enabled);
const valid={lighting:captured,objectSnapshot:prepared.value,materials,rayOriginM:OBJECT_CENTER,shopEmissionProfiles:assets.emitterProfiles,
  camera:photoCameraFromStudio(-.5,.2,3.45,subject.unitM,subject.fovDegrees,subject.subjectCenter).camera};
const burst=structuredClone(furnished);burst.shopLights.fixtures[0].exposureMode='burst';
const burstOptions={lighting:burst,materials:studioSceneMaterials(assets,prepared.value,burst),
  subject:studioScaleTransform(burst.scale,prepared.value.physical['radius-mm'],burst.table.enabled),
  rayOriginM:OBJECT_CENTER,shopEmissionProfiles:assets.emitterProfiles};
assert.throws(()=>buildStudioPhotoFixtures(burstOptions),/timing-packet/);
const timedBurst=buildStudioPhotoFixtures({...burstOptions,timedExposure:true});
assert(timedBurst.trigger_emitters.length>0);
assert(timedBurst.trigger_emitters.every(item=>timedBurst.shop_material_ids.includes(item.material)));
const unfiltered=structuredClone(captured);unfiltered.controls['key-filter-left-off']=true;
unfiltered.shopLights.fixtures[0].filterOff=true;
const unfilteredMaterials=studioSceneMaterials(assets,prepared.value,unfiltered);
const unfilteredBlob=buildStudioPhotoFixtures({lighting:unfiltered,materials:unfilteredMaterials,
  subject:studioScaleTransform(unfiltered.scale,prepared.value.physical['radius-mm'],unfiltered.table.enabled),
  rayOriginM:OBJECT_CENTER,shopEmissionProfiles:assets.emitterProfiles});
assert(unfilteredBlob.emitter_profiles.records.some(record=>record.filter===null&&record.profile==='led_cool_white'));
assert(unfilteredBlob.emitter_profiles.records.some(record=>record.filter===null&&record.profile==='fluorescent_cool_white'));
for(const [mutate,pattern]of [
  [p=>p.controls['light-strength-left']=-10,/Invalid light-strength-left/],
  [p=>p.controls['light-strength-left']=4001,/Invalid light-strength-left/],
  [p=>p.controls['floor-material-a']='nonexistent-material',/Unavailable option for floor-material-a/],
  [p=>p.controls['key-filter-left']='red',/Invalid key-filter-left/],
  [p=>p.controls['key-shape']='invented',/Unavailable option for key-shape/],
  [p=>p.scale.unitMm=0,/Studio unit/],
  [p=>p.poses.poses[0].pos[0]=NaN,/pose/i],
]) {const lighting=structuredClone(captured);mutate(lighting);assert.throws(()=>buildStudioScene({...valid,lighting}),pattern);}
assert.throws(()=>buildStudioScene({...valid,camera:{...valid.camera,eye_m:[NaN,0,1]}}),/camera/);
assert.throws(()=>buildStudioScene({...valid,rayOriginM:[0,NaN,0]}),/ray origin/);
const invalidMaterials={...materials,pbr:{...materials.pbr,data:Array.from(materials.pbr.data)}};invalidMaterials.pbr.data[0]=NaN;
assert.throws(()=>buildStudioScene({...valid,materials:invalidMaterials}),/finite Studio PBR/);
const optical=structuredClone(captured),id=materials.ids[optical.table.material],stride=materials.pbr.stride;
optical.sceneEngines.stage='dielectric';
optical.materialEdits.materials.push({name:optical.table.material,tables:{pbr:{stride,data:Array.from(materials.pbr.data.slice(id*stride,(id+1)*stride))}},
  optical:{material:id,wavelengths_nm:[450,550,650],transmittance:[.8,.9,.95],reference_thickness_mm:1,concentration:1}});
const opticalMaterials=studioSceneMaterials(assets,prepared.value,optical);
const opticalScene=buildStudioScene({...valid,lighting:optical,materials:opticalMaterials});
assert(opticalScene.fixtures.material_engines.volumes.length>0);
assert.equal(opticalScene.fixtures.material_engines.optical_media.materials.length,1);
const incomplete=structuredClone(optical);delete incomplete.materialEdits.materials[0].optical;
assert.throws(()=>buildStudioScene({...valid,lighting:incomplete,materials:opticalMaterials}),/transmittance/);
assert.match(page,/const value=buildStudioPhotoFixtures\(/);assert.match(page,/return buildStudioFixtureFaces\(/);
assert.match(page,/return computeStudioFixtureRoles\(/);assert.match(page,/initializeStudioFixtureRows\(tables\.pbr\.data/);
globalThis.fetch=originalFetch;
console.log('PASS DOM-free shared Studio scene construction, input immutability and page delegation');
