import assert from 'node:assert/strict';
import {installStudioSceneBuilders,installScenePacking} from './meowdoku_scene_packer_harness.mjs';
import {buildStudioPhotoFixtures} from '../examples/meowdoku/web/studio_scene_builder.js';
import {defaultStudioTable} from '../examples/meowdoku/web/marble_studio_table.js';
import fs from 'node:fs';
import vm from 'node:vm';
import {shopLightEmissionRecords,defaultShopLights,shopLightEmitterAllocations,writeShopLightEmitterRows as writeRows,studioShopLightGeometry} from '../examples/meowdoku/web/marble_shop_lights.js';
import {STUDIO_FIXED_MATERIAL_ROWS,STUDIO_RUNTIME_MATERIAL_ROWS} from '../examples/meowdoku/web/marble_materials.js';
import {writeStudioFixtureRows,packStudioScene} from '../examples/meowdoku/web/marble_studio_scene.js';
import {sceneMaterialEngineContract} from '../examples/meowdoku/web/marble_scene_materials.js';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
import {isOpaquePreviewMaterial} from '../examples/meowdoku/web/marble_glass_preview.js';
const shopEmissionProfiles={records:['fluorescent_cool_white','fluorescent_warm_white','led_cool_white','led_warm_white'].map(key=>({key,revision:'a'.repeat(64),preview_rgb:[1,1,1],filter_rgb_matrix:[[1,0,0],[0,1,0],[0,0,1]]}))};
const writeShopLightEmitterRows=(data,stride,base,state,options={})=>writeRows(data,stride,base,state,{profiles:shopEmissionProfiles,...options});
const source=fs.readFileSync(process.env.STUDIO_MAIN_SOURCE??new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const ids={metal_silver:4,black_matte_paint:38,marble_clear_glass:14},base=39;
const data=new Float32Array((base+STUDIO_RUNTIME_MATERIAL_ROWS)*16);
for(const id of Object.values(ids)){data[id*16+3]=.8;data[id*16+7]=1;}
data[14*16+5]=1;data[14*16+7]=0;
const exposure={controls:{'room-scale':6},table:defaultStudioTable(),shopLights:defaultShopLights(),sceneEngines:{version:1,stage:'legacy',room:'legacy'}};
let room=structuredClone(exposure);
const context=vm.createContext({shopLightEmissionRecords,shopEmissionProfiles,structuredClone,STUDIO_FIXED_MATERIAL_ROWS,STUDIO_LIGHTS,shopLightEmitterAllocations,writeShopLightEmitterRows,
  studioShopLightGeometry,writeStudioFixtureRows,sceneMaterialEngineContract,packStudioScene,
  studioRenderState:channel=>channel==='exposure'?exposure:room,studioRoomScale:()=>6,studioFloorDrop:()=>3.98,studioRoomBounds,
  studioSubjectTransform:()=>({unitM:.3048,subjectCenter:[0,-.9,0]}),studioPhotoScale:()=>.3048,OBJECT_CENTER:[0,0,.18],
  studioLighting:null,ids,FIXTURE_MAT_BASE:base,FIXTURE_HOUSING_MAT:base+10,opticalControls:null,
  tables:{pbr:{data,table:{floats_per_record:16}}},read:(_,id,k)=>data[id*16+k],isOpaquePreviewMaterial,
  mesh:faces=>({faces,count:faces.length*3}),transparentMesh:faces=>({faces,count:faces.length*3}),dispose:()=>{},
  shopOpaque:null,shopGlass:null,shopTransparentFaces:[],shopMeshKey:null,shopGeometry:null,sceneGlassKey:null,
  computeFixtureRoles:()=>({fixtureRoles:[],roleColor:[],roleStrength:[]}),buildFixtureFaces:()=>[],
  checkerboardFaces:()=>[],backdropRollFaces:()=>[],tableFaces:()=>[],goboFaces:()=>[],lightStandFacesForStudio:()=>[],backdropSupportFacesForStudio:()=>[]});
installStudioSceneBuilders(context);
vm.runInContext(source.slice(source.indexOf('function shopLightState('),source.indexOf('function rebuildTable(')),context);
vm.runInContext(source.slice(source.indexOf('export function sceneForPhoto('),source.includes('// Read-only capture')?source.indexOf('// Read-only capture'):source.indexOf('\nfunction syncMaterial()')).replace('export function','function'),context);
installScenePacking(context);
const canonical=structuredClone(context.shopLightGeometryForStudio('exposure'));
assert.equal(canonical.faces.length,360);
writeShopLightEmitterRows(data,16,base+12,room.shopLights);
const fixedBefore=data.slice(0,(base+12)*16),liveBefore=data.slice();
const rendered=structuredClone(context.rebuildShopMeshes());
assert.deepEqual(rendered,canonical);assert.equal(context.shopOpaque.faces.length,360);
assert.equal(rendered.emitters.length,2);
const photo=structuredClone(context.sceneForPhoto());
assert.deepEqual(data,liveBefore,'Exposure capture never zeroes the live Studio buffer');
assert.deepEqual(photo.vertices,canonical.faces.flatMap(f=>f.slice(0,3).flatMap(p=>p.map((v,i)=>v*.3048+[0,0,.18][i]))));
assert.equal(photo.material_ids.length,360);
assert.equal(photo.emitter_profiles.records.length,2);
assert(photo.emitter_profiles.records.every(r=>r.profile==='fluorescent_cool_white'&&r.revision==='a'.repeat(64)&&r.gain===0));
assert.deepEqual(photo.material_engines.groups.find(g=>g.name==='room').ranges,[{start:0,count:360}]);
for(const emitter of canonical.emitters){const compact=photo.material_ids[emitter.triangles[0]];
  assert.deepEqual(photo.pbr.slice(compact*16+8,compact*16+11),[0,0,0]);
  assert.deepEqual(Array.from(data.slice(emitter.materialId*16+8,emitter.materialId*16+11)),[1,1,1]);}
room=structuredClone(exposure);room.shopLights.fixtures[0].source='led-array';room.shopLights.fixtures[0].dimmer=.23;
room.shopLights.fixtures[0].filter=[.3,.5,1];room.shopLights.topMaterial='marble_clear_glass';
writeShopLightEmitterRows(data,16,base+12,room.shopLights);
const changed=structuredClone(context.rebuildShopMeshes());
assert.equal(changed.faces.length,37128);assert.equal(changed.emitters.length,3);assert(context.shopTransparentFaces.length>0);
assert.deepEqual(structuredClone(context.sceneForPhoto()),photo,'B-only source, dimmer, filter and material edits leave A photo byte-identical');
assert.deepEqual(data.slice(0,(base+12)*16),fixedBefore,'Shared catalogue and original twelve runtime slots never move or change');
const failed=structuredClone(room);room.shopLights.fixtures[0].x=1000;
assert.throws(()=>context.rebuildShopMeshes(),/shop-long:.*does not fit/);
room=failed;
exposure.shopLights.fixtures[0].exposureMode='burst';const beforeBurst=data.slice();
assert.throws(()=>context.sceneForPhoto(),/shop-long:.*timing-packet/);assert.deepEqual(data,beforeBurst);
const timedBurst=structuredClone(context.sceneForPhoto(true,{timedExposure:true}));
assert(timedBurst.trigger_emitters.length>0);
assert(timedBurst.trigger_emitters.every(item=>timedBurst.shop_material_ids.includes(item.material)));
assert(timedBurst.trigger_emitters.every(item=>item.gain>0));
exposure.shopLights.fixtures[0].exposureMode='continuous';
exposure.shopLights.fixtures[0].dimmer=.25;exposure.shopLights.fixtures[0].filter=[.2,.4,.8];
const continuousPhoto=structuredClone(context.sceneForPhoto());
const continuousRow=continuousPhoto.material_ids[canonical.emitters[0].triangles[0]];
assert.deepEqual(continuousPhoto.pbr.slice(continuousRow*16+8,continuousRow*16+11),[.05,.1,.2].map(Math.fround));
assert.equal(continuousPhoto.emitter_profiles.records[0].gain,.25);
assert.deepEqual(continuousPhoto.emitter_profiles.records[0].filter,[.2,.4,.8]);
assert.equal(continuousPhoto.emitter_profiles.records[0].material,continuousRow);
assert.deepEqual(continuousPhoto.vertices,photo.vertices,'Continuous override changes emission, not geometry');
assert.deepEqual(data,beforeBurst,'Exposure override never mutates Studio material rows');
exposure.shopLights.fixtures[0].source='led-array';
const mixedPhoto=structuredClone(context.sceneForPhoto());
assert.equal(mixedPhoto.emitter_profiles.records.length,3);
assert.deepEqual(mixedPhoto.emitter_profiles.records.map(r=>r.profile),['led_cool_white','led_warm_white','fluorescent_cool_white']);
const mixedMesh=context.shopLightGeometryForStudio('exposure');
for(const e of mixedMesh.emitters){const material=mixedPhoto.material_ids[e.triangles[0]];
 assert(e.triangles.every(i=>mixedPhoto.material_ids[i]===material));
 assert(mixedPhoto.emitter_profiles.records.some(r=>r.material===material));}
delete exposure.shopLights;room=structuredClone(exposure);
assert.equal(context.shopLightFacesForStudio('exposure').length,0);
assert.equal(context.sceneForPhoto().vertices.length,0,'Legacy rigs acquire no shop fixtures');
assert.doesNotMatch(source,/shopLightPhongSamples|maxSamples:100/);
assert.match(source,/for \(const role of registeredFixtures\)/);
assert.match(source,/uNumLights"\),lightPos.length/);
assert.match(buildStudioPhotoFixtures.toString(),/collect\('room','Shop light'/);
if(source.includes('function captureCurrentStudioScene('))assert.match(source,/shopLightFacesForStudio\("studio",collect\("shop"\)\)/);
console.log('PASS actual shop bridge: one mesh in Phong/ray, independent shared emitter slots, zero photo emission with intact tubes, exact room ownership, glass partition, A/B byte isolation, legacy empty, placement errors and original timing gate.');
