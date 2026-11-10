import {installScenePackerInputs} from './meowdoku_scene_packer_harness.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {defaultLightStands,validateLightStands,setLightStandMaterial,LIGHT_STAND_PROFILE_PATHS,
  createLightStandControls,studioLightStandFaces,lightStandPlacementWarnings} from '../examples/meowdoku/web/marble_light_stand_settings.js';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
import {applyLightTools,defaultLightTools} from '../examples/meowdoku/web/marble_light_tools.js';
import {applyFixtureSizes,defaultFixtureSizes} from '../examples/meowdoku/web/marble_fixture_sizes.js';
import {wallRelativeFixtureDefaults} from '../examples/meowdoku/web/marble_light_placement.js';
import {tableLightRoles,tableLightModifierFaces} from '../examples/meowdoku/web/marble_table_lights.js';
import {defaultStudioTable,tableFloorDrop,FLOOR_FIXTURE_INDICES} from '../examples/meowdoku/web/marble_studio_table.js';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
import {resolveRoomProfile,roomSettingLeaves,validateRoomOverrides} from '../examples/meowdoku/web/marble_lighting_profiles.js';
import {createLightingPoseState,resolveLightingPoses} from '../examples/meowdoku/web/marble_lighting.js';
import {defaultStudioScale,studioScaleTransform} from '../examples/meowdoku/web/marble_studio_scale.js';
import {isOpaquePreviewMaterial} from '../examples/meowdoku/web/marble_glass_preview.js';
import {backdropFaces,defaultBackdropLayers} from '../examples/meowdoku/web/marble_backdrop.js';

const names=['black_matte_paint','metal_steel'],ids={polished_silver:7,black_matte_paint:79,metal_steel:8};
const fresh=defaultLightStands(),legacy=validateLightStands(undefined,[]);
assert.deepEqual(fresh,{version:1,enabled:true,material:'black_matte_paint',upperSupport:'ceiling',wideSupports:true,backdropSupport:true});
assert.deepEqual(legacy,{...fresh,enabled:false,upperSupport:'floor',wideSupports:false,backdropSupport:false});
assert.equal(validateLightStands({version:1,enabled:true,material:'black_matte_paint'},names).upperSupport,'floor','Saved enabled supports retain floor geometry');
assert.deepEqual(validateLightStands(JSON.parse(JSON.stringify(legacy)),[]),legacy);
assert.throws(()=>validateLightStands(fresh,[]),/unavailable/);
assert.throws(()=>validateLightStands({...legacy,material:'missing'},names),/unavailable/);
for(const bad of [null,{}, {...fresh,version:2},{...fresh,enabled:'true'},{...fresh,material:1}])assert.throws(()=>validateLightStands(bad,names));
assert.equal(setLightStandMaterial(fresh,'metal_steel',names).material,'metal_steel');
assert.equal(fresh.material,'black_matte_paint');
assert.throws(()=>setLightStandMaterial(fresh,'missing',names),/unavailable/);
assert.deepEqual(LIGHT_STAND_PROFILE_PATHS,['lightStands/enabled','lightStands/material','lightStands/upperSupport','lightStands/wideSupports','lightStands/backdropSupport']);

// The incumbent Room resolver and enumerator own both canonical leaves.
const validate=profile=>({...profile,lightStands:validateLightStands(profile.lightStands,names)});
const exposure=validate({version:1,lightStands:fresh}),saved=JSON.stringify(exposure);
assert.deepEqual(roomSettingLeaves(exposure).filter(l=>l.path.startsWith('lightStands/')).map(l=>l.path),LIGHT_STAND_PROFILE_PATHS);
const overrides={version:1,values:[{path:'lightStands/enabled',value:false},{path:'lightStands/material',value:'metal_steel'}]};
const room=resolveRoomProfile(exposure,overrides,validate);
assert.deepEqual(validateRoomOverrides(overrides,exposure,validate),overrides);
assert.deepEqual(room.lightStands,{...fresh,enabled:false,material:'metal_steel'});
assert.equal(JSON.stringify(exposure),saved);
assert.deepEqual(resolveRoomProfile(exposure,{version:1,values:[]},validate),exposure);
assert.throws(()=>resolveRoomProfile(exposure,{values:[{path:'lightStands/material',value:'missing'}]},validate),/unavailable/);

class Element {
  constructor(tag,doc){this.tagName=tag.toUpperCase();this.ownerDocument=doc;this.children=[];this.events={};this.value='';this.checked=false;}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=[...nodes];}
  setAttribute(k,v){this[k]=v;}
  addEventListener(k,f){(this.events[k]??=[]).push(f);}
  fire(k){for(const f of this.events[k]??[])f({target:this});}
}
const doc={createElement(tag){return new Element(tag,this);}};
const all=e=>[e,...e.children.flatMap(all)];
let changes=0,reject=false;
const container=new Element('div',doc),controls=createLightStandControls(container,names,{onChange:()=>changes++,
  validateChange:()=>{if(reject)throw Error('key-left: no side fits this room');}});
const find=id=>all(container).find(el=>el.id===id);
assert.deepEqual(controls.getState(),fresh);assert.equal(changes,0);
find('light-stands-material').value='metal_steel';find('light-stands-material').fire('change');
assert.equal(controls.getState().material,'metal_steel');assert.equal(changes,1);
find('light-stands-enabled').checked=false;find('light-stands-enabled').fire('change');
assert.equal(controls.getState().enabled,false);assert.equal(changes,2);
reject=true;find('light-stands-enabled').checked=true;find('light-stands-enabled').fire('change');
assert.equal(controls.getState().enabled,false);assert.equal(find('light-stands-enabled').checked,false);assert.match(find('light-stands-status').textContent,/key-left/);assert.equal(changes,2);
reject=false;controls.setState(fresh);assert.equal(changes,2);
controls.setPlacementWarnings([{lightId:'underlight',message:'Raise its housing above the floor.'}]);
assert.match(find('light-stands-status').textContent,/underlight: Raise/);
const copy=controls.getState();copy.enabled=false;assert.equal(controls.getState().enabled,true);
controls.setState(undefined);assert.equal(controls.getState().enabled,false);
const absentContainer=new Element('div',doc),absent=createLightStandControls(absentContainer,[]);
assert.match(all(absentContainer).find(el=>el.id==='light-stands-status').textContent,/unavailable/);
assert.equal(absent.getState().material,'black_matte_paint');
assert.throws(()=>validateLightStands(absent.getState(),[]),/unavailable/);
absent.setEnabled(false);assert.equal(absent.getState().enabled,false);

// Measure the actual default rig, not a hand-authored approximation of it.
const source=fs.readFileSync(process.env.STUDIO_MAIN_SOURCE??new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const html=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.html',import.meta.url),'utf8'),fields={};
for(const m of html.matchAll(/<input\b([^>]*\bid="([^"]+)"[^>]*)>/g))fields[m[2]]={value:/value="([^"]*)"/.exec(m[1])?.[1]??'',checked:/\bchecked\b/.test(m[1])};
for(const m of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
  const opts=[...m[2].matchAll(/<option([^>]*)>/g)],opt=opts.find(o=>o[1].includes('selected'))??opts[0];
  fields[m[1]]={value:opt?/value="([^"]*)"/.exec(opt[1])?.[1]??'':''};
}
const bounds=studioRoomBounds(Number(fields['room-scale'].value),tableFloorDrop(defaultStudioTable()));
const context=vm.createContext({ids,applyLightTools,defaultLightTools,wallRelativeFixtureDefaults,applyFixtureSizes,defaultFixtureSizes,
  $:id=>fields[id],STUDIO_LIGHTS,FIXTURE_MAT_BASE:90,studioLighting:null,ids:{polished_silver:39,polished_gold:40},
  studioTableState:()=>defaultStudioTable(),studioRoomBounds,studioRoomScale:()=>Number(fields['room-scale'].value),studioFloorDrop:()=>tableFloorDrop(defaultStudioTable()),
  FLOOR_FIXTURE_INDICES,tableLightRoles,tableLightModifierFaces,rgb:value=>[1,3,5].map(k=>parseInt(value.slice(k,k+2),16)/255)});
vm.runInContext(source.slice(source.indexOf('const ROOM_FIXTURES ='),source.indexOf('const currentCameraOptions =')),context);
vm.runInContext(source.slice(source.indexOf('function quadFace('),source.indexOf('function digitFaces(')),context);
vm.runInContext(source.slice(source.indexOf('function studioRenderState('),source.indexOf('function surfaceMaterial(')),context);
vm.runInContext(source.slice(source.indexOf('function computeFixtureRoles('),source.indexOf('export function sceneForPhoto(')),context);
const roles=structuredClone(vm.runInContext('computeFixtureRoles().fixtureRoles',context));
const options={bounds,unitMm:304.8,fixtureVertices:role=>structuredClone(context.buildFixtureFaces([role],0).flatMap(face=>face.slice(0,3)))};
const warnings=lightStandPlacementWarnings(fresh,roles,bounds.floorY,options);
console.log('Actual default stand placement:',JSON.stringify(warnings));
assert.deepEqual(warnings,[],'Every actual new-rig light needs a floor stand that fits the room');
const parts=[],faces=studioLightStandFaces(fresh,roles,ids,bounds.floorY,{...options,onPart:p=>parts.push(p)});
const supportTriangleCount=8*240+480+576;
assert.equal(parts.length,10);assert.equal(faces.length,supportTriangleCount);assert(faces.every(f=>f[3]===79));
assert.equal(parts.filter(p=>p.construction==='ceiling-light-support').length,1);
assert.equal(parts.find(p=>p.id==='fill-upper').components.length,12);
assert.equal(parts.find(p=>p.id==='fill-lower').construction,'dual-floor-light-stand');
assert.equal(parts.find(p=>p.id==='fill-lower').components.length,10);
assert.equal(roles.find(r=>r.roleIndex===6).pos[1],5.2,'Fresh upper softbox is two feet higher');
assert.equal(studioLightStandFaces({version:1,enabled:true,material:'black_matte_paint'},roles,ids,bounds.floorY,options).length,2400,'Saved absent support leaf preserves all ten floor stands');
assert.deepEqual(parts.map(p=>p.id),STUDIO_LIGHTS.map(l=>l.id));
assert.equal(studioLightStandFaces({...fresh,material:'metal_steel'},roles,ids,bounds.floorY,options)[0][3],8);
assert.deepEqual(studioLightStandFaces(undefined,roles,{},bounds.floorY,options),[],'Old profiles add no triangles');
assert.throws(()=>studioLightStandFaces(fresh,roles,{},bounds.floorY,options),/unavailable/);
const blocked=structuredClone(roles);blocked[0].pos[1]=-50;let callbacks=0;
assert(lightStandPlacementWarnings(fresh,blocked,bounds.floorY,options).some(w=>w.lightId==='key-left'));
assert.throws(()=>studioLightStandFaces(fresh,blocked,ids,bounds.floorY,{...options,onPart:()=>callbacks++}),/key-left:/);
assert.equal(callbacks,0,'A blocked rig never emits a partial part set');
console.log('PASS stand settings: explicit new/legacy defaults, unavailable material errors, shared material, UI rollback, A/B resolver independence, actual all-ten default room fit, stable identities and all-or-nothing mesh.');

// Execute the actual shared Phong builder and photo catalogue remapper after
// subject-lock resolution at the current new-rig scale (one Studio unit/foot).
const scale=process.env.STUDIO_UNIT_MM?{...defaultStudioScale(),unitMm:Number(process.env.STUDIO_UNIT_MM)}:defaultStudioScale(),table=defaultStudioTable();
const subject=studioScaleTransform(scale,Number(fields['radius-mm'].value),table.enabled);
const poseState=createLightingPoseState(roles.map((r,i)=>({id:STUDIO_LIGHTS[i].id,pos:r.pos,dir:r.dir,
  aimAtSubject:i===0||i===1||i===5||i===6||i>=7})));
const sizes=defaultFixtureSizes();
const exposureProfile={version:1,controls:Object.fromEntries(Object.entries(fields).map(([id,field])=>[id,field.value])),
  poses:poseState,fixtureSizes:sizes,lightTools:defaultLightTools(sizes),lightStands:fresh,table,scale};
let previewProfile=structuredClone(exposureProfile);
const pbr=new Float32Array(100*16);
pbr.set([.025,.025,.025,.8,0,0,1.5,1,0,0,0,0,0,0,0,0],79*16);
pbr.set([.5,.6,.7,.2,1,0,1.5,1,0,0,0,0,0,0,0,0],8*16);
pbr.set([.9,.95,1,.035,0,1,1.52,0,0,0,0,0,0,0,0,0],15*16);
Object.assign(context,{resolveLightingPoses,studioLightStandFaces,ids:{...ids,marble_clear_glass:15,polished_silver:39,polished_gold:40},FIXTURE_HOUSING_MAT:101,
  studioLighting:{renderState:channel=>channel==='exposure'?exposureProfile:previewProfile,strengths:{getSolo:()=>null}},
  studioSubjectTransform:()=>subject,studioPhotoScale:()=>subject.unitM,OBJECT_CENTER:[0,0,0],
  tables:{pbr:{table:{floats_per_record:16},data:pbr}},checkerboardFaces:()=>[],backdropRollFaces:()=>[],tableFaces:()=>[],goboFaces:()=>[],
  mesh:faces=>({faces,count:faces.length*3}),transparentMesh:faces=>({faces,count:faces.length*3}),dispose:()=>{},
  read:(_,id,column)=>pbr[id*16+column],isOpaquePreviewMaterial,
  standOpaque:null,standGlass:null,standTransparentFaces:[],standMeshKey:null,sceneGlassKey:null});
vm.runInContext(source.slice(source.indexOf('function lightStandFacesForStudio('),source.indexOf('function rebuildTable(')),context);
vm.runInContext(source.slice(source.indexOf('export function sceneForPhoto('),source.indexOf('\nfunction syncMaterial()')).replace('export function','function'),context);
const resolvedRoles=structuredClone(context.computeFixtureRoles(subject.subjectCenter,'exposure').fixtureRoles);
assert.deepEqual(lightStandPlacementWarnings(fresh,resolvedRoles,bounds.floorY,options),[],'Subject-lock resolution at foot scale keeps every stand available');
const sharedFaces=structuredClone(context.lightStandFacesForStudio('exposure',resolvedRoles));
assert.equal(sharedFaces.length,supportTriangleCount);
installScenePackerInputs(context);
const photoBefore=structuredClone(context.sceneForPhoto());
assert.ok(Buffer.byteLength(JSON.stringify(photoBefore))<2*1024*1024-65536,
  "Stand scene leaves64KiB config headroom under bounded2MiB MRP1 config limit");
assert.deepEqual(photoBefore.vertices.slice(-supportTriangleCount*9),sharedFaces.flatMap(f=>f.slice(0,3).flatMap(p=>p.map(v=>v*subject.unitM))));
const standPhotoIds=photoBefore.material_ids.slice(-supportTriangleCount);assert.equal(new Set(standPhotoIds).size,1);
assert.deepEqual(photoBefore.pbr.slice(standPhotoIds[0]*16,standPhotoIds[0]*16+16),Array.from(pbr.slice(79*16,80*16)));
assert.equal(photoBefore.pbr[standPhotoIds[0]*16+8],0,'The catalogue stand row is never mistaken for an emitter');
context.rebuildStandMeshes(resolvedRoles);assert.equal(context.standOpaque.faces.length,supportTriangleCount);assert.equal(context.standTransparentFaces.length,0);
previewProfile={...previewProfile,lightStands:{...fresh,material:'marble_clear_glass'}};
context.rebuildStandMeshes(resolvedRoles);assert.equal(context.standOpaque.faces.length,0);assert.equal(context.standTransparentFaces.length,supportTriangleCount);
assert(context.standTransparentFaces.every(f=>f[3]===15),'Transparent stand finishes use the existing shared glass sort input');
assert.deepEqual(structuredClone(context.sceneForPhoto()),photoBefore,'B-only material changes leave every photo vertex/material byte unchanged');
previewProfile={...previewProfile,lightStands:{...previewProfile.lightStands,upperSupport:'floor'}};
context.rebuildStandMeshes(resolvedRoles);assert.equal(context.standTransparentFaces.length,2880);
assert.deepEqual(structuredClone(context.sceneForPhoto()),photoBefore,'B-only upper support replacement leaves A photo payload unchanged');
previewProfile={...previewProfile,lightStands:{...previewProfile.lightStands,enabled:false}};
context.rebuildStandMeshes(resolvedRoles);assert.equal(context.standOpaque.faces.length,0);assert.equal(context.standTransparentFaces.length,0);
assert.deepEqual(structuredClone(context.sceneForPhoto()),photoBefore,'B-only stand visibility leaves A ray payload unchanged');
assert(/\.\.\.standTransparentFaces,(\.\.\.shopTransparentFaces,)?\.\.\.faces/.test(source),'Ordinary Phong stand glass shares table/subject ordering');
if(source.includes('function captureCurrentStudioScene('))assert(source.includes('lightStandFacesForStudio("studio",displayed.fixtureRoles,collect("stand"))'),'Actual scene capture reads the same stand builder');
console.log('PASS actual stand bridge: foot-scale subject locks, identical Phong/ray vertices, catalogue-row remap without emission collision, opaque/transparent partition, and byte-identical A photo payload under B overrides.');

previewProfile=structuredClone(exposureProfile);
context.backdropRollFaces=(channel,cb)=>backdropFaces({count:1,material:'metal_steel',sweep:true,layers:defaultBackdropLayers(),
  wallZ:bounds.backZ,floorY:bounds.floorY},context.ids,cb);
const holderFaces=structuredClone(context.backdropSupportFacesForStudio('exposure'));
assert.equal(holderFaces.length,528);const holderPhoto=structuredClone(context.sceneForPhoto());
context.standMeshKey=null;context.rebuildStandMeshes(resolvedRoles);
assert.equal(context.standOpaque.faces.length,supportTriangleCount+528);
const allRoomRanges=holderPhoto.material_engines.groups.find(g=>g.name==='room').ranges;
assert(allRoomRanges.some(r=>r.count>=528),'Holder parts retain their actual room owner');
previewProfile.lightStands.backdropSupport=false;context.rebuildStandMeshes(resolvedRoles);
assert.equal(context.standOpaque.faces.length,supportTriangleCount);
assert.deepEqual(structuredClone(context.sceneForPhoto()),holderPhoto,'B-only holder visibility leaves A photo payload unchanged');
assert.equal(studioLightStandFaces({...fresh,wideSupports:false},resolvedRoles,ids,bounds.floorY,options).length,9*240+576,'Wide-box opt-out removes only the second lower stand');
console.log('PASS wide/holder bridge: actual lower fill has opposed supports; real backdrop holder joins Phong and room-owned photo geometry, shared material and B-only visibility isolate correctly.');
