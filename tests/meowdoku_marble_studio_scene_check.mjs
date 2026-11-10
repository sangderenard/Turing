// Actual authoring builders and current-Studio capture, CPU only. No native
// contact admission, ray job, active browser, or server publication occurs.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {packStudioScene,writeStudioFixtureRows} from '../examples/meowdoku/web/marble_studio_scene.js';
import {studioRoomFaces,studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
import {studioTableFaces,defaultStudioTable,tableFloorDrop} from '../examples/meowdoku/web/marble_studio_table.js';
import {studioGoboFaces} from '../examples/meowdoku/web/marble_studio_gobos.js';
import {backdropFaces,defaultBackdropLayers} from '../examples/meowdoku/web/marble_backdrop.js';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
import {writeShopLightEmitterRows} from '../examples/meowdoku/web/marble_shop_lights.js';
import {STUDIO_FIXED_MATERIAL_ROWS} from '../examples/meowdoku/web/marble_materials.js';
import {tableLightModifierFaces} from '../examples/meowdoku/web/marble_table_lights.js';
const source=fs.readFileSync(process.env.STUDIO_TEST_WEB_ROOT?`${process.env.STUDIO_TEST_WEB_ROOT}/marble_studio.js`:new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const ids={marble_clear_glass:0,white_plaster:1,black_felt:2};
const parts=[];
function coverage(prefix,build) {
  const found=[],withMetadata=build(part=>found.push({...part,id:`${prefix}/${part.id}`}));
  assert.deepEqual(withMetadata,build(undefined),'Metadata does not change renderer geometry');
  assert.deepEqual(structuredClone(found.flatMap(p=>p.faces)),structuredClone(withMetadata),'Each exact rendered face has one source identity');
  parts.push(...found);return withMetadata;
}
const table=({...defaultStudioTable(),width:5.2,length:3,height:3.02,material:"marble_clear_glass",skirt:{enabled:false,material:"black_felt"}}),floorY=studioRoomBounds(3,tableFloorDrop(table)).floorY;
const room=cb=>studioRoomFaces({scale:3,floorDrop:tableFloorDrop(table),windowOn:true,glassMaterial:0,material:()=>1,onSurface:cb});
const furniture=cb=>studioTableFaces(table,ids,floorY,cb);
const gobos=cb=>studioGoboFaces({version:1,gobos:[{id:7,enabled:true,shape:'ellipse',width:2,height:3,x:2,y:1,z:0,
  pitch:23,yaw:-67,roll:121,front:'black_felt',back:'white_plaster'}]},ids,cb);
const backdrops=cb=>backdropFaces({count:2,material:'white_plaster',sweep:true,layers:defaultBackdropLayers(),wallZ:-9.3,floorY},ids,cb);
coverage('room',room);coverage('table',furniture);coverage('gobo',gobos);coverage('backdrop',backdrops);
assert.equal(new Set(parts.filter(p=>p.id.startsWith('table/')).map(p=>p.bodyId)).size,1,'Table construction stays one compound');
assert.equal(new Set(parts.filter(p=>p.id.startsWith('room/')).map(p=>p.bodyId)).size,1);
assert.deepEqual(parts.filter(p=>p.id.startsWith('gobo/')).map(p=>p.id),['gobo/7']);
const context=vm.createContext({STUDIO_LIGHTS,tableLightModifierFaces,Math,structuredClone,lightStandFacesForStudio:()=>[],
  shopLightFacesForStudio:()=>[],shopLightState:()=>undefined,writeShopLightEmitterRows,STUDIO_FIXED_MATERIAL_ROWS,backdropSupportFacesForStudio:()=>[]});
// Existing fixture primitives and their existing authoring function, not a
// second fixture generator or a simplified collider used only by tests.
vm.runInContext(source.slice(source.indexOf('function quadFace('),source.indexOf('function stringFaces(')),context);
vm.runInContext(source.slice(source.indexOf('function buildFixtureFaces('),source.indexOf('// The ray-traced photo\'s')),context);
const fixtureRoles=[{pos:[0,4,1],dir:[0,-1,0],halfW:.5,halfH:.5,mat:3,roleIndex:0,round:false,shape:'bare'},
  {pos:[2,3,1],dir:[0,0,-1],radius:.2,mat:4,roleIndex:2,round:true,shape:'bare',gobo:true}];
coverage('fixture',cb=>context.buildFixtureFaces(fixtureRoles,5,cb));
const pbr=new Float32Array(6*20);for(let i=0;i<6;i++){pbr[i*20+3]=.7;pbr[i*20+7]=1;}
const prior=pbr.slice();writeStudioFixtureRows(pbr,20,3,[[1,.5,0],[0,1,.25]],[2,0]);
assert.equal(pbr[3*20+8],2);assert.equal(pbr[3*20+9],1);assert.equal(pbr[4*20],0);assert.equal(pbr[4*20+1],1);
assert.deepEqual(pbr.slice(0,60),prior.slice(0,60));
const model=[2,0,0,0,0,2,0,0,0,0,2,0,3,-1,5,1],n=[0,0,1];
const subjectFaces=[[[0,0,0],[1,0,0],[0,1,0],0,[n,n,n]]];
const options={channel:'studio',unitM:.01,rayOriginM:[0,0,.18],parts,pbr:{data:pbr,stride:20},
  subject:{id:'subject',faces:subjectFaces,model,radius:2,motion:{mode:'off'}},camera:{view:model,fovDegrees:30,aspect:2},authoring:{room:'actualStudio'}};
const packed=packStudioScene(options),allFaces=parts.flatMap(p=>p.faces);
assert.equal(packed.collisionReady,false);assert.equal(packed.ray.material_ids.length,allFaces.length);
assert.deepEqual(packed.ray.vertices,allFaces.flatMap(f=>f.slice(0,3).flatMap(p=>p.map((v,i)=>v*.01+options.rayOriginM[i]))));
assert.deepEqual(packed.ray.normals,allFaces.flatMap(f=>Array.from(f[4]).flat()));
assert.deepEqual(packed.subject.ray.vertices,[.03,-.01,.22999999999999998,.05,-.01,.22999999999999998,.03,.01,.22999999999999998]);
assert.deepEqual(packed.ray.pbr,Array.from({length:6},(_,i)=>Array.from(pbr.slice(i*20,i*20+16))).flat());
for(const part of packed.parts)for(let i=0;i<part.triangleCount;i++) {
  assert.deepEqual(packed.triangleSources[part.firstTriangle+i],{partId:part.id,bodyId:part.bodyId,triangle:i});
  assert.deepEqual(structuredClone(part.faces[i]),structuredClone(parts.find(p=>p.id===part.id).faces[i]));
}
model[12]=99;pbr[0]=99;parts[0].faces[0][0][0]=999;
assert.equal(packed.subject.position[0],3);assert.notEqual(packed.ray.pbr[0],99);assert.notEqual(packed.parts[0].faces[0][0][0],999);
for(const override of [{channel:undefined},{unitM:0},{pbr:{data:[NaN,...new Array(15).fill(0)],stride:16}},
  {parts:[parts[0],parts[0]]},{subject:{...options.subject,model:[]}}])assert.throws(()=>packStudioScene({...options,...override}));

// Execute the actual bootstrap capture function against actual shared builders.
// It must select Studio, not Exposure, and keep the displayed pose/camera.
const lighting={controls:{checker:true}},geometry={interior:subjectFaces,glass:[]};
const displayed={revision:4,engine:'visual',geometry,lighting,subject:{unitM:.01,subjectRadius:2,fovDegrees:30},
  model:[2,0,0,0,0,2,0,0,0,0,2,0,3,-1,5,1],view:new Float32Array(model),fovDegrees:24,aspect:2,
  orbit:{yaw:0,pitch:.2,distance:4,target:[0,0,0]},
  fixtureRoles,roleColor:[[1,.5,0],[0,1,.25]],roleStrength:[2,0],motion:{mode:'off',axis:'y',angle:0,roll:null}};
const controls={'preview-engine':{value:'visual'},'debug-coords':{checked:false}};
Object.assign(context,{lastStudioFrame:displayed,lastGeometry:geometry,studioSceneRevision:4,$:id=>controls[id],
  yaw:0,pitch:.2,distance:4,studioOrbitTarget:[0,0,0],
  studioRenderState:channel=>{assert.equal(channel,'studio');return lighting;},
  studioControlValue:(id,channel)=>{assert.equal(channel,'studio');return lighting.controls[id];},
  checkerboardFaces:(channel,cb)=>{assert.equal(channel,'studio');return room(cb);},
  backdropRollFaces:(channel,cb)=>{assert.equal(channel,'studio');return backdrops(cb);},
  tableFaces:(channel,cb)=>{assert.equal(channel,'studio');return furniture(cb);},
  goboFaces:(channel,cb)=>{assert.equal(channel,'studio');return gobos(cb);},
  tables:{pbr:{data:pbr,table:{floats_per_record:20}}},FIXTURE_MAT_BASE:3,FIXTURE_HOUSING_MAT:5,
  OBJECT_CENTER:[0,0,.18],writeStudioFixtureRows,packStudioScene,captureObjectSnapshot:()=>({id:'unsaved-current-design'})});
vm.runInContext(source.slice(source.indexOf('function captureCurrentStudioScene('),source.indexOf('\nfunction syncMaterial()')),context);
const before=structuredClone(displayed),captured=context.captureCurrentStudioScene();
assert.deepEqual(displayed,before,'Capturing does not mutate visible pose/camera');
assert.equal(captured.revision,4);assert.equal(captured.state.channel,'studio');
assert.equal(captured.state.camera.fovDegrees,24,'Actual displayed lens FOV, not former scale fallback');
assert.deepEqual(captured.state.subject.position,[3,-1,5]);assert.equal(captured.state.authoring.object.id,'unsaved-current-design');
lighting.controls.checker=false;const open=context.captureCurrentStudioScene();
assert(!open.state.parts.some(p=>p.id.startsWith('room/')||p.id.startsWith('backdrop/')));
for(const mutate of [()=>context.studioSceneRevision++,()=>controls['preview-engine'].value='particle',()=>controls['debug-coords'].checked=true]) {
  mutate();assert.throws(()=>context.captureCurrentStudioScene());context.studioSceneRevision=4;controls['preview-engine'].value='visual';controls['debug-coords'].checked=false;
}
context.yaw=.1;assert.throws(()=>context.captureCurrentStudioScene(),/camera movement/);context.yaw=0;
const motionSource=source.slice(source.indexOf('const mode=$("preview-motion").value,rollLocked='),source.indexOf('    lastStudioFrame={revision:'));
for(const [mode,locked]of [['off',false],['off',true],['spin',false]]) {
  const env=vm.createContext({$:id=>id==='inertia-lock'?{checked:locked}:{value:mode},
    axis:'z',angle:0,rollState:{position:[1,2,3],angularVelocity:[1,2,3]},subject:{radiusM:.01},structuredClone});
  const motion=structuredClone(vm.runInContext(`${motionSource}\nmotion`,env));
  assert.deepEqual(motion.velocityM,mode==='off'?(locked?[-0,0,0]:[-.03,0,.01]):null);
  assert.deepEqual(motion.angularVelocity,mode==='off'?(locked?[0,0,0]:[1,2,3]):null);
}
console.log(`PASS current Studio scene: ${packed.ray.material_ids.length} exact triangles, authored part ownership, unchanged builders, actual capture, stale/profile gates`);
