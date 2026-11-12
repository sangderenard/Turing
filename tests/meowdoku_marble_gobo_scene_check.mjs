import {installScenePackerInputs,installStudioSceneBuilders} from './meowdoku_scene_packer_harness.mjs';
import {applyLightTools,defaultLightTools,migrateLightTools} from '../examples/meowdoku/web/marble_light_tools.js';
import {wallRelativeFixtureDefaults} from "../examples/meowdoku/web/marble_light_placement.js";
import {applyFixtureSizes,defaultFixtureSizes} from "../examples/meowdoku/web/marble_fixture_sizes.js";
import {createLightingPoseState,resolveLightingPoses} from '../examples/meowdoku/web/marble_lighting.js';
import {defaultStudioScale,studioScaleTransform} from '../examples/meowdoku/web/marble_studio_scale.js';
// Execute the real full Studio furniture/scene packer. No GPU or server calls.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
import {tableLightRoles,tableLightModifierFaces} from '../examples/meowdoku/web/marble_table_lights.js';
import {defaultStudioTable,tableFloorDrop,studioTableFaces,FLOOR_FIXTURE_INDICES} from '../examples/meowdoku/web/marble_studio_table.js';
import {studioRoomBounds,studioRoomFaces} from '../examples/meowdoku/web/marble_room.js';
import {backdropFaces,defaultBackdropLayers} from '../examples/meowdoku/web/marble_backdrop.js';
import {concentricBackdropLayers} from '../examples/meowdoku/web/marble_studio_surfaces.js';
import {studioGoboFaces,MAX_STUDIO_GOBOS} from '../examples/meowdoku/web/marble_studio_gobos.js';
import {isOpaquePreviewMaterial} from '../examples/meowdoku/web/marble_glass_preview.js';
import {OBJECT_CENTER,photoCameraFromStudio,packMarblePhoto} from '../examples/meowdoku/web/marble_photo.js';
const source=fs.readFileSync(process.env.STUDIO_TEST_WEB_ROOT?`${process.env.STUDIO_TEST_WEB_ROOT}/marble_studio.js`:new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const html=fs.readFileSync(process.env.STUDIO_TEST_WEB_ROOT?`${process.env.STUDIO_TEST_WEB_ROOT}/marble_studio.html`:new URL('../examples/meowdoku/web/marble_studio.html',import.meta.url),'utf8');
const controls={};
for(const m of html.matchAll(/<input\b([^>]*\bid="([^"]+)"[^>]*)>/g))
  controls[m[2]]={value:/value="([^"]*)"/.exec(m[1])?.[1]??'',checked:/\bchecked\b/.test(m[1])};
for(const m of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
  const opts=[...m[2].matchAll(/<option([^>]*)>/g)],opt=opts.find(o=>o[1].includes('selected'))??opts[0];
  controls[m[1]]={value:opt?/value="([^"]*)"/.exec(opt[1])?.[1]??'':''};
}
for(const light of STUDIO_LIGHTS)controls[light.strengthId].value='10';
for(const surface of ['floor','wall','ceiling']) {
  controls[`${surface}-pattern`].value='solid';
  for(const key of ['a','b'])controls[`${surface}-material-${key}`].value='white_plaster';
}
controls['backdrop-layers'].value='6';controls['backdrop-material'].value='white_plaster';controls['backdrop-sweep'].checked=false;
const ids={polished_silver:3,black_felt:0,white_plaster:1,marble_clear_glass:2};
const pbr=new Float32Array(3*16);
for(const [id,color,transmission,opacity] of [[0,[.01,.01,.01],0,1],[1,[.9,.91,.92],0,1],[2,[.95,.97,1],1,.1]]) {
  pbr.set(color,id*16);pbr[id*16+3]=.3;pbr[id*16+5]=transmission;pbr[id*16+6]=1.52;pbr[id*16+7]=opacity;
}
const tableState=({...defaultStudioTable(),width:5.2,length:3,height:3.02,material:"marble_clear_glass",skirt:{enabled:false,material:"black_felt"}}),layers=concentricBackdropLayers(3,false,tableFloorDrop(tableState));
let state={version:1,gobos:Array.from({length:MAX_STUDIO_GOBOS},(_,i)=>({id:i+1,enabled:true,shape:'ellipse',width:1.2+i/10,height:2.1,
  x:-3+i*.7,y:1,z:1.5+i*.2,pitch:17,yaw:-87+i*31,roll:11,front:'black_felt',back:i%2?'white_plaster':'marble_clear_glass'}))};
const context=vm.createContext({applyLightTools,defaultLightTools:migrateLightTools,wallRelativeFixtureDefaults:roles=>roles,applyFixtureSizes,defaultFixtureSizes:()=>defaultFixtureSizes(true),$:id=>controls[id],ids,OBJECT_CENTER,STUDIO_LIGHTS,
  photo:{applyLensCamera:camera=>camera},
  tableLightRoles,tableLightModifierFaces,defaultStudioTable,tableFloorDrop,studioTableFaces,FLOOR_FIXTURE_INDICES,
  studioRoomBounds,studioRoomFaces,backdropFaces,defaultBackdropLayers,photoCameraFromStudio,studioGoboFaces,
  FIXTURE_MAT_BASE:3,studioLighting:null,
  defaultStudioScale:()=>({version:1,mode:'fixed',unitMm:10}),studioScaleTransform,studioOrbitTarget:[0,0,0],savedPhotoRadius:.01,yaw:-.5,pitch:.2,distance:3.45,
  rgb:v=>[1,3,5].map(k=>parseInt(v.slice(k,k+2),16)/255),
  tables:{pbr:{table:{floats_per_record:16},data:pbr}}});
installStudioSceneBuilders(context);
vm.runInContext(source.slice(source.indexOf('function studioRenderState('),source.indexOf('\nfunction syncMaterial()')).replace('export function','function'),context);
delete context.computeFixtureRoles;delete context.buildFixtureFaces;
context.shopLightFacesForStudio=()=>[]; // Explicit legacy fixture has no shop/rack catalogue.
installScenePackerInputs(context);
const copy=v=>JSON.parse(JSON.stringify(v)),run=code=>copy(vm.runInContext(code,context,{timeout:3000}));
const initial=run('computeFixtureRoles().fixtureRoles');
context.studioLighting={table:{getState:()=>tableState},backdrops:{getState:()=>layers},gobos:{getState:()=>state},
  strengths:{getStrengths:()=>STUDIO_LIGHTS.map(()=>1)},poses:{getPoses:()=>initial}};
const gobos=run('goboFaces()'),scene=run('sceneForPhoto()'),camera=run('studioPhotoCamera()');
const counts={fixtures:run('buildFixtureFaces(computeFixtureRoles().fixtureRoles,99).length'),room:run('checkerboardFaces().length'),backdrop:run('backdropRollFaces().length'),table:run('tableFaces().length'),gobos:gobos.length};
assert.equal(counts.gobos,1536);assert.equal(counts.backdrop,576);assert.equal(counts.table,60);
assert.equal(scene.material_ids.length,Object.values(counts).reduce((a,b)=>a+b,0));
assert(scene.material_ids.length<=5000,'Maximum panels with full rig stays within unchanged upload triangle limit');
assert(scene.pbr.length/16<=64);assert(scene.vertices.every(Number.isFinite));
tableState.skirt.enabled=true;
const skirtPreview=run('tableFaces()'),skirtScene=run('sceneForPhoto()');
assert.equal(skirtPreview.length,108);
assert.equal(skirtScene.material_ids.length,scene.material_ids.length+48);
assert.equal(skirtPreview.filter(face=>face[3]===ids.black_felt).length,48);
const isFelt=(s,id)=>Math.abs(s.pbr[id*16]-.01)<1e-6;
assert.equal(skirtScene.material_ids.filter(id=>isFelt(skirtScene,id)).length,scene.material_ids.filter(id=>isFelt(scene,id)).length+48);
tableState.skirt.enabled=false;
assert.deepEqual(run('sceneForPhoto()'),scene,'Removing skirt restores the original ray scene exactly');
const defaultKey=controls['key-shape'].value,defaultWindow=controls['window-glass'].checked;
const keyOptions=[...html.match(/<select id="key-shape">([\s\S]*?)<\/select>/)[1].matchAll(/<option value="([^"]+)"/g)].map(match=>match[1]);
for(const shape of keyOptions)for(const window of [false,true]) {
  controls['key-shape'].value=shape;controls['window-glass'].checked=window;
  const full=run('sceneForPhoto()');assert(full.material_ids.length<=5000);assert(full.pbr.length/16<=64);
}
controls['key-shape'].value=defaultKey;controls['window-glass'].checked=defaultWindow;
assert.deepEqual(scene.vertices.slice(-gobos.length*9),gobos.flatMap(f=>f.slice(0,3).flatMap(p=>p.map((v,i)=>OBJECT_CENTER[i]+v*.01))));
for(const [index,face] of gobos.entries()) {
  const compact=scene.material_ids[scene.material_ids.length-gobos.length+index];
  assert.deepEqual(scene.pbr.slice(compact*16,compact*16+16),Array.from(pbr.slice(face[3]*16,face[3]*16+16)));
}
// The preview consumes exactly the same faces, including both material sides,
// and places glass gobo faces into the existing shared transparency sort.
context.isOpaquePreviewMaterial=isOpaquePreviewMaterial;
context.read=(_table,id,offset)=>pbr[id*16+offset];context.dispose=()=>{};context.mesh=faces=>({faces});
context.tableOpaque=null;context.tableTransparentFaces=[];context.sceneGlassKey='old';
run('rebuildTable(); null');
const furniture=[...run('tableFaces()'),...gobos];
assert.deepEqual(copy(context.tableOpaque.faces),furniture.filter(f=>isOpaquePreviewMaterial(pbr[f[3]*16+5],pbr[f[3]*16+7])));
assert.deepEqual(copy(context.tableTransparentFaces),furniture.filter(f=>!isOpaquePreviewMaterial(pbr[f[3]*16+5],pbr[f[3]*16+7])));
assert.equal(context.sceneGlassKey,null);
const normal=[[0,0,1],[0,0,1],[0,0,1]],subject=[[0,0,0],[1,0,0],[0,1,0],0,normal];
const wire=packMarblePhoto({interior:[],glass:[subject]},'a'.repeat(64),{...camera,fixtures:scene});
const view=new DataView(wire),config=JSON.parse(new TextDecoder().decode(new Uint8Array(wire,16,view.getUint32(4,true))));
assert.deepEqual(config.fixtures,scene);
// The live callback calls the same sceneForPhoto. Disabled/legacy rows drop
// cleanly while the rest of the current rig and compact material order remain.
state={...state,gobos:state.gobos.map(gobo=>({...gobo,enabled:false}))};
const hidden=run('sceneForPhoto()');assert.equal(hidden.material_ids.length,scene.material_ids.length-gobos.length);
assert.deepEqual(hidden.vertices,scene.vertices.slice(0,-gobos.length*9));
state=undefined;assert.deepEqual(run('sceneForPhoto()'),hidden);
if(process.argv.includes('--json'))console.log(JSON.stringify({wire:Buffer.from(wire).toString('base64'),scene,counts}));
else console.log(`Gobo scene: ${scene.material_ids.length} triangles, ${scene.pbr.length/16} materials; exact preview/Phong partition, photo/live wire, front/back rows, legacy and limits passed`);



// Exact production scene/camera functions: arbitrary Room-only changes must not
// leak into any photo/live fixture triangles, materials, or camera conversion.
const base={controls:Object.fromEntries(Object.entries(controls).map(([id,el])=>[id,['window-glass','backdrop-sweep','checker'].includes(id)?el.checked:el.value])),
 poses:createLightingPoseState(initial.map((p,i)=>({id:STUDIO_LIGHTS[i].id,pos:p.pos,dir:p.dir,aimAtSubject:false}))),
 backdrops:copy(layers),table:copy(tableState),scale:{version:1,mode:'fixed',unitMm:10},gobos:{version:1,gobos:[]}};
let roomProfile=copy(base);
context.resolveLightingPoses=resolveLightingPoses;
context.studioLighting.renderState=channel=>channel==='exposure'?base:roomProfile;
const protectedScene=run('sceneForPhoto()'),protectedCamera=run('studioPhotoCamera()');
const previewBefore={lights:run('computeFixtureRoles()'),room:run('checkerboardFaces()'),table:run('tableFaces()'),backdrop:run('backdropRollFaces()')};
roomProfile.controls['key-filter-left']='#123456';roomProfile.controls['light-strength-left']=999;
roomProfile.controls['key-shape']='bare';roomProfile.poses.poses[0].pos=[-8,6,7];
roomProfile.controls['room-scale']=5;roomProfile.controls['wall-material-a']='black_felt';
roomProfile.table.width=9;roomProfile.table.material='black_felt';roomProfile.backdrops[0].width=13;
roomProfile.scale.unitMm=20;roomProfile.gobos={version:1,gobos:[{id:42,enabled:true,shape:'rectangle',width:2,height:3,x:1,y:2,z:3,pitch:0,yaw:0,roll:0,front:'black_felt',back:'white_plaster'}]};
assert.deepEqual(run('sceneForPhoto()'),protectedScene);assert.deepEqual(run('studioPhotoCamera()'),protectedCamera);
assert.notDeepEqual(run('computeFixtureRoles()'),previewBefore.lights);
assert.notDeepEqual(run('checkerboardFaces()'),previewBefore.room);assert.notDeepEqual(run('tableFaces()'),previewBefore.table);
assert.notDeepEqual(run('backdropRollFaces()'),previewBefore.backdrop);assert.ok(run('goboFaces().length')>0);
if(!process.argv.includes('--json'))console.log('Exact scene/camera isolation: room-only filters, strength, pose, shape, scale, table, backdrop, walls and gobos change Phong inputs with byte-identical ray payload');

const extentPoses=createLightingPoseState(initial.map((role,i)=>({...role,id:STUDIO_LIGHTS[i].id,aimAtSubject:true})));
const underExtent=extentPoses.poses.find(p=>p.id==='underlight');underExtent.pos=[0,-2,0];underExtent.aimAtSubject=false;underExtent.pitch=90;
context.extentPoses=extentPoses;
const actualUnder=run('buildFixtureFaces([computeFixtureRoles([0,0,0],"exposure",extentPoses).fixtureRoles[9]],0)');
const underY=actualUnder.flatMap(f=>f.slice(0,3).map(p=>p[1]));
assert.ok(Math.min(...underY)<-2,'Actual rear housing contributes to lower bound');
assert.ok(Math.max(...underY)>-2,'Actual front grid contributes to upper bound');
underExtent.pitch=0;
const tiltedUnder=run('buildFixtureFaces([computeFixtureRoles([0,0,0],"exposure",extentPoses).fixtureRoles[9]],0)');
const tiltedY=tiltedUnder.flatMap(f=>f.slice(0,3).map(p=>p[1]));
assert.ok(Math.max(...tiltedY)-Math.min(...tiltedY)>Math.max(...underY)-Math.min(...underY),'Tilted grid changes physical height extent');
if(!process.argv.includes('--json'))console.log('Actual pose-override geometry includes housing/grid extents and changes with aim');


const sizeBefore=run('computeFixtureRoles([0,0,0],"exposure")');
context.customSizes=defaultFixtureSizes();
const doubled=run('computeFixtureRoles([0,0,0],"exposure",null,customSizes)');
for(const i of [0,1,5,6,9]){const size=customSize(i);assert.equal(doubled.fixtureRoles[i].halfW,sizeBefore.fixtureRoles[i].halfW*size.width);assert.equal(doubled.fixtureRoles[i].halfH,sizeBefore.fixtureRoles[i].halfH*size.height);}
function customSize(i){return context.customSizes.lights[i];} // Includes the real table-fitted underlight dimensions.
const oldSizeFaces=run('buildFixtureFaces(computeFixtureRoles([0,0,0],"exposure").fixtureRoles,99)');
const newSizeFaces=run('buildFixtureFaces(computeFixtureRoles([0,0,0],"exposure",null,customSizes).fixtureRoles,99)');
assert.equal(newSizeFaces.length,oldSizeFaces.length);assert.notDeepEqual(newSizeFaces,oldSizeFaces);
assert.deepEqual(doubled.roleStrength,sizeBefore.roleStrength,'Sizing does not rewrite power sliders');
context.customSizes.lights[3].snoot=false;
assert.equal(run('buildFixtureFaces(computeFixtureRoles([0,0,0],"exposure",null,customSizes).fixtureRoles,99).length'),oldSizeFaces.length-24);
if(!process.argv.includes('--json'))console.log('Actual fixture dimensions reach shared emitter/housing/grid geometry; kicker snoot removes exactly one tube');

const oldGridScene=run('sceneForPhoto()');
base.fixtureSizes=defaultFixtureSizes(true);base.fixtureSizes.lights[9].gridCellsX=32;base.fixtureSizes.lights[9].gridCellsY=32;
const denseGridScene=run('sceneForPhoto()');
assert.equal(denseGridScene.material_ids.length,oldGridScene.material_ids.length+600);
assert.ok(denseGridScene.material_ids.length<=5000,'Maximum grid plus full existing room/gobo fixture stays inside original scene budget');
base.fixtureSizes.lights[9].grid=false;
assert.equal(run('sceneForPhoto().material_ids.length'),oldGridScene.material_ids.length-240);
if(!process.argv.includes('--json'))console.log('Maximum 32x32 grid retains full-scene triangle budget; grid off removes only its 240 grid/apron triangles');

// Verify new wall-relative defaults with the actual shared emitter/housing mesh.
context.studioLighting=null;context.wallRelativeFixtureDefaults=wallRelativeFixtureDefaults;
context.defaultFixtureSizes=defaultFixtureSizes;context.defaultLightTools=defaultLightTools;context.defaultStudioScale=defaultStudioScale;
const newRoles=run('computeFixtureRoles().fixtureRoles'),newBounds=studioRoomBounds(Number(controls["room-scale"].value),tableFloorDrop(defaultStudioTable()));
for(const i of [0,1,2,3,4,5]){
 context.testRole=newRoles[i];const vertices=run('buildFixtureFaces([testRole],0)').flatMap(f=>f.slice(0,3));
 assert.ok(vertices.every(p=>p[0]>=newBounds.leftX+.019999&&p[0]<=newBounds.rightX-.019999&&p[2]>=newBounds.backZ+.019999&&p[2]<=newBounds.frontZ-.019999),'Whole default housing fits horizontal room bounds');
 if(i>=2)assert.ok(Math.min(...vertices.map(p=>p[1]))>=newBounds.floorY+.019999,'Floor fixture rests above floor');
}
assert.ok(newRoles[0].pos[2]>7&&newRoles[1].pos[2]>7&&newRoles[5].pos[2]>7,'Keys/lower fill moved towards camera wall');
for(const i of [3,4])assert.ok(Math.abs(Math.abs(newRoles[i].pos[0]/newRoles[i].pos[2])-1)<1e-9,'Kickers start45 degrees from back axis');
assert.ok(newRoles[2].pos[2]<-7,'Background moved near back wall');
if(!process.argv.includes('--json'))console.log('New default keys/fill/back group use actual mesh wall/floor fit, with45-degree kickers');


// Large overhead/lower defaults are physical emitter dimensions, not UI-only factors.
assert.equal(newRoles[6].halfW*2,8);assert.equal(newRoles[6].halfH*2,8);
assert.equal(newRoles[5].halfW*2,4);assert.equal(newRoles[5].halfH*2,2);
context.testRole=newRoles[6];const overheadVertices=run('buildFixtureFaces([testRole],0)').flatMap(f=>f.slice(0,3));
assert.ok(overheadVertices.every(p=>p[1]>=1-.000001&&p[1]<=newBounds.ceilingY+.000001),'Whole overhead housing/grid stays above subject and below ceiling');
assert.ok(overheadVertices.every(p=>p[0]>=newBounds.leftX&&p[0]<=newBounds.rightX&&p[2]>=newBounds.backZ&&p[2]<=newBounds.frontZ),'Whole overhead fits room');
if(!process.argv.includes('--json'))console.log('Overhead 8x8 and lower fill 4x2 actual emitters; overhead housing/deep grid clears subject/ceiling/room');

for(const i of [3,4])assert.ok(Math.abs(newRoles[i].pos[1]-(-1+newRoles[i].radius+.02))<1e-12,'New kicker entire emitter disk clears tabletop with0.02units margin');
