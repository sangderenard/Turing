import {installScenePackerInputs,installStudioSceneBuilders} from './meowdoku_scene_packer_harness.mjs';
import {applyLightTools,defaultLightTools,migrateLightTools} from '../examples/meowdoku/web/marble_light_tools.js';
import {applyFixtureSizes,defaultFixtureSizes} from "../examples/meowdoku/web/marble_fixture_sizes.js";
// Production emitter faces, independent poses and the actual MRP1 upload.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
import {tableLightRoles,tableLightModifierFaces,studioPlacementWarnings} from '../examples/meowdoku/web/marble_table_lights.js';
import {TABLE_TOP_Y,TABLE_THICKNESS,defaultStudioTable,tableFloorDrop} from '../examples/meowdoku/web/marble_studio_table.js';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
import {createLightingPoseState,resolveLightingPoses} from '../examples/meowdoku/web/marble_lighting.js';
import {OBJECT_CENTER,packMarblePhoto} from '../examples/meowdoku/web/marble_photo.js';
const source=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const html=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.html',import.meta.url),'utf8');
const nodes={};
for(const match of html.matchAll(/<input\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
  const attributes=Object.fromEntries(Array.from(match[1].matchAll(/([\w-]+)="([^"]*)"/g),a=>[a[1],a[2]]));nodes[match[2]]=attributes;
}
nodes['key-shape']={value:'snoot'};
const context=vm.createContext({applyLightTools,defaultLightTools:migrateLightTools,wallRelativeFixtureDefaults:roles=>roles,applyFixtureSizes,defaultFixtureSizes:()=>defaultFixtureSizes(true),$:id=>nodes[id],STUDIO_LIGHTS,tableLightRoles,tableLightModifierFaces,OBJECT_CENTER,
  FIXTURE_MAT_BASE:38,studioLighting:null,studioTableState:()=>({enabled:false}),lightStandFacesForStudio:()=>[],
  rgb:v=>[1,3,5].map(k=>parseInt(v.slice(k,k+2),16)/255),
  tables:{pbr:{table:{floats_per_record:16},data:[]}},tableFaces:()=>[], goboFaces:()=>[],checkerboardFaces:()=>[],backdropRollFaces:()=>[],studioSubjectTransform:()=>({subjectCenter:[0,0,0]}),studioPhotoScale:()=>.025});
installStudioSceneBuilders(context);
vm.runInContext(source.slice(source.indexOf('function studioRenderState('),source.indexOf('// Capture actual page values;')),context);
installScenePackerInputs(context);
const run=code=>JSON.parse(JSON.stringify(vm.runInContext(code,context,{timeout:1000})));
const rig=run('computeFixtureRoles()');
assert.equal(rig.fixtureRoles.length,10);
assert.deepEqual(rig.fixtureRoles[2].pos,[0,-.95,-2.22]);
assert.deepEqual(rig.fixtureRoles[3].pos,[-1.92,-.95,-3.12]);
assert.deepEqual(rig.fixtureRoles[4].pos,[1.92,-.95,-3.12]);
const additions=rig.fixtureRoles.slice(7);
assert.ok(Math.abs(additions[2].pos[1] - (TABLE_TOP_Y - TABLE_THICKNESS - .48)) < 1e-12,
  'Fresh underlight is 0.20 Studio units lower than the former .28 clearance');
const sub=(a,b)=>a.map((v,i)=>v-b[i]),dot=(a,b)=>a.reduce((n,v,i)=>n+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
function rayHit(origin,direction,face) {
  const [a,b,c]=face,e1=sub(b,a),e2=sub(c,a),h=cross(direction,e2),det=dot(e1,h);
  if(Math.abs(det)<1e-10)return false;
  const s=sub(origin,a),u=dot(s,h)/det;if(u<0||u>1)return false;
  const q=cross(s,e1),v=dot(direction,q)/det;if(v<0||u+v>1)return false;
  return dot(e2,q)/det>1e-8;
}
for(const role of rig.fixtureRoles) {
  context.role=role;const faces=run('buildFixtureFaces([role],48)'),emitter=faces.filter(f=>f[3]===role.mat),housing=faces.filter(f=>f[3]===48);
  assert.equal(emitter.length,role.round?12:2);
  assert.ok(housing.length>=10);
  if(role.roleIndex>=7)assert.ok(housing.length>12);
  for(const [a,b,c,,normals] of emitter) {
    assert.ok(dot(cross(sub(b,a),sub(c,a)),role.dir)>0,'ray winding follows the declared optical direction');
    assert.deepEqual(normals[0],role.dir);
  }
  if(role.tableStand) {
    assert.ok(Math.abs(Math.min(...faces.flatMap(f=>f.slice(0,3).map(p=>p[1])))-TABLE_TOP_Y)<1e-12);
    assert.ok(!housing.some(f=>rayHit(role.pos,role.dir,f)),'snoot remains open along its aim');
    const sideways=[-role.dir[2],0,role.dir[0]].map((v,i)=>v+.2*role.dir[i]);
    assert.ok(housing.some(f=>rayHit(role.pos,sideways,f)),'snoot blocks off-axis light');
  } else if(role.grid) {
    assert.equal(role.shape,'grid');assert.equal(role.pos[0],0);assert.equal(role.pos[2],0);assert.deepEqual(role.dir,[0,1,0]);
    assert.ok(Math.max(...faces.flatMap(f=>f.slice(0,3).map(p=>p[1])))<TABLE_TOP_Y-TABLE_THICKNESS);
    const grid=tableLightModifierFaces(role,48);
    assert.equal(grid.length,(7+9+4)*12);
    for(const [a,b,c,,ns] of grid)assert.ok(dot(cross(sub(b,a),sub(c,a)),ns[0])>0);
    const cellCenter=[.15,role.pos[1],.15];
    assert.ok(!grid.some(f=>rayHit(cellCenter,[0,1,0],f)),'grid cells are genuinely open');
    assert.ok(grid.some(f=>rayHit(cellCenter,[1,.1,0],f)),'opaque grid bars stop shallow rays');
    for(const point of [[role.halfH*1.12,role.pos[1]-.01,.15],[.15,role.pos[1]-.01,role.halfW*1.12]])
      assert.ok(grid.some(f=>rayHit(point,[0,1,0],f)),'Opaque apron blocks former emitter-to-can spill gap');
    const crackOrigin=[role.halfH*1.1,role.pos[1]+.001,.15];
    assert.ok(grid.some(f=>rayHit(crackOrigin,[1,0,0],f)),'Apron seals the former .002 stand-off crack');
  }
}
const poseState=createLightingPoseState(additions.map((role,i)=>({...role,id:STUDIO_LIGHTS[i+7].id,aimAtSubject:true})));
const poseIds=STUDIO_LIGHTS.slice(7).map(l=>l.id);
for(const target of [[0,0,0],[.3,.5,-.4]])resolveLightingPoses(poseState,poseIds,target).forEach(p=>{
  const delta=sub(target,p.pos),n=Math.hypot(...delta);p.dir.forEach((v,i)=>assert.ok(Math.abs(v-delta[i]/n)<1e-12));
});
poseState.poses[0].aimAtSubject=false;poseState.poses[0].yaw=90;poseState.poses[0].pitch=0;
assert.ok(resolveLightingPoses(poseState,poseIds,[2,3,4])[0].dir[0]>.999999);
// All roles active: final three must not disappear behind an old seven/eight
// emitter bound, and material slots must not overlap housing or public rows.
for(const light of STUDIO_LIGHTS)nodes[light.strengthId].value='10';
const scene=run('sceneForPhoto()');
assert.equal(scene.pbr.length,11*16);assert.equal(new Set(rig.fixtureRoles.map(r=>r.mat)).size,10);
assert.deepEqual(rig.fixtureRoles.map(r=>r.mat),[38,39,40,41,42,43,44,45,46,47]);
assert.ok(scene.material_ids.length<5000);
for(let i=0;i<10;i++){assert.ok(scene.material_ids.includes(i));assert.deepEqual(scene.pbr.slice(i*16+8,i*16+11),[1,1,1]);}
assert.deepEqual(scene.pbr.slice(10*16+8,10*16+11),[0,0,0]);
const subject=[[0,0,0],[1,0,0],[0,1,0],0,[[0,0,1],[0,0,1],[0,0,1]]];
const wire=packMarblePhoto({interior:[],glass:[subject]},'a'.repeat(64),{fixtures:scene});
const view=new DataView(wire),header=JSON.parse(new TextDecoder().decode(new Uint8Array(wire,16,view.getUint32(4,true))));
assert.deepEqual(header.fixtures,scene);
// The simple height editor writes this same pose state. Preview roles and the
// actual photo payload must translate the emitter identically, not a UI-only Y.
const movedPoses=createLightingPoseState(rig.fixtureRoles.map((role,i)=>({
  id:STUDIO_LIGHTS[i].id,pos:role.pos,dir:role.dir,aimAtSubject:i===9})));
const oldUnderY=movedPoses.poses[9].pos[1];movedPoses.poses[9].pos[1]=-2.15;
context.studioLighting={poses:{getPoses:()=>resolveLightingPoses(movedPoses,STUDIO_LIGHTS.map(l=>l.id))},
  strengths:{getStrengths:()=>STUDIO_LIGHTS.map(()=>1)}};
const movedRig=run('computeFixtureRoles()'), movedScene=run('sceneForPhoto()');
assert.equal(movedRig.fixtureRoles[9].pos[1],-2.15);
assert.deepEqual(movedRig.fixtureRoles[9].dir,[0,1,0]);
for(let i=0;i<scene.material_ids.length;i++)if(scene.material_ids[i]===9) {
  for(let k=0;k<9;k++)assert.ok(Math.abs(movedScene.vertices[i*9+k]-scene.vertices[i*9+k]-
    (k%3===1?(-2.15-oldUnderY)*.025:0))<1e-12,'Photo emitter follows preview vertical translation');
}
const movedWire=packMarblePhoto({interior:[],glass:[subject]},'a'.repeat(64),{fixtures:movedScene});
const movedView=new DataView(movedWire);
assert.deepEqual(JSON.parse(new TextDecoder().decode(new Uint8Array(movedWire,16,movedView.getUint32(4,true)))).fixtures,movedScene);
const beforePlacement=JSON.parse(JSON.stringify(movedPoses));
for(const [index,pos] of [[7,[-1.2,-.72,.9]],[8,[2,-.72,-.5]],[6,[0,0,.3]]])movedPoses.poses[index].pos=pos;
const placedRig=run('computeFixtureRoles()'),placedScene=run('sceneForPhoto()');
for(const index of [6,7,8]) {
  assert.deepEqual(placedRig.fixtureRoles[index].pos,movedPoses.poses[index].pos);
  assert.deepEqual(placedRig.fixtureRoles[index].dir,movedRig.fixtureRoles[index].dir,'Manual direction is unchanged by placement');
  const delta=sub(movedPoses.poses[index].pos,beforePlacement.poses[index].pos);
  for(let i=0;i<scene.material_ids.length;i++)if(scene.material_ids[i]===index)for(let k=0;k<9;k++)
    assert.ok(Math.abs(placedScene.vertices[i*9+k]-movedScene.vertices[i*9+k]-delta[k%3]*.025)<1e-12,
      'Stage XZ / top-fill Y positions translate the same real photo emitters');
}
const warningPoses=resolveLightingPoses(movedPoses,STUDIO_LIGHTS.map(l=>l.id));
const table=({...defaultStudioTable(),width:5.2,length:3,height:3.02,material:"marble_clear_glass",skirt:{enabled:false,material:"black_felt"}}),bounds=studioRoomBounds(3,tableFloorDrop(table));
let warnings=studioPlacementWarnings(table,warningPoses,bounds);
assert.match(warnings['fill-upper'],/overlaps.*subject/);
warningPoses[6].pos=[3,0,.3];warningPoses[7].pos=[2.5,-.72,0];
warnings=studioPlacementWarnings(table,warningPoses,bounds);
assert.equal(warnings['fill-upper'],undefined);
assert.match(warnings['table-left'],/footprint/,'Whole .28-wide stand matters, not just its centre');
warningPoses[7].pos=[2.4,-.72,1.4];
assert.match(studioPlacementWarnings(table,warningPoses,bounds)['table-left'],/footprint/,'Whole .24-deep base matters');
warningPoses[7].pos=[2.4,-.72,1.3];
assert.equal(studioPlacementWarnings(table,warningPoses,bounds)['table-left'],undefined);
warningPoses[7].pos[1]=-.7;assert.match(studioPlacementWarnings(table,warningPoses,bounds)['table-left'],/floats/);
warningPoses[7].pos[1]=-.8;assert.match(studioPlacementWarnings(table,warningPoses,bounds)['table-left'],/intersects/);
assert.match(studioPlacementWarnings({...table,enabled:false},warningPoses,bounds)['table-left'],/no tabletop support/);
const beforeWarning=JSON.parse(JSON.stringify(warningPoses));studioPlacementWarnings(table,warningPoses,bounds);
assert.deepEqual(warningPoses,beforeWarning,'Advisory checks never mutate authored geometry');
if(process.argv.includes('--json'))console.log(JSON.stringify({scene,wire:Buffer.from(wire).toString('base64'),directions:rig.fixtureRoles.map(r=>r.dir)}));
else console.log('Table lights: two resting/open snoots, rectangular opaque grid, all10 ray normals, material routes and MRP1 payload passed');
