import {installScenePackerInputs,installStudioSceneBuilders} from './meowdoku_scene_packer_harness.mjs';
import {applyLightTools,defaultLightTools,migrateLightTools} from '../examples/meowdoku/web/marble_light_tools.js';
import {applyFixtureSizes,defaultFixtureSizes} from "../examples/meowdoku/web/marble_fixture_sizes.js";
import {defaultStudioScale,studioScaleTransform} from '../examples/meowdoku/web/marble_studio_scale.js';
// Actual furniture -> sceneForPhoto -> MRP1, plus closed geometry and can rays.
// CPU-only: no GPU job, live server, saved object mutation or publication.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {defaultStudioTable, validateStudioTable, studioTableFaces, tableFloorDrop,
  FLOOR_FIXTURE_INDICES, TABLE_TOP_Y, TABLE_THICKNESS} from '../examples/meowdoku/web/marble_studio_table.js';
import {studioGoboFaces} from '../examples/meowdoku/web/marble_studio_gobos.js';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
import {catEyeMarble} from '../examples/meowdoku/web/marble_mesh.js';
import {geometrySha256} from '../examples/meowdoku/web/marble_object.js';
import {OBJECT_CENTER, packMarblePhoto, photoCameraFromStudio, validatePhotoFocus} from '../examples/meowdoku/web/marble_photo.js';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
import {tableLightRoles, tableLightModifierFaces} from '../examples/meowdoku/web/marble_table_lights.js';
import {transparentTriangleOrder, triangleCentres} from '../examples/meowdoku/web/marble_glass_preview.js';
const source=fs.readFileSync(process.env.STUDIO_TEST_WEB_ROOT?`${process.env.STUDIO_TEST_WEB_ROOT}/marble_studio.js`:new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const html=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.html',import.meta.url),'utf8');
const copy=x=>JSON.parse(JSON.stringify(x)), near=(a,b)=>assert(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const ids={marble_clear_glass:1,white_plaster:2,marble_milk_glass:3};
let state=({...defaultStudioTable(),width:5.2,length:3,height:3.02,material:"marble_clear_glass",skirt:{enabled:false,material:"black_felt"}});
assert.deepEqual(validateStudioTable({version:1,enabled:true,material:'marble_clear_glass'},Object.keys(ids)),{...state,material:'marble_clear_glass'},
  'Existing v1 furniture keeps its original dimensions');
assert.equal(validateStudioTable(undefined,Object.keys(ids)).enabled,false);
for(const value of [null,{}, {...state,version:2},{...state,enabled:1},{...state,material:'missing'}])
  assert.throws(()=>validateStudioTable(value,Object.keys(ids)));
for(const key of ['width','length','height'])for(const bad of [null,true,'3',NaN,Infinity,-1,0,21])
  assert.throws(()=>validateStudioTable({...state,[key]:bad},Object.keys(ids)));
assert.deepEqual(studioTableFaces({...state,enabled:false},ids),[]);
const bounds=studioRoomBounds(3,tableFloorDrop(state)), table=studioTableFaces(state,ids,bounds.floorY);
assert.equal(table.length,60);
assert.equal(Math.min(...table.flatMap(f=>f.slice(0,3).map(p=>p[1]))),bounds.floorY);
assert.equal(Math.max(...table.flatMap(f=>f.slice(0,3).map(p=>p[1]))),TABLE_TOP_Y);
for(let start=0;start<table.length;start+=12) {
  const solid=table.slice(start,start+12),edges=new Map(); let volume=0;
  for(const [a,b,c,id,ns] of solid) {
    assert.equal(id,ids.marble_clear_glass);
    assert(dot(cross(sub(b,a),sub(c,a)),ns[0])>0);
    volume+=dot(a,cross(b,c))/6;
    for(const [p,q] of [[a,b],[b,c],[c,a]]) {
      const keys=[p.join(','),q.join(',')],key=keys.slice().sort().join('|'),prev=edges.get(key)||[0,0];
      edges.set(key,[prev[0]+1,prev[1]+(keys[0]<keys[1]?1:-1)]);
    }
  }
  assert(volume>0);for(const edge of edges.values())assert.deepEqual(edge,[2,0]);
}
for(const f of table.slice(12))for(const p of f.slice(0,3))assert(p[1]<=TABLE_TOP_Y-TABLE_THICKNESS);

const controls={'room-scale':{value:'3'},'radius-mm':{value:'99'}};
for(const m of html.matchAll(/<input\b([^>]*\bid="([^"]+)"[^>]*)>/g))
  controls[m[2]]={value:/value="([^"]*)"/.exec(m[1])?.[1]??'',checked:/\bchecked\b/.test(m[1])};
for(const m of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
  const opts=[...m[2].matchAll(/<option([^>]*)>/g)],opt=opts.find(o=>o[1].includes('selected'))??opts[0];
  controls[m[1]]={value:opt?/value="([^"]*)"/.exec(opt[1])?.[1]??'':''};
}
controls['radius-mm'].value='99'; // saved radius, not later edited DOM, owns photo units
const pbr=new Float32Array(3*16);pbr.set([1,1,1,.01,0,1,1.52,0],16);
const context=vm.createContext({applyLightTools,defaultLightTools:migrateLightTools,wallRelativeFixtureDefaults:roles=>roles,applyFixtureSizes,defaultFixtureSizes:()=>defaultFixtureSizes(true),$:id=>controls[id],ids,OBJECT_CENTER,STUDIO_LIGHTS,studioRoomBounds,
  photo:{applyLensCamera:camera=>camera},
  defaultStudioTable,studioTableFaces,tableFloorDrop,FLOOR_FIXTURE_INDICES,studioGoboFaces,
  defaultStudioScale,studioScaleTransform,studioOrbitTarget:[0,0,0],savedPhotoRadius:.01,studioLighting:{table:{getState:()=>state},scale:{getState:()=>({version:1,mode:"legacy"})}},photoCameraFromStudio,yaw:-.5,pitch:.2,distance:3.45,
  tables:{pbr:{table:{floats_per_record:16},data:pbr}},
  checkerboardFaces:()=>[],backdropRollFaces:()=>[],
  computeFixtureRoles:()=>({fixtureRoles:[],roleColor:STUDIO_LIGHTS.map(()=>[1,1,1]),roleStrength:STUDIO_LIGHTS.map(()=>0)}),
  buildFixtureFaces:()=>[],
});
installStudioSceneBuilders(context);
vm.runInContext(source.slice(source.indexOf('function studioRenderState('),source.indexOf('// Capture actual page values;')),context);
vm.runInContext(source.slice(source.indexOf('function studioRoomScale('),source.indexOf('function checkerboardFaces(')),context);
vm.runInContext(source.slice(source.indexOf('function perspective('),source.indexOf('const loc =')),context);
vm.runInContext(source.slice(source.indexOf('export function sceneForPhoto('),source.indexOf('\nfunction syncMaterial()')).replace('export function','function'),context);
context.shopLightFacesForStudio=()=>[]; // This focused fixture predates shop/rack material rows.
installScenePackerInputs(context);
const run=code=>copy(vm.runInContext(code,context,{timeout:1000}));
const geometry=catEyeMarble({glass:0,eye:0,accent:0},{kind:'clear',rings:8,segments:12});
assert.equal(Math.min(...geometry.glass.flatMap(f=>f.slice(0,3).map(p=>p[1]))),-1);
const geometryHash=await geometrySha256(geometry),cases=[];
for(const radiusMm of [10,25]) {
  context.savedPhotoRadius=radiusMm/1000;
  const scale=run('studioPhotoScale()'),scene=run('sceneForPhoto()');near(scale,radiusMm/1000);
  assert.deepEqual(scene.vertices,table.flatMap(f=>f.slice(0,3).flatMap(p=>p.map((x,i)=>OBJECT_CENTER[i]+x*scale))));
  const topY=Math.max(...scene.vertices.filter((_,i)=>i%3===1));near(topY,OBJECT_CENTER[1]-scale);
  const tableId=scene.material_ids[0];assert.deepEqual(scene.pbr.slice(tableId*16,tableId*16+16),Array.from(pbr.slice(16,32)));
  const camera=run('studioPhotoCamera()');near(camera.camera.focus_distance_m,3.45*scale);
  near(camera.camera.fov_degrees,360/7);
  const rasterFocal=run('Array.from(perspective(1))[5]');
  const photoFocal=1/Math.tan(camera.camera.fov_degrees*Math.PI/360);
  assert(Math.abs(photoFocal-rasterFocal)<1e-7,'Same vertical projection as the actual raster perspective');
  assert(photoFocal/3.45<1,'The default marble fits vertically without cropping');
  near(Math.hypot(...camera.camera.eye_m.map((v,i)=>v-OBJECT_CENTER[i]))/scale,3.45);
  const wire=packMarblePhoto(geometry,'a'.repeat(64),{...camera,fixtures:scene});
  cases.push({radiusMm,geometryHash,wire:Buffer.from(wire).toString('base64')});
}
for(const dimensions of [{width:2,length:2,height:.3},{width:8,length:5,height:5},{width:20,length:20,height:10}]) {
  state={...({...defaultStudioTable(),width:5.2,length:3,height:3.02,material:"marble_clear_glass",skirt:{enabled:false,material:"black_felt"}}),...dimensions};
  const resizedFloor=studioRoomBounds(3,tableFloorDrop(state)).floorY;
  const resized=studioTableFaces(state,ids),scene=run('sceneForPhoto()'),scale=run('studioPhotoScale()');
  assert.equal(resized.length,60);
  near(Math.min(...resized.flatMap(f=>f.slice(0,3).map(p=>p[1]))),resizedFloor);
  near(TABLE_TOP_Y-resizedFloor,dimensions.height);
  assert.equal(Math.max(...resized.flatMap(f=>f.slice(0,3).map(p=>p[1]))),-1,'Guaranteed marble-top contact');
  assert.equal(Math.max(...resized.flatMap(f=>f.slice(0,3).map(p=>p[0]))),dimensions.width/2);
  assert.equal(Math.max(...resized.flatMap(f=>f.slice(0,3).map(p=>p[2]))),dimensions.length/2);
  for(let start=12;start<60;start+=12) {
    const points=resized.slice(start,start+12).flatMap(f=>f.slice(0,3));
    const min=axis=>Math.min(...points.map(p=>p[axis])),max=axis=>Math.max(...points.map(p=>p[axis]));
    near(max(0)-min(0),.18);near(max(2)-min(2),.18);
    near(Math.abs((max(0)+min(0))/2),dimensions.width/2-.25);
    near(Math.abs((max(2)+min(2))/2),dimensions.length/2-.25);
    near(min(1),resizedFloor);near(max(1),TABLE_TOP_Y-TABLE_THICKNESS);
  }
  assert.deepEqual(scene.vertices,resized.flatMap(f=>f.slice(0,3).flatMap(p=>p.map((x,i)=>OBJECT_CENTER[i]+x*scale))),
    'Resized preview geometry is the exact photograph/live scene geometry');
  near(run('studioPhotoCamera().camera.focus_distance_m'),3.45*scale);
  assert.equal(await geometrySha256(geometry),geometryHash,'Furniture never changes the saved subject');
}
state={...({...defaultStudioTable(),width:5.2,length:3,height:3.02,material:"marble_clear_glass",skirt:{enabled:false,material:"black_felt"}}),enabled:false};near(run('studioPhotoScale()'),.11/3.45);assert.deepEqual(run('tableFaces()'),[]);
assert.equal(run('studioPhotoCamera().camera.fov_degrees'),18);
near(photoCameraFromStudio(0,0,3.45).camera.focus_distance_m,.11);
const macro=photoCameraFromStudio(0,0,3.45,.001);
assert.doesNotThrow(()=>validatePhotoFocus(macro.camera),'3.45 mm analytical focus no longer hits the former 20 mm app floor');
assert.doesNotThrow(()=>packMarblePhoto(geometry,'a'.repeat(64),macro));
assert.throws(()=>validatePhotoFocus({...macro.camera,focus_distance_m:.0009}),/numerical range/);
assert.doesNotThrow(()=>packMarblePhoto(geometry,'a'.repeat(64)),'Built-in framing remains available at small radii');
assert.doesNotThrow(()=>validatePhotoFocus(photoCameraFromStudio(0,0,20,.001).camera));
assert(source.includes('getPose:()=>validatePhotoFocus('),'Live camera-only updates also validate focus before posting');

// Execute the actual combined-table/subject transparent mesh path. One shared
// world-space sort includes both objects, and cached vertices rebuild on pose.
const identity=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
context.sceneGlassKey=null;context.sceneGlass=null;context.tableTransparentFaces=table;context.standTransparentFaces=[];context.shopTransparentFaces=[];
context.marbleTransparentFaces=geometry.glass;context.dispose=()=>{};
context.transparentMesh=faces=>({faces});
context.transformPoint=(m,[x,y,z])=>[m[0]*x+m[4]*y+m[8]*z+m[12],m[1]*x+m[5]*y+m[9]*z+m[13],m[2]*x+m[6]*y+m[10]*z+m[14]];
context.testModel=identity;
const combined=vm.runInContext('transparentSet(testModel)',context);
assert.equal(combined.faces.length,table.length+geometry.glass.length);
assert.deepEqual(copy(combined.faces.slice(0,table.length)),table);
assert.equal(vm.runInContext('transparentSet(testModel)',context),combined,'Static geometry is cached');
context.testModel=identity.map((v,i)=>i===12?v+.5:v);
const moved=vm.runInContext('transparentSet(testModel)',context);
assert.notEqual(moved,combined);
near(moved.faces[table.length][0][0],geometry.glass[0][0][0]+.5);
const centres=triangleCentres(moved.faces),order=transparentTriangleOrder(centres,identity);
for(let i=3;i<order.length;i+=3)assert(centres[order[i-3]/3][2]<=centres[order[i]/3][2]);

// The production background can encloses the same nonzero emitter. Its open
// forward tube blocks off-axis camera rays while the axial wall beam escapes.
const fixtureContext=vm.createContext({applyLightTools,defaultLightTools:migrateLightTools,wallRelativeFixtureDefaults:roles=>roles,applyFixtureSizes,defaultFixtureSizes:()=>defaultFixtureSizes(true),$:id=>controls[id],STUDIO_LIGHTS,
  FIXTURE_MAT_BASE:10,studioLighting:null,studioTableState:()=>({...defaultStudioTable(),width:5.2,length:3,height:3.02,material:"marble_clear_glass",skirt:{enabled:false,material:"black_felt"}}),
  studioRoomBounds,studioRoomScale:()=>3,studioFloorDrop:()=>3,FLOOR_FIXTURE_INDICES,tableLightRoles,tableLightModifierFaces,
  rgb:value=>[1,3,5].map(k=>parseInt(value.slice(k,k+2),16)/255)});
installStudioSceneBuilders(fixtureContext);
vm.runInContext(source.slice(source.indexOf('function studioRenderState('),source.indexOf('// Capture actual page values;')),fixtureContext);
const roles=copy(vm.runInContext('computeFixtureRoles().fixtureRoles',fixtureContext));
const background=roles[2];assert.equal(background.shroudLength,1.2);assert.equal(background.gobo,true);
fixtureContext.testRole=background;
const can=copy(vm.runInContext('buildFixtureFaces([testRole],99)',fixtureContext));
const housing=can.filter(f=>f[3]===99),emitter=can.filter(f=>f[3]===background.mat);
assert.equal(emitter.length,12);assert(emitter.every(f=>f[5]===background.mat));
const projected=housing.flatMap(f=>f.slice(0,3).map(p=>dot(sub(p,background.pos),background.dir)));
near(Math.max(...projected),1.2);near(Math.min(...projected),-.3*1.4);
function hit(direction) {
  for(const [a,b,c] of housing) {
    const e1=sub(b,a),e2=sub(c,a),p=cross(direction,e2),det=dot(e1,p);
    if(Math.abs(det)<1e-12)continue;
    const tvec=sub(background.pos,a),u=dot(tvec,p)/det;if(u<0||u>1)continue;
    const q=cross(tvec,e1),v=dot(direction,q)/det;if(v<0||u+v>1)continue;
    if(dot(e2,q)/det>1e-6)return true;
  }
  return false;
}
assert.equal(hit(background.dir),false,'Open forward axis preserves the background beam');
assert.equal(hit(sub([0,1,3.45],background.pos)),true,'Back-facing camera ray meets opaque housing');
assert.equal(hit(sub([3,0,-2],background.pos)),true,'Off-axis source ray meets extended side wall');
for(const index of FLOOR_FIXTURE_INDICES) {
  fixtureContext.testRole=roles[index];
  const faces=copy(vm.runInContext('buildFixtureFaces([testRole],99)',fixtureContext));
  const low=Math.min(...faces.flatMap(f=>f.slice(0,3).map(p=>p[1])));
  if(roles[index].defaultFloorHeight!==undefined) {
    near(roles[index].pos[1],bounds.floorY+roles[index].defaultFloorHeight);
    assert(low>bounds.floorY,'Wall-relative kickers preserve their authored raised height above the real floor');
  } else near(low,bounds.floorY+.02);
}
if(process.argv.includes('--json'))console.log(JSON.stringify(cases));
else console.log('Studio table: closed supports, saved-radius contact/scene/camera, legacy scale, clear-glass PBR and real background can rays passed');


assert.equal(defaultStudioTable().material,"marble_milk_glass","New tables use milk glass without rewriting legacy fixture materials");

const skirtIds={marble_milk_glass:3,black_felt:4,white_plaster:2};
const newTable=defaultStudioTable(),parts=[];
assert.deepEqual(newTable.skirt,{enabled:true,material:'black_felt'});
const skirtFaces=studioTableFaces(newTable,skirtIds,undefined,p=>parts.push(p)).filter(f=>f[3]===4);
assert.equal(skirtFaces.length,48);
assert.equal(parts.filter(p=>p.id.startsWith('skirt-')).length,4);
for(const axis of [0,2])for(const direction of [-1,1])assert.ok(skirtFaces.some(f=>f[4][0][axis]===direction),'Skirt has inward and outward faces');
assert.ok(skirtFaces.every(f=>f.slice(0,3).every(p=>Math.abs(p[0])>=newTable.width/2 || Math.abs(p[2])>=newTable.length/2)),'Open center preserves underlight path');
assert.equal(studioTableFaces({...newTable,skirt:{enabled:false,material:'black_felt'}},skirtIds).length,60);
assert.equal(studioTableFaces({...newTable,skirt:{enabled:true,material:'white_plaster'}},skirtIds).filter(f=>f[3]===2).length,48);
assert.throws(()=>validateStudioTable({...newTable,skirt:{enabled:true,material:'missing'}},Object.keys(skirtIds)));
if(!process.argv.includes('--json'))console.log('Two-sided skirt: four closed panels, shared materials, open center, removal and validation passed');

const defaultDimensions=defaultStudioTable();
assert.deepEqual([defaultDimensions.width,defaultDimensions.length,defaultDimensions.height],[8,4,4]);
assert.equal(tableFloorDrop(defaultDimensions),3.98);
assert.equal(tableFloorDrop({enabled:true}),3,'Missing old height retains old floor drop');
const absentTable=validateStudioTable(undefined,Object.keys(ids));
assert.deepEqual([absentTable.width,absentTable.length,absentTable.height],[5.2,3,3.02]);
assert.equal(absentTable.enabled,false);
const legacyPartial=validateStudioTable({version:1,enabled:true,material:'marble_clear_glass',width:9},Object.keys(ids));
assert.deepEqual([legacyPartial.width,legacyPartial.length,legacyPartial.height],[9,3,3.02]);
const newFaces=studioTableFaces({...defaultDimensions,skirt:{enabled:false,material:'black_felt'}},ids);
const newVertices=newFaces.flatMap(f=>f.slice(0,3));
assert.deepEqual([Math.min(...newVertices.map(p=>p[0])),Math.max(...newVertices.map(p=>p[0]))],[-4,4]);
assert.deepEqual([Math.min(...newVertices.map(p=>p[2])),Math.max(...newVertices.map(p=>p[2]))],[-2,2]);
assert.deepEqual([Math.min(...newVertices.map(p=>p[1])),Math.max(...newVertices.map(p=>p[1]))],[-5,-1]);
assert.equal(studioScaleTransform(defaultStudioScale(),10,true).unitM,.3048);
if(!process.argv.includes('--json'))console.log('New table 8x4x4 bounds, one foot/unit, and unchanged legacy/partial dimensions passed');
