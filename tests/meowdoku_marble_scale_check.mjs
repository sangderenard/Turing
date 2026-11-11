import {installScenePackerInputs,installStudioSceneBuilders} from './meowdoku_scene_packer_harness.mjs';
// Real normalized meshes and MRP1 payloads, without a browser, server or GPU.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {defaultStudioScale, validateStudioScale, studioScaleTransform, studioFrameDistance, studioCameraClip,
  radiusFromSlider, sliderFromRadius} from '../examples/meowdoku/web/marble_studio_scale.js';
import {defaultStudioTable, tableFloorDrop, studioTableFaces} from '../examples/meowdoku/web/marble_studio_table.js';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
import {studioPlacementWarnings} from '../examples/meowdoku/web/marble_table_lights.js';
import {OBJECT_CENTER, photoCameraFromStudio, packMarblePhoto} from '../examples/meowdoku/web/marble_photo.js';
import {buildMarbleObjectGeometry, geometrySha256, analyticSpheresForDesign, massFromSavedMarbleObject,
  verifySavedMarbleObject} from '../examples/meowdoku/web/marble_object.js';
import {objectSnapshotKey} from '../examples/meowdoku/web/marble_object_snapshot.js';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
const copy = x => x === undefined ? undefined : JSON.parse(JSON.stringify(x));
const near = (a,b) => assert.ok(Math.abs(a-b) < 1e-12 * Math.max(1,Math.abs(a),Math.abs(b)), `${a} != ${b}`);
assert.deepEqual(defaultStudioScale(),{version:1,mode:'fixed',unitMm:304.8});
const state = {version:1,mode:'fixed',unitMm:10}; // Explicit saved ten-mm rig must retain prior behavior.
for (const radius of [.1,1,10,100,200]) near(radiusFromSlider(sliderFromRadius(radius)),radius);
assert.deepEqual(validateStudioScale(),{version:1,mode:'legacy'});
for (const bad of [null,{}, {version:2,mode:'fixed',unitMm:10}, {version:1,mode:'bad'},
  ...[0,NaN,Infinity,1001,'10'].map(unitMm=>({version:1,mode:'fixed',unitMm}))]) assert.throws(()=>validateStudioScale(bad));
for (const radius of [0,NaN,Infinity,201,'10']) assert.throws(()=>studioScaleTransform(state,radius,true));
for (const radius of [.1,10,25,200]) {
  const t=studioScaleTransform(state,radius,true);
  near(t.subjectCenter[1]-t.subjectRadius,-1); near(t.unitM,.01); near(t.radiusM,radius/1000);
  assert.deepEqual(studioScaleTransform(state,radius,false).subjectCenter,[0,0,0]);
  const legacy=studioScaleTransform(undefined,radius,true);near(legacy.subjectRadius,1);near(legacy.unitM,radius/1000);
  const off=studioScaleTransform(undefined,radius,false);near(off.unitM,.11/3.45);assert.equal(off.fovDegrees,18);
}

const source=fs.readFileSync(process.env.STUDIO_TEST_WEB_ROOT?`${process.env.STUDIO_TEST_WEB_ROOT}/marble_studio.js`:new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const nodes={'radius-mm':{value:'10',setCustomValidity(v){this.validation=v;}},'marble-scale-out':{},
  'marble-scale-slider':{},'mass-report':{},'room-scale':{value:'3'}};
let table=({...defaultStudioTable(),width:5.2,length:3,height:3.02,material:"marble_clear_glass",skirt:{enabled:false,material:"black_felt"}}),profile=state;
const pbr=[.71,.82,.93,.04,0,1,1.52,.1,0,0,0,0,.12,.23,.34,.45];
const context=vm.createContext({$:id=>nodes[id],OBJECT_CENTER,STUDIO_LIGHTS,ids:{marble_clear_glass:0},lightStandFacesForStudio:()=>[],
  photo:{applyLensCamera:camera=>camera},
  defaultStudioScale,studioScaleTransform,studioFrameDistance,sliderFromRadius,defaultStudioTable,
  tableFloorDrop,studioTableFaces,studioRoomBounds,photoCameraFromStudio,
  studioLighting:{table:{getState:()=>table},scale:{getState:()=>profile,sync(){}}},
  savedPhotoRadius:.01,lastPreviewSubjectTransform:null,studioOrbitTarget:[0,0,0],yaw:-.5,pitch:.2,distance:3.45,
  tables:{pbr:{table:{floats_per_record:16},data:new Float32Array(pbr)}},
  checkerboardFaces:()=>[],backdropRollFaces:()=>[],goboFaces:()=>[],
  computeFixtureRoles:()=>({fixtureRoles:[],roleColor:STUDIO_LIGHTS.map(()=>[1,1,1]),roleStrength:STUDIO_LIGHTS.map(()=>0)}),
  buildFixtureFaces:()=>[],
});
installStudioSceneBuilders(context);
vm.runInContext(source.slice(source.indexOf('function studioRenderState('),source.indexOf('// Capture actual page values;')),context);
vm.runInContext(source.slice(source.indexOf('function studioRoomScale('),source.indexOf('function goboFaces(')),context);
vm.runInContext(source.slice(source.indexOf('export function sceneForPhoto('),source.indexOf('\nfunction syncMaterial()')).replace('export function','function'),context);
installScenePackerInputs(context);
const run = code => copy(vm.runInContext(code,context,{timeout:1000}));
run('syncSubjectSizeControls()');
const valid=run('studioPreviewSubjectTransform()');
for(const invalid of ['', '0', 'Infinity', '201']) {
  nodes['radius-mm'].value=invalid;run('syncSubjectSizeControls()');
  assert.deepEqual(run('studioPreviewSubjectTransform()'),valid,'Partial number retains last derived preview transform');
  assert.match(nodes['marble-scale-out'].textContent,/last valid size/);
  assert.throws(()=>vm.runInContext('studioSubjectTransform()',context));
  near(run('studioSubjectTransform(true).radiusM'),.01,'Accepted saved photo stays coherent while editing');
  assert.doesNotThrow(()=>run('studioPhotoCamera(true)'), 'Saved camera does not depend on an unfinished current radius');
}
nodes['radius-mm'].value='25';run('syncSubjectSizeControls()');
near(run('studioPreviewSubjectTransform().subjectRadius'),2.5);assert.equal(nodes['radius-mm'].validation,'');
assert.match(nodes['marble-scale-out'].textContent,/50.00 mm/);
assert(source.includes('const subject=studioPreviewSubjectTransform();'),'Animation uses the recovered preview view');
// Exercise the exact radius/density input registration, not just the pure
// transform. Both bubble and ordinary paths must reject before calling work.
const physicalEvents={};
for(const field of ['radius-mm','shell-density','eye-density','accent-density','confetti-density','enamel-density','mark-density']) {
  nodes[field]??={value:'2500'};
  nodes[field].addEventListener=(type,fn)=>{assert.equal(type,'input');(physicalEvents[field]??=[]).push(fn);};
}
nodes['bubble-count']={value:'0'};
let massCalls=0,rebuildCalls=0;
context.lastGeometry={};context.reportMass=()=>{assert(Number(nodes['radius-mm'].value)>0);massCalls++;};
context.rebuild=()=>{assert(Number(nodes['radius-mm'].value)>0);rebuildCalls++;};
vm.runInContext(source.slice(source.indexOf('  for(const field of ["radius-mm","shell-density"'),
  source.indexOf('  $("marble-scale-slider").addEventListener')),context);
for(const bubbles of ['0','2']) for(const invalid of ['', '0', 'Infinity', '201']) {
  nodes['bubble-count'].value=bubbles;nodes['radius-mm'].value=invalid;
  for(const field of ['radius-mm','shell-density']) for(const fn of physicalEvents[field]) assert.doesNotThrow(fn);
  assert.equal(massCalls,0);assert.equal(rebuildCalls,0);
  assert.equal(nodes['radius-mm'].value,invalid,'Invalid authored data is not replaced by cached preview size');
  assert.match(nodes['mass-report'].textContent,/waiting for a valid marble radius/);
  assert.throws(()=>vm.runInContext('studioSubjectTransform()',context));
}
nodes['radius-mm'].value='25';nodes['bubble-count'].value='0';physicalEvents['radius-mm'].forEach(fn=>fn());
assert.equal(massCalls,1);assert.equal(rebuildCalls,0);assert.equal(nodes['radius-mm'].validation,'');
nodes['bubble-count'].value='2';physicalEvents['radius-mm'].forEach(fn=>fn());
assert.equal(massCalls,1);assert.equal(rebuildCalls,1);
// An independent geometry/material refresh entering reportMass directly also
// stops before the strict marbleMass function sees an incomplete radius.
const eventMass=context.reportMass;
vm.runInContext(source.slice(source.indexOf('function reportMass('),source.indexOf('// Pure (no WebGL calls) face builder')),context);
nodes['radius-mm'].value='';context.marbleMass=()=>{throw new Error('Invalid radius reached marbleMass');};
assert.doesNotThrow(()=>vm.runInContext('reportMass({})',context));
nodes['radius-mm'].value='25';context.reportMass=eventMass;run('syncSubjectSizeControls()');
const sliderEvents=[],frameEvents=[];
nodes['marble-scale-slider'].addEventListener=(type,fn)=>sliderEvents.push(fn);
nodes['frame-marble']={addEventListener:(type,fn)=>frameEvents.push(fn)};
let radiusInputs=0,cameraChanges=0;
nodes['radius-mm'].dispatchEvent=event=>{assert.equal(event.type,'input');assert.equal(event.bubbles,true);radiusInputs++;};
context.Event=class {constructor(type,options){this.type=type;Object.assign(this,options);}};
context.radiusFromSlider=radiusFromSlider;context.canvas={clientWidth:400,clientHeight:800};
context.liveRay={cameraChanged:()=>cameraChanges++};context.photo={applyLensCamera:camera=>camera,
  lensFov:value=>value,lensMinimumFocus:()=>.001,focusLensAt:value=>{context.focusedAt=value;},status:message=>{context.photoStatus=message;}};
nodes['photo-camera']={value:'builtin'};
vm.runInContext(source.slice(source.indexOf('  $("marble-scale-slider").addEventListener'),source.indexOf('  for(const field of ["paint-a-color"')),context);
nodes['marble-scale-slider'].value=String(sliderFromRadius(25));sliderEvents[0]();
assert.equal(nodes['radius-mm'].value,'25');assert.equal(radiusInputs,1,'Slider emits the same actual radius edit used by autosave');
const priorTarget=copy(context.studioOrbitTarget),priorDistance=context.distance;frameEvents[0]();
assert.equal(cameraChanges,1);near(context.studioOrbitTarget[1],1.5);assert(context.distance>priorDistance);
context.studioOrbitTarget=priorTarget;context.distance=priorDistance;
const originalScene=run('sceneForPhoto()'),originalCamera=run('studioPhotoCamera()');
const design={seed:17,'base-shape':'marble',kind:'clear',shell:'glass',eye:'glass',accent:'glass',
  'confetti-material':'glass','mark-material':'glass',coverage:0,'band-count':0,'bubble-count':2,
  width:28,length:75,curve:18,strands:3,thickness:35,twist:90,tip:135,flare:18,phase:0,
  'orientation-yaw':0,'orientation-pitch':0,'orientation-roll':0,'tube-count':2,
  'tube-radius':24,'tube-turns':220,'confetti-count':220,'band-width':18,'band-curve':120};
const ids={shell:0,eye:0,accent:0,confetti:0,mark:0,coated:0,paintA:0,paintB:0};
const geometry=buildMarbleObjectGeometry(design,ids);
design.boundary_media=geometry.boundaryMedia;
design.analytic_spheres=analyticSpheresForDesign(design,geometry,0,{stride:16,data:pbr});
const hash=await geometrySha256(geometry), beforePbr=JSON.stringify(pbr), descriptorBytes=JSON.stringify(design);
const saved={v:1,generator:'meowdoku-marble-object-v1',name:'Scale proof',digest:'a'.repeat(64),
  geometry_sha256:hash,design,materials:{index:{glass:0},extra:{coated:0,paintA:0,paintB:0},
    tables:{pbr:{stride:16,data:pbr},enamel:{stride:16,data:Array(16).fill(0)}}},
  physical:Object.fromEntries(['shell','eye','accent','confetti','enamel','mark'].map(role=>[`${role}-density`,2500]))};
const cases=[];
let initialMass;
for (const radiusMm of [1,10,25,200]) {
  saved.physical['radius-mm']=radiusMm;context.savedPhotoRadius=radiusMm/1000;
  saved.derived=massFromSavedMarbleObject(saved,geometry);
  if(!initialMass)initialMass=copy(saved.derived);
  near(saved.derived.massKg,initialMass.massKg*radiusMm**3);
  saved.derived.inertiaKgM2.forEach((row,i)=>row.forEach((v,j)=>near(v,initialMass.inertiaKgM2[i][j]*radiusMm**5)));
  await verifySavedMarbleObject(saved);
  assert.equal(await geometrySha256(geometry),hash);assert.equal(JSON.stringify(pbr),beforePbr);assert.equal(JSON.stringify(design),descriptorBytes);
  const scene=run('sceneForPhoto()'),camera=run('studioPhotoCamera()'),center=run('studioPhotoSubjectCenter()');
  assert.deepEqual(scene,originalScene,'Room/table/material bytes do not resize with marble');
  assert.deepEqual(camera.camera.target_m,center,'Linked photographic view aims at the actual saved subject centre');
  near(camera.camera.focus_distance_m,originalCamera.camera.focus_distance_m);
  assert.equal(camera.camera.fov_degrees,originalCamera.camera.fov_degrees);
  context.photographicCameraBinding.hold(originalCamera);
  assert.deepEqual(run('studioPhotoCamera()'),originalCamera,'Held photographic pose is unchanged by subject size');
  context.photographicCameraBinding.follow();
  near(center[1]-radiusMm/1000,-.01);
  const transform=studioScaleTransform(profile,radiusMm,true);
  near(center[1],transform.subjectCenter[1]*transform.unitM);
  const wire=packMarblePhoto(geometry,saved.digest,{...camera,object_center_m:center,fixtures:scene,sphereCoverage:'exact'});
  const live=packMarblePhoto(geometry,saved.digest,{...camera,object_center_m:center,fixtures:scene,sphereCoverage:'exact',
    live:{session_id:'b'.repeat(32),resolution:256}});
  const header=buffer=>JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,16,new DataView(buffer).getUint32(4,true))));
  const h=header(wire),l=header(live);delete l.live;assert.deepEqual(l,h,'Live and still share physical scene');
  cases.push({saved:copy(saved),wire:Buffer.from(wire).toString('base64')});
}
const key=objectSnapshotKey(saved);profile={version:1,mode:'fixed',unitMm:25};run('sceneForPhoto()');
assert.equal(objectSnapshotKey(saved),key,'Rig-only edits do not become object edits');
assert(!('radiusMm' in profile));
profile={version:1,mode:'legacy'};table={...table,enabled:false};
near(run('studioPhotoScale()'),.11/3.45);assert.deepEqual(run('studioPhotoSubjectCenter()'),OBJECT_CENTER);
assert.deepEqual(run('studioPhotoCamera()'),photoCameraFromStudio(-.5,.2,3.45),'Legacy physical camera is byte-for-byte unchanged');
const small=studioScaleTransform(state,1,true),frame=studioFrameDistance(small,.5);
assert(frame>=.001/small.unitM);assert(studioFrameDistance(studioScaleTransform(state,200,true),.5)>14);
assert(frame*small.unitM<.02,'Small subjects no longer hit the obsolete 21mm framing floor');
const framed=photoCameraFromStudio(0,0,frame,small.unitM,small.fovDegrees,small.subjectCenter);
assert.deepEqual(framed.camera.target_m,OBJECT_CENTER.map((v,i)=>v+small.subjectCenter[i]*small.unitM));
const giant=studioScaleTransform(state,200,true);
const warning=studioPlacementWarnings(({...defaultStudioTable(),width:5.2,length:3,height:3.02,material:"marble_clear_glass",skirt:{enabled:false,material:"black_felt"}}),[],studioRoomBounds(3,3),{center:giant.subjectCenter,radius:giant.subjectRadius});
assert.match(warning.table,/beyond the room/);assert.match(warning.table,/overhangs/);
if(process.argv.includes('--json'))console.log(JSON.stringify({cases}));
else console.log('Marble scale: actual mesh/PBR identity, mass r^3/inertia r^5, fixed rig and camera, contact, migration, invalid-edit recovery and still/live wire passed');

// Exposure snapshots use current radius without changing the saved-object owner.
table=({...defaultStudioTable(),width:5.2,length:3,height:3.02,material:"marble_clear_glass",skirt:{enabled:false,material:"black_felt"}});profile=state;nodes['radius-mm'].value='2';context.savedPhotoRadius=.025;
near(run('studioSubjectTransform(true).radiusM'),.025);
near(run('studioSubjectTransform(false,"exposure").radiusM'),.002);
assert.notDeepEqual(run('studioPhotoSubjectCenter(false)'),run('studioPhotoSubjectCenter(true)'));
near(context.savedPhotoRadius,.025);

const shop=studioScaleTransform(defaultStudioScale(),25,true);
near(shop.unitM,.3048);near(shop.radiusM,.025);near(shop.subjectCenter[1]-shop.subjectRadius,-1);
near(shop.subjectRadius,.025/.3048);
near(8*shop.unitM,2.4384);near(4*shop.unitM,1.2192);
assert.deepEqual(validateStudioScale({version:1,mode:'fixed',unitMm:10}),state);
if(!process.argv.includes('--json'))console.log('New shop one-foot units preserve independent marble radius and saved mm rigs');

// Tangency uses physical radius and actual table top, never a memorized centre.
for(const unitMm of [.1,10,200,304.8])for(const radiusMm of [.1,1,10,25,100,200]){
 const t=studioScaleTransform({version:1,mode:'fixed',unitMm},radiusMm,true);
 const table=defaultStudioTable(),faces=studioTableFaces(table,{[table.material]:0,black_felt:1});
 const top=Math.max(...faces.flatMap(f=>f.slice(0,3).map(v=>v[1])));
 near(t.subjectCenter[1]-t.subjectRadius,top);
 near(t.subjectRadius*t.unitM,radiusMm/1000);
 // A sphere's only minimum-Y point is centre+[0,-radius,0].
 for(const theta of [.001,.1,Math.PI/2,Math.PI])
  assert.ok(t.subjectCenter[1]-t.subjectRadius*Math.cos(theta)>top);
}
if(!process.argv.includes('--json'))console.log('Physical marble size and unique spherical table-contact point survive all supported rig units/radii');

for(const radius of [.1,1,10,25,200])for(const scale of [1,6,8]){
 const t=studioScaleTransform(defaultStudioScale(),radius,true),d=studioFrameDistance(t,1,.001);
 const room=studioRoomBounds(scale,tableFloorDrop(defaultStudioTable())),clip=studioCameraClip(t,d,room);
 assert.ok(clip.near>0&&clip.near<d-t.subjectRadius,'Near plane clears front of framed sphere');
 for(const x of [room.leftX,room.rightX])for(const y of [room.floorY,room.ceilingY])for(const z of [room.backZ,room.frontZ])
  assert.ok(clip.far>d+Math.hypot(x-t.subjectCenter[0],y-t.subjectCenter[1],z-t.subjectCenter[2]));
}
if(!process.argv.includes('--json'))console.log('Adaptive clipping preserves1mm macro focus and full room extents at one-foot scale');
