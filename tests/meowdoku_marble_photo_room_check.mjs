import {installScenePackerInputs} from './meowdoku_scene_packer_harness.mjs';
// CPU-only proof of the actual Studio scene packer, without a WebGL canvas.
import assert from 'node:assert/strict';
import {writeStudioFixtureRows} from '../examples/meowdoku/web/marble_studio_scene.js';
import fs from 'node:fs';
import vm from 'node:vm';
import {backdropFaces, defaultBackdropLayers} from '../examples/meowdoku/web/marble_backdrop.js';
import {studioRoomBounds,studioRoomFaces} from '../examples/meowdoku/web/marble_room.js';
import {concentricBackdropLayers} from '../examples/meowdoku/web/marble_studio_surfaces.js';
import {packMarblePhoto, OBJECT_CENTER} from '../examples/meowdoku/web/marble_photo.js';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
const roomBase=STUDIO_LIGHTS.length+1;

const source = fs.readFileSync(process.env.STUDIO_SOURCE ?? new URL('../examples/meowdoku/web/marble_studio.js', import.meta.url), 'utf8');
const start = source.indexOf('export function sceneForPhoto(');
const end = source.indexOf('\nfunction syncMaterial()', start);
assert.ok(start >= 0 && end > start);
// Execute production sceneForPhoto unchanged, supplying only its scene inputs.
const layers = defaultBackdropLayers();
layers[0] = {material:'stone', width:7, height:6};
layers[1] = {material:'metal', width:3, height:4};
const ids = {stone:1, metal:3};
const backdrop = backdropFaces({count:2, material:'stone', sweep:true, layers,wallZ:studioRoomBounds(3).backZ}, ids);
const n = [[0,0,1],[0,0,1],[0,0,1]];
const floor = studioRoomFaces({scale:3,material:()=>ids.stone});
const fixture = [[[1,1,-1],[2,1,-1],[1,2,-1],0,n]];
const pbr = new Float32Array(4*16);
for (const [id,color] of [[1,[.2,.3,.4]],[3,[.8,.7,.1]]]) {
  pbr.set(color,id*16); pbr[id*16+3]=.8; pbr[id*16+7]=1;
}
const context = vm.createContext({
  lightStandFacesForStudio:()=>[], // Explicit legacy rig without floor stands.
  OBJECT_CENTER, STUDIO_LIGHTS, tables:{pbr:{table:{floats_per_record:16},data:pbr}},
  computeFixtureRoles:()=>({fixtureRoles:[],roleColor:STUDIO_LIGHTS.map(()=>[1,1,1]),roleStrength:STUDIO_LIGHTS.map((_,i)=>i===0?1:0)}),
  buildFixtureFaces:(_roles,_housing,cb)=>{const faces=fixture.map(f=>[...f.slice(0,3),f[3]+context.FIXTURE_MAT_BASE,...f.slice(4)]);cb?.({id:'test-emitter',closed:false,faces});return faces;}, checkerboardFaces:(_channel,cb)=>{cb?.({id:'room',closed:false,faces:floor});return floor;}, backdropRollFaces:(_channel,cb)=>{cb?.({id:'backdrop',closed:false,faces:backdrop});return backdrop;},
  tableFaces:()=>[], goboFaces:()=>[], studioSubjectTransform:()=>({subjectCenter:[0,0,0]}),studioPhotoScale:()=>.11/3.45,
});
vm.runInContext(source.slice(start,end).replace('export function','function'),context,{timeout:1000});
installScenePackerInputs(context);
const scene=JSON.parse(JSON.stringify(vm.runInContext('sceneForPhoto()',context,{timeout:1000})));
const all=[...fixture,...floor,...backdrop], scale=.11/3.45;
assert.equal(scene.material_ids.length,all.length);
assert.equal(scene.vertices.length,all.length*9);
assert.deepEqual(scene.vertices,all.flatMap(face=>face.slice(0,3).flatMap(point=>point.map((x,i)=>OBJECT_CENTER[i]+x*scale))));
const expectedIds=[0,...floor.map(()=>roomBase),...backdrop.map(face=>face[3]===1?roomBase:roomBase+1)];
assert.deepEqual(scene.material_ids,expectedIds);
assert.deepEqual(scene.pbr.slice(roomBase*16,(roomBase+1)*16),Array.from(pbr.slice(1*16,2*16)));
assert.deepEqual(scene.pbr.slice((roomBase+1)*16,(roomBase+2)*16),Array.from(pbr.slice(3*16,4*16)));
const subject=[[0,0,0],[1,0,0],[0,1,0],0,n];
const wire=packMarblePhoto({interior:[],glass:[subject]},'a'.repeat(64),{fixtures:scene});
const view=new DataView(wire);
const request=JSON.parse(new TextDecoder().decode(new Uint8Array(wire,16,view.getUint32(4,true))));
assert.deepEqual(request.fixtures,scene);
assert.equal(request.room,undefined);
assert.equal(request.exposure_seconds,30);
// The default/inverted concentric mats use the same production face arrays as
// preview. The photo packer preserves every triangle and both material rows.
for(const inverted of [false,true]) {
  const concentric=backdropFaces({count:6,material:'white_plaster',sweep:false,
    layers:concentricBackdropLayers(3,inverted),wallZ:studioRoomBounds(3).backZ},
  {white_plaster:1,black_felt:3});
  context.backdropRollFaces=(_channel,cb)=>{cb?.({id:"backdrop",closed:false,faces:concentric});return concentric;};
  const circularScene=JSON.parse(JSON.stringify(vm.runInContext('sceneForPhoto()',context,{timeout:1000})));
  const circularFaces=[...fixture,...floor,...concentric];
  assert.deepEqual(circularScene.vertices,circularFaces.flatMap(face=>face.slice(0,3).flatMap(point=>point.map((x,i)=>OBJECT_CENTER[i]+x*scale))));
  assert.deepEqual(circularScene.material_ids,[0,...floor.map(()=>roomBase),...concentric.map(face=>face[3]===1?roomBase:roomBase+1)]);
  assert.deepEqual(circularScene.pbr.slice(roomBase*16,(roomBase+1)*16),Array.from(pbr.slice(1*16,2*16)));
  assert.deepEqual(circularScene.pbr.slice((roomBase+1)*16,(roomBase+2)*16),Array.from(pbr.slice(3*16,4*16)));
  const packed=packMarblePhoto({interior:[],glass:[subject]},'a'.repeat(64),{fixtures:circularScene});
  const packedView=new DataView(packed);
  const payload=JSON.parse(new TextDecoder().decode(new Uint8Array(packed,16,packedView.getUint32(4,true))));
  assert.deepEqual(payload.fixtures,circularScene);
}
if(process.argv.includes('--json')) console.log(JSON.stringify({wire:Buffer.from(wire).toString('base64'),scene,roomBase}));
else console.log('Studio room: authored backdrop geometry/materials packed exactly once with no procedural room');

// Surface colour is independent of light power in both actual photo packing
// and the shared Phong/detached-capture writer; emission keeps its old law.
for (const color of [[0,0,0],[1,1,1],[.2,.6,.1]]) for (const strength of [0,2]) {
  context.computeFixtureRoles=()=>({fixtureRoles:[],roleColor:STUDIO_LIGHTS.map(()=>color),roleStrength:STUDIO_LIGHTS.map(()=>strength)});
  const photo=JSON.parse(JSON.stringify(vm.runInContext('sceneForPhoto()',context)));
  const data=new Float32Array(STUDIO_LIGHTS.length*16);
  writeStudioFixtureRows(data,16,0,STUDIO_LIGHTS.map(()=>color),STUDIO_LIGHTS.map(()=>strength));
  for(let i=0;i<STUDIO_LIGHTS.length;i++) {
    assert.deepEqual(photo.pbr.slice(i*16,i*16+3),color.map(Math.fround));
    for(let j=0;j<3;j++) {
      assert.equal(data[i*16+j],Math.fround(color[j]));
      assert.equal(photo.pbr[i*16+8+j],Math.fround(color[j]*strength));
      assert.equal(data[i*16+8+j],Math.fround(color[j]*strength));
    }
    assert.equal(photo.pbr[i*16+5],0);assert.equal(photo.pbr[i*16+7],1);
    assert.equal(data[i*16+5],0);assert.equal(data[i*16+7],1);
  }
}
if(!process.argv.includes('--json'))console.log('PASS filter surface independent of emission: black, white, coloured; off/on; Phong/capture/photo parity');
