import assert from 'node:assert/strict';
import fs from 'node:fs';
import {CAMERA_BACK_PRESETS,cameraBack,sensorGate,cameraWithBack} from '../examples/meowdoku/web/marble_camera_back.js';
import {cameraWithLens,createStudioLensControls} from '../examples/meowdoku/web/marble_lens_controls.js';
import {packMarblePhoto} from '../examples/meowdoku/web/marble_photo.js';
const base={eye_m:[0,0,.07],target_m:[0,0,.18],focus_distance_m:.11,fov_degrees:18,aperture_radius_m:.002};
const custom={key:'custom',focusMm:110,focalMm:50,fNumber:8};
const nearly=(a,b)=>assert(Math.abs(a-b)<1e-12,`${a} != ${b}`);
for(const p of CAMERA_BACK_PRESETS)for(const aspect of [.5,1,1.5,2,10]){
  const back=cameraBack(p.key),gate=sensorGate(back,aspect);
  assert(gate.width_m<=p.width_m+1e-15);assert(gate.height_m<=p.height_m+1e-15);
  nearly(gate.width_m/gate.height_m,aspect);
  const cam=cameraWithLens(base,{...custom,focalMm:100,back},[],aspect);
  nearly(Math.tan(cam.fov_degrees*Math.PI/360),gate.height_m/.2);
  nearly(cam.aperture_radius_m,.1/16);assert.equal(cam.focus_distance_m,.11);
  assert.deepEqual(cameraWithBack(cam,back,aspect),cam);
}
const aps=cameraBack('aps_c');
const original=cameraWithLens(base,{key:'unchanged',back:aps});
assert(original.fov_degrees<18);
assert.deepEqual(cameraWithLens(original,{key:'unchanged',back:aps}),original);
assert.equal(cameraWithLens(original,{key:'unchanged'}).fov_degrees,18);
const narrow=cameraWithLens(base,{...custom,focalMm:500,fNumber:8,back:cameraBack('micro_four_thirds')});
assert(narrow.fov_degrees<2.75);
const decode=buffer=>JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,16,new DataView(buffer).getUint32(4,true))));
for(const resolution of [128,256,512,768,1024,2048,4096]){
  const config=decode(packMarblePhoto({interior:[],glass:[]},'a'.repeat(64),{resolution,camera:narrow}));
  assert.equal(config.camera.resolution,resolution);nearly(config.camera.fov_degrees,narrow.fov_degrees);
}
const live=decode(packMarblePhoto({interior:[],glass:[]},'a'.repeat(64),{resolution:256,camera:original,live:{aspect:2.01}}));
assert.deepEqual(live.camera.sensor_gate,sensorGate(aps,2.01));
assert.throws(()=>cameraBack('custom',0,24));assert.throws(()=>cameraBack('bad'));
class Element{
  constructor(value=''){this.value=value;this.children=[];this.dataset={};this.listeners={};}
  addEventListener(name,fn){this.listeners[name]=fn;} fire(name){this.listeners[name]?.();} append(child){this.children.push(child);}
}
const values={'photo-lens':'unchanged','photo-focus-mm':'110','photo-focal-mm':'75.76501818','photo-f-number':'18.94125455',
  'photo-camera-back':'reference','photo-back-width-mm':'36','photo-back-height-mm':'24'};
const elements=new Map(),events=[];
for(const id of [...Object.keys(values),'photo-lens-controls','photo-lens-status','photo-focus-slider','photo-focal-slider',
  'photo-aperture-slider','photo-lens-inventory','photo-focus-diagnostic','photo-back-status'])elements.set(id,new Element(values[id]));
const $=id=>elements.get(id),doc={getElementById:$,createElement:()=>new Element(),defaultView:{Event},dispatchEvent:e=>events.push(e.type)};
const controls=createStudioLensControls(doc,{fetcher:async()=>({ok:true,json:async()=>({schema:'studio-lens-catalog-v1',profiles:[]})})});
await controls.ready;
assert.equal($('photo-camera-back').children.length,6);
$('photo-camera-back').value='aps_c';$('photo-camera-back').fire('change');
assert(controls.applyCamera(base).fov_degrees<18);assert($('photo-back-width-mm').disabled);
assert.match($('photo-back-status').textContent,/centered/);
$('photo-camera-back').value='custom';$('photo-camera-back').fire('change');
assert(!$('photo-back-width-mm').disabled);
$('photo-back-width-mm').value='';$('photo-back-width-mm').fire('input');
assert.throws(()=>controls.applyCamera(base));assert.equal($('photo-lens-status').dataset.valid,'false');
$('photo-back-width-mm').value='36';$('photo-back-height-mm').value='24';$('photo-back-width-mm').fire('input');
assert.equal(controls.applyCamera(base).camera_back.key,'custom');
const wide=controls.applyCamera(base,2);nearly(wide.sensor_gate.width_m,.036);nearly(wide.sensor_gate.height_m,.018);
controls.focusAt(.11,{camera:wide,custom:true});
const refocused=controls.applyCamera(base,2);nearly(refocused.fov_degrees,wide.fov_degrees);nearly(refocused.aperture_radius_m,wide.aperture_radius_m);
$('photo-camera-back').value='reference';$('photo-camera-back').fire('change');
assert.equal(controls.applyCamera(base).camera_back,undefined);assert(events.includes('studio-lens-change'));
const html=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.html',import.meta.url),'utf8');
assert.match(html,/id="photo-camera-back"/);assert.match(html,/value="512" selected>512 [x\u00d7] 512/);
console.log('PASS camera back presets, crop geometry, unchanged pixel choices, live full aspect, repeated application, invalid/recovered custom edit, focus conversion and reset');
