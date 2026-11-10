import assert from 'node:assert/strict';
import fs from 'node:fs';
import {cameraWithLens,cameraFocusDiagnostic,createStudioLensControls,fovFromFocal,focalFromFov} from '../examples/meowdoku/web/marble_lens_controls.js';
import {validatePhotoFocus,packMarblePhoto} from '../examples/meowdoku/web/marble_photo.js';
const all=JSON.parse(fs.readFileSync(new URL('../examples/meowdoku/lens_contract_catalog.json',import.meta.url)));
const profiles=all.contracts.filter(c=>c.kind==='authored-thin-profile');
const references=all.contracts.filter(c=>c.kind!=='authored-thin-profile');
const catalog={schema:'studio-lens-catalog-v1',profiles,references};
const base={eye_m:[0,0,.07],target_m:[0,0,.18],focus_distance_m:.11,fov_degrees:18,aperture_radius_m:.002};
const custom={key:'custom',focusMm:5,focalMm:50,fNumber:4};
const derived=cameraWithLens(base,custom,profiles);
assert.equal(derived.focus_distance_m,.005);assert.equal(derived.aperture_radius_m,.00625);
assert(Math.abs(derived.fov_degrees-26.99146656)<1e-8);
assert.deepEqual(derived.eye_m,base.eye_m);assert.deepEqual(derived.target_m,base.target_m);
for(const mm of [1,5,19])assert.doesNotThrow(()=>validatePhotoFocus(cameraWithLens(base,{...custom,focusMm:mm},profiles)));
for(const change of [{focusMm:.9},{focusMm:NaN},{focalMm:300},{fNumber:.9},{key:'gullstrand_eye'}])
  assert.throws(()=>cameraWithLens(base,{...custom,...change},profiles));
const macro=profiles.find(p=>p.key.toLowerCase().includes('macro'));
assert.throws(()=>cameraWithLens(base,{...custom,key:macro.key},profiles),/Profile focus/);
const valid=cameraWithLens(base,{...custom,key:macro.key,focusMm:230},profiles);
assert.equal(valid.lens_selection.source_sha256,macro.source_sha256);
assert(Math.abs(focalFromFov(fovFromFocal(.05))-.05)<1e-14);
const bytes=packMarblePhoto({interior:[],glass:[]},'a'.repeat(64),{camera:valid});
const view=new DataView(bytes),wire=JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,16,view.getUint32(4,true))));
assert.deepEqual(wire.camera.lens_selection,valid.lens_selection);

class Element{
 constructor(value=''){this.value=value;this.children=[];this.dataset={};this.listeners={};}
 addEventListener(name,fn){this.listeners[name]=fn;} fire(name){this.listeners[name]?.();}
 append(child){this.children.push(child);}
}
function harness(fetcher){
 const values={'photo-lens':'unchanged','photo-focus-mm':'110','photo-focal-mm':'75.76501818','photo-f-number':'18.94125455'};
 const elements=new Map(),events=[];
 for(const name of ['photo-lens-controls','photo-lens','photo-lens-status','photo-focus-mm','photo-focal-mm','photo-f-number',
  'photo-focus-slider','photo-focal-slider','photo-aperture-slider','photo-lens-inventory','photo-focus-diagnostic'])elements.set(name,new Element(values[name]));
 const doc={getElementById:id=>elements.get(id),createElement:()=>new Element(),defaultView:{Event},dispatchEvent:e=>events.push(e.type)};
 const controls=createStudioLensControls(doc,{fetcher});
 return {controls,$:id=>elements.get(id),events};
}
const h=harness(async()=>({ok:true,json:async()=>catalog}));await h.controls.ready;
assert.equal(h.$('photo-lens').children.length,20);
assert.equal(h.$('photo-lens').children.filter(c=>c.disabled).length,13);
assert(h.$('photo-focus-mm').disabled);
h.$('photo-lens').value='custom';h.$('photo-lens').fire('change');
assert(!h.$('photo-focal-slider').disabled);
h.$('photo-focus-mm').value='5';h.$('photo-focus-mm').fire('input');
assert.equal(h.controls.applyCamera(base).focus_distance_m,.005);
assert.equal(h.controls.minimumFocus(),.001);
h.controls.focusAt(.012);assert.equal(h.controls.applyCamera(base).focus_distance_m,.012);
const last=h.controls.fov(18);
h.$('photo-focus-mm').value='';h.$('photo-focus-mm').fire('input');
assert.throws(()=>h.controls.applyCamera(base));assert.equal(h.controls.fov(18),last);
h.$('photo-lens').value=macro.key;h.$('photo-lens').fire('change');
assert.equal(h.$('photo-focus-mm').value,'230');assert(h.$('photo-focal-slider').disabled);
assert.equal(h.controls.applyCamera(base).lens_selection.key,macro.key);
assert.equal(h.controls.minimumFocus(),.23);
assert.throws(()=>h.controls.focusAt(.01),/Axial focus/);
const macroCamera=h.controls.applyCamera({...base,eye_m:[0,0,.20312604158822956]});
const diagnostic=cameraFocusDiagnostic(macroCamera);
assert(diagnostic.warning);assert(diagnostic.blurFraction>0);
assert.equal(h.$('photo-focus-diagnostic').dataset.warning,'true');
assert.match(h.$('photo-focus-diagnostic').textContent,/23.13 mm away; axial focus 230.00 mm/);
assert.throws(()=>h.controls.focusAt(.023,{custom:true}),/actual camera/);
assert.equal(h.$('photo-lens').value,macro.key,'Failed custom conversion is transactional');
h.controls.focusAt(diagnostic.distanceM,{camera:macroCamera,custom:true});
const focused=h.controls.applyCamera(macroCamera);
assert.equal(focused.lens_selection.key,null,'Explicit custom mode clears physical profile provenance');
assert.equal(focused.focus_distance_m,diagnostic.distanceM);
assert(Math.abs(focused.fov_degrees-macroCamera.fov_degrees)<1e-12);
assert(Math.abs(focused.aperture_radius_m-macroCamera.aperture_radius_m)<1e-12);
assert.equal(cameraFocusDiagnostic(focused).warning,false);
h.$('photo-lens').value='unchanged';h.$('photo-lens').fire('change');
assert.equal(h.controls.fov(18),18);assert.equal(h.controls.applyCamera(base).lens_selection,null);
const failed=harness(async()=>{throw Error('offline');});await failed.controls.ready;
failed.$('photo-lens').value='custom';failed.$('photo-lens').fire('change');
assert.doesNotThrow(()=>failed.controls.applyCamera(base));assert.match(failed.$('photo-lens-status').textContent,/offline/);
console.log('PASS lens controls: source identities, prime limits, millimetres, aperture, sub-2cm custom focus, invalid edits, failed load, mode reset and wire provenance');

const mirror = profiles.find(p=>p.key==='mirror_500mm_f8');
const mirrorSettings={key:mirror.key,focusMm:5000,focalMm:500,fNumber:8};
const mirrorCam=cameraWithLens(base,mirrorSettings,profiles);
assert.equal(mirrorCam.aperture_radius_m,.03125);
assert.doesNotThrow(()=>packMarblePhoto({interior:[],glass:[]},'a'.repeat(64),{camera:mirrorCam}));
for(const change of [{focalMm:499},{focalMm:501},{fNumber:7.9},{fNumber:8.1}])assert.throws(()=>cameraWithLens(base,{...mirrorSettings,...change},profiles));
h.$('photo-lens').value=mirror.key;h.$('photo-lens').fire('change');
assert.equal(h.controls.applyCamera(base).lens_selection.f_number,8);
assert(h.$('photo-focal-slider').disabled);assert(h.$('photo-aperture-slider').disabled);
assert.equal(h.$('photo-f-number').max,'8');
h.$('photo-lens').value='custom';h.$('photo-lens').fire('change');
assert.equal(h.$('photo-focal-slider').max,'500');assert(!h.$('photo-aperture-slider').disabled);
assert.equal(h.$('photo-f-number').max,'64');
console.log('PASS mirror nominal specs, fixed f/8, 500mm bounds and mode recovery');
