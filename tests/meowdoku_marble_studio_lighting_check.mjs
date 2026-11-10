import {applyLightTools} from '../examples/meowdoku/web/marble_light_tools.js';
import {resolveLightingPoses} from '../examples/meowdoku/web/marble_lighting.js';
import {defaultStudioScale} from "../examples/meowdoku/web/marble_studio_scale.js";
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {defaultBackdropLayers, backdropFaces, validateBackdropLayers, BACKDROP_CIRCLE_SEGMENTS} from '../examples/meowdoku/web/marble_backdrop.js';
import {concentricBackdropLayers, createStudioSurfaceControls, ROOM_SURFACE_FIELDS, STUDIO_SURFACE_PRESETS} from '../examples/meowdoku/web/marble_studio_surfaces.js';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
import {setupStudioLighting, validateStudioLighting, LIGHTING_PROFILE_FIELDS, STUDIO_LIGHT_IDS} from '../examples/meowdoku/web/marble_studio_lighting.js';
import {STORAGE_KEY} from '../examples/meowdoku/web/marble_lighting_profiles.js';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
import {defaultStudioTable, tableFloorDrop} from '../examples/meowdoku/web/marble_studio_table.js';

const copy = value => JSON.parse(JSON.stringify(value));
const names = ['stone_onyx', 'metal_gold', 'metal_silver', 'ceramic_porcelain', 'stucco_plaster',
  'marble_milk_glass', 'black_felt', 'white_gloss_paint', 'white_plaster', 'black_gloss_paint', 'marble_clear_glass', 'black_matte_paint',
  'calibration_white', 'calibration_red', 'calibration_green', 'calibration_blue',
  'calibration_yellow', 'calibration_magenta', 'calibration_cyan', 'calibration_gray20', 'polished_silver'];
const ids = Object.fromEntries(names.map((name, i) => [name, i]));
const defaults = defaultBackdropLayers();
for (let count = 0; count <= 6; count++) for (const sweep of [false, true]) {
  const faces = backdropFaces({count, sweep, material: names[0], layers: defaults}, ids);
  assert.equal(faces.length, count * (sweep ? 4 : 2));
  for (let i = 0; i < count; i++) {
    const f = faces[i * (sweep ? 4 : 2)], w = 4 - .22 * i;
    assert.deepEqual(f.slice(0, 3), [[-w,-w,-3.1+.012*(i+1)],[w,-w,-3.1+.012*(i+1)],[w,w,-3.1+.012*(i+1)]]);
    if (sweep) assert.ok(Math.abs(faces[i*4+2][0][2] - (-3.1+1.2-.22*i*.3)) < 1e-12);
  }
}
const custom = copy(defaults); custom[1] = {material: 'metal_gold', width: 3, height: 5};
const faces = backdropFaces({count: 2, sweep: false, material: names[0], layers: custom}, ids);
assert.deepEqual(faces[2].slice(0, 4), [[-1.5,-2.5,-3.076],[1.5,-2.5,-3.076],[1.5,2.5,-3.076],ids.metal_gold]);
assert.equal(faces[0][3], ids.stone_onyx);
assert.equal(backdropFaces({count: 1, material: 'calibration', sweep: true, layers: custom},ids).length,16);
assert.equal(new Set(backdropFaces({count: 6, material: 'calibration', layers: custom},ids).map(f=>f[3])).size,8);
for (const mutate of [a=>a.pop(),a=>a[0].material='missing',a=>a[0].width=NaN,a=>a[0].height=64.01,a=>a[0].width=.39,
  a=>a[0].shape='unknown',a=>a[0].centerY=Infinity,a=>a[0].centerY=32.01]) {
  const bad=copy(defaults); mutate(bad); assert.throws(()=>validateBackdropLayers(bad,names));
}
for (const count of [-1,7,1.5,NaN]) assert.throws(()=>backdropFaces({count,layers:defaults,material:names[0]},ids));

// Event/DOM harness uses the actual HTML bounds and options. No WebGL needed.
class Element {
  constructor(tag,doc) { this.tagName=tag;this.ownerDocument=doc;this.children=[];this.dataset={};this.events={};this.value='';this.type='';this.min='';this.max=''; }
  append(...nodes) { this.children.push(...nodes); }
  appendChild(node) { this.append(node); return node; }
  replaceChildren(...nodes) { this.children=nodes; }
  setAttribute(key,value) { this[key]=value; }
  addEventListener(type,fn) { (this.events[type]??=[]).push(fn); }
  fire(type) { for (const fn of this.events[type]??[]) fn(); }
  get options() { return this.children.filter(n=>n?.tagName==='option'); }
}
const all=node=>[node,...(node.children??[]).filter(n=>typeof n==='object').flatMap(all)];
const elements={};
const doc={events:{},addEventListener(type,handler){(this.events[type]??=[]).push(handler);},createElement:tag=>new Element(tag,doc),getElementById:id=>elements[id]??Object.values(elements).flatMap(all).find(n=>n.id===id)};
const html=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.html',import.meta.url),'utf8');
for (const match of html.matchAll(/<(input|select|output|div)\b([^>]*\bid="([^"]+)"[^>]*)>([\s\S]*?)(?=<\/select>|<\/output>|<\/div>|$)/g)) {
  // Input tags are also parsed separately below because they are void elements.
  const [,tag,attrs,id,body]=match, el=new Element(tag,doc); el.id=id;
  for(const a of attrs.matchAll(/([\w-]+)="([^"]*)"/g)) el[a[1]]=a[2];
  if(tag==='select') for(const o of body.matchAll(/<option([^>]*)>(.*?)<\/option>/g)) {
    const option=new Element('option',doc); option.value=/value="([^"]*)"/.exec(o[1])?.[1]??o[2]; el.append(option);
    if(!el.value||o[1].includes('selected'))el.value=option.value;
  }
  elements[id]=el;
}
for(const m of html.matchAll(/<input\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
  const el=new Element('input',doc);el.id=m[2];
  for(const a of m[1].matchAll(/([\w-]+)="([^"]*)"/g))el[a[1]]=a[2];
  el.checked=/\bchecked\b/.test(m[1]);elements[el.id]=el;
}
// Parse selects independently so earlier wrapping divs cannot consume them.
for(const m of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
  const el=new Element('select',doc);el.id=m[1];
  for(const o of m[2].matchAll(/<option([^>]*)>(.*?)<\/option>/g)) {
    const option=new Element('option',doc);option.value=/value="([^"]*)"/.exec(o[1])?.[1]??o[2];el.append(option);
    if(!el.value||o[1].includes('selected'))el.value=option.value;
  }
  elements[el.id]=el;
}
for(const id of ['light-pose-controls','lighting-profiles','backdrop-layer-controls','studio-surface-controls','studio-surface-restore-controls','studio-table-controls','studio-gobo-controls','studio-scale-controls'])elements[id]=new Element('div',doc);
for(const id of ['underlight-height-status','top-fill-height-status','table-left-placement-status','table-right-placement-status'])
  elements[id]=new Element('output',doc);
for(const id of LIGHTING_PROFILE_FIELDS.filter(id=>id.includes('material'))) {
  const el=elements[id];for(const name of names){const o=new Element('option',doc);o.value=name;el.append(o);}el.value='stone_onyx';
}
const storage=new Map();globalThis.localStorage={getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,v)};
let stale=0,room=0,backdrop=0,clearanceVertices=null;
const fixtures=STUDIO_LIGHT_IDS.map((id,i)=>(id==='underlight'?{pos:[0,-1.6,0],dir:[0,1,0]}:{pos:[i-3,2,-2],dir:[0,0,1]}));
const editor=setupStudioLighting(doc,fixtures,names,{lightGuideRole:(id,subject,state,sizes,tools)=>{const index=STUDIO_LIGHT_IDS.indexOf(id),pose=resolveLightingPoses(state,STUDIO_LIGHT_IDS,subject)[index];return applyLightTools([{...pose,round:true,roleIndex:index,radius:.2*sizes.lights[index].width}],tools,sizes,elements['key-shape'].value)[0];},fixtureVertices:(...args)=>clearanceVertices?.(...args),onChange:()=>stale++,onRoomChange:()=>room++,onBackdropChange:()=>backdrop++});
assert.equal(editor.poses.getState().poses.length,10);
assert(editor.renderState('studio').shopLights.fixtures.every(f=>f.dimmer===.5),'Fresh B shop lights start at half power');
assert(editor.renderState('exposure').shopLights.fixtures.every(f=>f.dimmer===1&&f.exposureMode==='off'),'Fresh A photo sources remain off');
assert.deepEqual([editor.table.getState().width,editor.table.getState().length,editor.table.getState().height],[8,4,4],'Fresh Studio UI adopts new table dimensions');
assert.deepEqual(['width','length','height'].map(k=>doc.getElementById('studio-table-'+k).value),['8','4','4']);
assert.equal(stale,0);
assert.equal(storage.size,0,'Initialization never rewrites browser profiles');
elements['backdrop-layers'].value='2';editor.syncBackdropVisibility();
const panels=elements['backdrop-layer-controls'].children;assert.equal(panels.filter(p=>!p.hidden).length,2);
const width=doc.getElementById('backdrop-layer-1-width');width.value='3';width.fire('input');
const material=doc.getElementById('backdrop-layer-1-material');material.value='metal_gold';material.fire('change');
assert.equal(backdrop,2);assert.equal(stale,2);
elements['backdrop-layers'].value='0';editor.syncBackdropVisibility();assert.equal(elements['backdrop-layer-controls'].hidden,true);
elements['backdrop-layers'].value='2';editor.syncBackdropVisibility();assert.equal(editor.backdrops.getState()[1].width,3);
elements['light-x'].value='-300';elements['light-x'].fire('input');
assert.equal(editor.poses.getState().poses[0].pos[0],-3);assert.equal(editor.poses.getState().poses[1].pos[0],3);
const pose=editor.poses.getState();pose.poses[3].pos=[-4,1,0];pose.poses[3].yaw=90;pose.poses[3].aimAtSubject=true;editor.poses.setState(pose);
const profileNodes=all(elements['lighting-profiles']);
const name=profileNodes.find(n=>n['aria-label']==='Lighting profile name');
const selection=profileNodes.find(n=>n['aria-label']==='Saved lighting profiles');
const save=profileNodes.find(n=>n.textContent==='Save/update profile'),load=profileNodes.find(n=>n.textContent==='Load selected');
name.value='Cross light';save.fire('click');
assert.ok(storage.has(STORAGE_KEY));
const saved=JSON.parse(storage.get(STORAGE_KEY)).profiles[0].state;
assert.equal(saved.poses.poses[3].aimAtSubject,true);assert.equal(saved.backdrops[1].material,'metal_gold');
assert.equal(saved.backdrops[1].width,3);
assert.equal(saved.controls['room-scale'],6);
assert.equal(saved.strengths.lights.length,10);
assert.ok(saved.strengths.lights.every(light=>light.exposure===null));
elements['room-scale'].value='5';
const changed=editor.poses.getState();changed.poses[3].pos=[2,2,2];editor.poses.setState(changed);
width.value='5';width.fire('input');elements['light-strength-left'].value='0';
selection.value='Cross light';load.fire('click');
assert.deepEqual(editor.poses.getState(),saved.poses);assert.deepEqual(editor.backdrops.getState(),saved.backdrops);
assert.equal(Number(elements['light-strength-left'].value),saved.controls['light-strength-left']);
assert.equal(Number(elements['room-scale'].value),6);
assert.equal(room,1);assert.equal(backdrop,4);
const fields=Object.fromEntries(LIGHTING_PROFILE_FIELDS.map(id=>[id,elements[id]]));
const legacy=copy(saved);delete legacy.scale;delete legacy.controls['room-scale'];
delete legacy.strengths;
legacy.poses.poses=legacy.poses.poses.slice(0,7);
for(const [i,id] of ['key-filter-left','key-filter-right','background-filter','kicker-left-filter',
  'kicker-right-filter','fill-lower-filter','fill-upper-filter'].entries())legacy.controls[id]=`#${(0x123456+i*0x10203).toString(16).padStart(6,'0')}`;
for(const light of STUDIO_LIGHTS.slice(7))delete legacy.controls[light.strengthId];
for(const id of ['table-left-filter','table-right-filter','under-filter'])delete legacy.controls[id];
delete legacy.surfaces;
delete legacy.table;
delete legacy.gobos;
assert.equal(validateStudioLighting(legacy,fields,names).controls['room-scale'],1);
const migrated=validateStudioLighting(legacy,fields,names);
assert.deepEqual(migrated.poses.poses.slice(0,7),legacy.poses.poses);
assert.equal(migrated.poses.poses.length,10);
for(const [id,value] of Object.entries(legacy.controls))assert.equal(migrated.controls[id],value);
for(const light of STUDIO_LIGHTS.slice(7))assert.equal(migrated.controls[light.strengthId],0);
for(const id of ['table-left-filter','table-right-filter','under-filter'])assert.equal(migrated.controls[id],'#ffffff');
assert.equal(migrated.table.enabled,false);
assert.equal(migrated.surfaces.concentric,false);
const legacySplit=copy(legacy);legacySplit.strengths={version:1,lights:STUDIO_LIGHTS.slice(0,7).map(({id},i)=>({id,exposure:i}))};
const migratedSplit=validateStudioLighting(legacySplit,fields,names);
assert.ok(migratedSplit.strengths.lights.every(light=>light.exposure===null));
assert.deepEqual(STUDIO_LIGHTS.slice(0,7).map(l=>migratedSplit.controls[l.strengthId]/10),[0,1,2,3,4,5,6]);
assert.deepEqual(migratedSplit.roomOverrides.values.filter(v=>v.path.startsWith('controls/')).map(v=>v.value),STUDIO_LIGHTS.slice(0,7).map(l=>legacySplit.controls[l.strengthId]));
assert.ok(validateStudioLighting(legacySplit,fields,names).strengths.lights.slice(7).every(light=>light.exposure===null));
assert.ok(validateStudioLighting(legacy,fields,names).strengths.lights.every(light=>light.exposure===null));
for(const mutate of [s=>s.controls['room-scale']=9,s=>s.controls['light-x']=Infinity,s=>s.controls['backdrop-layers']=1.1,s=>s.controls['floor-material-a']='gone',s=>s.controls['key-filter-left']='bad',s=>s.poses.poses[0].id='wrong',s=>s.backdrops[0].height=65]){
  const bad=copy(saved);mutate(bad);assert.throws(()=>validateStudioLighting(bad,fields,names));
}
// Profiles persist independent exposure overrides and Studio's existing slider
// values together. Legacy profile application must clear any current overrides.
for (const [i,light] of STUDIO_LIGHTS.entries()) {
  const link=doc.getElementById(`${light.id}-levels-linked`),exposure=doc.getElementById(`${light.id}-exposure-strength`);
  link.checked=false;link.fire('change');exposure.value=String(i*2);exposure.fire('input');
  elements[light.strengthId].value=String((i+1)*30);elements[light.strengthId].fire('input');
}
name.value='Split lights';save.fire('click');
const splitSaved=JSON.parse(storage.get(STORAGE_KEY)).profiles.find(p=>p.name==='Split lights').state;
assert.ok(splitSaved.strengths.lights.every(light=>light.exposure===null));
assert.deepEqual(STUDIO_LIGHTS.map(l=>splitSaved.controls[l.strengthId]/10),[0,2,4,6,8,10,12,14,16,18]);
editor.strengths.setState(undefined);
elements['light-strength-left'].value='4000';
selection.value='Split lights';load.fire('click');
assert.deepEqual(editor.strengths.getState(),splitSaved.strengths);
assert.deepEqual(STUDIO_LIGHTS.map(l=>editor.renderState('studio').controls[l.strengthId]/10),[3,6,9,12,15,18,21,24,27,30]);
assert.deepEqual(editor.strengths.getStrengths('exposure'),[0,2,4,6,8,10,12,14,16,18]);
const withLegacy=JSON.parse(storage.get(STORAGE_KEY));withLegacy.profiles.push({name:'Legacy',state:legacy});storage.set(STORAGE_KEY,JSON.stringify(withLegacy));
selection.value='Legacy';load.fire('click');
assert.ok(editor.strengths.getState().lights.every(light=>light.exposure===null));
assert.deepEqual(editor.strengths.getStrengths('studio'),editor.strengths.getStrengths('exposure'));
assert.equal(elements['room-scale'].value,'1');
selection.value='Cross light';
// A corrupt stored profile cannot partly apply fields or disturb current mesh.
const before=copy(editor.poses.getState()),beforeRoom=room,beforeBackdrop=backdrop;
const corrupt=JSON.parse(storage.get(STORAGE_KEY));corrupt.profiles[0].state.backdrops[5].width=-4;storage.set(STORAGE_KEY,JSON.stringify(corrupt));
load.fire('click');assert.deepEqual(editor.poses.getState(),before);assert.equal(room,beforeRoom);assert.equal(backdrop,beforeBackdrop);
const beforeStrengths=editor.strengths.getState();
const invalidStrengthProfile=copy(splitSaved);invalidStrengthProfile.strengths.lights[4].exposure=-1;
storage.set(STORAGE_KEY,JSON.stringify({version:1,profiles:[{name:'Bad levels',state:invalidStrengthProfile}]}));
selection.value='Bad levels';load.fire('click');
assert.deepEqual(editor.strengths.getState(),beforeStrengths);assert.deepEqual(editor.poses.getState(),before);
assert.equal(room,beforeRoom);assert.equal(backdrop,beforeBackdrop);

// Simple controls are an explicit opt-in, not an initialization/profile migration.
assert.equal(editor.surfaces.getState().concentric,false);
assert.deepEqual(editor.backdrops.getState(),legacy.backdrops);
assert.equal(storage.size,1);
const roomBefore=Object.fromEntries(ROOM_SURFACE_FIELDS.map(id=>[id,elements[id].value]));
for(const preset of STUDIO_SURFACE_PRESETS) {
  const button=doc.getElementById(`studio-surface-${preset.id}`),was=stale,wasRoom=room;
  assert.equal(button.disabled,false);button.fire('click');
  assert.equal(button['aria-pressed'],'true');
  assert.equal(stale,was+1);assert.equal(room,wasRoom+1);
  button.fire('click');assert.equal(stale,was+1,'Repeated preset selection changes nothing');
  assert.equal(editor.surfaces.ownsControl(button.id),true);
  for(const id of ROOM_SURFACE_FIELDS) assert.equal(elements[id].value,id.endsWith('pattern')?'solid':preset.material);
  assert.deepEqual(editor.surfaces.getState().surfaceBackup,roomBefore);
}
doc.getElementById('studio-restore-surfaces').fire('click');
assert.deepEqual(Object.fromEntries(ROOM_SURFACE_FIELDS.map(id=>[id,elements[id].value])),roomBefore);
const rig=doc.getElementById('studio-concentric-enabled'),invert=doc.getElementById('studio-concentric-invert');
const customBackdrop=copy(editor.backdrops.getState()),customCount=elements['backdrop-layers'].value;
const customMaterial=elements['backdrop-material'].value,customSweep=elements['backdrop-sweep'].checked;
let was=stale;invert.checked=true;invert.fire('change');assert.equal(stale,was,'Off-state inversion changes no scene');
assert.deepEqual(editor.backdrops.getState(),customBackdrop);
rig.checked=true;rig.fire('change');assert.equal(stale,was+1);
assert.equal(editor.backdrops.getState().at(-1).material,'black_felt');
assert.equal(elements['backdrop-layers'].value,'6');assert.equal(elements['backdrop-sweep'].checked,false);
was=stale;rig.fire('change');assert.equal(stale,was,'Repeated checkbox event is idempotent');
const invertedGeometry=editor.backdrops.getState();
invert.checked=false;invert.fire('change');assert.equal(stale,was+1);
editor.backdrops.getState().forEach((layer,i)=>{
  assert.notEqual(layer.material,invertedGeometry[i].material);
  assert.deepEqual({...layer,material:''},{...invertedGeometry[i],material:''});
});
elements['room-scale'].value='3';elements['room-scale'].fire('input');
assert.deepEqual(editor.backdrops.getState(),concentricBackdropLayers(3,false));
assert.match(doc.getElementById('studio-concentric-status').textContent,/diameter 12.06; outer 26.32; centre height 5.01/);
// Save preserves actual triangles' settings, inversion and the prior custom rig.
storage.set(STORAGE_KEY,JSON.stringify({version:1,profiles:[]}));
name.value='Concentric';save.fire('click');
const concentricSaved=JSON.parse(storage.get(STORAGE_KEY)).profiles[0].state;
assert.equal(concentricSaved.surfaces.concentric,true);
assert.deepEqual(concentricSaved.surfaces.backdropBackup.layers,customBackdrop);
const tableToggle=doc.getElementById('studio-table-enabled');
tableToggle.checked=true;tableToggle.fire('change');
assert.deepEqual(editor.backdrops.getState(),concentricBackdropLayers(3,false,3));
assert.match(doc.getElementById('studio-concentric-status').textContent,/diameter 15.06; outer 27.77; centre height 3.51/);
name.value='Table and concentric';save.fire('click');
const tableRigSaved=JSON.parse(storage.get(STORAGE_KEY)).profiles.find(p=>p.name==='Table and concentric').state;
assert.equal(tableRigSaved.table.enabled,true);assert.equal(tableRigSaved.surfaces.concentric,true);
tableToggle.checked=false;tableToggle.fire('change');
assert.deepEqual(editor.backdrops.getState(),concentricSaved.backdrops);
selection.value='Table and concentric';load.fire('click');
assert.deepEqual(editor.backdrops.getState(),tableRigSaved.backdrops);assert.equal(tableToggle.checked,true);
selection.value='Concentric';load.fire('click');
assert.deepEqual(editor.backdrops.getState(),concentricSaved.backdrops);assert.equal(tableToggle.checked,false);
rig.checked=false;rig.fire('change');
assert.deepEqual(editor.backdrops.getState(),customBackdrop);
assert.equal(elements['backdrop-layers'].value,customCount);
assert.equal(elements['backdrop-material'].value,customMaterial);
assert.equal(elements['backdrop-sweep'].checked,customSweep);
selection.value='Concentric';load.fire('click');
assert.equal(rig.checked,true);assert.deepEqual(editor.backdrops.getState(),concentricSaved.backdrops);
const edits=doc.getElementById('backdrop-layer-1-width');edits.value='5';edits.fire('input');
assert.equal(rig.checked,false,'Advanced edits take ownership without restoring a default');
assert.equal(editor.backdrops.getState()[1].width,5);
doc.getElementById('studio-restore-backdrop').fire('click');
assert.deepEqual(editor.backdrops.getState(),customBackdrop);
for(const mutate of [s=>s.surfaces.inverted='yes',s=>s.surfaces.backdropBackup.layers[0].width=-1,
  s=>s.surfaces.backdropBackup=null,s=>s.backdrops[0].material='metal_gold',s=>s.controls['backdrop-layers']=5]) {
  const bad=copy(concentricSaved);mutate(bad);assert.throws(()=>validateStudioLighting(bad,fields,names));
}
// The old profile has neither simple state nor new shape fields. Loading it
// after an active rig keeps its exact authored materials, rectangles and count.
storage.set(STORAGE_KEY,JSON.stringify({version:1,profiles:[{name:'Old exact',state:legacy}]}));
selection.value='Old exact';load.fire('click');
assert.equal(rig.checked,false);assert.equal(invert.checked,false);
assert.deepEqual(editor.backdrops.getState(),legacy.backdrops);
assert.equal(elements['backdrop-layers'].value,String(legacy.controls['backdrop-layers']));
assert.equal(editor.surfaces.getState().backdropBackup,null);
// Serving a new UI with an older material snapshot must expose the shortage,
// not relabel another material or apply an incomplete four-finish preset.
const unavailableContainer=new Element('div',doc),beforeUnavailable=editor.backdrops.getState(),beforeUnavailableStale=stale;
createStudioSurfaceControls(unavailableContainer,fields,editor.backdrops,names.filter(name=>!['white_plaster','white_gloss_paint','black_gloss_paint'].includes(name)),
  {onChange:()=>stale++,onRoomChange:()=>room++,onBackdropChange:()=>backdrop++,syncBackdropVisibility:editor.syncBackdropVisibility});
const unavailableNodes=all(unavailableContainer);
for(const id of ['studio-surface-white-matte','studio-surface-white-gloss','studio-surface-black-gloss','studio-concentric-enabled'])
  assert.equal(unavailableNodes.find(node=>node.id===id).disabled,true);
assert.equal(unavailableNodes.find(node=>node.id==='studio-surface-black-matte').disabled,false);
unavailableNodes.find(node=>node.id==='studio-surface-white-matte').fire('click');
assert.equal(stale,beforeUnavailableStale);assert.deepEqual(editor.backdrops.getState(),beforeUnavailable);

for (const scale of [1,3,6]) for (const inverted of [false,true]) for (const floorDrop of [0,3]) {
  const layers=concentricBackdropLayers(scale,inverted,floorDrop),bounds=studioRoomBounds(scale,floorDrop);
  const inner=layers.at(-1),outer=layers[0],height=bounds.ceilingY-bounds.floorY;
  assert.equal(inner.width,height);assert.equal(inner.centerY,(bounds.ceilingY+bounds.floorY)/2);
  assert.equal(outer.width,.98*Math.hypot(bounds.rightX-bounds.leftX,height));
  assert.equal(inner.material,inverted?'black_felt':'white_plaster');
  const fs=backdropFaces({count:6,material:'white_plaster',sweep:false,layers,wallZ:bounds.backZ},ids);
  assert.equal(fs.length,6*BACKDROP_CIRCLE_SEGMENTS);
  assert.deepEqual(backdropFaces({count:6,material:'white_plaster',sweep:true,layers,wallZ:bounds.backZ},ids),fs);
  layers.forEach((layer,i)=>{
    assert.equal(layer.width,layer.height);assert.equal(layer.shape,'ellipse');
    if(i) {assert.ok(layer.width<layers[i-1].width);assert.notEqual(layer.material,layers[i-1].material);}
    const f=fs[i*BACKDROP_CIRCLE_SEGMENTS];
    assert.equal(f[3],ids[layer.material]);assert.deepEqual(f[0],[0,layer.centerY,bounds.backZ+.012*(i+1)]);
    assert.equal(f[1][0],layer.width/2);
    // A point halfway through each ring is inside the rear wall at a diagonal
    // angle. Thus every band has visible area despite natural wall clipping.
    const radius=i===5?inner.width/4:(layer.width+layers[i+1].width)/4;
    const angle=Math.atan2(height,bounds.rightX-bounds.leftX);
    assert.ok(radius*Math.cos(angle)<bounds.rightX);
    assert.ok(layer.centerY+radius*Math.sin(angle)<bounds.ceilingY);
  });
}
// Table material, enabled state and floor-mounted poses survive profile loads.
const beforeTablePoses=editor.poses.getState(),tableMaterial=doc.getElementById('studio-table-material');
tableToggle.checked=true;tableToggle.fire('change');
editor.poses.getState().poses.forEach((pose,i)=>assert.equal(pose.pos[1],beforeTablePoses.poses[i].pos[1]-([2,3,4,5].includes(i)?3:0)));
const tablePoses=editor.poses.getState(),beforeTableMaterialStale=stale;
tableMaterial.value='white_plaster';tableMaterial.fire('change');
assert.deepEqual(editor.poses.getState(),tablePoses);assert.equal(stale,beforeTableMaterialStale+1);
name.value='White table';save.fire('click');
const whiteTableSaved=JSON.parse(storage.get(STORAGE_KEY)).profiles.find(p=>p.name==='White table').state;
assert.deepEqual(whiteTableSaved.table,{...defaultStudioTable(),width:5.2,length:3,height:3.02,material:'white_plaster',skirt:{enabled:false,material:'black_felt'}});
tableMaterial.value='marble_clear_glass';tableMaterial.fire('change');
tableToggle.checked=false;tableToggle.fire('change');
selection.value='White table';const beforeTableLoadStale=stale;load.fire('click');
assert.equal(stale,beforeTableLoadStale+1,'Profile invalidates only after the whole scene is applied');
assert.deepEqual(editor.table.getState(),whiteTableSaved.table);assert.deepEqual(editor.poses.getState(),whiteTableSaved.poses);
const badTable=copy(whiteTableSaved);badTable.table.material='missing';
storage.set(STORAGE_KEY,JSON.stringify({version:1,profiles:[{name:'Bad table',state:badTable}]}));
selection.value='Bad table';load.fire('click');
assert.deepEqual(editor.table.getState(),whiteTableSaved.table);assert.deepEqual(editor.poses.getState(),whiteTableSaved.poses);
console.log('studio lighting: legacy + split profiles, table materials/poses, surface matrix, concentric inversion/geometry and Advanced restoration passed');

// Optional gobo state is profile-owned. It never migrates older storage on read
// and invalid material/pose data cannot partially apply any other scene field.
assert.deepEqual(editor.gobos.getState(),{version:1,gobos:[]});
doc.getElementById('studio-gobo-add').fire('click');
let goboId=editor.gobos.getState().gobos[0].id;
for(const [key,value,event] of [['shape','ellipse','change'],['front','metal_gold','change'],['back','white_plaster','change'],
  ['width','2.7','input'],['height','1.3','input'],['x','-3.5','input'],['y','1.7','input'],['z','4.1','input'],
  ['pitch','19','input'],['yaw','-37','input'],['roll','83','input']]) {
  const node=doc.getElementById(`studio-gobo-${goboId}-${key}`);node.value=value;node.fire(event);
}
storage.set(STORAGE_KEY,JSON.stringify({version:1,profiles:[]}));
name.value='With gobo';save.fire('click');
const goboSaved=JSON.parse(storage.get(STORAGE_KEY)).profiles.find(p=>p.name==='With gobo').state;
assert.deepEqual(goboSaved.gobos,editor.gobos.getState());
editor.gobos.setState(undefined);selection.value='With gobo';const beforeGoboLoad=stale;load.fire('click');
assert.deepEqual(editor.gobos.getState(),goboSaved.gobos);assert.equal(stale,beforeGoboLoad+1);
assert.deepEqual(editor.table.getState(),goboSaved.table);assert.deepEqual(editor.strengths.getState(),goboSaved.strengths);
assert.deepEqual(editor.surfaces.getState(),goboSaved.surfaces);assert.deepEqual(editor.poses.getState(),goboSaved.poses);
for(const change of [value=>value.gobos.gobos[0].front='missing',value=>value.gobos.gobos[0].yaw=181]) {
  const bad=copy(goboSaved);change(bad);bad.controls['room-scale']=6;
  const text=JSON.stringify({version:1,profiles:[{name:'Invalid gobo',state:bad}]});storage.set(STORAGE_KEY,text);
  selection.value='Invalid gobo';const was=stale;load.fire('click');
  assert.deepEqual(editor.gobos.getState(),goboSaved.gobos);assert.equal(stale,was);
  assert.equal(Number(elements['room-scale'].value),goboSaved.controls['room-scale']);
  assert.equal(storage.get(STORAGE_KEY),text,'Rejected source bytes are preserved');
}
const oldNoGobos=copy(goboSaved);delete oldNoGobos.gobos;
const oldText=JSON.stringify({version:1,profiles:[{name:'Before gobos',state:oldNoGobos}]});storage.set(STORAGE_KEY,oldText);
selection.value='Before gobos';load.fire('click');
assert.deepEqual(editor.gobos.getState(),{version:1,gobos:[]});
assert.deepEqual(editor.table.getState(),oldNoGobos.table);assert.deepEqual(editor.poses.getState(),oldNoGobos.poses);
assert.equal(storage.get(STORAGE_KEY),oldText,'Legacy profiles load without an implicit rewrite');
assert.match(html,/<details id="studio-surface-advanced"[^>]*>[\s\S]*<details id="studio-gobo-advanced"[^>]*><summary>Gobos<\/summary>/);
assert(!/<details id="studio-gobo-advanced"[^>]*\bopen\b/.test(html));
console.log('Gobo profiles: exact roundtrip with all other state, legacy empty/no-rewrite and atomic rejection passed');

// The simple height is only a shortcut to the profile-owned underlight pose.
const height=elements['underlight-height'], underIndex=STUDIO_LIGHT_IDS.indexOf('underlight');
const beforeHeight=editor.poses.getState(), beforeHeightChanges=stale;
height.value='-1.83';height.fire('input');
const afterHeight=editor.poses.getState();
assert.equal(afterHeight.poses[underIndex].pos[1],-1.83);
const expectedHeight=copy(beforeHeight);expectedHeight.poses[underIndex].pos[1]=-1.83;
assert.deepEqual(afterHeight,expectedHeight,'Shortcut changes only underlight Y, retaining aim and other poses');
assert.equal(stale,beforeHeightChanges+1);
for(const bad of ['', 'NaN', '51', '-51']) {
  height.value=bad;height.fire('input');
  assert.deepEqual(editor.poses.getState(),afterHeight);
  assert.equal(stale,beforeHeightChanges+1,'Invalid input does not invalidate the scene');
}
height.value='-1.83';height.fire('input');
assert.equal(elements['underlight-height-status'].textContent,'','Restoring the unchanged valid height clears the error');
assert.equal(stale,beforeHeightChanges+1);
const advancedHeight=copy(afterHeight);advancedHeight.poses[underIndex].pos[1]=-2.125;
editor.poses.setState(advancedHeight);assert.equal(height.value,'-2.125');
storage.set(STORAGE_KEY,JSON.stringify({version:1,profiles:[]}));
name.value='Underlight position';save.fire('click');
const heightSaved=JSON.parse(storage.get(STORAGE_KEY)).profiles[0].state;
assert.equal(heightSaved.poses.poses[underIndex].pos[1],-2.125);
assert.equal(heightSaved.controls['underlight-height'],undefined,'No competing profile field is added');
height.value='-4';height.fire('input');selection.value='Underlight position';load.fire('click');
assert.equal(height.value,'-2.125');assert.deepEqual(editor.poses.getState(),heightSaved.poses);
const oldUnder=copy(heightSaved);oldUnder.poses.poses[underIndex].pos[1]=-1.4;
const oldUnderText=JSON.stringify({version:1,profiles:[{name:'Original underlight',state:oldUnder}]});
storage.set(STORAGE_KEY,oldUnderText);selection.value='Original underlight';load.fire('click');
assert.equal(height.value,'-1.4');assert.equal(storage.get(STORAGE_KEY),oldUnderText);
assert(editor.ownsPoseShortcut('underlight-height'));assert(!editor.ownsPoseShortcut('light-under-strength'));
console.log('Underlight height: exact pose-only editing, validation, Advanced sync and old/new profile preservation passed');

// Every placement shortcut authors the existing pose, not a second profile field.
const near=(a,b)=>assert(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const displayNear=(a,b)=>assert(Math.abs(a-b)<=.0005,`${a} != ${b} to 3 decimals`);
let placement=editor.poses.getState();
placement.poses[7]={...placement.poses[7],pos:[-1.85,-.72,.1],yaw:17,pitch:-23,aimAtSubject:false};
placement.poses[8]={...placement.poses[8],pos:[1.85,-.72,.1],yaw:-41,pitch:32,aimAtSubject:true};
placement.poses[6]={...placement.poses[6],pos:[0,3.2,.3],aimAtSubject:true};
const legacyPlacementTable={...defaultStudioTable(),width:5.2,length:3,height:3.02};
editor.table.setState(legacyPlacementTable);editor.poses.setState(placement);
const leftArc=elements['table-left-arc'],leftRadius=elements['table-left-radius'];
const rightArc=elements['table-right-arc'],rightRadius=elements['table-right-radius'];
for(const [input,id] of [[leftArc,'table-left-arc'],[leftRadius,'table-left-radius'],[rightArc,'table-right-arc'],[rightRadius,'table-right-radius']]) {
  assert(editor.ownsPoseShortcut(id));assert(input);
}
displayNear(Number(leftArc.value),Math.atan2(-1.85,.1)*180/Math.PI);
displayNear(Number(leftRadius.value),Math.hypot(1.85,.1,.72));
const beforeArc=editor.poses.getState(),beforeArcStale=stale;
leftArc.value='-120';leftArc.fire('input');
const afterArc=editor.poses.getState(),expectedArc=copy(beforeArc),originalRadius=Math.hypot(1.85,.1);
expectedArc.poses[7].pos[0]=originalRadius*Math.sin(-120*Math.PI/180);
expectedArc.poses[7].pos[2]=originalRadius*Math.cos(-120*Math.PI/180);
for(let i=0;i<3;i++)near(afterArc.poses[7].pos[i],expectedArc.poses[7].pos[i]);
assert.deepEqual({...afterArc.poses[7],pos:expectedArc.poses[7].pos},expectedArc.poses[7]);assert.equal(stale,beforeArcStale+1);
leftRadius.value='2.2';leftRadius.fire('input');
near(editor.poses.getState().poses[7].pos[0],2.2*originalRadius/Math.hypot(originalRadius,.72)*Math.sin(-120*Math.PI/180));
near(editor.poses.getState().poses[7].pos[2],2.2*originalRadius/Math.hypot(originalRadius,.72)*Math.cos(-120*Math.PI/180));
assert.deepEqual(editor.poses.getState().poses[8],beforeArc.poses[8]);
const beforeRight=editor.poses.getState();rightArc.value='60';rightArc.fire('input');rightRadius.value='2';rightRadius.fire('input');
assert.deepEqual(editor.poses.getState().poses[7],beforeRight.poses[7]);
near(editor.poses.getState().poses[8].pos[0],Math.sqrt(3)*originalRadius/Math.hypot(originalRadius,.72));near(editor.poses.getState().poses[8].pos[2],originalRadius/Math.hypot(originalRadius,.72));
assert.equal(editor.poses.getState().poses[8].aimAtSubject,true);
assert.equal(editor.poses.getState().poses[8].yaw,beforeRight.poses[8].yaw);
assert.equal(editor.poses.getState().poses[8].pitch,beforeRight.poses[8].pitch);
near(editor.poses.getState().poses[8].pos[1],2*(-.72)/Math.hypot(originalRadius,.72));
const beforeBadArc=editor.poses.getState(),beforeBadArcStale=stale;
for(const bad of ['', 'NaN', '-181', '181']) {leftArc.value=bad;leftArc.fire('input');assert.deepEqual(editor.poses.getState(),beforeBadArc);}
for(const bad of ['', '-1', '71', 'Infinity']) {rightRadius.value=bad;rightRadius.fire('input');assert.deepEqual(editor.poses.getState(),beforeBadArc);}
assert.equal(stale,beforeBadArcStale);
rightRadius.value='70';rightRadius.fire('input');assert.deepEqual(editor.poses.getState(),beforeBadArc,'Reject X/Z beyond pose bounds');
assert.match(elements['table-right-placement-status'].textContent,/outside -50 to 50/);
editor.poses.setState(beforeBadArc);
const poseNodes=all(elements['light-pose-controls']);
const poseSelect=poseNodes.find(n=>n.dataset.lightingPose==='fixture');poseSelect.value='table-left';poseSelect.fire('change');
const advancedX=poseNodes.find(n=>n.dataset.lightingPose==='X'&&n.type==='number');
advancedX.value='-3.5';advancedX.fire('input');
const advancedLeft=editor.poses.getState().poses[7];
displayNear(Number(leftRadius.value),Math.hypot(...advancedLeft.pos));
displayNear(Number(leftArc.value),Math.atan2(advancedLeft.pos[0],advancedLeft.pos[2])*180/Math.PI);
assert.match(elements['table-left-placement-status'].textContent,/footprint/);
assert.equal(advancedLeft.pos[0],-3.5,'Warning never relocates a saved/Advanced pose');

const topHeight=elements['top-fill-height'],beforeTop=editor.poses.getState();
topHeight.value='0';topHeight.fire('input');
const atSubject=editor.poses.getState(),expectedTop=copy(beforeTop);expectedTop.poses[6].pos[1]=0;
assert.deepEqual(atSubject,expectedTop);assert.match(elements['top-fill-height-status'].textContent,/overlaps.*subject/);
assert.equal(topHeight.min,'0');assert(editor.ownsPoseShortcut('top-fill-height'));
const atSubjectStale=stale;topHeight.fire('input');assert.equal(stale,atSubjectStale,'Unchanged valid height is idempotent');
for(const bad of ['', '-.01', '51', 'NaN']) {topHeight.value=bad;topHeight.fire('input');assert.deepEqual(editor.poses.getState(),atSubject);}
const clearFill=copy(atSubject);clearFill.poses[6].pos=[3,0,.3];editor.poses.setState(clearFill);
assert.equal(elements['top-fill-height-status'].textContent,'');assert.equal(topHeight.value,'0');

// Resizing never moves stage lamps. Only explicit height edits translate the
// existing floor-mounted slots, while slab contact and subject remain fixed.
const tableWidth=doc.getElementById('studio-table-width'),tableLength=doc.getElementById('studio-table-length');
const tableHeight=doc.getElementById('studio-table-height');
const beforeDimensions=editor.poses.getState();
tableWidth.value='9';tableWidth.fire('input');tableLength.value='5';tableLength.fire('input');
assert.deepEqual(editor.poses.getState(),beforeDimensions);
assert.equal(editor.table.getState().width,9);assert.equal(editor.table.getState().length,5);
assert.match(elements['table-left-placement-status'].textContent,/below the tabletop/);
assert(editor.table.ownsControl(tableWidth.id));assert(editor.table.ownsControl(tableHeight.id));
tableHeight.value='5';tableHeight.fire('input');
editor.poses.getState().poses.forEach((pose,i)=>{
  const expected=copy(beforeDimensions.poses[i]);if([2,3,4,5].includes(i))expected.pos[1]+=3-4.98;
  assert.deepEqual(pose,expected);
});
near(tableFloorDrop(editor.table.getState()),4.98);
const afterDimensions=editor.table.getState(),afterDimensionPoses=editor.poses.getState(),afterDimensionStale=stale;
for(const [input,badValues] of [[tableWidth,['','1.99','20.01','NaN']],[tableLength,['0','Infinity']],[tableHeight,['.29','10.01']]])
  for(const bad of badValues) {input.value=bad;input.fire('input');assert.deepEqual(editor.table.getState(),afterDimensions);assert.deepEqual(editor.poses.getState(),afterDimensionPoses);}
assert.equal(stale,afterDimensionStale);
const nearlyLimited=copy(afterDimensionPoses);nearlyLimited.poses[2].pos[1]=-49;editor.poses.setState(nearlyLimited);
const beforeLimitedStale=stale;tableHeight.value='10';tableHeight.fire('input');
assert.deepEqual(editor.table.getState(),afterDimensions);assert.deepEqual(editor.poses.getState(),nearlyLimited);
assert.equal(stale,beforeLimitedStale);assert.match(doc.getElementById('studio-table-dimension-status').textContent,/floor-mounted light/);
editor.poses.setState(afterDimensionPoses);
elements['room-scale'].value='1';elements['room-scale'].fire('input');
assert.match(doc.getElementById('studio-table-placement-status').textContent,/room walls/);
elements['room-scale'].value='3';elements['room-scale'].fire('input');
assert.doesNotMatch(doc.getElementById('studio-table-placement-status').textContent,/room walls/);
tableHeight.value='.3';tableHeight.fire('input');
assert.match(doc.getElementById('studio-table-placement-status').textContent,/below the room floor/);
assert.equal(editor.poses.getState().poses[9].pos[1],afterDimensionPoses.poses[9].pos[1]);
tableHeight.value='5';tableHeight.fire('input');
// New dimensions and all manual/locked poses round-trip; no polar/Y duplicates.
storage.set(STORAGE_KEY,JSON.stringify({version:1,profiles:[]}));name.value='Sized placement';save.fire('click');
const sized=JSON.parse(storage.get(STORAGE_KEY)).profiles[0].state;
assert.deepEqual(sized.table,{version:1,enabled:true,material:'marble_milk_glass',width:9,length:5,height:5,skirt:{enabled:true,material:'black_felt'}});
for(const id of ['table-left-arc','table-left-radius','table-right-arc','table-right-radius','top-fill-height'])assert.equal(sized.controls[id],undefined);
tableWidth.value='2';tableWidth.fire('input');leftRadius.value='.1';leftRadius.fire('input');
selection.value='Sized placement';load.fire('click');assert.deepEqual(editor.poses.getState(),sized.poses);assert.deepEqual(editor.table.getState(),sized.table);
assert.equal(tableWidth.value,'9');assert.equal(tableLength.value,'5');assert.equal(tableHeight.value,'5');
displayNear(Number(leftRadius.value),Math.hypot(...sized.poses.poses[7].pos));
const preDimension=copy(sized);delete preDimension.table.width;delete preDimension.table.length;delete preDimension.table.height;
preDimension.poses.poses[6].pos[1]=-2.25;
const preDimensionText=JSON.stringify({version:1,profiles:[{name:'Before dimensions',state:preDimension}]});
storage.set(STORAGE_KEY,preDimensionText);selection.value='Before dimensions';load.fire('click');
assert.deepEqual(editor.table.getState(),legacyPlacementTable);assert.deepEqual(editor.poses.getState(),preDimension.poses);
assert.equal(topHeight.value,'-2.25','Saved below-subject poses load without clamping to the new shortcut bounds');
assert.equal(storage.get(STORAGE_KEY),preDimensionText,'Reading an old profile never rewrites its bytes');
console.log('Placement controls: independent polar poses, top-fill clearance, dimension limits/rollback, floor translation, Advanced sync and exact old/new profiles passed');

// Fixed scale is a rig-only profile field. Subject radius remains object state.
assert.deepEqual(saved.scale,defaultStudioScale());
assert.deepEqual(validateStudioLighting(legacy,fields,names).scale,{version:1,mode:'legacy'});
const unit=doc.getElementById('studio-unit-mm'),scaleMode=doc.getElementById('studio-scale-mode');
editor.scale.setState({version:1,mode:'fixed',unitMm:10});
elements['radius-mm'].value='1';elements['radius-mm'].fire('input');
const oldTopPose=copy(editor.poses.getState().poses[6]);
displayNear(Number(topHeight.value),oldTopPose.pos[1]+.9);
topHeight.value='0';topHeight.fire('input');displayNear(editor.poses.getState().poses[6].pos[1],-.9);
const tinyTop=copy(editor.poses.getState().poses[6]);
elements['radius-mm'].value='25';elements['radius-mm'].fire('input');
assert.deepEqual(editor.poses.getState().poses[6],tinyTop,'Resizing changes relative readout, never authored light pose');
displayNear(Number(topHeight.value),-2.4);
for(const v of ['', '0', '1001', 'Infinity']) {
  unit.value=v;unit.fire('input');assert.deepEqual(editor.scale.getState(),{version:1,mode:'fixed',unitMm:10});
}
unit.value='20';unit.fire('input');assert.deepEqual(editor.scale.getState(),{version:1,mode:'fixed',unitMm:20});
storage.set(STORAGE_KEY,JSON.stringify({version:1,profiles:[]}));name.value='Independent';save.fire('click');
const fixed=JSON.parse(storage.get(STORAGE_KEY)).profiles[0].state;
assert.equal(fixed.controls['radius-mm'],undefined);assert.equal(fixed.scale.unitMm,20);
unit.value='15';unit.fire('input');elements['radius-mm'].value='10';elements['radius-mm'].fire('input');
selection.value='Independent';load.fire('click');assert.deepEqual(editor.scale.getState(),fixed.scale);
assert.equal(elements['radius-mm'].value,'10','Loading a rig never overwrites physical object radius');
const legacyScale=copy(fixed);delete legacyScale.scale;
const oldScaleText=JSON.stringify({version:1,profiles:[{name:'Legacy scale',state:legacyScale}]});
storage.set(STORAGE_KEY,oldScaleText);selection.value='Legacy scale';load.fire('click');
assert.deepEqual(editor.scale.getState(),{version:1,mode:'legacy'});assert.equal(storage.get(STORAGE_KEY),oldScaleText);
elements['radius-mm'].value='25';elements['radius-mm'].fire('input');
scaleMode.value='fixed';scaleMode.fire('change');assert.deepEqual(editor.scale.getState(),{version:1,mode:'fixed',unitMm:25});
const invalidScale=copy(fixed);invalidScale.scale.unitMm=0;
assert.throws(()=>validateStudioLighting(invalidScale,fields,names),/Studio unit/);
console.log('Scale profiles: fixed/legacy migration, no storage rewrite or duplicate radius, atomic invalid units and relative top-fill reach passed');

// Quick slots reuse the complete validated profile owner, not partial slider copies.
const quickSaved = new Map();
for (const [index,slot] of ['C','D'].entries()) {
  elements['light-strength-left'].value = String(10+index);
  doc.getElementById(`lighting-store-${slot}`).fire('click');
  quickSaved.set(slot, editor.strengths.getStrengths());
  assert.equal(doc.getElementById(`lighting-quick-${slot}`).disabled, false);
}
for (const slot of ['D','C','D','C']) {
  doc.getElementById('key-right-solo').fire('click');
  const before = stale;
  doc.getElementById(`lighting-quick-${slot}`).fire('click');
  assert.deepEqual(editor.strengths.getStrengths(), quickSaved.get(slot));
  assert.equal(editor.strengths.getSolo(), null);
  assert.equal(stale, before+1);
}
console.log('Quick lighting A/B/C/D: complete profile recall, repeated switching, and transient solo restoration passed.');

// A owns exposure; B overrides are resolved only for the room/Phong route.
assert.equal(doc.getElementById('lighting-quick-A').textContent,'A (exposure)');
assert.equal(doc.getElementById('lighting-quick-B').textContent,'B (room)');
doc.getElementById('lighting-quick-B').fire('click');
assert.equal(editor.roomOverrides.element.hidden,false);
const exposureBaseline=copy(editor.renderState('exposure'));
let beforeOverride=stale;
function roomEdit(path,value) {
  const id=path.replaceAll('/','-');let lock=doc.getElementById('room-unlock-'+id);
  if(!lock.checked){lock.checked=true;lock.fire('change');}
  const input=doc.getElementById('room-value-'+id);
  if(typeof value==='boolean')input.checked=value;else input.value=String(value);
  input.fire('change');
}
const roomGroups=editor.roomOverrides.element.children.at(-1);roomGroups.children[0].open=true;const originalRoomGroup=roomGroups.children[0];
roomEdit('controls/key-filter-left','#123456');
assert.equal(roomGroups.children[0].open,true,'Editing a value must not collapse its group');
assert.equal(roomGroups.children[0],originalRoomGroup,'Editing preserves the active DOM branch for bubbling event containment');
roomEdit('controls/light-strength-left',10);
roomEdit('poses/poses/@key-left/pos/0',-9);
roomEdit('table/width',8);
roomEdit('backdrops/0/width',12);
roomEdit('backdrops/0/shape','ellipse');roomEdit('backdrops/0/centerY',2);
roomEdit('controls/wall-material-a','black_felt');
roomEdit('scale/unitMm',20);
assert.equal(stale,beforeOverride,'Room edits never invalidate photo/live');
assert.deepEqual(editor.renderState('exposure').controls,exposureBaseline.controls);
assert.deepEqual(editor.renderState('exposure').poses,exposureBaseline.poses);
assert.deepEqual(editor.renderState('exposure').table,exposureBaseline.table);
const resolved=editor.renderState('studio');
assert.equal(resolved.controls['key-filter-left'],'#123456');assert.equal(resolved.controls['light-strength-left'],100);
assert.equal(resolved.backdrops[0].shape,'ellipse');assert.equal(resolved.backdrops[0].centerY,2);
assert.equal(resolved.poses.poses[0].pos[0],-9);assert.equal(resolved.table.width,8);assert.equal(resolved.backdrops[0].width,12);assert.equal(resolved.scale.unitMm,20);
const badRoom=doc.getElementById('room-value-table-width');badRoom.value='-1';badRoom.fire('change');
assert.equal(editor.renderState('studio').table.width,8);
const lock=doc.getElementById('room-unlock-controls-light-strength-left');lock.checked=false;lock.fire('change');
assert.equal(editor.renderState('studio').controls['light-strength-left'],exposureBaseline.controls['light-strength-left']);
elements['light-strength-left'].value='170';elements['light-strength-left'].fire('input');
assert.equal(editor.renderState('studio').controls['light-strength-left'],170);
assert.equal(editor.renderState('studio').controls['key-filter-left'],'#123456','Unlocked override survives A edits');
const roomSaved=copy(editor.renderState('studio'));doc.getElementById('lighting-store-D').fire('click');
roomEdit('table/width',9);doc.getElementById('lighting-quick-D').fire('click');
assert.deepEqual(editor.renderState('studio'),roomSaved);
doc.getElementById('lighting-quick-A').fire('click');assert.equal(editor.roomOverrides.element.hidden,true);
console.log('A exposure/B room: independent filter, level, pose, material, backdrop, table and scale overrides; relock/follow, invalid edits, full slot persistence and zero ray invalidations passed');

const roomBeforeCalibration=copy(editor.renderState('studio'));
elements['lighting-profiles'].fire('room-calibration-start');
assert.deepEqual(editor.renderState('studio'),editor.renderState('exposure'));
elements['lighting-profiles'].fire('room-calibration-end');
assert.deepEqual(editor.renderState('studio'),roomBeforeCalibration);
console.log('Calibration temporarily follows exposure and restores the Room override state unchanged');

const outline=doc.getElementById('studio-concentric-shape');
outline.value='rectangle';outline.fire('change');
assert.equal(editor.surfaces.getState().shape,'rectangle');
const rectangular=concentricBackdropLayers(3,false,3,'rectangle');
assert.ok(rectangular.every(layer=>layer.shape==='rectangle'));
assert.throws(()=>concentricBackdropLayers(3,false,3,'invalid'));
console.log('Concentric outline selection and invalid-shape rejection passed');

if(!editor.surfaces.getState().concentric){rig.checked=true;rig.fire('change');}
assert.ok(editor.backdrops.getState().every(layer=>layer.shape==='rectangle'));
name.value='Rectangular outline';save.fire('click');
const rectangleProfile=JSON.parse(storage.get(STORAGE_KEY)).profiles.find(p=>p.name==='Rectangular outline');
assert.ok(rectangleProfile,JSON.stringify(profileNodes.filter(n=>n.role==='status'||n['role']==='status').map(n=>n.textContent)));
assert.equal(rectangleProfile.state.surfaces.shape,'rectangle');
assert.ok(validateStudioLighting(rectangleProfile.state,fields,names).backdrops.every(layer=>layer.shape==='rectangle'));
outline.value='ellipse';outline.fire('change');
selection.value='Rectangular outline';load.fire('click');
assert.equal(outline.value,'rectangle');
console.log('Rectangular rig full-profile validation/save/load and authored geometry passed');

const skirtToggle=doc.getElementById('studio-table-skirt-enabled'),skirtPicker=doc.getElementById('studio-table-skirt-material');
skirtToggle.checked=true;skirtToggle.fire('change');skirtPicker.value='white_plaster';skirtPicker.fire('change');
name.value='Fabric skirt';save.fire('click');
const skirtProfile=JSON.parse(storage.get(STORAGE_KEY)).profiles.find(p=>p.name==='Fabric skirt');
assert.deepEqual(skirtProfile.state.table.skirt,{enabled:true,material:'white_plaster'});
skirtToggle.checked=false;skirtToggle.fire('change');assert.equal(editor.table.getState().skirt.enabled,false);
selection.value='Fabric skirt';load.fire('click');assert.equal(skirtToggle.checked,true);assert.equal(skirtPicker.value,'white_plaster');
console.log('Skirt controls, material choice and exact profile roundtrip passed');

// Controlled fixture extents exercise UI rejection independently of geometry.
clearanceVertices=(id,state)=>{const y=state.poses.find(p=>p.id===id).pos[1];return [[0,y-.2,0],[0,y+.3,0]];};
const safeState=editor.poses.getState();
safeState.poses.find(p=>p.id==='underlight').pos=[0,-2,0];
safeState.poses.find(p=>p.id==='fill-upper').pos=[0,5,0];editor.poses.setState(safeState);
const underControl=elements['underlight-height'],overControl=elements['top-fill-height'];
let current=copy(editor.poses.getState());underControl.value='-1.2';underControl.fire('input');
assert.deepEqual(editor.poses.getState(),current,'Reject underlight grid crossing tabletop underside');
assert.match(elements['underlight-height-status'].textContent,/underlight|Underlight/);
underControl.value='-49';underControl.fire('input');assert.deepEqual(editor.poses.getState(),current,'Reject underlight below floor');
underControl.value='-2.1';underControl.fire('input');assert.equal(editor.poses.getState().poses.find(p=>p.id==='underlight').pos[1],-2.1);
current=copy(editor.poses.getState());overControl.value='0';overControl.fire('input');assert.deepEqual(editor.poses.getState(),current,'Reject overlight through subject top');
overControl.value='49';overControl.fire('input');assert.deepEqual(editor.poses.getState(),current,'Reject overlight above ceiling');
const previousMax=Number(overControl.max);elements['room-scale'].value='4';elements['room-scale'].fire('input');assert.notEqual(Number(overControl.max),previousMax,'Room edits refresh actual extent bounds');
console.log('Underlight floor/table and overlight subject/ceiling rejection and dynamic limits passed');

const widthControl=doc.getElementById('fixture-size-key-left-width');
widthControl.value='3';widthControl.fire('change');
assert.equal(editor.fixtureSizes.getState().lights[0].width,3);
name.value='Fixture dimensions';save.fire('click');
widthControl.value='4';widthControl.fire('change');selection.value='Fixture dimensions';load.fire('click');
assert.equal(widthControl.value,'3');
assert.equal(editor.renderState('exposure').fixtureSizes.lights[0].width,3);
doc.getElementById('lighting-quick-B').fire('click');
roomEdit('fixtureSizes/lights/@key-left/width',2.5);
assert.equal(editor.renderState('studio').fixtureSizes.lights[0].width,2.5);
assert.equal(editor.renderState('exposure').fixtureSizes.lights[0].width,3);
console.log('Fixture dimension UI save/load and Room-only size override passed');

doc.getElementById('lighting-quick-B').fire('click');
const roomPoseBefore=copy(editor.renderState('studio').poses);
roomEdit('poses/poses/@underlight/pos/1',-1.2);
assert.deepEqual(editor.renderState('studio').poses,roomPoseBefore,'Room cannot move its underlight through tabletop');
assert.equal(editor.renderState('exposure').poses.poses.find(p=>p.id==='underlight').pos[1],-2.1);
console.log('Room underlight clipping rejection preserves both channel poses');

doc.getElementById('lighting-quick-A').fire('click');
const gridCells=doc.getElementById('light-tool-underlight-grid-cellsX'),gridDepth=doc.getElementById('light-tool-underlight-grid-depth');
gridCells.value='3';gridCells.fire('change');gridDepth.value='.3';gridDepth.fire('change');
name.value='Underlight grid';save.fire('click');
const gridProfile=JSON.parse(storage.get(STORAGE_KEY)).profiles.find(p=>p.name==='Underlight grid').state;
assert.equal(gridProfile.lightTools.attachments.find(t=>t.id==='underlight-grid').settings.cellsX,3);assert.equal(gridProfile.lightTools.attachments.find(t=>t.id==='underlight-grid').settings.depth,.3);
gridCells.value='4';gridCells.fire('change');selection.value='Underlight grid';load.fire('click');assert.equal(doc.getElementById('light-tool-underlight-grid-cellsX').value,'3');
console.log('Underlight grid controls and lighting-profile roundtrip passed');

const targetPitch=doc.getElementById('light-tool-underlight-grid-pitch-width');
const gridWidth=1.8*editor.fixtureSizes.getState().lights[9].width;
targetPitch.value=String(gridWidth/5);targetPitch.fire('change');
assert.equal(editor.lightTools.getState().attachments.find(t=>t.id==='underlight-grid').settings.cellsX,5);
assert.match(doc.getElementById('light-tool-underlight-grid-pitch').textContent,/Cell pitch/);
const pitchState=JSON.stringify(editor.lightTools.getState());targetPitch.value='0';targetPitch.fire('change');assert.equal(JSON.stringify(editor.lightTools.getState()),pitchState);
console.log('Direct requested cell pitch derives whole tessellation count and rejects invalid size');

doc.getElementById('lighting-quick-B').fire('click');assert.equal(editor.fixtureSizes.element.hidden,true);
doc.getElementById('lighting-quick-A').fire('click');assert.equal(editor.fixtureSizes.element.hidden,false);
console.log('Canonical Exposure accessories hide in Room editing mode');

clearanceVertices=null;elements['radius-mm'].value='10';editor.scale.setState({version:1,mode:'fixed',unitMm:10});
doc.getElementById('lighting-quick-A').fire('click');
const radialState=editor.poses.getState();radialState.poses.find(p=>p.id==='kicker-left').pos=[-2,2,-2];radialState.poses.find(p=>p.id==='key-left').pos=[-2,2,2];radialState.poses.find(p=>p.id==='table-left').pos=[-2,0,2];editor.poses.setState(radialState);
for(const id of ['key-left','kicker-left']){elements[id+'-arc'].value='45';elements[id+'-arc'].fire('input');elements[id+'-radius'].value='2';elements[id+'-radius'].fire('input');}
const radials=editor.poses.getState();near(radials.poses[0].pos[0],-Math.sqrt(2));near(radials.poses[0].pos[2],Math.sqrt(2));near(radials.poses[3].pos[0],-Math.sqrt(2));near(radials.poses[3].pos[2],-Math.sqrt(2));
elements['kicker-left-arc'].value='180';elements['kicker-left-arc'].fire('input');near(editor.poses.getState().poses[3].pos[2],2);
elements['kicker-left-wall-gap'].value='2';elements['kicker-left-wall-gap'].fire('input');near(editor.poses.getState().poses[3].pos[2],3.1*Number(elements['room-scale'].value)-2);
elements['table-left-arc'].value='-45';elements['table-left-arc'].fire('input');elements['table-left-elevation'].value='30';elements['table-left-elevation'].fire('input');elements['table-left-radius'].value='2';elements['table-left-radius'].fire('input');
const stage3d=editor.poses.getState().poses[7];near(stage3d.pos[0],-Math.sqrt(1.5));near(stage3d.pos[1],1);near(stage3d.pos[2],Math.sqrt(1.5));assert.equal(stage3d.aimAtSubject,radialState.poses[7].aimAtSubject);
name.value='Radial placement';save.fire('click');const radialSaved=copy(editor.poses.getState());elements['table-left-elevation'].value='-20';elements['table-left-elevation'].fire('input');selection.value='Radial placement';load.fire('click');assert.deepEqual(editor.poses.getState(),radialSaved);
const beforeWallResize=copy(editor.poses.getState());elements['room-scale'].value='5';elements['room-scale'].fire('input');assert.deepEqual(editor.poses.getState(),beforeWallResize,'Room resize updates views without silently moving saved lights');
console.log('Key/kicker reference angles, back-wall inset, stage spherical controls and exact pose persistence passed');

// Attachment lifecycle is per light and all saved settings use stable tool IDs.
clearanceVertices=null;
doc.getElementById('lighting-quick-A').fire('click');
const toolsBefore=editor.lightTools.getState();
assert.equal(doc.getElementById('light-tool-key-left-grid-depth'),undefined);
const attach=doc.getElementById('light-tool-attach-key-left');attach.value='grid';attach.fire('change');
assert.ok(editor.lightTools.getState().attachments.some(t=>t.id==='key-left-grid'));
assert.ok(doc.getElementById('light-tool-key-left-grid-depth'));
const keyDepth=doc.getElementById('light-tool-key-left-grid-depth');keyDepth.value='.6';keyDepth.fire('change');
const exposureDepth=editor.lightTools.getState().attachments.find(t=>t.id==='key-left-grid').settings.depth;
assert.equal(exposureDepth,.6);
doc.getElementById('lighting-quick-B').fire('click');
roomEdit('lightTools/attachments/@key-left-grid/settings/depth',.8);
assert.equal(editor.renderState('studio').lightTools.attachments.find(t=>t.id==='key-left-grid').settings.depth,.8);
assert.equal(editor.renderState('exposure').lightTools.attachments.find(t=>t.id==='key-left-grid').settings.depth,.6);
doc.getElementById('lighting-quick-A').fire('click');
doc.getElementById('light-tool-remove-key-left-grid').fire('click');
assert.equal(doc.getElementById('light-tool-key-left-grid-depth'),undefined);
assert.ok(!editor.renderState('exposure').roomOverrides.values.some(v=>v.path.includes('@key-left-grid/')));
const again=doc.getElementById('light-tool-attach-key-left');again.value='grid';again.fire('change');
assert.equal(editor.lightTools.getState().attachments.filter(t=>t.lightId==='key-left').length,1);
assert.equal(editor.lightTools.getState().attachments.find(t=>t.id==='key-left-grid').settings.depth,.2);
name.value='Modular tools';save.fire('click');
const toolsSaved=JSON.parse(storage.get(STORAGE_KEY)).profiles.find(p=>p.name==='Modular tools').state;
assert.deepEqual(toolsSaved.lightTools,editor.lightTools.getState());
doc.getElementById('light-tool-remove-key-left-grid').fire('click');selection.value='Modular tools';load.fire('click');
assert.deepEqual(editor.lightTools.getState(),toolsSaved.lightTools);
console.log('Modular tools: per-light attach/remove, attached-only settings, repeated attachment, stable IDs, save/load and A/B isolation passed');

// A former Room override on the legacy underlight fields becomes a tool leaf.
const oldToolsProfile=copy(editor.renderState('exposure'));delete oldToolsProfile.lightTools;
oldToolsProfile.roomOverrides={version:1,values:[{path:'fixtureSizes/lights/@underlight/gridDepth',value:.7},{path:'fixtureSizes/lights/@kicker-left/snoot',value:false}]};
const migratedTools=validateStudioLighting(oldToolsProfile,Object.fromEntries(LIGHTING_PROFILE_FIELDS.map(id=>[id,elements[id]])),names);
assert.ok(migratedTools.roomOverrides.values.some(v=>v.path==='lightTools/attachments/@underlight-grid/settings/depth'&&v.value===.7));
assert.ok(migratedTools.roomOverrides.values.some(v=>v.path==='lightTools/attachments/@kicker-left-snoot/enabled'&&v.value===false));
assert.ok(!migratedTools.lightTools.attachments.some(t=>t.lightId==='fill-upper'));
assert.deepEqual(migratedTools.fixtureSizes,oldToolsProfile.fixtureSizes);
console.log('Legacy profile tool migration preserves old dimensions and maps prior Room accessory overrides');

// A shape edits must validate the attached grid before shrinking its footprint.
doc.getElementById('lighting-quick-A').fire('click');
const shapeControl=elements['key-shape'];shapeControl.value='bare';shapeControl.fire('change');
const denseCells=doc.getElementById('light-tool-key-left-grid-cellsX');denseCells.value='32';denseCells.fire('change');
const denseState=copy(editor.renderState('exposure')),denseTools=copy(editor.lightTools.getState());
assert.equal(denseTools.attachments.find(t=>t.id==='key-left-grid').settings.cellsX,32);
for(const event of ['input','change']){
 shapeControl.value='snoot';shapeControl.fire(event);
 assert.equal(shapeControl.value,'bare','Invalid footprint shrink immediately restores the previous accepted shape');
 assert.deepEqual(editor.renderState('exposure'),denseState);
 assert.deepEqual(editor.lightTools.getState(),denseTools);
 assert.match(doc.getElementById('key-shape-status').textContent,/cell pitch/);
}
name.value='Guarded dense grid';save.fire('click');
const denseSaved=JSON.parse(storage.get(STORAGE_KEY)).profiles.find(p=>p.name==='Guarded dense grid').state;
assert.deepEqual(denseSaved,denseState,'A rejected shape never leaks into profile storage');
denseCells.value='4';denseCells.fire('change');shapeControl.value='snoot';shapeControl.fire('change');
assert.equal(editor.renderState('exposure').controls['key-shape'],'snoot','A compatible narrower shape still works');
assert.equal(doc.getElementById('key-shape-status').textContent,'');
selection.value='Guarded dense grid';load.fire('click');
shapeControl.value='grid';shapeControl.fire('change');
assert.equal(shapeControl.value,'bare','Loading a profile resets the accepted-shape rollback target');
assert.deepEqual(editor.renderState('exposure'),denseSaved);
console.log('A key shape guard: immediate dense-grid shrink rejection, exact state/storage preservation, compatible edit and profile-load rollback target passed');

shapeControl.value='snoot';let propagationStopped=false,ancestorRan=false;
const shapeEvent={target:shapeControl,stopImmediatePropagation(){propagationStopped=true;},stopPropagation(){propagationStopped=true;}};
for(const handler of doc.events.input){handler(shapeEvent);if(propagationStopped)break;}
if(!propagationStopped){ancestorRan=true;editor.renderState('exposure');}
assert.equal(shapeControl.value,'bare');assert.equal(ancestorRan,false,'Invalid shape is rolled back during document capture before ancestor render/invalidation handlers');
assert.deepEqual(editor.renderState('exposure'),denseSaved);
console.log('Shape guard runs before ancestor capture readers and prevents an invalid intermediate render');

// One saved stand record belongs to A; B overrides only its two real leaves.
assert.deepEqual(saved.lightStands,{version:1,enabled:true,material:'black_matte_paint',upperSupport:'ceiling',wideSupports:true,backdropSupport:true});
const noStandProfile=copy(saved);delete noStandProfile.lightStands;
const noStandBytes=JSON.stringify(noStandProfile);
assert.equal(validateStudioLighting(noStandProfile,fields,names).lightStands.enabled,false);
assert.equal(JSON.stringify(noStandProfile),noStandBytes,'Legacy stand migration never rewrites input');
editor.lightStands.setState({version:1,enabled:true,material:'black_matte_paint'});editor.invalidateRenderState();
const exposureStands=copy(editor.renderState('exposure').lightStands),standPose=copy(editor.poses.getState());
doc.getElementById('lighting-quick-B').fire('click');
assert.equal(editor.lightStands.element.hidden,true);
const standEnableLock=doc.getElementById('room-unlock-lightStands-enabled');standEnableLock.checked=true;standEnableLock.fire('change');
const roomStandEnabled=doc.getElementById('room-value-lightStands-enabled');roomStandEnabled.checked=false;roomStandEnabled.fire('change');
const standMaterialLock=doc.getElementById('room-unlock-lightStands-material');standMaterialLock.checked=true;standMaterialLock.fire('change');
const roomStandMaterial=doc.getElementById('room-value-lightStands-material');roomStandMaterial.value='metal_gold';roomStandMaterial.fire('change');
assert.deepEqual(editor.renderState('exposure').lightStands,exposureStands);
assert.deepEqual(editor.renderState('studio').lightStands,{version:1,enabled:false,material:'metal_gold',upperSupport:'floor',wideSupports:false,backdropSupport:false});
assert.deepEqual(editor.poses.getState(),standPose);
roomStandMaterial.value='missing';roomStandMaterial.fire('change');assert.equal(editor.renderState('studio').lightStands.material,'metal_gold');
doc.getElementById('lighting-quick-A').fire('click');assert.equal(editor.lightStands.element.hidden,false);
name.value='Shared floor stands';save.fire('click');
const savedStands=JSON.parse(storage.get(STORAGE_KEY)).profiles.find(p=>p.name==='Shared floor stands').state;
assert.deepEqual(savedStands.lightStands,exposureStands);
assert(savedStands.roomOverrides.values.some(v=>v.path==='lightStands/material'&&v.value==='metal_gold'));
editor.lightStands.setEnabled(false);selection.value='Shared floor stands';load.fire('click');
assert.deepEqual(editor.lightStands.getState(),exposureStands);
assert.deepEqual(editor.renderState('studio').lightStands,{version:1,enabled:false,material:'metal_gold',upperSupport:'floor',wideSupports:false,backdropSupport:false});
console.log('Floor stands: new enabled/legacy disabled, canonical shared material, A/B visibility/material isolation, invalid material rollback, saved poses and profile roundtrip passed');

{
// New independent engine settings use existing named/quick profile persistence.
elements['scene-stage-engine'].value='dielectric';elements['scene-room-engine'].value='legacy';
editor.invalidateRenderState();name.value='Scene engine roundtrip';save.fire('click');
const sceneProfile=JSON.parse(storage.get(STORAGE_KEY)).profiles.find(p=>p.name===name.value).state;
assert.deepEqual(sceneProfile.sceneEngines,{version:1,stage:'dielectric',room:'legacy'});
elements['scene-stage-engine'].value='legacy';elements['scene-room-engine'].value='dielectric';editor.invalidateRenderState();
selection.value='Scene engine roundtrip';load.fire('click');
assert.equal(elements['scene-stage-engine'].value,'dielectric');assert.equal(elements['scene-room-engine'].value,'legacy');
const edited=copy(sceneProfile);edited.materialEdits={version:1,materials:[{name:'marble_milk_glass',tables:{pbr:{stride:16,data:[1,1,1,.1,0,.42,1.52,.79,0,0,0,0,0,0,0,0]}},optical:{material:4,wavelengths_nm:[450,550,650],transmittance:[null,.9,.8],reference_thickness_mm:null,concentration:1}}]};
assert.deepEqual(validateStudioLighting(copy(edited),elements,names).materialEdits,edited.materialEdits);
const historical=copy(sceneProfile);delete historical.sceneEngines;delete historical.materialEdits;
const migrated=validateStudioLighting(historical,elements,names);
assert.deepEqual(migrated.sceneEngines,{version:1,stage:'legacy',room:'legacy'});assert(!('materialEdits'in migrated));
assert.equal(historical.sceneEngines,undefined,'Validation never rewrites historical source profiles');
console.log('PASS independent scene-engine save/load, exact shared buffer/draft roundtrip and read-only Legacy migration.');

}

// The guide's floor-relative height field edits only the existing pose owner.
doc.getElementById('lighting-quick-A').fire('click');
const beforeGuideHeight=copy(editor.poses.getState()),guideFloor=studioRoomBounds(Number(elements['room-scale'].value),tableFloorDrop(editor.table.getState())).floorY;
const kickerHeight=doc.getElementById('kicker-left-height');assert.ok(kickerHeight);assert.ok(editor.ownsPoseShortcut(kickerHeight.id));
kickerHeight.value='4.25';kickerHeight.fire('input');
assert.equal(editor.poses.getState().poses[3].pos[1],guideFloor+4.25);
for(let i=0;i<10;i++)if(i!==3)assert.deepEqual(editor.poses.getState().poses[i],beforeGuideHeight.poses[i]);
assert.equal(editor.poses.getState().poses[3].aimAtSubject,beforeGuideHeight.poses[3].aimAtSubject);
name.value='Kicker guide height';save.fire('click');const savedGuidePose=copy(editor.poses.getState());
kickerHeight.value='4.5';kickerHeight.fire('input');selection.value='Kicker guide height';load.fire('click');
assert.deepEqual(editor.poses.getState(),savedGuidePose);assert.equal(kickerHeight.value,'4.25');
const guidePoseString=JSON.stringify(editor.poses.getState());kickerHeight.value='-1';kickerHeight.fire('input');assert.equal(JSON.stringify(editor.poses.getState()),guidePoseString);
editor.kickerGuides.dispose();
console.log('Kicker height integration: floor-relative edit uses existing pose, preserves other lights/lock, saves/loads exact pose and rejects invalid height');

// New shop records use the same A/B leaf owner and named/quick serialization.
{
assert.equal(saved.shopLights.fixtures.length,2);assert(saved.shopLights.fixtures.every(f=>f.enabled));
const historical=copy(saved);delete historical.shopLights;historical.roomOverrides.values=historical.roomOverrides.values.filter(v=>!v.path.startsWith('shopLights/'));const historyBytes=JSON.stringify(historical);
assert.deepEqual(validateStudioLighting(historical,fields,names).shopLights.fixtures,[]);
assert.equal(JSON.stringify(historical),historyBytes);
const a=copy(editor.renderState('exposure')),poseBytes=JSON.stringify(editor.poses.getState());
doc.getElementById('lighting-quick-B').fire('click');assert.equal(editor.shopLights.element.hidden,true);
for(const [path,value]of [['shopLights/fixtures/@shop-long/source','led-array'],['shopLights/fixtures/@shop-long/dimmer',.2],
 ['shopLights/fixtures/@shop-wide/enabled',false],['shopLights/backingMaterial','metal_gold']]){
 const suffix=path.replaceAll('/','-'),lock=doc.getElementById('room-unlock-'+suffix);assert(lock,path);lock.checked=true;lock.fire('change');
 const field=doc.getElementById('room-value-'+suffix);if(typeof value==='boolean')field.checked=value;else field.value=String(value);field.fire('change');
}
assert.deepEqual({...editor.renderState('exposure'),roomOverrides:a.roomOverrides},a);
assert.equal(editor.renderState('studio').shopLights.fixtures[0].source,'led-array');
assert.equal(editor.renderState('studio').shopLights.fixtures[0].dimmer,.2);
assert.equal(editor.renderState('studio').shopLights.fixtures[1].enabled,false);
assert.equal(editor.renderState('studio').shopLights.backingMaterial,'metal_gold');
assert.equal(JSON.stringify(editor.poses.getState()),poseBytes);
assert(!doc.getElementById('room-value-shopLights-fixtures-@shop-long-exposureMode'),'No unsupported burst editor');
name.value='Ceiling shop independence';save.fire('click');
const both=JSON.parse(storage.get(STORAGE_KEY)).profiles.find(p=>p.name===name.value).state;
assert.deepEqual(both.shopLights,a.shopLights);assert(both.roomOverrides.values.some(v=>v.path==='shopLights/fixtures/@shop-long/source'));
doc.getElementById('lighting-quick-A').fire('click');assert.equal(editor.shopLights.element.hidden,false);
doc.getElementById('shop-light-remove-shop-long').fire('click');
assert(!editor.renderState('exposure').shopLights.fixtures.some(f=>f.id==='shop-long'));
assert(!editor.renderState('exposure').roomOverrides.values.some(v=>v.path.startsWith('shopLights/fixtures/@shop-long/')));
selection.value='Ceiling shop independence';load.fire('click');
assert.deepEqual(editor.renderState('exposure'),both);
assert.equal(editor.renderState('studio').shopLights.fixtures[0].source,'led-array');
console.log('Ceiling shops: new two-fixture/legacy empty, full A/B source/power/visibility/material independence, saved pose preservation, stable-ID removal pruning and complete profile roundtrip passed');
}

{
const before=copy(editor.renderState('exposure')),posesBefore=JSON.stringify(editor.poses.getState());
assert.equal(before.lightStands.upperSupport,'floor','An explicit old support record remains a floor stand');
doc.getElementById('lighting-quick-B').fire('click');
const lock=doc.getElementById('room-unlock-lightStands-upperSupport');lock.checked=true;lock.fire('change');
const field=doc.getElementById('room-value-lightStands-upperSupport');field.value='ceiling';field.fire('change');
assert.equal(editor.renderState('exposure').lightStands.upperSupport,'floor');
assert.equal(editor.renderState('studio').lightStands.upperSupport,'ceiling');
assert.equal(JSON.stringify(editor.poses.getState()),posesBefore);
name.value='Upper ceiling support';save.fire('click');
const savedSupport=JSON.parse(storage.get(STORAGE_KEY)).profiles.find(p=>p.name===name.value).state;
assert(savedSupport.roomOverrides.values.some(v=>v.path==='lightStands/upperSupport'&&v.value==='ceiling'));
lock.checked=false;lock.fire('change');assert.equal(editor.renderState('studio').lightStands.upperSupport,'floor');
selection.value='Upper ceiling support';load.fire('click');assert.equal(editor.renderState('studio').lightStands.upperSupport,'ceiling');
assert.equal(JSON.stringify(editor.poses.getState()),posesBefore);
doc.getElementById('lighting-quick-A').fire('click');
const upper=doc.getElementById('light-stands-upper-support');upper.value='ceiling';upper.fire('change');
assert.equal(editor.renderState('exposure').lightStands.upperSupport,'ceiling');
upper.value='invalid';upper.fire('change');assert.equal(editor.renderState('exposure').lightStands.upperSupport,'ceiling');
assert.equal(JSON.stringify(editor.poses.getState()),posesBefore);
console.log('Upper support: saved floor migration, fresh ceiling default, canonical choice, B-only support independence, profile persistence and unchanged saved poses passed');
}

// Lower-fill convenience controls share the canonical pose and profile state.
{
const edit=(id,value,event='input')=>{const el=doc.getElementById(id);if(typeof value==='boolean')el.checked=value;else el.value=String(value);el.fire(event);};
const lower=()=>editor.poses.getState().poses.find(p=>p.id==='fill-lower');
const old=copy(editor.poses.getState()),other=old.poses.filter(p=>p.id!=='fill-lower');
edit('lower-fill-wall-gap',2);edit('lower-fill-height',1.25);
assert.equal(lower().pos[2],studioRoomBounds(Number(doc.getElementById('room-scale').value),tableFloorDrop(editor.table.getState())).frontZ-2);
assert.equal(lower().pos[1],1.25);assert.deepEqual(editor.poses.getState().poses.filter(p=>p.id!=='fill-lower'),other);
edit('lower-fill-lock',false,'change');
for(const [yaw,pitch,want] of [[0,0,[0,0,-1]],[180,0,[0,0,1]],[0,90,[0,1,0]],[0,-90,[0,-1,0]]]){
 edit('lower-fill-yaw',yaw);edit('lower-fill-pitch',pitch);
 const d=resolveLightingPoses(editor.poses.getState(),STUDIO_LIGHT_IDS).find(p=>p.id==='fill-lower').dir;
 assert(d.every((v,i)=>Math.abs(v-want[i])<1e-12));
}
const bytes=JSON.stringify(lower());edit('lower-fill-wall-gap',-1);assert.equal(JSON.stringify(lower()),bytes);
edit('lower-fill-yaw',181);assert.equal(JSON.stringify(lower()),bytes);
name.value='Lower fill aiming';save.fire('click');edit('lower-fill-height',2);selection.value=name.value;load.fire('click');assert.equal(JSON.stringify(lower()),bytes);
const a=copy(editor.renderState('exposure'));doc.getElementById('lighting-quick-B').fire('click');
const path='poses-poses-@fill-lower-yaw',unlock=doc.getElementById('room-unlock-'+path);
assert(unlock);unlock.checked=true;unlock.fire('change');const yaw=doc.getElementById('room-value-'+path);yaw.value='45';yaw.fire('change');
assert.equal(editor.renderState('exposure').poses.poses.find(p=>p.id==='fill-lower').yaw,a.poses.poses.find(p=>p.id==='fill-lower').yaw);
assert.equal(editor.renderState('studio').poses.poses.find(p=>p.id==='fill-lower').yaw,45);
doc.getElementById('lighting-quick-A').fire('click');
console.log('Lower fill: wall-relative readout, canonical height/lock, forward/back/up/down rays, invalid edit rejection, exact profiles and B-only aim isolation passed');
}

{
const grid=editor.lightTools.getState().attachments.find(t=>t.kind==='grid');assert(grid);
const interior=doc.getElementById(`light-tool-${grid.id}-interiorMaterial`),outer=doc.getElementById(`light-tool-${grid.id}-outerMaterial`);
interior.value='polished_silver';interior.fire('change');outer.value='metal_gold';outer.fire('change');
assert.equal(editor.lightTools.getState().attachments.find(t=>t.id===grid.id).settings.interiorMaterial,'polished_silver');
assert.equal(editor.lightTools.getState().attachments.find(t=>t.id===grid.id).settings.outerMaterial,'metal_gold');
name.value='Grid reflector materials';save.fire('click');outer.value='housing';outer.fire('change');selection.value=name.value;load.fire('click');
assert.equal(editor.lightTools.getState().attachments.find(t=>t.id===grid.id).settings.outerMaterial,'metal_gold');
const prior=JSON.stringify(editor.lightTools.getState());const bad=doc.getElementById(`light-tool-${grid.id}-interiorMaterial`);bad.value='missing_material';bad.fire('change');assert.equal(JSON.stringify(editor.lightTools.getState()),prior);
console.log('Grid material selectors: shared names, independent tunnel/exterior surfaces, save/load and invalid material rollback passed');
}
{
const tool=editor.fixtureSizes.tools.getState().attachments.find(t=>t.kind==='grid'),prefix=`light-tool-${tool.id}-accent-`;
const set=(key,value)=>{const input=doc.getElementById(prefix+key);assert(input);if(typeof value==='boolean')input.checked=value;else input.value=String(value);input.fire('change');};
set('enabled',true);set('pattern','columns');set('scope','corners');set('size',2);set('width',.15);set('material','metal_gold');
const accent=editor.fixtureSizes.tools.getState().attachments.find(t=>t.id===tool.id).settings.accent;
assert.deepEqual(accent,{enabled:true,material:'metal_gold',pattern:'columns',size:2,scope:'corners',width:.15});
name.value='Grid accent profile';save.fire('click');set('enabled',false);selection.value=name.value;load.fire('click');
assert.deepEqual(editor.fixtureSizes.tools.getState().attachments.find(t=>t.id===tool.id).settings.accent,accent);
set('size',0);assert.deepEqual(editor.fixtureSizes.tools.getState().attachments.find(t=>t.id===tool.id).settings.accent,accent);
console.log('Grid accent advanced controls: canonical fields, exact save/load and invalid pattern size rollback passed');
}
{
// The shared editor resolves the same margin-owned dimensions before rendering/saving.
const pose=editor.poses.getState();Object.assign(pose.poses.find(p=>p.id==='underlight'),{pos:[0,-1.6,0],aimAtSubject:true,pitch:90});editor.poses.setState(pose);
const {defaultFixtureSizes}=await import('../examples/meowdoku/web/marble_fixture_sizes.js');
editor.table.setState({...editor.table.getState(),width:8,length:4});editor.fixtureSizes.setState(defaultFixtureSizes());editor.invalidateRenderState();
assert.equal(doc.getElementById('fixture-size-underlight-width'),undefined);
const margin=doc.getElementById('underlight-shortMargin');margin.value='.12';margin.fire('change');
assert.equal(editor.fixtureSizes.getState().underlightFit.shortMargin,.12);
const before=editor.fixtureSizes.getState().lights[9].width;
const tableLength=doc.getElementById('studio-table-length');tableLength.value='5';tableLength.fire('input');
assert.ok(editor.fixtureSizes.getState().lights[9].width>before);
const snapshot=JSON.stringify(editor.fixtureSizes.getState());margin.value='10';margin.fire('change');assert.equal(JSON.stringify(editor.fixtureSizes.getState()),snapshot);
editor.invalidateRenderState();const profile=editor.renderState('exposure');
assert.deepEqual(validateStudioLighting(profile,elements,names).fixtureSizes,profile.fixtureSizes);
name.value='Underlight table margins';save.fire('click');margin.value='.2';margin.fire('change');selection.value=name.value;load.fire('click');assert.equal(editor.fixtureSizes.getState().underlightFit.shortMargin,.12);
console.log('Underlight autosize UI: margins only, resize propagation, impossible-margin rollback, profile roundtrip and reload passed');
}
{
// Existing A/B controls and saved profiles own the off action; shops are excluded.
doc.getElementById('lighting-quick-B').fire('click');
for(const {strengthId}of STUDIO_LIGHTS){elements[strengthId].value='70';roomEdit('controls/'+strengthId,130);}
editor.invalidateRenderState();const priorA=copy(editor.renderState('exposure')),priorB=copy(editor.renderState('studio'));
const strengthPaths=new Set(STUDIO_LIGHTS.map(l=>'controls/'+l.strengthId));
const retained=priorA.roomOverrides.values.filter(v=>!strengthPaths.has(v.path));
const off=doc.getElementById('studio-lights-off');assert(off);off.fire('click');
for(const channel of ['exposure','studio']){
 const state=editor.renderState(channel),prior=channel==='exposure'?priorA:priorB;
 assert.ok(STUDIO_LIGHTS.every(l=>state.controls[l.strengthId]===0));
 assert.deepEqual(state.shopLights,prior.shopLights);assert.deepEqual(state.poses,prior.poses);
 assert.deepEqual(state.fixtureSizes,prior.fixtureSizes);assert.deepEqual(state.lightTools,prior.lightTools);
 const {writeStudioFixtureRows}=await import('../examples/meowdoku/web/marble_studio_scene.js');
 const data=new Float32Array(12*16).fill(7),colors=STUDIO_LIGHTS.map(()=>[.8,.3,1]);
 writeStudioFixtureRows(data,16,0,colors,STUDIO_LIGHTS.map(l=>state.controls[l.strengthId]/10));
 for(let i=0;i<10;i++)assert.deepEqual(Array.from(data.slice(i*16+8,i*16+11)),[0,0,0]);
 assert.ok(data.slice(10*16).every(n=>n===7),'The original PBR writer never touches shop rows');
}
assert.deepEqual(editor.renderState('exposure').roomOverrides.values,retained);
name.value='Studio off shops intact';save.fire('click');elements['light-strength-left'].value='90';editor.invalidateRenderState();selection.value=name.value;load.fire('click');
assert.ok(STUDIO_LIGHTS.every(l=>editor.renderState('exposure').controls[l.strengthId]===0));
assert.ok(STUDIO_LIGHTS.every(l=>editor.renderState('studio').controls[l.strengthId]===0));
assert.deepEqual(editor.renderState('studio').shopLights,priorB.shopLights);
console.log('Studio-off button: both A/B zero, shop and geometry state unchanged, original shared ray/Phong PBR emissions zero, saved profile reload passed');
}
// Run the unchanged registration loop to the point immediately before GL upload.
{
 const vm=await import('node:vm');const code=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
 const start=code.indexOf('    const coneAngles={bare:null'),end=code.indexOf('    gl.uniform1i(loc(P,"uNumLights")',start);
 assert.ok(start>0&&end>start);
 const fixtureRoles=STUDIO_LIGHTS.map((l,i)=>({pos:[i,0,0],dir:[0,1,0],round:false,halfW:.5,halfH:.5,shape:'bare',mat:i}));
 const data=new Float32Array(11*16);data.set([3,2,1],10*16+8);
 const context=vm.createContext({fixtureRoles,roleColor:STUDIO_LIGHTS.map(()=>[1,1,1]),roleStrength:STUDIO_LIGHTS.map(l=>editor.renderState('studio').controls[l.strengthId]/10),displayedShop:{emitters:[{position:[0,4,0],surfaceAreaStudio:2,groupId:10,materialId:10}]},tables:{pbr:{data,table:{floats_per_record:16}}},view:[],transformPoint:(_,p)=>p});
 vm.runInContext(code.slice(start,end),context);
 assert.deepEqual(Array.from(vm.runInContext('lightIntensity',context)),[0,0,0,0,0,0,0,0,0,0,2]);
 assert.deepEqual(Array.from(vm.runInContext('lightColor[10]',context)),[3,2,1]);
 console.log('Original Phong registration loop: ten Studio GPU intensity inputs zero, separate shop intensity/color unchanged (CPU boundary test, not GPU pixels).');
}

{
// Direct overhead controls are views of the same saved and renderer-resolved pose.
clearanceVertices=null;
const overhead=()=>editor.poses.getState().poses.find(p=>p.id==='fill-upper');
const change=(field,value)=>{const el=elements['overhead-fill-'+field];if(field==='lock')el.checked=value;else el.value=String(value);el.fire(field==='lock'?'change':'input');};
const ceiling=()=>studioRoomBounds(Number(elements['room-scale'].value),tableFloorDrop(editor.table.getState())).ceilingY;
const before=copy(editor.poses.getState());
change('drop',4);assert.equal(overhead().pos[1],ceiling()-4);
change('x',1.5);change('z',-2);assert.deepEqual(overhead().pos,[1.5,ceiling()-4,-2]);
assert.deepEqual(editor.poses.getState().poses.filter(p=>p.id!=='fill-upper'),before.poses.filter(p=>p.id!=='fill-upper'));
assert(elements['overhead-fill-yaw'].disabled);const locked=copy(overhead());change('yaw',90);assert.deepEqual(overhead(),locked);
change('lock',false);assert.equal(elements['overhead-fill-pitch'].disabled,false);
change('yaw',90);change('pitch',-30);
const dir=editor.poses.getPoses()[6].dir;
assert.ok(Math.abs(dir[0]-Math.sqrt(3)/2)<1e-12&&Math.abs(dir[1]+.5)<1e-12&&Math.abs(dir[2])<1e-12);
const manual=copy(overhead());change('lock',true);assert.notDeepEqual(editor.poses.getPoses()[6].dir,dir);change('lock',false);assert.deepEqual(overhead(),manual);
for(const [field,value]of [['drop',-1],['drop',''],['x',51],['yaw',181],['pitch',-91]]){change(field,value);assert.deepEqual(overhead(),manual);}
const oldRoom=elements['room-scale'].value;elements['room-scale'].value=oldRoom==='7'?'6':'7';elements['room-scale'].fire('input');assert.deepEqual(overhead(),manual);assert.ok(Math.abs(Number(elements['overhead-fill-drop'].value)-(ceiling()-manual.pos[1]))<.00051);
elements['room-scale'].value=oldRoom;elements['room-scale'].fire('input');
const advanced=editor.poses.getState();advanced.poses[6].pos[1]-=1;editor.poses.setState(advanced);assert.ok(Math.abs(Number(elements['overhead-fill-drop'].value)-(ceiling()-overhead().pos[1]))<.00051);
name.value='Overhead gimbal';save.fire('click');const saved=copy(overhead());change('yaw',-45);selection.value=name.value;load.fire('click');assert.deepEqual(overhead(),saved);
clearanceVertices=(id,state)=>id==='fill-upper'?[[0,ceiling()+1,0]]:null;
change('drop',5);assert.deepEqual(overhead(),saved);assert.match(elements['top-fill-height-status'].textContent,/below the ceiling/);
clearanceVertices=null;
console.log('Overhead controls: ceiling-drop conversion, XYZ/Advanced sync, manual gimbal direction, tracking restore, invalid/clearance rollback, room-resize preservation and saved profile roundtrip passed.');
}
