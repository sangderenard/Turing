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
const profileNodes=all(elements['lighting-profiles']);
const name=profileNodes.find(n=>n['aria-label']==='Lighting profile name');
const selection=profileNodes.find(n=>n['aria-label']==='Saved lighting profiles');
const save=profileNodes.find(n=>n.textContent==='Save/update profile'),load=profileNodes.find(n=>n.textContent==='Load selected');
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
