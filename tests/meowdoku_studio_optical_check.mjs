import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {prepareObjectSnapshot,objectSnapshotKey} from '../examples/meowdoku/web/marble_object_snapshot.js';
import {verifySavedMarbleObject,buildMarbleObjectGeometry,materialIdsForDesign} from '../examples/meowdoku/web/marble_object.js';
import {OPTICAL_PROVENANCE,illustrativeOpticalRecord,opticalMaterialIds,restoreOpticalSurfaceRows} from '../examples/meowdoku/web/marble_optical_authoring.js';
import {createOpticalMaterialControls,OPTICAL_RUNTIME_ENABLED} from '../examples/meowdoku/web/marble_studio_optical.js';
import {createRodBundleWarning} from '../examples/meowdoku/web/marble_optical_diagnostic.js';

const index={clear:0,blue:1,green:2,metal:3,mark:4},extra={coated:5,paintA:6,paintB:7};
const design={seed:17,'base-shape':'marble',kind:'cat_eye',shell:'clear',eye:'blue',accent:'green',
  'confetti-material':'metal','mark-material':'mark',width:28,length:75,curve:0,strands:2,
  'ribbon-layout':'axial',thickness:35,twist:90,tip:135,flare:18,phase:45,
  'orientation-yaw':0,'orientation-pitch':0,'orientation-roll':0,'tube-count':2,
  'tube-radius':24,'tube-turns':220,'confetti-count':220,coverage:0,'band-count':0,
  'band-width':18,'band-curve':120,'bubble-count':0};
const physical={'radius-mm':10,'shell-density':2500,'eye-density':2800,'accent-density':2600,
  'confetti-density':7800,'enamel-density':3000,'mark-density':2500};
const tables=Object.fromEntries(['pbr','phong','enamel','texstack'].map(name=>[name,{stride:16,data:Array(8*16).fill(0)}]));
for(let i=0;i<8;i++)tables.pbr.data.splice(i*16,11,.2+i*.06,.6,.9,.1,0,.8,1.52+i*.01,.3,0,0,0);
const original={v:1,name:'Optical test',generator:'meowdoku-marble-object-v1',design,physical,materials:{index,extra,tables}};
const geometry=buildMarbleObjectGeometry(design,materialIdsForDesign(design,index,extra));
const ids=opticalMaterialIds(geometry),records=ids.map(id=>illustrativeOpticalRecord(id,tables.pbr));
const originalJSON=JSON.stringify(original);
const legacy=await prepareObjectSnapshot(original);
assert.equal(legacy.value.design.optical_authoring,undefined);
assert.equal(legacy.value.design.optical_media,undefined);
assert.equal(legacy.value.design.region_media,undefined);
for(const mode of ['legacy','glass_rod','injection_dye']) {
  const input=structuredClone(original);
  input.design.optical_authoring={version:1,mode,provenance:OPTICAL_PROVENANCE,materials:structuredClone(records)};
  const before=JSON.stringify(input),key=objectSnapshotKey(input);
  input.design.optical_authoring.materials[0].concentration=.25;
  assert.notEqual(key,objectSnapshotKey(input),'optical edit must participate in autosave identity');
  const edited=JSON.stringify(input),prepared=await prepareObjectSnapshot(input),{value}=prepared;
  assert.equal(prepared.authoringKey,objectSnapshotKey(input),'save retains raw authoring identity before canonical projection');
  const host=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
  const context=vm.createContext({
    $:id=>({'preview-engine':{value:'visual'},'preview-motion':{value:'off'},'inertia-lock':{checked:true}}[id]),
    rollState:null,studioLighting:{strengths:{getSolo:()=>null}},lastGeometry:geometry,
    savedObjectSnapshot:prepared,objectKey:prepared.authoringKey,objectSnapshotKey,captureObjectSnapshot:()=>input,
    studioBuilderLighting:()=>({}),computeFixtureRoles:()=>({}),studioSubjectTransform:()=>({unitM:.01,subjectCenter:[0,0,0]}),
    yaw:0,pitch:0,distance:1,
  });
  vm.runInContext(host.slice(host.indexOf('function fieldSceneKey('),host.indexOf('let fieldMaterialProjection=')),context);
  assert.doesNotThrow(()=>vm.runInContext('fieldSceneKey({saved:true})',context),'canonical optical save accepts its exact raw authoring state');
  const opacity=input.materials.tables.pbr.data[7];input.materials.tables.pbr.data[7]=opacity+.01;
  assert.throws(()=>vm.runInContext('fieldSceneKey({saved:true})',context),/Save the current object/,'raw edit is rejected even before queued objectKey updates');
  input.materials.tables.pbr.data[7]=opacity;
  context.savedObjectSnapshot={value,geometry};
  assert.throws(()=>vm.runInContext('fieldSceneKey({saved:true})',context),/Save the current object/,'missing authoring receipt fails closed');
  assert.equal(JSON.stringify(input),edited,'snapshot never mutates input');
  await verifySavedMarbleObject(JSON.parse(JSON.stringify(value)));
  const repeated=await prepareObjectSnapshot(value);
  assert.deepEqual(repeated.value,value,'saved optical snapshot re-prepares without losing original surface rows');
  assert.equal(value.geometry_sha256,legacy.value.geometry_sha256,'optics do not perturb the mesh');
  if(mode==='legacy') {
    assert.equal(value.design.region_media,undefined);assert.equal(value.design.optical_media,undefined);
    assert.deepEqual(value.materials.tables.pbr,input.materials.tables.pbr);continue;
  }
  for(const r of value.design.optical_authoring.materials) {
    assert.equal(value.materials.tables.pbr.data[r.material*16+5],1);
    assert.equal(value.materials.tables.pbr.data[r.material*16+7],0);
    assert.equal(value.materials.tables.pbr.data[r.material*16+6],mode==='injection_dye'?records[0].ior:r.ior);
    assert.equal(value.design.optical_media.materials.find(x=>x.material===r.material).ior,undefined);
  }
  assert.equal(Boolean(value.design.region_media.axial_bundle),mode==='glass_rod');
  const restored=structuredClone(value);restoreOpticalSurfaceRows(restored);
  assert.deepEqual(restored.materials.tables.pbr,input.materials.tables.pbr,'returning to legacy retains pre-opt-in surface rows');
  for(const mutate of [
    x=>x.design.region_media.regions[1].count--,
    x=>x.design.optical_media.materials[0].transmittance[0]=.99,
    x=>x.materials.tables.pbr.data[7]=.4,
    x=>x.design.optical_authoring.materials[0].ior=NaN,
    x=>x.design.optical_authoring.legacy_pbr[0].pop(),
  ]) { const bad=structuredClone(value);mutate(bad);await assert.rejects(()=>verifySavedMarbleObject(bad)); }
}
assert.equal(JSON.stringify(original),originalJSON);
for(const mode of ['glass_rod','injection_dye']) {
  const input=structuredClone(original);input.design['bubble-count']=8;
  input.design.optical_authoring={version:1,mode,provenance:OPTICAL_PROVENANCE,materials:structuredClone(records)};
  // The explicit volume choice converts these captured opaque surface rows.
  for(const id of ids){input.materials.tables.pbr.data[id*16+5]=0;input.materials.tables.pbr.data[id*16+7]=1;}
  const prepared=await prepareObjectSnapshot(input);
  assert.equal(prepared.geometry.bubbles.length,8);
  assert.equal(prepared.value.design.boundary_media,undefined);
  assert.equal(prepared.value.design.region_media.regions.filter(r=>r.mode==='air_bubble').length,8);
  assert.equal(prepared.value.derived.massExact,false,'shared-volume mechanics must not be described as exact');
  await verifySavedMarbleObject(JSON.parse(JSON.stringify(prepared.value)));
}
const forbiddenLegacy=structuredClone(original);forbiddenLegacy.design['bubble-count']=8;
await assert.rejects(()=>prepareObjectSnapshot(forbiddenLegacy),/disjoint/);

class Element extends EventTarget {
  constructor(tag){super();this.tag=tag;this.children=[];this.value='';this.style={};this.hidden=true;this.textContent='';}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  setAttribute(k,v){this[k]=v;}
}
function harness(runtimeEnabled=false){
  const elements=new Map(),$=id=>{if(!elements.has(id))elements.set(id,new Element('div'));return elements.get(id);};
  $('optical-mode').options=['legacy','glass_rod','injection_dye'].map(value=>({value,disabled:false}));
  const doc={getElementById:$,createElement:tag=>new Element(tag)};
  let changes=0;
  const context={geometry,design,shell:0,pbr:tables.pbr};
  const api=createOpticalMaterialControls({doc,getContext:()=>context,onChange(){changes++;},runtimeEnabled});
  const inputs=()=>{
    const out=[];function visit(el){if(el.tag==='input')out.push(el);for(const c of el.children)visit(c);}
    visit($('optical-material-rows'));return out;
  };
  return {$,api,context,inputs,get changes(){return changes;}};
}
assert.equal(OPTICAL_RUNTIME_ENABLED,true,'reviewed experimental photo authoring is enabled');
{
  const h=harness();assert.equal(h.api.capture(),undefined);assert.equal(h.changes,0);
  assert(h.$('optical-mode').options.slice(1).every(x=>x.disabled));
  h.$('optical-mode').value='glass_rod';h.$('optical-mode').dispatchEvent(new Event('change'));
  assert.equal(h.$('optical-mode').value,'legacy');assert.equal(h.api.capture(),undefined);
  const first=h.inputs()[0];first.value='.37';first.dispatchEvent(new Event('input'));
  assert.equal(h.api.capture().mode,'legacy');assert.equal(h.api.capture().materials[0].transmittance[0],.37);
  first.value='';first.dispatchEvent(new Event('input'));assert.throws(()=>h.api.capture(),/transmission/);
  h.api.restore({});assert.equal(h.api.capture(),undefined);assert.equal(h.$('optical-mode').value,'legacy');
}
{
  const h=harness(true);h.$('optical-mode').value='injection_dye';h.$('optical-mode').dispatchEvent(new Event('change'));
  assert.equal(h.api.capture().mode,'injection_dye');
  assert(h.inputs().filter(x=>x['aria-label']==='Glass rod IOR').every(x=>x.disabled));
  const saved=h.api.capture();h.api.restore({optical_authoring:saved});assert.deepEqual(h.api.capture(),saved);
  h.$('optical-mode').value='legacy';h.$('optical-mode').dispatchEvent(new Event('change'));
  assert.equal(h.api.capture().mode,'legacy');assert.equal(h.changes,2);
}
{
  const node=new Element('output'),warning=createRodBundleWarning({getElementById:()=>node},'warning');
  const bad={rod_bundle_diagnostic:{unresolved_samples:17,policy:'reject',warning:'Native cut event is unresolved.'}};
  warning.observe(bad,'A/first/0');assert.equal(node.hidden,false);assert.match(node.textContent,/INVALID IMAGE: 17/);
  assert.match(node.textContent,/not a valid converged result/);assert.match(node.textContent,/Native cut event/);
  warning.observe(null,'B/next/1');assert.equal(node.hidden,false);
  warning.displayed({},'B/next/1');assert.equal(node.hidden,false,'missing diagnostics cannot certify a replacement');
  const valid={rod_bundle_diagnostic:{unresolved_samples:0}};
  warning.displayed(valid,'A/first/0');assert.equal(node.hidden,false,'same invalid image stays invalid');
  warning.observe(valid,'B/next/1');assert.equal(node.hidden,false,'status alone does not replace displayed pixels');
  warning.displayed(valid,'B/next/1');assert.equal(node.hidden,true);assert.equal(warning.invalid,false);
}
const html=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.html',import.meta.url),'utf8');
for(const id of ['optical-material-controls','optical-material-rows','live-optical-warning','photo-optical-warning'])
  assert.equal(html.split(`id="${id}"`).length,2,`${id} is wired exactly once`);
assert.match(html,/Illustrative uncalibrated/);
assert(!html.includes('<details id="optical-material-controls"'));
assert(!html.includes('optical-surface-controls-note'));
console.log('PASS optical panel, gated opt-in, immutable save/load/re-save, legacy restoration, spectral edits and persistent invalid-image diagnostics.');

// Neutrality applies only to NEW clear-host records, never to authored data.
assert.deepEqual(illustrativeOpticalRecord(0,tables.pbr,{neutralHost:true}).transmittance,[1,1,1]);
{
 const h=harness(true);h.context.design={...design,shell:'marble_clear_glass'};
 h.api.restore({});h.$('optical-mode').value='glass_rod';h.$('optical-mode').dispatchEvent(new Event('change'));
 assert.deepEqual(h.api.capture().materials.find(x=>x.material===0).transmittance,[1,1,1]);
 assert.equal(h.$('opacity').disabled,true);assert.equal(h.$('transmission').disabled,true);
 const authored=h.api.capture();authored.materials.find(x=>x.material===0).transmittance=[1,.97,.93];
 h.api.restore({optical_authoring:authored});assert.deepEqual(h.api.capture(),authored);
 h.$('optical-mode').value='legacy';h.$('optical-mode').dispatchEvent(new Event('change'));
 assert.equal(h.$('opacity').disabled,false);assert(h.inputs().every(x=>x.disabled));
 assert.deepEqual(h.api.capture().materials.find(x=>x.material===0).transmittance,[1,.97,.93]);
}
console.log('PASS neutral new clear host, existing authored spectrum preserved, per-mode legacy slider warning');

{
 const h=harness(true);assert.equal(h.api.mode,'glass_rod');assert.equal(h.$('optical-mode').value,'glass_rod');
 assert.equal(h.api.capture().mode,'glass_rod');
 // Loading a historical design without optical authoring stays Legacy.
 h.api.restore({});assert.equal(h.api.mode,'legacy');assert.equal(h.api.capture(),undefined);
 // Loading either saved version retains that exact choice and spectrum.
 for(const mode of ['legacy','glass_rod','injection_dye']) {
   const saved={version:1,mode,provenance:OPTICAL_PROVENANCE,materials:structuredClone(records)};
   h.api.restore({optical_authoring:saved});assert.deepEqual(h.api.capture(),saved);
 }
}
console.log('PASS three versions: fresh Glass rod default; historical Legacy and authored modes preserved');

{
 const h=harness(true),id=ids[0],offset=id*h.context.pbr.stride+6;
 h.context.selectedMaterial=id;h.api.refresh();
 assert.equal(h.inputs().length,5,'one selected material, three samples, thickness and concentration');
 assert(!h.inputs().some(x=>/IOR/.test(x['aria-label'])),'no duplicate IOR editor');
 const old=h.context.pbr.data[offset];h.context.pbr.data[offset]=1.61;
 assert.equal(h.api.capture().materials.find(r=>r.material===id).ior,h.context.pbr.data[offset]);
 h.context.selectedMaterial=999;h.api.refresh();assert.equal(h.inputs().length,5);
 assert(h.inputs().every(x=>x.disabled));assert.equal(h.$('opacity').disabled,false);
 h.context.selectedMaterial=ids.find(x=>x!==0);h.$('optical-mode').value='injection_dye';h.$('optical-mode').dispatchEvent(new Event('change'));
 assert.equal(h.$('ior').disabled,true);assert(h.inputs().every(x=>!x.disabled));
 h.context.selectedMaterial=0;h.api.refresh();assert.equal(h.$('ior').disabled,false);
 assert(h.$('optical-material-rows').children.every(x=>x.tag==='label'));
 h.context.pbr.data[offset]=old;
}
console.log('PASS shared selected-material editor, canonical PBR IOR, unsupported-volume notice');

{
 const h=harness(true),subject=h.api.capture();
 h.context.selectedGroup='stage';h.context.selectedMaterial=3;h.context.sceneEngines={stage:'dielectric',room:'legacy'};h.api.refresh();
 assert(h.inputs().every(x=>!x.disabled));assert.deepEqual(h.inputs().slice(0,4).map(x=>x.value),['','','','']);
 assert.equal(h.$('opacity').disabled,true);assert.equal(h.$('ior').disabled,false);
 assert.deepEqual(h.api.capture(),subject,'Selecting another scene group never changes the subject engine or spectrum');
 const values=['.8','.9','.95','12','1.2'];h.inputs().forEach((input,i)=>{input.value=values[i];input.dispatchEvent(new Event('input'));});
 const scene=h.api.sceneRecord(3);assert.deepEqual(scene.transmittance,[.8,.9,.95]);assert.equal(scene.reference_thickness_mm,12);
 h.context.sceneEngines.stage='legacy';h.api.refresh();assert(h.inputs().every(x=>x.disabled));assert.equal(h.$('opacity').disabled,false);
 assert.deepEqual(h.api.sceneRecord(3),scene,'Switching engines preserves authored optical data');
 h.api.restore({optical_authoring:subject});assert.deepEqual(h.api.sceneRecord(3),scene,'Loading a subject preserves unrelated shared scene spectra');
 h.context.selectedMaterial=subject.materials[0].material;h.api.refresh();assert.deepEqual(h.api.sceneRecord(h.context.selectedMaterial).transmittance,subject.materials[0].transmittance,'The same material ID uses the same record');
}
console.log('PASS scene targets reuse the original five optical controls, blank unmeasured spectra, PBR IOR, subject isolation and same-ID record sharing.');
