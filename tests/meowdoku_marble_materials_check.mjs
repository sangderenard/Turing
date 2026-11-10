import assert from 'node:assert/strict';
import {restoreOpticalSurfaceRows,illustrativeOpticalRecord,opticalMaterialIds,OPTICAL_PROVENANCE} from '../examples/meowdoku/web/marble_optical_authoring.js';
import {prepareObjectSnapshot} from '../examples/meowdoku/web/marble_object_snapshot.js';
import fs from 'node:fs';
import vm from 'node:vm';
import {createHash} from 'node:crypto';
import {STUDIO_FIXED_MATERIAL_ROWS, STUDIO_RUNTIME_MATERIAL_ROWS, studioMaterialExtras, studioMaterialBase, savedMaterialRows} from '../examples/meowdoku/web/marble_materials.js';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
import {DESIGN_FIELDS, PHYSICAL_FIELDS, applyDesign, applyPhysical, readDesign,
  buildMarbleObjectGeometry, geometryFromSavedMarbleObject, materialIdsForDesign, geometrySha256,
  massFromSavedMarbleObject, verifySavedMarbleObject} from '../examples/meowdoku/web/marble_object.js';
import {packMarblePhoto} from '../examples/meowdoku/web/marble_photo.js';

const copy = value => JSON.parse(JSON.stringify(value));
const source = fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js', import.meta.url), 'utf8');
const legacyNames = ('cat x_mark heart metal_gold metal_silver metal_copper metal_brass metal_bronze ' +
  'metal_steel metal_chrome metal_aluminum metal_titanium metal_iron metal_oxidized_copper ' +
  'marble_clear_glass marble_milk_glass marble_cat_eye_blue marble_cat_eye_green marble_cat_eye_amber ' +
  'stone_onyx stone_jade ceramic_porcelain black_felt stucco_plaster calibration_white calibration_red ' +
  'calibration_green calibration_blue calibration_yellow calibration_magenta calibration_cyan calibration_gray20').split(' ');
const legacyIndex = Object.fromEntries(legacyNames.map((name, id) => [name, id]));
const index = {...legacyIndex, white_gloss_paint:35, white_plaster:36, black_gloss_paint:37};
const extra = {coated:32, paintA:33, paintB:34};
const strides = {pbr:16, phong:8, enamel:8, texstack:16};
function assetsFor(index, records, extras) {
  return {material_index:{...index}, ...(extras ? {material_extras:{...extras}} : {}),
    materials:Object.fromEntries(Object.entries(strides).map(([name,stride]) => [name, {
      records, floats_per_record:stride, data:Array.from({length:records*stride}, (_,i) => Math.fround(0.1+i/1000))
    }]))};
}
const assets = assetsFor(index, 38, extra), oldAssets = assetsFor(legacyIndex, 32);
assert.deepEqual(studioMaterialExtras(assets), extra);
assert.deepEqual(studioMaterialExtras(oldAssets), extra);
assert.equal(studioMaterialBase(assets, extra), 38);
assert.equal(studioMaterialBase(oldAssets, extra), 35);
assert.throws(() => studioMaterialExtras({...assets,material_extras:{coated:35,paintA:33,paintB:34}}), /IDs/);
assert.throws(() => studioMaterialExtras({...assets,material_extras:{coated:32,paintA:33,paintB:33}}), /IDs/);
assert.throws(() => studioMaterialBase({...assets,materials:{...assets.materials,enamel:{...assets.materials.enamel,records:37}}},extra), /float table/);

// Run the real texture allocation and boot-time fixture initialization.
function bootTables(assets) {
  const gl = new Proxy({}, {get:(_,key) => key === 'createTexture' ? () => ({}) :
    key.toUpperCase() === key ? 0 : () => {}});
  const context = vm.createContext({assets, ids:assets.material_index, gl,
    STUDIO_FIXED_MATERIAL_ROWS, STUDIO_RUNTIME_MATERIAL_ROWS, STUDIO_LIGHTS, studioMaterialExtras, studioMaterialBase,
    coatedId:0,paintAId:0,paintBId:0,tables:null,FIXTURE_MAT_BASE:0,FIXTURE_HOUSING_MAT:0,FIXTURE_LABEL_MAT:0});
  vm.runInContext(source.slice(source.indexOf('function floatTexture('), source.indexOf('function flatArray(')), context);
  const start = source.indexOf('    const extra=studioMaterialExtras(assets)');
  const end = source.indexOf('    const names=Object.keys(ids)',start);
  assert.ok(start >= 0 && end > start);
  vm.runInContext(source.slice(start,end),context);
  return context;
}
const boot = bootTables(assets), oldBoot = bootTables(oldAssets);
assert.equal(STUDIO_FIXED_MATERIAL_ROWS,12);assert.equal(STUDIO_RUNTIME_MATERIAL_ROWS,28);
assert.equal(boot.FIXTURE_MAT_BASE,38); assert.equal(boot.FIXTURE_HOUSING_MAT,48); assert.equal(boot.FIXTURE_LABEL_MAT,49);
assert.equal(oldBoot.FIXTURE_MAT_BASE,35); assert.equal(oldBoot.tables.pbr.data.length,(35+STUDIO_RUNTIME_MATERIAL_ROWS)*16);
for (const [name,stride] of Object.entries(strides)) {
  assert.equal(boot.tables[name].data.length,(38+STUDIO_RUNTIME_MATERIAL_ROWS)*stride);
  assert.deepEqual(Array.from(boot.tables[name].data.slice(0,38*stride)),assets.materials[name].data);
  assert.deepEqual(Array.from(oldBoot.tables[name].data.slice(0,32*stride)),oldAssets.materials[name].data);
}
function savedTables(rows) {
  return Object.fromEntries(Object.entries(strides).map(([name,stride]) => [name,
    {stride,data:Array.from({length:rows*stride},(_,i) => Math.fround(0.2+i/2000))}]));
}
const legacyMaterials = {index:{...legacyIndex},extra:{...extra},tables:savedTables(41)};
// Former eight-shop-slot snapshots remain readable after adding warm groups.
assert.doesNotThrow(()=>savedMaterialRows({index:{...index},extra:{...extra},tables:savedTables(38+20)},index,extra,boot.tables));
legacyMaterials.tables.enamel.data[32*8] = 360;
function applyRows(rows, tables) {
  for (const [name,values] of Object.entries(rows)) for (const row of values) tables[name].data.set(row.data,row.offset);
}
const original = JSON.stringify(legacyMaterials);
const untouched = Object.fromEntries(Object.entries(boot.tables).map(([name,t]) => [name,Array.from(t.data)]));
const rows = savedMaterialRows(legacyMaterials,index,extra,boot.tables);
assert.equal(JSON.stringify(legacyMaterials),original);
for (const [name,table] of Object.entries(boot.tables)) assert.deepEqual(Array.from(table.data),untouched[name]);
applyRows(rows,boot.tables);
for (const [name,stride] of Object.entries(strides)) {
  assert.deepEqual(Array.from(boot.tables[name].data.slice(0,35*stride)),legacyMaterials.tables[name].data.slice(0,35*stride));
  // Legacy fixture rows 35..40 must never overwrite room finishes or this rig.
  assert.deepEqual(Array.from(boot.tables[name].data.slice(35*stride)),untouched[name].slice(35*stride));
}

const invalid = [
  m => {delete m.tables.enamel;},
  m => {m.tables.unknown = m.tables.pbr;},
  m => {m.tables.enamel.stride = 16;},
  m => {m.tables.enamel.data.pop();},
  m => {m.tables.enamel.data.length = 32*8;},
  m => {m.tables.enamel.data.push(...Array(8).fill(0));},
  m => {m.tables.texstack.data[40*16] = NaN;},
  m => {m.tables.texstack.data[0] = Infinity;},
  m => {m.tables.texstack.data[0] = Number.MAX_VALUE;},
  m => {m.tables.texstack.data[0] = '1';},
  m => {m.tables.texstack.data[0] = null;},
  m => {delete m.tables.texstack.data[0];},
  m => {m.index.unknown = 35;},
  m => {m.index.cat = 37;},
  m => {m.index.cat = 1;},
  m => {delete m.index.cat;},
  m => {m.extra.coated = 31;},
  m => {m.extra.paintB = 35;},
  m => {m.extra.unknown = 42;},
];
for (const mutate of invalid) {
  const broken = copy(legacyMaterials); mutate(broken);
  const before = Object.values(boot.tables).map(t => Array.from(t.data));
  assert.throws(() => savedMaterialRows(broken,index,extra,boot.tables));
  assert.deepEqual(Object.values(boot.tables).map(t => Array.from(t.data)),before);
}
// A same-catalogue save restores authored edits, not stale lighting rows.
const currentMaterials = {index:{...index},extra:{...extra},tables:savedTables(38+STUDIO_RUNTIME_MATERIAL_ROWS)};
const previousRigMaterials = {...currentMaterials,tables:savedTables(47)};
assert.equal(savedMaterialRows(previousRigMaterials,index,extra,boot.tables).pbr.length,38);
assert.equal(savedMaterialRows({...currentMaterials,tables:savedTables(50)},index,extra,boot.tables).pbr.length,38);
currentMaterials.tables.pbr.data[35*16+3] = 0.417;
const runtime = Object.fromEntries(Object.entries(boot.tables).map(([name,t]) => [name,Array.from(t.data.slice(38*strides[name]))]));
applyRows(savedMaterialRows(currentMaterials,index,extra,boot.tables),boot.tables);
for (const [name,stride] of Object.entries(strides)) {
  assert.deepEqual(Array.from(boot.tables[name].data.slice(0,38*stride)),Array.from(new Float32Array(currentMaterials.tables[name].data.slice(0,38*stride))));
  assert.deepEqual(Array.from(boot.tables[name].data.slice(38*stride)),runtime[name]);
}

const design={seed:17,'base-shape':'marble','die-shell-preset':'clear',kind:'hybrid',preset:'custom',
  shell:'marble_clear_glass',eye:'marble_cat_eye_blue',accent:'marble_milk_glass',
  'confetti-material':'metal_gold','mark-material':'stone_onyx',
  width:28,length:75,curve:18,strands:3,'ribbon-layout':'separated',thickness:35,twist:90,tip:135,flare:18,phase:0,
  'orientation-yaw':0,'orientation-pitch':0,'orientation-roll':0,'tube-count':2,
  'tube-radius':24,'tube-turns':220,'confetti-count':20,coverage:20,'band-count':2,
  'band-width':18,'band-curve':120,'paint-a-color':'#3769a8','paint-b-color':'#e7b35a',
  'marking-mode':'raised','mark-scale':72,'relief-depth':20,'dice-detail':24,
  'die-inclusion-size':100,'dice-bevel':40,'dice-face-relief':10,'compensate-raised':true,opaque:false};
const physical={'radius-mm':10,'shell-density':2500,'eye-density':2800,'accent-density':2600,
  'confetti-density':7800,'enamel-density':3000,'mark-density':2500,'sim-rho':1,'sim-radius':.01,
  'sim-mu':.35,'sim-e':.05,'sim-roll':0,'sim-irr':0};
const snapshot={v:1,name:'Material identity check',generator:'meowdoku-marble-object-v1',design,
  materials:legacyMaterials,physical};
const geometry=buildMarbleObjectGeometry(design,materialIdsForDesign(design,legacyIndex,extra));
snapshot.geometry_sha256=await geometrySha256(geometry);
snapshot.derived=massFromSavedMarbleObject(snapshot,geometry);
assert.deepEqual(new Set([...geometry.interior,...geometry.glass].map(face=>face[3])).has(32),true);
assert.equal(await geometrySha256(buildMarbleObjectGeometry(design,materialIdsForDesign(design,index,extra))),snapshot.geometry_sha256);

// Execute production loadObject with real mesh/mass verification and an inert DOM.
// The original selected digest must reach photo/live unchanged, without a POST.
async function loadSnapshot(value, digest, expectedVerified=true, duringVerify=null) {
  const nodes=Object.fromEntries([...DESIGN_FIELDS,...PHYSICAL_FIELDS,'saved-object','object-name'].map(id =>
    [id,{value:'',type:typeof value.design[id] === 'boolean'?'checkbox':typeof value.design[id] === 'number'?'range':'select',dispatchEvent(){}}]));
  nodes['saved-object'].value=digest;
  const document={getElementById:id=>nodes[id]};
  const tables=bootTables(assets).tables, saved=[],fetches=[],status=[];
  let rebuilt,restoredOptical;
  const context=vm.createContext({$,document,tables,ids:index,coatedId:32,paintAId:33,paintBId:34,
    applyingObject:false,objectKey:"",savedObjectSnapshot:null,objectLoadIntent:0,AbortSignal,structuredClone,
    restoreOpticalSurfaceRows,opticalControls:{restore(design){restoredOptical=design.optical_authoring;}},
    objectAutosave:{dirty:false,revision:0,acceptExternal(){this.revision++;}},
    objectEdited(){},captureObjectSnapshot:()=>value,objectSnapshotKey:()=>"test-key",syncSubjectSizeControls(){},
    savedMaterialRows,applyDesign,applyPhysical,Event:class {},seed:0,opaqueSaved:null,dieDefaultsApplied:false,savedPhotoRadius:null,
    photo:{saved:d=>saved.push(['photo',d])},liveRay:{saved:d=>saved.push(['live',d])},
    objectStatus:message=>status.push(message),syncMaterial(){},
    rebuild(){rebuilt=buildMarbleObjectGeometry(readDesign(document,context.seed),materialIdsForDesign(value.design,index,extra));},
    verifySavedMarbleObject:async v=>{
      try {return await verifySavedMarbleObject(v);}
      finally {tables.pbr.data[38*16+8]=123;duringVerify?.(context);} // A frame updates the fixture while verification awaits.
    },
    fetch:async(url,options)=>{fetches.push({url,options});return {ok:true,json:async()=>value};},
  });
  function $(id){return nodes[id];}
  vm.runInContext(source.slice(source.indexOf('function acceptSavedObject('),source.indexOf('async function saveObject()')),context);
  vm.runInContext(source.slice(source.indexOf('async function loadObject()'),source.indexOf('function objectLibraryControls()')),context);
  const before=JSON.stringify(value);
  await vm.runInContext('loadObject()',context);
  assert.deepEqual(restoredOptical,value.design.optical_authoring);
  assert.equal(JSON.stringify(value),before);
  assert.equal(await geometrySha256(rebuilt),await geometrySha256(geometryFromSavedMarbleObject(value)));
  assert.deepEqual(saved,expectedVerified?[['photo',digest],['live',digest]]:[]);
  assert.equal(fetches.length,1); assert.equal(fetches[0].options.method,undefined);
  assert.equal(tables.pbr.data[38*16+8],123);
  if(!expectedVerified) {
    assert.match(status.at(-1),/stored mesh receipt no longer matches.*original is preserved/i);
    return tables;
  }
  assert.equal(await geometrySha256(rebuilt),value.geometry_sha256);
  const wire=packMarblePhoto(rebuilt,digest),view=new DataView(wire);
  const request=JSON.parse(new TextDecoder().decode(new Uint8Array(wire,16,view.getUint32(4,true))));
  assert.equal(request.object_digest,digest);
  const wireGeometry=new Uint8Array(wire,16+view.getUint32(4,true));
  const count=view.getUint32(8,true),interleaved=new Uint8Array(count*76);
  for(let i=0;i<count;i++) {
    interleaved.set(wireGeometry.subarray(i*36,(i+1)*36),i*76);
    interleaved.set(wireGeometry.subarray(count*36+i*36,count*36+(i+1)*36),i*76+36);
    interleaved.set(wireGeometry.subarray(count*72+i*4,count*72+(i+1)*4),i*76+72);
  }
  assert.equal(createHash('sha256').update(interleaved).digest('hex'),value.geometry_sha256);
  return tables;
}
const loaded=await loadSnapshot(snapshot,'a'.repeat(64));
await assert.rejects(loadSnapshot(snapshot,'a'.repeat(64),true,context=>context.objectAutosave.revision++),/Object changed while loading/);
assert.equal(loaded.enamel.data[32*8],360);
assert.deepEqual(Array.from(loaded.pbr.data.slice(35*16,38*16)),assets.materials.pbr.data.slice(35*16,38*16));
const modern=copy(snapshot);modern.materials=currentMaterials;
modern.materials.tables.enamel.data[32*8]=360;
modern.derived=massFromSavedMarbleObject(modern,geometry);
const modernLoaded=await loadSnapshot(modern,'b'.repeat(64));
assert.equal(modernLoaded.pbr.data[35*16+3],Math.fround(.417));
const optical=copy(modern);
Object.assign(optical.design,{kind:'cat_eye',coverage:0,'band-count':0,'ribbon-layout':'axial',strands:2});
const opticalGeometry=buildMarbleObjectGeometry(optical.design,materialIdsForDesign(optical.design,index,extra));
for(const id of opticalMaterialIds(opticalGeometry)) {
  const row=optical.materials.tables.pbr.data,at=id*16;
  row[at+4]=0;row[at+5]=.7;row[at+6]=1.54;row[at+7]=.3;
  row[at+8]=row[at+9]=row[at+10]=0;
}
optical.design.optical_authoring={version:1,mode:'glass_rod',provenance:OPTICAL_PROVENANCE,
  materials:opticalMaterialIds(opticalGeometry).map(id=>illustrativeOpticalRecord(id,optical.materials.tables.pbr))};
const opticalSaved=(await prepareObjectSnapshot(optical)).value;
const opticalLoaded=await loadSnapshot(opticalSaved,'c'.repeat(64));
for(const record of opticalSaved.design.optical_authoring.materials) {
  const offset=record.material*16;
  assert.deepEqual(Array.from(opticalLoaded.pbr.data.slice(offset,offset+16)),
    Array.from(new Float32Array(optical.materials.tables.pbr.data.slice(offset,offset+16))),
    'optical load restores original editable rows while retaining the saved spectral profile');
}

if(process.argv[2]) {
  const path=process.argv[2],raw=fs.readFileSync(path),value=JSON.parse(raw);
  const digest=path.replaceAll('\\','/').split('/').at(-1).replace(/\.json$/,'');
  let verified=true;
  try {await verifySavedMarbleObject(value);}
  catch(error) {
    if(!/^saved mesh recipe no longer matches its triangles$|^saved mass or moment of inertia no longer matches the mesh$/.test(error.message))throw error;
    verified=false;
    console.log(`Existing saved receipt mismatch retained as inspection-only: ${error.message}`);
  }
  await loadSnapshot(value,digest,verified);
  assert.deepEqual(fs.readFileSync(path),raw);
  console.log(`Read-only saved object bytes and material IDs preserved: ${digest}`);
}
console.log('Material IDs: legacy/current roundtrip, original photo digest, authored-only row copy, fixture isolation and atomic invalid-table rejection passed');

// Appending disk-library profiles preserves every historical authored ID.
{
 const expandedIndex={...index,disk_glass:38,disk_felt:39};
 const expanded=bootTables(assetsFor(expandedIndex,40,extra));
 const saved={index:{...index},extra:{...extra},tables:savedTables(38+STUDIO_RUNTIME_MATERIAL_ROWS)};
 const before=Array.from(expanded.tables.pbr.data);
 applyRows(savedMaterialRows(saved,expandedIndex,extra,expanded.tables),expanded.tables);
 assert.deepEqual(Array.from(expanded.tables.pbr.data.slice(0,38*16)),saved.tables.pbr.data.slice(0,38*16));
 assert.deepEqual(Array.from(expanded.tables.pbr.data.slice(38*16)),before.slice(38*16));
 const gap=copy(saved);delete gap.index.white_plaster;
 assert.throws(()=>savedMaterialRows(gap,expandedIndex,extra,expanded.tables),/migration/);
}
console.log('PASS append-only disk catalogue migration preserves new profiles and active fixtures');
