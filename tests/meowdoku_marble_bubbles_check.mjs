import assert from 'node:assert/strict';
import {catEyeMarble} from '../examples/meowdoku/web/marble_mesh.js';
import {buildMarbleObjectGeometry,geometrySha256,massFromSavedMarbleObject,verifySavedMarbleObject,analyticSpheresForDesign,applyDesign} from '../examples/meowdoku/web/marble_object.js';
import {generateAirBubbles,phongInterior,validateBubbleInclusions} from '../examples/meowdoku/web/marble_bubbles.js';
const index={marble_clear_glass:1,marble_cat_eye_blue:2,marble_milk_glass:3,
  metal_gold:4,stone_onyx:5};
const extra={coated:6,paintA:7,paintB:8};
const design={seed:17,"base-shape":"marble","die-shell-preset":"clear",kind:"cat_eye",preset:"custom",
  shell:"marble_clear_glass",eye:"marble_cat_eye_blue",accent:"marble_milk_glass",
  "confetti-material":"metal_gold","mark-material":"stone_onyx",
  width:28,length:75,curve:18,strands:3,thickness:35,twist:90,tip:135,flare:18,phase:0,
  "orientation-yaw":0,"orientation-pitch":0,"orientation-roll":0,"tube-count":2,
  "tube-radius":24,"tube-turns":220,"confetti-count":220,coverage:0,"band-count":0,
  "band-width":18,"band-curve":120,"paint-a-color":"#3769a8","paint-b-color":"#e7b35a",
  "marking-mode":"raised","mark-scale":72,"relief-depth":20,"dice-detail":24,
  "die-inclusion-size":100,"dice-bevel":40,"dice-face-relief":10,"compensate-raised":true,opaque:false};
const physical={"radius-mm":10,"shell-density":2500,"eye-density":2800,
  "accent-density":2600,"confetti-density":7800,"enamel-density":3000,"mark-density":2500,
  "sim-rho":1,"sim-radius":0.01,"sim-mu":0.35,"sim-e":0.05,"sim-roll":0,"sim-irr":0};
const tables=Object.fromEntries(["pbr","phong","enamel","texstack"].map(name=>
  [name,{stride:16,data:Array(9*16).fill(0)}]));
const snapshot={v:1,name:"Clear D6",generator:"meowdoku-marble-object-v1",design,
  materials:{index,extra,tables},physical};
const ids={shell:1,eye:2,accent:3,confetti:4,mark:5,...extra};


// Lock the untouched legacy generator receipt, not a hash regenerated in this test.
assert.equal(await geometrySha256(catEyeMarble({glass:11,eye:12,accent:13})),'b6d1b1ca49dce15364944415f134184f4ab95224de8103e570d5b820c20dad9a');
for(const seed of [0,17,0x7fffffff]) {
 const all=generateAirBubbles({count:64,seed});
 assert.deepEqual(generateAirBubbles({count:8,seed}),all.slice(0,8));
 for(let i=0;i<all.length;i++) {
  const b=all[i];assert(Math.hypot(...b.center)+b.radius+.004<=.94);
  for(const a of all.slice(0,i))assert(Math.hypot(...b.center.map((x,k)=>x-a.center[k]))>=b.radius+a.radius+.004);
 }
}
for(let id=0;id<9;id++){tables.pbr.data[id*16+5]=1;tables.pbr.data[id*16+6]=1.5;tables.pbr.data[id*16+7]=0.1;}
design.kind='clear';design['ribbon-layout']='separated';
const legacy=buildMarbleObjectGeometry(design,ids);
assert.equal(await geometrySha256(legacy),await geometrySha256(buildMarbleObjectGeometry({...design,'bubble-count':0},ids)));
snapshot.geometry_sha256=await geometrySha256(legacy);snapshot.derived=massFromSavedMarbleObject(snapshot,legacy);
await verifySavedMarbleObject(snapshot);assert(!Object.hasOwn(design,'analytic_spheres'));
const root={getElementById:()=>({value:'8',type:'text'})};const bubbleElement={value:'8',type:'range'};root.getElementById=id=>id==='bubble-count'?bubbleElement:{value:'',type:'text'};
applyDesign(root,design);assert.equal(bubbleElement.value,'0');
design['bubble-count']=4;
const mesh=buildMarbleObjectGeometry(design,ids);const positions=mesh.bubbles.map(b=>[b.center,b.radius]);
for(const detail of ['medium','low','proxy'])assert.deepEqual(buildMarbleObjectGeometry(design,ids,{detail}).bubbles.map(b=>[b.center,b.radius]),positions);
assert.equal(phongInterior(mesh).length,legacy.interior.length);
assert.deepEqual(mesh.interior.slice(0,legacy.interior.length),legacy.interior);
assert.notDeepEqual(buildMarbleObjectGeometry({...design,seed:18},ids).bubbles[0].center,mesh.bubbles[0].center);
let end=0;for(const r of mesh.boundaryMedia.ranges){assert.equal(r.start,end);end+=r.count;}assert.equal(end,mesh.triangleCount);
design.boundary_media=mesh.boundaryMedia;snapshot.geometry_sha256=await geometrySha256(mesh);snapshot.derived=massFromSavedMarbleObject(snapshot,mesh);
assert(snapshot.derived.massKg<massFromSavedMarbleObject(snapshot,legacy).massKg);
await verifySavedMarbleObject(snapshot);
const spheres=analyticSpheresForDesign(design,mesh,ids.shell,tables.pbr);
assert.deepEqual(spheres.spheres[0],{start:mesh.interior.length,count:mesh.glass.length,center:[0,0,0],radius:1});
for(let i=0;i<mesh.bubbles.length;i++) {
 const d=spheres.spheres[i+1],b=mesh.bubbles[i];assert.deepEqual(d,{start:b.start,count:b.count,center:b.center,radius:b.radius});
 assert.equal(d.radius,b.radius);assert.notEqual(d.radius,Math.fround(b.radius),'keep generator doubles, not mesh float32 reconstruction');
}
design.analytic_spheres=spheres;await verifySavedMarbleObject(JSON.parse(JSON.stringify(snapshot)));
for(const mutate of [s=>s.spheres[0].radius=.99,s=>s.spheres[1].center[0]+=.001,s=>s.spheres[1].radius=Math.fround(s.spheres[1].radius),s=>s.spheres[0].count--]) {
 const bad=structuredClone(snapshot);mutate(bad.design.analytic_spheres);await assert.rejects(verifySavedMarbleObject(bad),/analytic sphere descriptors/);
}
for(const change of [{'base-shape':'d6'},{coverage:1},{'band-count':1}])assert.equal(analyticSpheresForDesign({...design,...change},mesh,ids.shell,tables.pbr),undefined);
const emission=structuredClone(tables.pbr);emission.data[ids.shell*16+8]=1;assert.equal(analyticSpheresForDesign(design,mesh,ids.shell,emission),undefined);
const opaque=structuredClone(tables.pbr);opaque.data[ids.shell*16+5]=0;opaque.data[ids.shell*16+7]=1;assert(analyticSpheresForDesign(design,legacy,ids.shell,opaque));
const mixed={...mesh,glass:mesh.glass.map((f,i)=>i===0?[...f.slice(0,3),ids.eye,f[4]]:f)};assert.equal(analyticSpheresForDesign(design,mixed,ids.shell,tables.pbr),undefined);
assert(!Object.hasOwn(design,'sphere_coverage'),'descriptors never select exposure mode');
console.log('PASS air bubbles: seeded prefix/containment/disjointness; legacy digest/reset; LOD/persistence/mass/Phong; optional analytic descriptors, eligibility and exact-double tamper checks');



// New explicit optical modes may cut air through inclusions; legacy stays
// disjoint and rejects the same axial recipe. No implicit mode migration.
const axial={...design,kind:'cat_eye','ribbon-layout':'axial',strands:3,'bubble-count':16};
assert.throws(()=>buildMarbleObjectGeometry(axial,ids),/disjoint/);
for(const mode of ['glass_rod','injection_dye']) {
 const opted={...axial,optical_authoring:{version:1,mode}};
 const cut=buildMarbleObjectGeometry(opted,ids);
 assert.equal(cut.bubbles.length,16);
 assert.equal(cut.mechanicsResolved,false);
 assert.match(cut.mechanicsNote,/mechanical mass/);
 for(const detail of ['medium','low','proxy'])assert.deepEqual(
   buildMarbleObjectGeometry(opted,ids,{detail}).bubbles.map(b=>[b.center,b.radius]),
   cut.bubbles.map(b=>[b.center,b.radius]));
}
console.log('PASS opt-in bubble/inclusion coexistence; legacy recipes and mechanical limitation retained.');
