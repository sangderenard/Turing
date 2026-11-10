import assert from 'node:assert/strict';
import fs from 'node:fs';
import {catEyeMarble} from '../examples/meowdoku/web/marble_mesh.js';
import {materialIdsForDesign,buildMarbleObjectGeometry} from '../examples/meowdoku/web/marble_object.js';
const mats={glass:1,eye:2,accent:3,confetti:4,ribbons:[10,11,12,13,14,15,16]};
for(const kind of ['cat_eye','spiral','rings','clear'])for(const angle of [0,.8]){
 const opts={kind,rings:12,segments:24,seed:17,orientationYaw:angle,orientationPitch:angle,confettiCount:500};
 const base=catEyeMarble(mats,opts),mixed=catEyeMarble(mats,{...opts,confettiEnabled:true});
 assert.equal(mixed.confetti.length,500);
 assert.deepEqual(mixed.interior.slice(0,base.interior.length),base.interior,'overlay must preserve existing decoration exactly');
 assert.deepEqual(mixed.glass,base.glass);
 assert.deepEqual(catEyeMarble(mats,{...opts,confettiEnabled:false}).interior,base.interior);
 for(const part of [...mixed.ribbons,...mixed.tubes]){
  const bounds=[0,1,2].map(i=>[Math.min(...mixed.interior.slice(part.start,part.start+part.count).flatMap(f=>f.slice(0,3).map(p=>p[i]))),Math.max(...mixed.interior.slice(part.start,part.start+part.count).flatMap(f=>f.slice(0,3).map(p=>p[i])))]);
  for(const chip of mixed.confetti)for(const f of mixed.interior.slice(chip.start,chip.start+chip.count))for(const p of f.slice(0,3)){
   assert(Math.hypot(...p)<1);
   assert(p.some((v,i)=>v<bounds[i][0]||v>bounds[i][1]),'chip vertex outside every decoration exclusion');
  }
 }
}
for(const kind of ['confetti','hybrid']){
 const opts={kind,rings:12,segments:24,seed:31};
 assert.deepEqual(catEyeMarble(mats,opts),catEyeMarble(mats,opts));
 assert.equal(catEyeMarble(mats,{...opts,confettiEnabled:false}).confetti.length,0);
}
const ribbonMesh=catEyeMarble(mats,{strandCount:7,rings:12,segments:24});
assert.deepEqual(ribbonMesh.ribbons.map(r=>r.material),mats.ribbons);
for(const r of ribbonMesh.ribbons)assert(ribbonMesh.interior.slice(r.start,r.start+r.count).every(f=>f[3]===r.material));
const index={glass:1,eye:2,accent:3,confetti:4,...Object.fromEntries(mats.ribbons.map((id,i)=>[`ribbon_${i+1}`,id]))};
const design={shell:'glass',eye:'eye',accent:'accent','confetti-material':'confetti','mark-material':'confetti','ribbon-materials':mats.ribbons.map((_,i)=>`ribbon_${i+1}`)};
assert.deepEqual(materialIdsForDesign(JSON.parse(JSON.stringify(design)),index,{}).ribbons,mats.ribbons);
assert.throws(()=>materialIdsForDesign({...design,'ribbon-materials':['missing']},index,{}),/ribbon materials/);
console.log('PASS additive confetti: 500 chips over rotated ribbons/spirals/rings/clear, preserved base geometry, disjoint exclusions, independent ribbon IDs and serialization');
