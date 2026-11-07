import assert from 'node:assert/strict';
import {diceSolid,fitInclusionsInsideDie} from '../examples/meowdoku/web/dice_mesh.js';
import {catEyeMarble} from '../examples/meowdoku/web/marble_mesh.js';
import {marbleMass} from '../examples/meowdoku/web/marble_mass.js';

const materials={glass:1,eye:2,accent:3,confetti:4};
for(const kind of ['d4','d6','d10','d20']) {
  const die=diceSolid(kind,{shellMaterial:1,markMaterial:5,bevelRadius:0.06,
    faceRelief:0.025,markingMode:'raised',reliefDepth:0.02,
    subdivisions:24});
  assert(die.safeInteriorRadius>0);
  let minVertex=Infinity;
  for(const triangle of die.glass) for(const p of triangle.slice(0,3))
    minVertex=Math.min(minVertex,Math.hypot(...p));
  assert(die.safeInteriorRadius<=minVertex+1e-9,
    'triangle-distance insphere may be smaller than any vertex radius');
  for(const interiorKind of ['cat_eye','confetti','hybrid','spiral','rings','clear']) {
    const source=catEyeMarble(materials,{kind:interiorKind,rings:24,segments:48,
      confettiCount:24,seed:17});
    const combined=fitInclusionsInsideDie(die,source,{gap:0.015,scale:0.8});
    assert.equal(combined.glass,die.glass);
    assert.equal(combined.interior.length,source.interior.length);
    for(const triangle of combined.interior) for(const p of triangle.slice(0,3))
      assert(Math.hypot(...p)<=die.safeInteriorRadius-0.015+1e-10,
        `${kind}/${interiorKind}: inclusion escapes insphere`);
    assert.equal(combined.ribbons.length,source.ribbons.length);
    assert.equal(combined.tubes.length,source.tubes.length);
    assert.equal(combined.confetti.length,source.confetti.length);
    for(let i=0;i<source.confetti.length;i++)
      assert.deepEqual(combined.confetti[i].center,
        source.confetti[i].center.map(x=>x*combined.inclusionFit.factor));
    const properties=marbleMass(combined,{radiusM:0.01,densityByRole:{
      shell:2500,eye:3200,accent:3400,confetti:3900}});
    assert(properties.massKg>0&&properties.inertiaKgM2.every((row,i)=>row[i]>0));
  }
  const source=catEyeMarble(materials,{kind:'cat_eye',rings:24,segments:48});
  const empty=fitInclusionsInsideDie(die,source,{scale:0});
  assert.equal(empty.interior.length,0);
  assert.equal(empty.ribbons.length,0);
  assert.throws(()=>fitInclusionsInsideDie(die,source,{gap:0}),RangeError);
  assert.throws(()=>fitInclusionsInsideDie(die,source,{gap:die.safeInteriorRadius}),RangeError);
  console.log(`${kind}: final-surface insphere and all six interior designs fit with closed mass`);
}
