import assert from 'node:assert/strict';
import {catEyeMarble} from '../examples/meowdoku/web/marble_mesh.js';
import {regionMediaFromGeometry,authoredOpticalMedia} from '../examples/meowdoku/web/marble_optical_authoring.js';
const mesh=catEyeMarble({glass:0,eye:1,accent:2},{kind:'cat_eye',ribbonLayout:'axial',strandCount:3,rings:12,segments:24});
const before=JSON.stringify(mesh);
for(const mode of ['glass_rod','injection_dye']) {
  const regions=regionMediaFromGeometry(mesh,0,mode);
  assert.equal(regions.regions.length,mesh.ribbons.length+1);
  assert(regions.regions.slice(1).every(r=>r.mode===mode));
  if(mode==='glass_rod') {
    assert.deepEqual(regions.axial_bundle.regions,[1,2,3]);
    assert.deepEqual(regions.axial_bundle.frame,{origin:[0,0,0],axes:[[1,0,0],[0,1,0],[0,0,1]],scale:1});
    regions.axial_bundle.frame.origin[0]=1;
    assert.equal(mesh.inclusionFrame.origin[0],0);
  } else assert.equal(regions.axial_bundle,undefined);
  const ids=[...new Set(regions.regions.map(r=>r.material))];
  const records=ids.map(material=>({material,wavelengths_nm:[380,780],transmittance:[.5,.9],reference_thickness_mm:1,concentration:1}));
  const optics=authoredOpticalMedia(records,regions);
  optics.materials[0].transmittance[0]=.8;
  assert.equal(records[0].transmittance[0],.5);
  assert.throws(()=>authoredOpticalMedia(records.slice(1),regions),/every volume/);
  records[0].transmittance[0]=0;
  assert.throws(()=>authoredOpticalMedia(records,regions),/transmission/);
}
assert.equal(JSON.stringify(mesh),before);
const bad=structuredClone(mesh);bad.ribbons[0].count--;
assert.throws(()=>regionMediaFromGeometry(bad,0,'glass_rod'),/Every generated face/);
assert.throws(()=>regionMediaFromGeometry(mesh,0,'last_added'),/Choose/);
for(const strandCount of [1,2,4,7]) {
  const options={kind:'cat_eye',ribbonLayout:'axial',strandCount,rings:12,segments:24,
    eyeLength:.85,eyeCurve:.28,ribbonTwist:-2.7,ribbonPhase:1.4,
    orientationYaw:.6,orientationPitch:-.4,orientationRoll:1.2};
  const plain=catEyeMarble({glass:0,eye:1,accent:2},{...options,orientationYaw:0,orientationPitch:0,orientationRoll:0});
  const oriented=catEyeMarble({glass:0,eye:1,accent:2},options);
  const {origin,axes,scale}=oriented.inclusionFrame;
  for(let f=0;f<plain.interior.length;f++)for(let v=0;v<3;v++)for(let k=0;k<3;k++) {
    const q=plain.interior[f][v];
    const expected=origin[k]+scale*q.reduce((sum,x,i)=>sum+x*axes[i][k],0);
    assert(Math.abs(oriented.interior[f][v][k]-expected)<1e-14,'frame follows actual oriented vertex');
  }
  const regions=regionMediaFromGeometry(oriented,0,'glass_rod');
  assert.equal(Boolean(regions.axial_bundle),strandCount>1);
  if(strandCount>1)assert.deepEqual(regions.axial_bundle.frame,oriented.inclusionFrame);
}
const fitted=catEyeMarble({glass:0,eye:1,accent:2},{kind:'cat_eye',ribbonLayout:'axial',rings:12,segments:24,
  eyeLength:.85,eyeCurve:.28,orientationRoll:Math.PI/2});
assert(fitted.inclusionFrame.scale<1,'centroid rotation exercises the real common envelope fit');
assert(Math.abs(Math.max(...fitted.interior.flatMap(f=>f.slice(0,3).map(p=>Math.hypot(...p))))-.94)<1e-14);
const separated=catEyeMarble({glass:0,eye:1,accent:2},{ribbonLayout:'separated',rings:12,segments:24});
assert.equal(regionMediaFromGeometry(separated,0,'glass_rod').axial_bundle,undefined);
const hybrid=catEyeMarble({glass:0,eye:1,accent:2},{kind:'hybrid',ribbonLayout:'axial',rings:12,segments:24,confettiCount:0});
assert.throws(()=>regionMediaFromGeometry(hybrid,0,'glass_rod'),/cat-eye ribbons/);
assert.equal(regionMediaFromGeometry(hybrid,0,'injection_dye').axial_bundle,undefined);
console.log('PASS pure optical authoring: both modes, exact common frames and fit, explicit axial bundles, immutable geometry.');
