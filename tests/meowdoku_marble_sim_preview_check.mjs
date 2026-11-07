import assert from "node:assert/strict";
import { marblePreviewSpec, livePreviewParticle, previewLawToScene } from "../examples/meowdoku/web/marble_sim_preview.js";

const base={manifest:"fountain3d.manifest.json",size:"small",view:[-1,1,0,2],depth:1,
  uniforms:{grav_z:-9.8,emit_rate:0.02,r0:0.01,rho0:1}};
const physical={"sim-radius":0.012,"sim-rho":2,"sim-mu":0.45,"sim-e":0.3,"sim-roll":0.08,"sim-irr":0.04};
const spec=marblePreviewSpec(base,physical,7);
assert.equal(spec.epoch,7);
assert.equal(spec.uniforms.r0,0.012);
assert.equal(spec.uniforms.rho0,2);
assert.equal(spec.uniforms.mu0,0.45);
assert.equal(spec.uniforms.emit_rate,1e9);
assert.equal(spec.uniforms.grav_z,-9.8);
assert.equal(base.uniforms.r0,0.01);
assert.throws(()=>marblePreviewSpec(base,{...physical,"sim-rho":2600},8),/native law/);
assert.equal(livePreviewParticle(new Float32Array([0,0,0,0,0])),null);
assert.deepEqual(livePreviewParticle(new Float32Array([1,2,3,0,120])),{x:1,y:2,z:3,radius:0.02});
const map=previewLawToScene(base.view,base.depth,0.012);
assert.ok(map.point([0,1,0]).every((v)=>Math.abs(v)<1e-12));
console.log("marble simulation preview: native field mapping and packed particle position");
