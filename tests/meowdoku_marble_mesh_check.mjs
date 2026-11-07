import { catEyeMarble } from "../examples/meowdoku/web/marble_mesh.js";
import { marbleMass, principalMoments } from "../examples/meowdoku/web/marble_mass.js";
import { rollingStep } from "../examples/meowdoku/web/marble_roll.js";

const assert = (condition, message) => { if (!condition) throw new Error(message); };
const mesh = catEyeMarble({ glass: 11, eye: 12, accent: 13 });
const reroll = catEyeMarble({ glass: 11, eye: 12, accent: 13 }, { seed: 31415 });
assert(JSON.stringify(mesh.interior[100][0]) !== JSON.stringify(reroll.interior[100][0]), "reroll changes dyed geometry");
assert(JSON.stringify(mesh.glass[100][0]) === JSON.stringify(reroll.glass[100][0]), "reroll retains the glass shell");
assert(mesh.interior.length > 1000 && mesh.glass.length > 1000, "both surfaces are triangulated");
assert(mesh.ribbons.length === 3, "three separate closed ribbon surfaces");
assert(mesh.ribbons.every((r) => r.count === mesh.glass.length), "each ribbon has its own full closed surface");
assert(mesh.ribbons.reduce((n, r) => n + r.count, 0) === mesh.interior.length, "ribbon ranges partition interior triangles");
assert(mesh.ribbons.every((r) => mesh.interior.slice(r.start, r.start + r.count).every((f) => f[3] === r.material)), "each ribbon has one material and geometric boundaries");
const volume = (faces) => faces.reduce((sum, [a,b,c]) => sum + (
  a[0]*(b[1]*c[2]-b[2]*c[1]) + a[1]*(b[2]*c[0]-b[0]*c[2]) + a[2]*(b[0]*c[1]-b[1]*c[0]))/6, 0);
assert(mesh.ribbons.every((r) => volume(mesh.interior.slice(r.start,r.start+r.count)) > 0), "each ribbon encloses a positively wound volume");
assert(mesh.triangleCount === mesh.interior.length + mesh.glass.length, "triangle count");
assert(new Set(mesh.interior.map((f) => f[3])).size === 2, "dyed and accent triangles");
assert(mesh.glass.every((f) => f[3] === 11), "glass is a separate material surface");
for (const [label, faces] of [["interior", mesh.interior], ["glass", mesh.glass]]) {
  for (const face of faces) {
    const [a, b, c, , normals] = face;
    assert(normals.length === 3 && normals.every((n) => n.length === 3 && n.every(Number.isFinite)), `${label} vertex normals`);
    const u = b.map((v, k) => v - a[k]), v = c.map((w, k) => w - a[k]);
    const cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]];
    assert(Math.hypot(...cross) > 1e-8, `${label} nondegenerate triangle`);
    const n = normals[0];
    assert(cross.reduce((sum, x, k) => sum + x * n[k], 0) > 0, `${label} winding matches normal`);
    for (const p of [a, b, c]) {
      const radius = Math.hypot(...p);
      assert(label === "glass" ? Math.abs(radius - 1) < 1e-9 : radius < 0.96, `${label} lies on its own surface`);
    }
  }
}
for (const invalid of [{ eyeWidth: 0.7 }, { rings: 3 }, { eyeLength: 1 }, { ribbonThickness: 0.2 }, { ribbonTip: 4 }]) {
  let threw = false; try { catEyeMarble({ glass: 1, eye: 2, accent: 3 }, invalid); } catch (e) { threw = e instanceof RangeError; }
  assert(threw, "invalid design rejected");
}
const detail = catEyeMarble({ glass: 11, eye: 12, accent: 13 }, { rings: 96, segments: 128, strandCount: 5, ribbonTwist: 2.4, ribbonTip: 2.3 });
assert(detail.triangleCount > 100000, "high detail studio mesh has over 100k actual triangles");
assert(detail.ribbons.length === 5, "high detail mesh keeps separate ribbons");
const phase = catEyeMarble({glass:11,eye:12,accent:13},{ribbonPhase:Math.PI/2});
assert(JSON.stringify(phase.interior[100][0]) !== JSON.stringify(mesh.interior[100][0]), "phase rotates ribbon geometry");
assert(JSON.stringify(phase.glass[100][0]) === JSON.stringify(mesh.glass[100][0]), "phase leaves bare shell geometry unchanged");
const direction=(face)=>face[1].map((x,k)=>x-face[0][k]);
assert(direction(phase.interior[100]).every((x,k)=>Math.abs(x-direction(mesh.interior[100])[k])<1e-9),
  "placement phase leaves ribbon local orientation unchanged");
const oriented=catEyeMarble({glass:11,eye:12,accent:13},{orientationYaw:0.2,orientationPitch:0.15,orientationRoll:-0.1});
assert(direction(oriented.interior[100]).some((x,k)=>Math.abs(x-direction(mesh.interior[100])[k])>1e-4),
  "orthogonal orientation controls rotate actual inclusion triangles");
assert(oriented.glass.every((f,i)=>JSON.stringify(f[0])===JSON.stringify(mesh.glass[i][0])),
  "inclusion orientation leaves shell unchanged");
const decorated = catEyeMarble({glass:11,eye:12,accent:13,enamel:14,confetti:15,paintA:16,paintB:17},
  {kind:"hybrid",confettiCount:100,enamelCoverage:0.3,paintBandCount:7,seed:42});
assert(decorated.confetti.length === 100 && decorated.ribbons.length === 3, "hybrid has discrete confetti and ribbons");
assert(decorated.confetti.every((c)=>volume(decorated.interior.slice(c.start,c.start+c.count))>0), "each chip is a positively wound closed tetrahedron");
assert(new Set(decorated.glass.map((f)=>f[3])).size === 4, "paint bands, enamel and bare shell assign real triangle materials");
assert(decorated.confetti.every((c)=>decorated.interior.slice(c.start,c.start+c.count).every((f)=>f.slice(0,3).every((p)=>Math.hypot(...p)<0.96))), "chips stay inside glass");
const clear=catEyeMarble({glass:11,eye:12,accent:13},{kind:"clear"});
assert(clear.interior.length===0 && clear.glass.length>0,"clear shell omits all inclusions");
const densities={shell:2500,eye:2500,accent:2500,enamel:3000,confetti:7800};
const clearMass=marbleMass(clear,{radiusM:0.01,densityByRole:densities});
const analyticMass=4*Math.PI/3*0.01**3*2500;
assert(Math.abs(clearMass.massKg/analyticMass-1)<0.01,"closed shell mass approaches analytic sphere");
const analyticInertia=0.4*analyticMass*0.01**2;
assert(Math.abs(clearMass.inertiaKgM2[0][0]/analyticInertia-1)<0.02,"tetrahedral rotational inertia approaches analytic sphere");
const principal=principalMoments([[2,0.3,0],[0.3,4,0],[0,0,8]]);
assert(principal.every((p,i)=>i===0||principal[i-1].momentKgM2<=p.momentKgM2),"principal moments sorted");
assert(Math.abs(principal.reduce((s,p)=>s+p.momentKgM2,0)-14)<1e-9,"principal moments preserve tensor trace");
assert(Math.abs(principal[0].axis.reduce((s,x,i)=>s+x*[[2,0.3,0],[0.3,4,0],[0,0,8]][i].reduce((t,y,j)=>t+y*principal[0].axis[j],0),0)-principal[0].momentKgM2)<1e-9,"principal axis matches eigenvalue");
const hybridMass=marbleMass(decorated,{radiusM:0.01,densityByRole:densities,enamelMaterial:14,enamelThicknessM:300e-9});
assert(hybridMass.massKg>clearMass.massKg && hybridMass.components.length===decorated.ribbons.length+decorated.confetti.length+2,"inclusions and coating contribute mass");
assert(Math.abs(hybridMass.components.reduce((s,c)=>s+c.massKg,0)-hybridMass.massKg)<1e-12,
  "component masses account for displaced shell volume exactly");
assert(!hybridMass.validForRoll && hybridMass.mechanicsNote.includes("thin film"),"thin-film enamel estimate gates real rolling");
const uncoatedHybrid=marbleMass(decorated,{radiusM:0.01,densityByRole:densities,enamelMaterial:14});
assert(uncoatedHybrid.validForRoll,"closed disjoint hybrid volumes provide rolling inertia");
const startRoll={position:[0,0,0],orientation:[1,0,0,0],angularVelocity:[0,0,0]};
const rolled=rollingStep(startRoll,clearMass,[0.001,0,0],0.01,0.01);
assert(rolled.angularAcceleration[2]<0 && rolled.position[0]>0,"applied force rolls marble along floor");
const anisotropic={...clearMass,inertiaKgM2:[[2e-7,0,0],[0,4e-7,0],[0,0,8e-7]]};
const rollX=rollingStep(startRoll,anisotropic,[0.001,0,0],0.01,0.01);
const rollZ=rollingStep(startRoll,anisotropic,[0,0,0.001],0.01,0.01);
assert(Math.abs(rollX.angularAcceleration[2])<Math.abs(rollZ.angularAcceleration[0]),
  "full anisotropic tensor changes torque response by axis");
let gated=false;try{rollingStep(startRoll,{...clearMass,validForRoll:false},[0.001,0,0],0.01,0.01);}catch(e){gated=e instanceof RangeError;}
assert(gated,"rolling rejects unresolved mass and inertia");
for(const kind of ["spiral","rings"]) {
  const design=catEyeMarble({glass:11,eye:12,accent:13},{kind,tubeCount:2});
  assert(design.tubes.length===2 && design.interior.length>1000,`${kind} consists of triangulated tubes`);
  for(const tube of design.tubes) {
    const faces=design.interior.slice(tube.start,tube.start+tube.count);
    assert(volume(faces)>0,`${kind} tube encloses positive volume`);
    assert(faces.every((face)=>face.slice(0,3).every((p)=>Math.hypot(...p)<0.96)),`${kind} tube stays within shell`);
  }
  const properties=marbleMass(design,{radiusM:0.01,densityByRole:densities});
  assert(properties.components.length===3 && properties.massKg>0,`${kind} tubes contribute to mass`);
}
const confettiOnly=catEyeMarble({glass:11,eye:12,accent:13,confetti:15},{kind:"confetti",confettiCount:500,seed:19});
assert(confettiOnly.confetti.length===500,"uniform fill packs 500 closed chips");
assert(confettiOnly.confetti.every((a,i)=>confettiOnly.confetti.slice(i+1).every((b)=>
  Math.hypot(...a.center.map((x,k)=>x-b.center[k]))>0.027)),"confetti chip centers remain separated");
console.log(`PASS cat's-eye marble: ${mesh.interior.length} inclusion and ${mesh.glass.length} shell triangles`);
