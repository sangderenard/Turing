import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {catEyeMarble} from "../examples/meowdoku/web/marble_mesh.js";
import {marbleMass} from "../examples/meowdoku/web/marble_mass.js";
import {DESIGN_FIELDS, readDesign, applyDesign, buildMarbleObjectGeometry,
  geometrySha256, massFromSavedMarbleObject, verifySavedMarbleObject} from "../examples/meowdoku/web/marble_object.js";

const materials={glass:11,eye:12,accent:13};
const legacy=catEyeMarble(materials);
assert.equal(await geometrySha256(legacy),
  "b6d1b1ca49dce15364944415f134184f4ab95224de8103e570d5b820c20dad9a",
  "legacy triangles and normals must keep their pre-change digest");
const near=(a,b)=>assert(Math.abs(a-b)<1e-9,`${a} != ${b}`);
for(const count of [2,3,4,5,6,7]) {
  const geometry=catEyeMarble(materials,{ribbonLayout:"axial",strandCount:count,
    ribbonPhase:Math.PI/4,seed:17});
  const {interior,ribbons,parameters:o}=geometry;
  assert.equal(ribbons.length,count);
  for(let strand=0;strand<count;strand++) {
    const angle=Math.PI*strand/count,c=Math.cos(angle),s=Math.sin(angle);
    let volume=0;
    for(let f=0;f<ribbons[0].count;f++) {
      const base=interior[f],face=interior[ribbons[strand].start+f];
      const [a,b,d]=face;
      volume+=(a[0]*(b[1]*d[2]-b[2]*d[1])+a[1]*(b[2]*d[0]-b[0]*d[2])+a[2]*(b[0]*d[1]-b[1]*d[0]))/6;
      const u=b.map((v,k)=>v-a[k]),v=d.map((w,k)=>w-a[k]);
      const cross=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
      assert(Math.hypot(...cross)>1e-8,"nondegenerate axial face");
      assert(cross.reduce((sum,x,k)=>sum+x*face[4][0][k],0)>0,"outward axial winding");
      for(let j=0;j<3;j++) {
        const [x,y,z]=base[j],centre=o.eyeCurve*(1-(y/o.eyeLength)**2);
        near(face[j][0],centre+(x-centre)*c-z*s);
        near(face[j][1],y);
        near(face[j][2],(x-centre)*s+z*c);
        assert(Math.hypot(...face[j])<0.96,"axial band inside shell");
        assert(face[4][j].every(Number.isFinite),"finite axial normal");
        near(Math.hypot(...face[4][j]),1);
      }
    }
    assert(volume>0,"each ribbon is a closed positively wound volume");
  }
  const mass=marbleMass(geometry,{radiusM:0.01,densityByRole:{shell:2500,eye:2500,accent:2500}});
  assert(!mass.massExact&&!mass.validForRoll,"overlapping material volumes cannot claim exact mechanics");
  assert.match(mass.mechanicsNote,/overlap/);
}
assert.throws(()=>catEyeMarble(materials,{ribbonLayout:"unknown"}),RangeError);

// Read the actual Studio's initial controls, not a second copy of its defaults.
const html=readFileSync(new URL("../examples/meowdoku/web/marble_studio.html",import.meta.url),"utf8");
const elements=new Map();
for(const id of DESIGN_FIELDS) {
  const input=html.match(new RegExp(`<input[^>]* id="${id}"[^>]*>`));
  const select=html.match(new RegExp(`<select id="${id}">([\\s\\S]*?)</select>`));
  assert(input||select,`missing Studio control ${id}`);
  const type=input?.[0].match(/type="([^"]+)"/)?.[1]||"select-one";
  const value=input?.[0].match(/value="([^"]*)"/)?.[1]||select?.[1].match(/<option value="([^"]+)"/)?.[1]||"";
  elements.set(id,{type,value,checked:!!input?.[0].includes(" checked")});
}
const root={getElementById:id=>elements.get(id)};
const design=readDesign(root,1);
assert.equal(design["ribbon-layout"],"axial");assert.equal(design.strands,2);
assert.equal(design.phase,45);assert.equal(design.twist,90);
assert.equal(design.curve,0,"default ribbons share the straight longitudinal axis");
assert.equal(design.eye,"marble_cat_eye_blue");assert.equal(design.accent,"marble_cat_eye_green");
const index={marble_clear_glass:1,marble_cat_eye_blue:2,marble_cat_eye_green:3,metal_gold:4};
const extra={coated:5,paintA:6,paintB:7};
const ids={shell:1,eye:2,accent:3,confetti:4,mark:4,...extra};
// These selects are populated from the catalogue at startup.
design.shell="marble_clear_glass";design["confetti-material"]="metal_gold";design["mark-material"]="metal_gold";
const physical={"radius-mm":10,"shell-density":2500,"eye-density":2500,"accent-density":2500,
  "confetti-density":7800,"enamel-density":3000,"mark-density":2500};
for(const layout of [undefined,"axial"]) {
  const d={...design};if(layout===undefined)delete d["ribbon-layout"];
  else d["ribbon-layout"]=layout;
  const geometry=buildMarbleObjectGeometry(d,ids);
  const saved={v:1,name:"Axial regression",generator:"meowdoku-marble-object-v1",design:d,physical,
    materials:{index,extra,tables:{enamel:{stride:8,data:Array(64).fill(0)}}},
    geometry_sha256:await geometrySha256(geometry)};
  saved.derived=massFromSavedMarbleObject(saved,geometry);
  const restored=JSON.parse(JSON.stringify(saved));
  const verified=await verifySavedMarbleObject(restored);
  assert.equal(verified.mass.massExact,layout===undefined);
  applyDesign(root,d);
  assert.equal(root.getElementById("ribbon-layout").value,layout||"separated");
  assert.equal(await geometrySha256(buildMarbleObjectGeometry(readDesign(root,1),ids)),saved.geometry_sha256,
    "loading then reading a saved design preserves its exact geometry");
}
console.log("PASS axial defaults: co-located rotated bands 2..7; legacy digest; new/legacy saved-design roundtrips; honest mechanics");
