import { catEyeMarble } from "../examples/meowdoku/web/marble_mesh.js";

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
console.log(`PASS cat's-eye marble: ${mesh.interior.length} inclusion and ${mesh.glass.length} shell triangles`);
