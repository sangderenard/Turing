import { catEyeMarble } from "../examples/meowdoku/web/marble_mesh.js";

const assert = (condition, message) => { if (!condition) throw new Error(message); };
const mesh = catEyeMarble({ glass: 11, eye: 12, accent: 13 });
const reroll = catEyeMarble({ glass: 11, eye: 12, accent: 13 }, { seed: 31415 });
assert(JSON.stringify(mesh.interior[100][0]) !== JSON.stringify(reroll.interior[100][0]), "reroll changes dyed geometry");
assert(JSON.stringify(mesh.glass[100][0]) === JSON.stringify(reroll.glass[100][0]), "reroll retains the glass shell");
assert(mesh.interior.length > 1000 && mesh.glass.length > 1000, "both surfaces are triangulated");
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
for (const invalid of [{ eyeWidth: 0.7 }, { rings: 3 }, { eyeLength: 1 }]) {
  let threw = false; try { catEyeMarble({ glass: 1, eye: 2, accent: 3 }, invalid); } catch (e) { threw = e instanceof RangeError; }
  assert(threw, "invalid design rejected");
}
console.log(`PASS cat's-eye marble: ${mesh.interior.length} inclusion and ${mesh.glass.length} shell triangles`);
