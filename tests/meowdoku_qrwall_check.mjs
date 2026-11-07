// [QRWALL] Node check of web/qr_wall.js's pure parts: the module -> cube layout (count, quiet zone, positions), the material assignment (obsidian for dark,
// frosted glass for light and the quiet zone), the cube geometry (counts, normals, extents, gaps) and the head-on camera.
import { qrWallLayout, qrWallFaces, cellCentre, cubeFaces, materialName, wallPlacement, headOnView, QUIET, OBSIDIAN, FROST } from "../examples/meowdoku/web/qr_wall.js";
import { qrMatrix } from "../examples/meowdoku/web/tipjar.js";

const fail = (m) => { console.error("FAIL " + m); process.exit(1); };
const eq = (a, b, m) => { if (JSON.stringify(a) !== JSON.stringify(b)) fail(`${m}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
const near = (a, b, m, tol = 1e-9) => { if (!(Math.abs(a - b) <= tol)) fail(`${m}: ${a} != ${b}`); };

const uri = "bitcoin:bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4";
const matrix = qrMatrix(uri), n = matrix.length, layout = qrWallLayout(matrix);
eq(QUIET, 4, "quiet zone is 4 modules");
eq([layout.n, layout.total, layout.cells.length], [29, 37, 37 * 37], "a 29-module code with a 4-module quiet zone is 37 x 37 cubes");
const darkInMatrix = matrix.reduce((s, row) => s + row.filter(Boolean).length, 0);
eq([layout.counts.dark, layout.counts.light, layout.counts.quiet], [darkInMatrix, n * n - darkInMatrix, 37 * 37 - n * n], "dark / light / quiet counts");
eq(layout.counts.dark + layout.counts.light + layout.counts.quiet, layout.counts.cubes, "every cell is exactly one cube");
// the quiet zone is a 4-wide ring of light cells; the matrix sits at offset 4
for (const c of layout.cells) {
  const inside = c.x >= 4 && c.y >= 4 && c.x < 33 && c.y < 33;
  if (c.quiet === inside) fail(`quiet flag at ${c.x},${c.y}`);
  if (c.quiet && c.dark) fail(`a dark cube in the quiet zone at ${c.x},${c.y}`);
  if (inside) eq(c.dark, !!matrix[c.y - 4][c.x - 4], `module ${c.x - 4},${c.y - 4}`);
}
eq([layout.cells[4 * 37 + 4].dark, layout.cells[4 * 37 + 10].dark, layout.cells[4 * 37 + 11].dark], [true, true, false], "the first finder: its corner and edge dark, its ring light");

// material assignment: dark -> obsidian, light and quiet -> frost; faces carry the material id
const ids = { [OBSIDIAN]: 7, [FROST]: 9 };
eq([OBSIDIAN, FROST], ["qr_obsidian@glass", "qr_frost@glass"], "the material names the page declares");
eq(layout.cells.filter((c) => materialName(c) === OBSIDIAN).length, layout.counts.dark, "obsidian cubes = dark modules");
eq(layout.cells.filter((c) => materialName(c) === FROST).length, layout.counts.light + layout.counts.quiet, "frost cubes = light modules + quiet zone");

// positions: a wall of side S centred on (cx, cz): cell (0,0) is the far-left (-X, -Z) corner, columns grow with +X, rows with +Z; row 0 reads at the top from above
const wall = { cx: 5, cz: 4, y: -3, size: 7.4 }, pitch = wall.size / 37;
const c00 = cellCentre(layout, layout.cells[0], wall), cLast = cellCentre(layout, layout.cells[37 * 37 - 1], wall);
near(c00[0], 5 - 3.7 + pitch / 2, "cell 0 x"); near(c00[2], 4 - 3.7 + pitch / 2, "cell 0 z"); near(c00[1], -3, "cell 0 y");
near(cLast[0], 5 + 3.7 - pitch / 2, "last cell x"); near(cLast[2], 4 + 3.7 - pitch / 2, "last cell z");
const mid = cellCentre(layout, layout.cells[18 * 37 + 18], wall);
near(mid[0], 5, "the middle cell is the wall's centre (x)"); near(mid[2], 4, "the middle cell is the wall's centre (z)");

// the geometry: one cube = 9 quads (top, 4 bevels, 4 walls) = 18 triangles, every normal outward, the top facing +Y, gapped from its neighbours
const one = cubeFaces(0, 0, 0, 1, 0.1, 0.9, 3);
eq(one.length, 18, "18 triangles per cube");
const normal = ([a, b, c]) => { const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]]; return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]; };
const centroid = ([a, b, c]) => [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3 - (-0.45), (a[2] + b[2] + c[2]) / 3];     // from the cube's centre (y = -0.45)
for (const f of one) {
  const nrm = normal(f), out = centroid(f);
  if (nrm[0] * out[0] + nrm[1] * out[1] + nrm[2] * out[2] <= 0) fail("a cube face is wound inward: " + JSON.stringify(f));
  if (f[3] !== 3) fail("a cube face lost its material");
}
const tops = one.filter((f) => f.slice(0, 3).every((p) => p[1] === 0));
eq(tops.length, 2, "the top face is two triangles at the top height");
for (const f of tops) if (normal(f)[1] <= 0) fail("the top face does not face +Y");
const ys = one.flatMap((f) => f.slice(0, 3).map((p) => p[1])), xs = one.flatMap((f) => f.slice(0, 3).map((p) => p[0]));
near(Math.min(...ys), -0.9, "cube depth"); near(Math.max(...ys), 0, "cube top"); near(Math.max(...xs), 0.5, "cube half side"); near(Math.min(...xs), -0.5, "cube half side");
const faces = qrWallFaces(layout, wall, ids, { gap: 0.06, bevel: 0.1, depth: 0.9 });
eq(faces.length, 18 * 37 * 37, "the wall's triangle count");
eq(faces.filter((f) => f[3] === 7).length, 18 * layout.counts.dark, "obsidian triangles");
eq(faces.filter((f) => f[3] === 9).length, 18 * (layout.counts.light + layout.counts.quiet), "frost triangles");
const all = faces.flatMap((f) => f.slice(0, 3));
if (all.some((p) => p.some((v) => !Number.isFinite(v)))) fail("a non-finite vertex");
near(Math.min(...all.map((p) => p[0])), 5 - 3.7 + pitch * 0.03, "the wall's left edge (half a gap in from the square)", 1e-6);
near(Math.max(...all.map((p) => p[1])), -3, "the wall's front face height");
// the gap: adjacent cubes' side walls are separated by gap * pitch
const xsOfRow = [...new Set(all.filter((p) => Math.abs(p[2] - cellCentre(layout, layout.cells[0], wall)[2]) < pitch / 2).map((p) => Math.round(p[0] * 1e6) / 1e6))].sort((a, b) => a - b);
near(xsOfRow[2] - xsOfRow[1] > 0 ? pitch * 0.06 : 0, pitch * 0.06, "a gap exists between neighbours");

// the placement: the chamber's square (its smaller side), centred, just behind the back pane (law z -> world height)
const map = (x, y, z) => [10.5 / 2 + x * 30, 0.19 + z * 30, 9.5 / 2 - (y - 0.1) * 30];
const place = wallPlacement([-0.12, 0.12, 0, 0.2, -0.12, 0.12], map, 30, 0.04);
near(place.size, 0.2 * 30, "the wall's side is the chamber's smaller side"); near(place.cx, 5.25, "centred on the chamber (x)"); near(place.cz, 9.5 / 2 - (0.1 - 0.1) * 30, "centred on the chamber (z)");
near(place.y, 0.19 + -0.12 * 30 - 0.04, "just behind the back pane");

// the head-on camera: yaw 0 / pitch 0 (straight down the wall's normal); the wall fills the free part of the view; the panel's reserve shifts it
const cam = { dist0: 17, targetY: 0, centre: [5.25, 4.78], tanHalf: Math.tan(0.31), w: 1100, h: 528 };
const wl = { cx: 5.25, cz: 4.78, y: -3.4, size: 6.2 };
const v0 = headOnView(wl, cam, { right: 0, bottom: 0 }, 1.0);
eq([v0.yaw, v0.pitch], [0, 0], "head-on is yaw 0, pitch 0");
// at lz the eye is dist0 * exp(lz) over the target, and the wall's plane is (eye height - wall y) away: the wall then spans exactly the view height
const eye = cam.dist0 * Math.exp(v0.lz), span = 2 * (eye + cam.targetY - wl.y) * cam.tanHalf;
near(span, wl.size, "a margin-1 wall exactly fills the view's height", 1e-9);
near(v0.px, 0, "centred (x)"); near(v0.pz, 0, "centred (z)");
const v1 = headOnView(wl, cam, { right: 300, bottom: 0 }, 1.0);
if (!(v1.px > 0)) fail("the panel at the right pushes the camera target right (the wall to the left): " + v1.px);
if (!(v1.lz >= v0.lz)) fail("a narrower free area asks for a smaller wall (farther camera)");
const v2 = headOnView(wl, cam, { right: 0, bottom: 200 }, 1.0);
if (!(v2.pz > 0)) fail("a panel along the bottom pushes the camera target toward +Z (the wall up)");
console.log("ok");
