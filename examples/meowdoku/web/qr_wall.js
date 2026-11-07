// [QRWALL] THE TIP JAR'S QR WALL (render only).  The jar's `bitcoin:` URI as a wall of glass CUBES behind the jar's back pane: dark modules are
// OBSIDIAN cubes, light modules (and the 4-module quiet zone) are FROSTED glass cubes, so the wall is a real, scannable QR code that is also
// part of the scene.  This file is the pure part (the module -> cube layout, the cube geometry, the head-on camera); app.js builds the mesh
// through its own litMesh / material path and draws it with the housing.  Nothing here simulates and the law is not touched.
//
// WHERE THE WALL IS.  The 3-D fountain's chamber is a box in the board's frame: law x -> world +X, law y -> world -Z, law z (the slab's depth) ->
// world height.  Seen from above (the default view tips 31 degrees off straight down) the BACK pane is the one at law z = -depth, under the
// jar, so the wall lies in the horizontal plane just behind it; it reads upright from the front (column 0 at the left, row 0 at the top, which
// is world -Z), and looking straight down at it is the HEAD-ON view.

export const QUIET = 4;                       // the QR standard's quiet zone, in modules
export const OBSIDIAN = "qr_obsidian@glass";  // the scene materials (app.js uploadMaterials declares them)
export const FROST = "qr_frost@glass";

// The module grid of a matrix (rows of booleans, true = dark) with its quiet zone: row-major cells { x, y, dark, quiet } over (n + 2 quiet)^2.
export function qrWallLayout(matrix, quiet = QUIET) {
  const n = matrix.length, total = n + 2 * quiet, cells = [];
  let dark = 0, light = 0, border = 0;
  for (let y = 0; y < total; y++) for (let x = 0; x < total; x++) {
    const inside = x >= quiet && y >= quiet && x < quiet + n && y < quiet + n;
    const isDark = inside && !!matrix[y - quiet][x - quiet];
    cells.push({ x, y, dark: isDark, quiet: !inside });
    if (isDark) dark++; else if (inside) light++; else border++;
  }
  return { n, quiet, total, cells, counts: { dark, light, quiet: border, cubes: total * total } };
}

// The scene material a module takes: obsidian for dark, frosted glass for light and for the quiet zone.
export const materialName = (cell) => (cell.dark ? OBSIDIAN : FROST);

// The world position (centre of the cube's top face) of a layout cell on a wall of side `size` centred at (cx, cz), top faces at height y.
export function cellCentre(layout, cell, wall) {
  const pitch = wall.size / layout.total;
  return [wall.cx - wall.size / 2 + (cell.x + 0.5) * pitch, wall.y, wall.cz - wall.size / 2 + (cell.y + 0.5) * pitch];
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// One bevelled cube, open at the back (never seen: the wall is the back): a top face, four 45 degree bevels, four walls.  `s` the side, `b` the
// bevel, `depth` the height of the cube, (cx, cz) the centre, `yTop` the top face's height.  Faces are triangles [a, b, c, material] wound
// outward (litMesh takes the normal from the winding).
export function cubeFaces(cx, cz, yTop, s, b, depth, m, out = []) {
  const h = s / 2, k = h - b, yb = yTop - b, y0 = yTop - depth;
  const ring = (r, y) => [[cx - r, y, cz - r], [cx + r, y, cz - r], [cx + r, y, cz + r], [cx - r, y, cz + r]];
  const I = ring(k, yTop), O = ring(h, yb), B = ring(h, y0);
  const quad = (p, q, r, t, outward) => {                       // two triangles, wound so the face looks along `outward`
    const flip = dot(cross(sub(q, p), sub(r, p)), outward) < 0;
    const [a, c, d] = flip ? [p, r, q] : [p, q, r], [a2, c2, d2] = flip ? [p, t, r] : [p, r, t];
    out.push([a, c, d, m], [a2, c2, d2, m]);
  };
  quad(I[0], I[1], I[2], I[3], [0, 1, 0]);
  const side = [[0, 0, -1], [1, 0, 0], [0, 0, 1], [-1, 0, 0]];
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4, n = side[i];
    quad(O[i], O[j], I[j], I[i], [n[0], 1, n[2]]);              // the bevel
    quad(B[i], B[j], O[j], O[i], n);                            // the wall
  }
  return out;
}

// The whole wall as triangles: `wall` = { cx, cz, y, size }; `ids` = { [OBSIDIAN]: material id, [FROST]: material id }.  Cubes are gapped by
// `gap` of the pitch and bevelled by `bevel` of the pitch, so the wall reads as glass blocks but a module stays a module.
export function qrWallFaces(layout, wall, ids, { gap = 0.06, bevel = 0.1, depth = 0.9 } = {}) {
  const pitch = wall.size / layout.total, s = pitch * (1 - gap), faces = [];
  for (const cell of layout.cells) {
    const c = cellCentre(layout, cell, wall);
    cubeFaces(c[0], c[2], wall.y, s, pitch * bevel, pitch * depth, ids[materialName(cell)], faces);
  }
  return faces;
}

// The wall's square inside the jar: centred on the chamber's footprint in the board's frame, as large as the chamber's height (the smaller
// side of its box) allows, lying just behind the back pane.  `bounds` is the chamber mesh's [x0, x1, y0, y1, z0, z1] in law units; `map` is
// the law -> world map of the page (lawToWorld, in which law z is the height).  `behind` is the world gap behind the pane.
export function wallPlacement(bounds, map, scale, behind = 0.04) {
  const [x0, x1, y0, y1, z0] = bounds;
  const c = map((x0 + x1) / 2, (y0 + y1) / 2, z0);
  return { cx: c[0], cz: c[2], y: c[1] - behind, size: Math.min(x1 - x0, y1 - y0) * scale };
}

// THE HEAD-ON VIEW: the camera goal that faces the wall straight on (looking straight down at the board's plane: yaw 0, pitch 0 on a 90 degree
// tilt) and fills the free part of the view with it.  `cam` = { dist0 (the camera's distance at zero zoom), targetY, centre: [x, z] the camera
// target's rest position, tanHalf (tan of half the vertical field of view), w, h (the canvas, px) }; `reserve` the px the page's panel covers
// at the right / bottom of the canvas.  Returns { yaw, pitch, lz, px, pz }.
export function headOnView(wall, cam, reserve = { right: 0, bottom: 0 }, margin = 1.08) {
  const availW = Math.max(40, cam.w - (reserve.right || 0)), availH = Math.max(40, cam.h - (reserve.bottom || 0));
  const u = wall.size * margin / Math.min(availW, availH);                  // world units per px at the wall's plane
  const dist = (u * cam.h / 2) / cam.tanHalf - (cam.targetY - wall.y);      // the eye's height over the camera target
  const tx = wall.cx + (reserve.right || 0) / 2 * u, tz = wall.cz + (reserve.bottom || 0) / 2 * u;   // the wall centred in the free area
  return { yaw: 0, pitch: 0, lz: Math.log(Math.max(dist, 1e-3) / cam.dist0), px: tx - cam.centre[0], pz: tz - cam.centre[1] };
}
