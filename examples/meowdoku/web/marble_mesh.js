// Layered cat's-eye marble geometry for the page's existing litMesh/Phong path.
// Coordinates are in marble radii. The closed dyed spindle sits inside the
// closed glass shell; neither surface is a painted texture or a shader trick.

const TAU = 2 * Math.PI;

function unit(x, y, z) {
  const length = Math.hypot(x, y, z) || 1;
  return [x / length, y / length, z / length];
}

function checked(options) {
  const o = { rings: 24, segments: 48, eyeWidth: 0.28, eyeLength: 0.75,
    eyeCurve: 0.18, strandCount: 3, seed: 0, ...options };
  for (const key of ["rings", "segments", "strandCount", "seed"])
    if (!Number.isInteger(o[key])) throw new RangeError(`${key} must be an integer`);
  if (o.rings < 8 || o.rings > 96 || o.segments < 12 || o.segments > 192 || o.strandCount < 1 || o.strandCount > 7)
    throw new RangeError("marble tessellation outside supported range");
  if (o.seed < 0 || o.seed > 0x7fffffff) throw new RangeError("seed outside supported range");
  if (!(o.eyeWidth >= 0.08 && o.eyeWidth <= 0.42 && o.eyeLength >= 0.4 && o.eyeLength <= 0.85 &&
        o.eyeCurve >= -0.28 && o.eyeCurve <= 0.28))
    throw new RangeError("cat's-eye dimensions outside the glass interior");
  return o;
}

function surface(faces, point, normal, rings, segments, material) {
  for (let i = 0; i < rings; i++) for (let j = 0; j < segments; j++) {
    const a = point(i, j), b = point(i + 1, j), c = point(i + 1, j + 1), d = point(i, j + 1);
    const na = normal(i, j), nb = normal(i + 1, j), nc = normal(i + 1, j + 1), nd = normal(i, j + 1);
    if (i < rings - 1) faces.push([a, c, b, material, [na, nc, nb]]);
    if (i > 0) faces.push([a, d, c, material, [na, nd, nc]]);
  }
}

/**
 * Build separately drawable opaque interior and transparent glass exterior.
 * `materials` contains the integer IDs in the page's MaterialDatabase table.
 * The renderer must draw `interior` before `glass` with depth writes off for
 * glass, as it already does for translucent scene geometry.
 */
export function catEyeMarble(materials, options = {}) {
  const o = checked(options), interior = [], glass = [];
  for (const key of ["glass", "eye", "accent"])
    if (!Number.isInteger(materials[key]) || materials[key] < 0) throw new TypeError(`missing material ${key}`);
  const sphere = (radius) => (i, j) => {
    const theta = Math.PI * i / o.rings, phi = TAU * j / o.segments;
    return [radius * Math.sin(theta) * Math.cos(phi), radius * Math.cos(theta), radius * Math.sin(theta) * Math.sin(phi)];
  };
  const shellPoint = sphere(1);
  surface(glass, shellPoint, (i, j) => unit(...shellPoint(i, j)), o.rings, o.segments, materials.glass);

  // The dyed inclusion is a curved, pointed spindle. Its longitudinal ridges
  // are real enclosed surfaces; alternating material strips form the cat's
  // eye and are lit through the independently drawn shell.
  const point = (i, j) => {
    const theta = Math.PI * i / o.rings, phi = TAU * j / o.segments;
    const s = Math.sin(theta), y = o.eyeLength * Math.cos(theta);
    const phase = (o.seed % 997) / 997 * TAU;
    const wave = 1 + 0.07 * Math.cos(3 * phi + 2 * theta + phase)
      + 0.045 * Math.sin(5 * phi - 3 * theta + 1.7 * phase);
    const radius = o.eyeWidth * Math.pow(s, 0.78) * wave;
    const bend = o.eyeCurve * (1 - y * y) + 0.06 * Math.sin(2 * theta + phase) * s * s;
    return [bend + radius * Math.cos(phi), y, radius * Math.sin(phi)];
  };
  // Tangents of the parametric surface provide the normal even where the
  // spindle bends. This avoids lighting the dyed region as a flat decal.
  const normal = (i, j) => {
    if (i <= 0) return [0, 1, 0];
    if (i >= o.rings) return [0, -1, 0];
    const eps = 0.001, p = (u, v) => point(Math.max(0, Math.min(o.rings, u)), v);
    const a = p(i + eps, j), b = p(i - eps, j), c = p(i, j + eps), d = p(i, j - eps);
    const t = a.map((v, k) => v - b[k]), q = c.map((v, k) => v - d[k]);
    return unit(q[1]*t[2]-q[2]*t[1], q[2]*t[0]-q[0]*t[2], q[0]*t[1]-q[1]*t[0]);
  };
  for (let i = 0; i < o.rings; i++) for (let j = 0; j < o.segments; j++) {
    const a = point(i, j), b = point(i+1, j), c = point(i+1, j+1), d = point(i, j+1);
    const m = Math.floor(j * o.strandCount / o.segments) % 2 ? materials.accent : materials.eye;
    if (i < o.rings - 1) interior.push([a, c, b, m, [normal(i,j), normal(i+1,j+1), normal(i+1,j)]]);
    if (i > 0) interior.push([a, d, c, m, [normal(i,j), normal(i,j+1), normal(i+1,j+1)]]);
  }
  return { interior, glass, triangleCount: interior.length + glass.length,
    parameters: o };
}
