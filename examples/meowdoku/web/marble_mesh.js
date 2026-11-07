// Layered cat's-eye marble geometry for the page's existing litMesh/Phong path.
// Coordinates are in marble radii. Each dyed ribbon is a closed, thin volume
// inside the separate closed glass shell; neither is a painted texture.

const TAU = 2 * Math.PI;

function unit(x, y, z) {
  const length = Math.hypot(x, y, z) || 1;
  return [x / length, y / length, z / length];
}

function checked(options) {
  const o = { rings: 24, segments: 48, eyeWidth: 0.28, eyeLength: 0.75,
    eyeCurve: 0.18, strandCount: 3, ribbonThickness: 0.035,
    ribbonTwist: 0.9, ribbonTip: 1.35, ribbonFlare: 0.18, seed: 0, ...options };
  for (const key of ["rings", "segments", "strandCount", "seed"])
    if (!Number.isInteger(o[key])) throw new RangeError(`${key} must be an integer`);
  if (o.rings < 8 || o.rings > 128 || o.segments < 12 || o.segments > 256 || o.strandCount < 1 || o.strandCount > 7)
    throw new RangeError("marble tessellation outside supported range");
  if (o.seed < 0 || o.seed > 0x7fffffff) throw new RangeError("seed outside supported range");
  if (!(o.eyeWidth >= 0.08 && o.eyeWidth <= 0.42 && o.eyeLength >= 0.4 && o.eyeLength <= 0.85 &&
        o.eyeCurve >= -0.28 && o.eyeCurve <= 0.28))
    throw new RangeError("cat's-eye dimensions outside the glass interior");
  if (!(o.ribbonThickness >= 0.008 && o.ribbonThickness <= 0.08 && o.ribbonTwist >= -3.2 && o.ribbonTwist <= 3.2 &&
        o.ribbonTip >= 0.6 && o.ribbonTip <= 3.0 && o.ribbonFlare >= 0 && o.ribbonFlare <= 0.6))
    throw new RangeError("ribbon geometry outside the glass interior");
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
  const o = checked(options), interior = [], glass = [], ribbons = [];
  for (const key of ["glass", "eye", "accent"])
    if (!Number.isInteger(materials[key]) || materials[key] < 0) throw new TypeError(`missing material ${key}`);
  const sphere = (radius) => (i, j) => {
    const theta = Math.PI * i / o.rings, phi = TAU * j / o.segments;
    return [radius * Math.sin(theta) * Math.cos(phi), radius * Math.cos(theta), radius * Math.sin(theta) * Math.sin(phi)];
  };
  const shellPoint = sphere(1);
  surface(glass, shellPoint, (i, j) => unit(...shellPoint(i, j)), o.rings, o.segments, materials.glass);

  // Each blade has its own closed cross-section, tapering to two points. The
  // broad faces, narrow bevel-like sides and curled tips therefore present
  // different normals to the real Phong light. The blades twist with the
  // shared curve but are offset in depth, like dyed glass drawn through a
  // clear marble, rather than sectors of one broad spindle.
  for (let strand = 0; strand < o.strandCount; strand++) {
    const lane = strand - (o.strandCount - 1) / 2;
    const phase = ((o.seed + strand * 619) % 997) / 997 * TAU;
    const point = (i, j) => {
      const theta = Math.PI * i / o.rings, phi = TAU * j / o.segments;
      const s = Math.max(0, Math.sin(theta)), y = o.eyeLength * Math.cos(theta);
      const t = y / o.eyeLength, centre = o.eyeCurve * (1 - t*t);
      const angle = 0.38 * lane + o.ribbonTwist * t + 0.18 * Math.sin(theta + phase);
      const ux = Math.cos(angle), uz = Math.sin(angle), vx = -uz, vz = ux;
      const width = o.eyeWidth * (0.46 + 0.08 * Math.sin(phase + strand)) * Math.pow(s, o.ribbonTip)
        * (1 + o.ribbonFlare * Math.sin(2 * theta + phase) * s);
      const thick = o.ribbonThickness * Math.pow(s, Math.max(0.6, o.ribbonTip * 0.7));
      // A superellipse cross-section gives broad, flat-looking faces and
      // narrow rounded edges without open seams at a twist or at a tip.
      const p = Math.sign(Math.cos(phi)) * Math.sqrt(Math.abs(Math.cos(phi)));
      const q = Math.sign(Math.sin(phi)) * Math.sqrt(Math.abs(Math.sin(phi)));
      const spread = Math.min(0.12, 0.30 / Math.max(1, (o.strandCount - 1) / 2));
      const offsetX = lane * spread, offsetZ = lane * Math.min(0.07, spread * 0.7);
      const sway = 0.024 * Math.sin(2 * theta + phase) * s*s;
      return [centre + offsetX + sway + width * p * ux + thick * q * vx,
        y, offsetZ + width * p * uz + thick * q * vz];
    };
    const normal = (i, j) => {
      if (i <= 0) return [0, 1, 0];
      if (i >= o.rings) return [0, -1, 0];
      const eps = 0.001, p = (u, v) => point(Math.max(0, Math.min(o.rings, u)), v);
      const a = p(i + eps, j), b = p(i - eps, j), c = p(i, j + eps), d = p(i, j - eps);
      const t = a.map((v, k) => v - b[k]), q = c.map((v, k) => v - d[k]);
      return unit(q[1]*t[2]-q[2]*t[1], q[2]*t[0]-q[0]*t[2], q[0]*t[1]-q[1]*t[0]);
    };
    const start = interior.length, material = strand % 2 ? materials.accent : materials.eye;
    surface(interior, point, normal, o.rings, o.segments, material);
    ribbons.push({ start, count: interior.length - start, material });
  }
  return { interior, glass, triangleCount: interior.length + glass.length,
    ribbons, parameters: o };
}
