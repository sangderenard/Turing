// THE WORLD CURVE: one control per game that has a curve, from a small round planet to a flat one to an orthographic one.
//
// Render-only.  The control is a slider position u in [0, 1] with four regions:
//
//   [0, U_ROUND]        a round world of radius R, logarithmic from R_MIN (very round) to R_MAX (large)
//   (U_ROUND, U_FLAT)   the last of the round: the curvature k = 1/R eases to 0 (R grows without bound)
//   [U_FLAT, U_ORTHO)   FLAT: R = infinity, k = exactly 0 (the page's planetDrop returns 0, nothing is tilted)
//   [U_ORTHO, 1]        ORTHOGRAPHIC FLAT: flat, and the scene camera is an orthographic projection framing the same view
//
// The page eases what the slider asks for (``stepCurve``): the curvature k by a critically damped spring (it reaches 0
// exactly, not asymptotically), and the perspective -> orthographic blend ``s`` by another.  The blend is a DOLLY ZOOM
// whose limit is the orthographic camera: the eye backs away along the view axis while the field of view narrows so the
// plane through the camera's target keeps the same half-height, d * tan(fov / 2) = H.  As d -> infinity the perspective
// matrix tends to the orthographic one with half-height H, so the end of the blend (the true orthographic matrix at the
// last distance) differs from the step before it by a sub-pixel amount: no pop.  Lighting and shaders are untouched: the
// lights and the specular term are in view space, and a far eye is the orthographic view direction.
//
// Pure arithmetic, tested in node (tests/meowdoku_world_curve_check.mjs).  The world radius of a TRUE sphere world is the
// law's (a compiled constant); this only drives how the page DRAWS the curve.
import { springStep } from "./world_zoom.js";

export const CURVE = {
  rMin: 6, rMax: 600,            // world units: very round ... large
  uRound: 0.8, uFlat: 0.9, uOrtho: 0.95,
  defaultRadius: 36,             // app.js PLANET.radius, the curve the terrain games always had
  omegaK: 8, omegaS: 5,          // 1/s: the curvature's ease and the dolly's
  kEps: 1e-6, kFlatEps: 1e-4, sEps: 1e-3,
  dollyMax: 5000,                // the farthest the dolly goes, in multiples of the perspective distance, before it IS the ortho camera
  fovy: 0.62,                    // the scene camera's vertical field of view (app.js camera())
  near: 0.1, far: 300,
};
const clamp01 = (u) => { const v = Number(u); return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0; };
export const clampCurveU = clamp01;

// ---------------------------------------------------------------- the slider
// notch (slider position) -> { mode: "round" | "flat" | "ortho", R, k }.  R = Infinity and k = 0 for flat and ortho.
export function uToCurve(u) {
  u = clamp01(u);
  if (u >= CURVE.uOrtho) return { mode: "ortho", R: Infinity, k: 0 };
  if (u >= CURVE.uFlat) return { mode: "flat", R: Infinity, k: 0 };
  const lr = Math.log(CURVE.rMax / CURVE.rMin);
  if (u <= CURVE.uRound) { const R = CURVE.rMin * Math.exp(lr * u / CURVE.uRound); return { mode: "round", R, k: 1 / R }; }
  // the last stretch: k falls linearly from 1/R_MAX to 0 at the flat notch
  const k = (1 / CURVE.rMax) * (CURVE.uFlat - u) / (CURVE.uFlat - CURVE.uRound);
  return { mode: "round", R: 1 / k, k };
}
// radius -> the notch that gives it (a finite radius only; Infinity is the flat notch)
export function radiusToU(R) {
  R = Number(R);
  if (!(R > 0)) return CURVE.uRound;
  if (!Number.isFinite(R)) return CURVE.uFlat;
  const lr = Math.log(CURVE.rMax / CURVE.rMin);
  if (R <= CURVE.rMax) return clamp01(CURVE.uRound * Math.log(Math.max(R, CURVE.rMin) / CURVE.rMin) / lr);
  const k = 1 / R;     // between R_MAX and flat: invert the linear k stretch
  return clamp01(CURVE.uFlat - (CURVE.uFlat - CURVE.uRound) * k * CURVE.rMax);
}
// the notch centres a click lands on
export const NOTCH = { flat: (CURVE.uFlat + CURVE.uOrtho) / 2, ortho: (CURVE.uOrtho + 1) / 2 };
export function curveLabel(u) {
  const c = uToCurve(u);
  return c.mode === "ortho" ? "orthographic flat" : c.mode === "flat" ? "flat" : `R ${c.R < 100 ? c.R.toFixed(1) : Math.round(c.R)}`;
}
// what the page eases toward: the curvature and the orthographic blend (0 perspective, 1 orthographic)
export const curveTarget = (u) => { const c = uToCurve(u); return { k: c.k, s: c.mode === "ortho" ? 1 : 0 }; };

// ---------------------------------------------------------------- the ease
// One step of the eased state { k, kv, s, sv }.  Both are critically damped (no overshoot); k reaches 0 exactly (FLAT is
// exactly flat) and s reaches exactly 0 / 1 (the true perspective / the true orthographic camera).
export function stepCurve(st, target, dt) {
  const a = springStep(st.k, st.kv || 0, target.k, dt, CURVE.omegaK);
  const b = springStep(st.s, st.sv || 0, target.s, dt, CURVE.omegaS);
  const kEps = target.k === 0 ? CURVE.kFlatEps : CURVE.kEps;
  const k = Math.abs(a.x - target.k) < kEps ? target.k : Math.max(0, a.x);
  const s = Math.abs(b.x - target.s) < CURVE.sEps ? target.s : Math.min(1, Math.max(0, b.x));
  return { k, kv: k === target.k ? 0 : a.v, s, sv: s === target.s ? 0 : b.v };
}
export const curveSettled = (st, target) => st.k === target.k && st.s === target.s;
export const initialCurve = (target) => ({ k: target.k, kv: 0, s: target.s, sv: 0 });

// ---------------------------------------------------------------- the dolly zoom
// The camera parameters at blend s for a perspective camera at distance d0 from its target with vertical field fovy0.
// The half-height of the plane through the target is H = d0 tan(fovy0 / 2) at EVERY s; s = 0 returns the camera as it was
// (exactly), s = 1 the orthographic camera with half-height H at the farthest distance.  The distance grows geometrically
// in s (a dolly is even in log distance), the depth range keeps its width (so near is still ahead of everything the
// original near plane saw).
export function dollyParams(d0, fovy0 = CURVE.fovy, s = 0, near0 = CURVE.near, far0 = CURVE.far) {
  s = clamp01(s);
  const H = d0 * Math.tan(fovy0 / 2);
  if (s === 0) return { d0, dist: d0, fovy: fovy0, halfHeight: H, near: near0, far: far0, ortho: false };
  const dist = d0 * Math.pow(CURVE.dollyMax, s);
  const front = d0 - near0, back = far0 - d0;           // the scene in front of / behind the target plane, as the original range had it
  const near = dist - front, far = dist + back;
  return { d0, dist, fovy: s >= 1 ? 0 : 2 * Math.atan(H / dist), halfHeight: H, near, far, ortho: s >= 1 };
}
// The projection of dollyParams, column-major (16 numbers), applied to coordinates in the view of the ORIGINAL camera (the eye
// at d0, not moved): the dolly's translation back along the axis is folded into the matrix, in closed form, and the whole
// matrix is divided by the distance.  The numbers stay of order 1/H however far the eye goes (the textbook matrix for an eye at
// 5000 d0 has depth terms of order d^2 / range that float32 cannot hold), so the page's float32 MVP products keep their
// depth resolution, and lighting (done in the original view) is exactly what it was.  A homogeneous scale changes nothing on
// the screen; clip.w is (dist - z0) / dist, which is 1 on the target's plane (z0 = -d0) and what sizeDepth reads.
//   s = 0 returns the textbook perspective matrix itself.
export function projectionMatrix(p, aspect) {
  const R = p.far - p.near, H = p.halfHeight, d = p.dist, delta = p.dist - p.d0;
  if (!p.ortho && delta === 0) {
    const f = 1 / Math.tan(p.fovy / 2), inv = 1 / (p.near - p.far);
    return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (p.far + p.near) * inv, -1, 0, 0, 2 * p.far * p.near * inv, 0];
  }
  if (p.ortho) return [1 / (H * aspect), 0, 0, 0, 0, 1 / H, 0, 0, 0, 0, -2 / R, 0, 0, 0, (2 * delta - (p.far + p.near)) / R, 1];
  const a = -(p.far + p.near) / R, b = -2 * p.far * p.near / R;
  return [1 / (H * aspect), 0, 0, 0, 0, 1 / H, 0, 0, 0, 0, a / d, -1 / d, 0, 0, (b - a * delta) / d, delta / d];
}
// what a particle of world radius r spans on the screen is r * proj[5] * (H / 2) / clip.w; clip.w is the eye-space depth
// of a perspective projection and 1 for an orthographic one (proj[11] = 0, proj[15] = 1)
export const clipW = (proj, eyeZ) => proj[11] * eyeZ + proj[15];
// the depth a particle sizes by: |eye depth| for a perspective camera, 1 for an orthographic one
export const sizeDepth = (proj, eyeZ) => Math.max(1e-3, Math.abs(clipW(proj, eyeZ)));

// ---------------------------------------------------------------- the camera as numbers, and picking
const vsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const vcross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const vnorm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
// the camera's axes (the same as app.js M.lookAt): right, up, and back (the eye looks along -back)
export function lookAxes(eye, target, up) {
  const z = vnorm(vsub(eye, target)), x = vnorm(vcross(up, z)), y = vcross(z, x);
  return { x, y, z };
}
// cam = { eye, target, up, aspect, params } (params from dollyParams).  A world point -> its normalised device x, y.
export function projectPoint(cam, p) {
  const ax = lookAxes(cam.eye, cam.target, cam.up), r = vsub(p, cam.eye);
  const vx = vdot(ax.x, r), vy = vdot(ax.y, r), vz = vdot(ax.z, r), P = cam.params;
  if (P.ortho) return [vx / (P.halfHeight * cam.aspect), vy / P.halfHeight];
  const t = Math.tan(P.fovy / 2);
  return [vx / (-vz * t * cam.aspect), vy / (-vz * t)];
}
// the ray through the normalised device point (x right, y up): { origin, dir }.  Perspective: from the eye; orthographic:
// parallel to the axis from the point on the camera's plane.
export function pickRay(cam, ndc) {
  const ax = lookAxes(cam.eye, cam.target, cam.up), P = cam.params, fwd = [-ax.z[0], -ax.z[1], -ax.z[2]];
  if (P.ortho) {
    const a = ndc[0] * P.halfHeight * cam.aspect, b = ndc[1] * P.halfHeight;
    return { origin: [cam.eye[0] + ax.x[0] * a + ax.y[0] * b, cam.eye[1] + ax.x[1] * a + ax.y[1] * b, cam.eye[2] + ax.x[2] * a + ax.y[2] * b], dir: fwd };
  }
  const t = Math.tan(P.fovy / 2), a = ndc[0] * t * cam.aspect, b = ndc[1] * t;
  return { origin: cam.eye.slice(), dir: vnorm([fwd[0] + ax.x[0] * a + ax.y[0] * b, fwd[1] + ax.x[1] * a + ax.y[1] * b, fwd[2] + ax.x[2] * a + ax.y[2] * b]) };
}
// the horizontal plane y = height under the pointer: [x, z], or null when the plane is not ahead of the ray
export function pickPlane(cam, ndc, height) {
  const r = pickRay(cam, ndc);
  if (Math.abs(r.dir[1]) < 1e-12) return null;
  const t = (height - r.origin[1]) / r.dir[1];
  return t >= 0 ? [r.origin[0] + t * r.dir[0], r.origin[2] + t * r.dir[2]] : null;
}
