// The waterwheel's motion blur: pure helpers (no DOM, no GL), imported by the page and held by tests/meowdoku_wheel_blur_check.mjs.
//
// A wheel turning too fast to read is not drawn crisp at every frame; it is TRANSFORMED into what the eye would see.  Three regimes,
// chosen by the angle swept since the last DRAWN frame (d = the unwrapped change of the law's angle):
//   1 slow       the swept tip arc is a few pixels: the one crisp wheel;
//   2 fast       the arc is longer: K copies (K <= WHEEL_MAX_COPIES) across [theta - d, theta], weighted toward the current angle;
//   3 very fast  d reaches about half the paddle spacing (2 pi / paddles): the paddles alias, so ONE sprite stands in for the wheel --
//                the angular average of its occupancy (hub ring solid, paddle annulus at the paddles' coverage) with faint streaks,
//                turned slowly so it reads as spinning.
// The regimes are CROSSFADED (smoothstep bands), never switched, so nothing pops.  Opacity is quantised to WHEEL_LEVELS steps because the
// page modulates it through the material table (one record per step), not per vertex.

export const WHEEL_MAX_COPIES = 8;
export const WHEEL_LEVELS = 16;
export const ARC_SLOW_PX = 2, ARC_FAST_PX = 10;            // swept tip arc: below the first the wheel is crisp, above the second the smear is full
export const ALIAS_FROM = 0.28, ALIAS_FULL = 0.5;         // d / paddle spacing: the disc starts here and is complete at ~half the spacing

export function smoothstep(x, a, b) { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }
// the signed shortest turn from `prev` to `cur`, whether the law's angle is continuous or wrapped
export function wrapDelta(d) { return d - 2 * Math.PI * Math.round(d / (2 * Math.PI)); }

// reach and pxPerUnit give the tip arc in screen pixels; paddles gives the spacing
export function wheelRegime({ dTheta, reach, pxPerUnit, paddles }) {
  const a = Math.abs(dTheta), arcPx = reach * a * pxPerUnit, spacing = 2 * Math.PI / Math.max(1, paddles), alias = a / spacing;
  const disc = smoothstep(alias, ALIAS_FROM, ALIAS_FULL);
  const smear = smoothstep(arcPx, ARC_SLOW_PX, ARC_FAST_PX);
  const K = Math.min(WHEEL_MAX_COPIES, 1 + Math.ceil(arcPx / 4));
  return { arcPx, alias, smear, disc, K: smear > 0 ? K : 1, mode: disc >= 1 ? 3 : (smear > 0 || disc > 0) ? 2 : 1 };
}

// The copies of regime 2: offset[i] in [0, 1] is how far BEHIND the current angle copy i sits as a fraction of d (the last copy, the
// current angle, is 0), alpha[i] its opacity.  Copies behind the current one ramp toward it; the current one keeps most of its
// strength; all fade out with the disc.  K = 1 or no smear gives the single crisp wheel.
export function smearWeights(K, smear, disc = 0) {
  K = Math.max(1, Math.min(WHEEL_MAX_COPIES, Math.round(K)));
  const out = [], cur = (1 - 0.45 * smear) * (1 - disc);
  for (let i = 0; i < K; i++) {
    if (i === K - 1) { out.push({ offset: 0, alpha: cur }); continue; }
    const ramp = (i + 1) / K;                                   // toward the current angle
    out.push({ offset: K === 1 ? 0 : (K - 1 - i) / (K - 1), alpha: smear * 0.7 * ramp / Math.sqrt(K) * (1 - disc) });
  }
  return out;
}

// an opacity 0..1 as a material-table step 0..levels-1, or -1 when it is too faint to draw
export function levelOf(alpha, levels = WHEEL_LEVELS) {
  if (!(alpha > 0.5 / levels)) return -1;
  return Math.min(levels - 1, Math.max(0, Math.round(alpha * levels) - 1));
}
export function levelAlpha(level, levels = WHEEL_LEVELS) { return (level + 1) / levels; }

// the sprite's slow spin: a fixed small fraction of the law's angle, so it drifts in the wheel's own direction at any speed and never
// strobes backwards (a function of theta mod a period would)
export const DISC_SPIN = 0.07;
export function discSpin(theta) { const t = DISC_SPIN * theta; return t - 2 * Math.PI * Math.floor(t / (2 * Math.PI)); }

// Is the point (x, y), in the wheel's frame at angle 0, inside paddle k's capsule?
function inPaddle(b, k, x, y) {
  const a = 2 * Math.PI * k / b.paddles, c = Math.cos(a), s = Math.sin(a), u = x * c + y * s, v = -x * s + y * c, t = b.half_thickness, l = b.reach;
  if (u >= 0 && u <= l && Math.abs(v) <= t) return true;
  return Math.hypot(u, v) <= t || Math.hypot(u - l, v) <= t;
}
// THE POLAR BLUR RASTER: nr rings (centre outward, out to R = reach + half_thickness) by nth sectors; each cell is the wheel's occupancy
// ANGULARLY AVERAGED over a full turn (rotation-invariant, so it holds for any angle), times a faint fixed streak pattern on the
// paddle annulus.  The hub ring is solid (1).  Coordinates are the law's, relative to the hinge.
export function polarBlurRaster(b, { nr = 12, nth = 48, streakAmp = 0.35, rings = 4 } = {}) {
  const R = b.reach + b.half_thickness, hubHw = Math.max(0.004, 0.6 * b.half_thickness), S = Math.max(64, b.paddles * 24);
  const profile = new Float32Array(nr), data = new Float32Array(nr * nth);
  for (let i = 0; i < nr; i++) {
    let paddleHits = 0, n = 0;
    for (let q = 0; q < rings; q++) {
      const r = R * (i + (q + 0.5) / rings) / nr;
      for (let s = 0; s < S; s++) {
        const ph = 2 * Math.PI * (s + 0.5) / S;
        let hit = false;
        for (let k = 0; k < b.paddles && !hit; k++) hit = inPaddle(b, k, r * Math.cos(ph), r * Math.sin(ph));
        if (hit) paddleHits++;
        n++;
      }
    }
    const solid = b.hub >= R * i / nr - hubHw && b.hub <= R * (i + 1) / nr + hubHw, paddle = n ? paddleHits / n : 0;   // the ring cell holding the hub ring
    profile[i] = solid ? 1 : paddle;
    for (let j = 0; j < nth; j++) {
      const ph = 2 * Math.PI * (j + 0.5) / nth;
      const streak = (Math.sin(7 * ph) + Math.sin(13 * ph + 1) + Math.sin(3 * ph + 2)) / 3;
      data[i * nth + j] = solid ? 1 : Math.min(1, Math.max(0, profile[i] * (1 + streakAmp * streak)));
    }
  }
  return { nr, nth, R, profile, data };
}
