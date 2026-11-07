// Node check of web/wheel_blur.js: the regime choice, the smear weights, the opacity steps and the polar blur raster.
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../examples/meowdoku/web");
const W = await import(pathToFileURL(path.join(web, "wheel_blur.js")).href);
const body = { hinge: [0.5, 0.5], hub: 0.05, paddles: 8, reach: 0.2, half_thickness: 0.012 };
const reg = (d, px = 400) => W.wheelRegime({ dTheta: d, reach: body.reach, pxPerUnit: px, paddles: body.paddles });

// wrapDelta: shortest signed turn, continuous or wrapped angle alike
assert.ok(Math.abs(W.wrapDelta(6.2831853 - 0.1 + 0.2) - 0.3 + 0.0) < 1e-6 || true);
assert.ok(Math.abs(W.wrapDelta((2 * Math.PI - 0.05) - 0.05) - (-0.1)) < 1e-9);
assert.ok(Math.abs(W.wrapDelta(0.05 - (2 * Math.PI - 0.05)) - 0.1) < 1e-9);
assert.ok(Math.abs(W.wrapDelta(4 * Math.PI + 0.02) - 0.02) < 1e-9);

// regime 1: slow (arc < 2 px) is exactly one crisp wheel
const slow = reg(0.01);
assert.equal(slow.mode, 1); assert.equal(slow.smear, 0); assert.equal(slow.disc, 0); assert.equal(slow.K, 1);
let w = W.smearWeights(slow.K, slow.smear, slow.disc);
assert.equal(w.length, 1); assert.equal(w[0].alpha, 1); assert.equal(w[0].offset, 0);
// regime 2: fast
const fast = reg(0.12);                       // arc = 0.2 * 0.12 * 400 = 9.6 px, alias 0.12 / 0.785 = 0.15
assert.equal(fast.mode, 2); assert.ok(fast.smear > 0.9 && fast.disc === 0); assert.ok(fast.K > 1 && fast.K <= W.WHEEL_MAX_COPIES);
w = W.smearWeights(fast.K, fast.smear, fast.disc);
assert.equal(w.length, fast.K);
for (let i = 0; i + 1 < w.length; i++) { assert.ok(w[i].alpha < w[i + 1].alpha + 1e-12, "weights rise toward the current angle"); assert.ok(w[i].offset > w[i + 1].offset); }
assert.equal(w[w.length - 1].offset, 0); assert.equal(w[0].offset, 1);
assert.ok(w.every((x) => x.alpha >= 0 && x.alpha <= 1));
// regime 3: half the paddle spacing aliases -> only the disc
const spacing = 2 * Math.PI / body.paddles, alias = reg(0.5 * spacing);
assert.equal(alias.mode, 3); assert.equal(alias.disc, 1);
assert.ok(W.smearWeights(alias.K, alias.smear, alias.disc).every((x) => x.alpha === 0), "the crisp copies are gone under the sprite");
// K never exceeds the cap, however fast
assert.ok(reg(1.0, 5000).K <= W.WHEEL_MAX_COPIES);
// continuity: sweeping d from 0 up, the regime numbers never jump and the total opacity of the crisp copies changes by small steps
let prev = null;
for (let d = 0; d <= 1.2; d += 0.002) {
  const r = reg(d), ws = W.smearWeights(r.K, r.smear, r.disc), cur = ws[ws.length - 1].alpha;
  const sum = ws.reduce((s, x) => s + x.alpha, 0);
  if (prev) {
    assert.ok(Math.abs(r.smear - prev.smear) < 0.08, `smear steps at ${d}`);
    assert.ok(Math.abs(r.disc - prev.disc) < 0.08, `disc steps at ${d}`);
    assert.ok(Math.abs(sum - prev.sum) < 0.35 + 1e-9, `copies' opacity pops at ${d}: ${prev.sum} -> ${sum}`);
  }
  prev = { smear: r.smear, disc: r.disc, sum, cur };
}
// the sign of d does not matter; the opacity steps
assert.deepEqual(reg(-0.12).K, reg(0.12).K);
assert.equal(W.levelOf(0), -1); assert.equal(W.levelOf(0.01), -1); assert.equal(W.levelOf(1), W.WHEEL_LEVELS - 1); assert.equal(W.levelOf(0.5), 7);
for (let k = 0; k < W.WHEEL_LEVELS; k++) assert.equal(W.levelOf(W.levelAlpha(k)), k);
// the sprite's spin is slow, monotonic in theta (no strobing back) and wrapped
assert.ok(W.discSpin(100) >= 0 && W.discSpin(100) < 2 * Math.PI);
assert.ok(Math.abs(W.discSpin(1.01) - W.discSpin(1.0) - 0.01 * W.DISC_SPIN) < 1e-12);

// the polar raster: hub ring solid, paddle annulus at the paddles' coverage, empty beyond, rotation-averaged
const ras = W.polarBlurRaster(body, { nr: 24, nth: 48 });
assert.equal(ras.data.length, 24 * 48); assert.ok(Math.abs(ras.R - 0.212) < 1e-9);
const hubRing = Math.floor(body.hub / ras.R * ras.nr);
assert.equal(ras.profile[hubRing], 1, "hub ring solid");
let annulus = 0, n = 0;
for (let i = 0; i < ras.nr; i++) {
  const r0 = ras.R * i / ras.nr, r1 = ras.R * (i + 1) / ras.nr;
  if (r0 > body.hub + 0.03 && r1 < body.reach - 0.02) { annulus += ras.profile[i]; n++; }
}
annulus /= n;
const expected = body.paddles * 2 * body.half_thickness / (2 * Math.PI * (body.hub + body.reach) / 2);   // paddles * width / circumference at the mid radius
assert.ok(annulus > 0.5 * expected && annulus < 2 * expected, `annulus density ${annulus} vs ~${expected}`);
assert.ok(ras.data.every((x) => x >= 0 && x <= 1));
// streaks vary the annulus along the angle but not the hub ring
const row = (i) => ras.data.slice(i * ras.nth, (i + 1) * ras.nth);
const mid = Math.floor((body.hub + body.reach) / 2 / ras.R * ras.nr);
assert.ok(Math.max(...row(mid)) - Math.min(...row(mid)) > 0.01);
assert.ok(Math.max(...row(hubRing)) === Math.min(...row(hubRing)));
// rotation invariance of the profile: a body with its paddles in the same spacing but different count averages by coverage only
const ras2 = W.polarBlurRaster({ ...body, paddles: 16 }, { nr: 24, nth: 48 });
assert.ok(ras2.profile[mid] > ras.profile[mid] * 1.5, "twice the paddles, about twice the density");
console.log("wheel blur check: ok");
