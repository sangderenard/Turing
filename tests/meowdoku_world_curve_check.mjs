// Node check of web/world_curve.js (the world curve control: notch -> R / flat / ortho, the ease, the dolly zoom whose limit is the
// orthographic camera, particle sizing and picking under it) and of the control's share-link spec (app.js's own CURVE_SPEC and
// pageSpecs, carried by share.js serializeSpecs).
import fs from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../examples/meowdoku/web");
const C = await import(pathToFileURL(path.join(web, "world_curve.js")).href);
const S = await import(pathToFileURL(path.join(web, "share.js")).href);
const P = await import(pathToFileURL(path.join(web, "particle_styles.js")).href);
const rl = await import(pathToFileURL(path.join(web, "replay_layers.js")).href);
const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ""} ${a} vs ${b} (tol ${tol})`);
const { CURVE } = C;

// ---------------------------------------------------------------- the slider: notch -> R / flat / ortho
{
  const lo = C.uToCurve(0), big = C.uToCurve(CURVE.uRound), flat = C.uToCurve(CURVE.uFlat), ortho = C.uToCurve(CURVE.uOrtho);
  assert.equal(lo.mode, "round"); near(lo.R, CURVE.rMin, 1e-9); near(lo.k, 1 / CURVE.rMin, 1e-12);
  assert.equal(big.mode, "round"); near(big.R, CURVE.rMax, 1e-6);
  assert.deepEqual([flat.mode, flat.R, flat.k], ["flat", Infinity, 0]);
  assert.deepEqual([ortho.mode, ortho.R, ortho.k], ["ortho", Infinity, 0]);
  assert.deepEqual(C.uToCurve(1), ortho); assert.deepEqual(C.uToCurve(C.NOTCH.flat), flat); assert.deepEqual(C.uToCurve(C.NOTCH.ortho), ortho);
  // the whole range: the radius only grows, the curvature only falls, and nothing between the large notch and the flat one is a jump
  let prev = C.uToCurve(0);
  for (let i = 1; i <= 1000; i++) {
    const c = C.uToCurve(i / 1000);
    assert.ok(c.k <= prev.k + 1e-15, `k falls at ${i}`); assert.ok(c.R >= prev.R - 1e-9);
    if (c.mode === "round" && prev.mode === "round") assert.ok(prev.k - c.k < 0.01, "no jump in k");
    prev = c;
  }
  // clamps
  assert.deepEqual(C.uToCurve(-5), C.uToCurve(0)); assert.deepEqual(C.uToCurve(9), ortho); assert.deepEqual(C.uToCurve(NaN), C.uToCurve(0));
  // R <-> notch round trip, and the default notch is the radius the terrain games always had
  for (const R of [6, 12, 30, 36, 100, 599, 600, 700, 2000, 50000]) near(C.uToCurve(C.radiusToU(R)).R, R, R * 1e-9, `R ${R}`);
  assert.equal(C.radiusToU(Infinity), CURVE.uFlat);
  const spec = fs.readFileSync(path.join(web, "app.js"), "utf8").match(/const CURVE_SPEC = \{[^}]*value: ([0-9.]+) \}/);
  assert.ok(spec, "CURVE_SPEC is declared in app.js");
  near(Number(spec[1]), C.radiusToU(CURVE.defaultRadius), 1e-3, "CURVE_SPEC.value is the notch of R 36");
  assert.equal(C.curveLabel(C.NOTCH.flat), "flat"); assert.equal(C.curveLabel(1), "orthographic flat"); assert.equal(C.curveLabel(0), "R 6.0");
  assert.deepEqual(C.curveTarget(C.NOTCH.ortho), { k: 0, s: 1 }); assert.deepEqual(C.curveTarget(C.NOTCH.flat), { k: 0, s: 0 });
  assert.equal(C.curveTarget(0.3).s, 0);
}

// ---------------------------------------------------------------- the ease: critically damped, exact at the end
{
  const run = (st, target, seconds = 4, dt = 1 / 60) => {
    const trace = [st];
    for (let t = 0; t < seconds; t += dt) { st = C.stepCurve(st, target, dt); trace.push(st); if (C.curveSettled(st, target)) break; }
    return { st, trace };
  };
  // round -> FLAT: k reaches exactly 0, never negative, never rises
  let a = run(C.initialCurve(C.curveTarget(C.radiusToU(36))), C.curveTarget(C.NOTCH.flat));
  assert.equal(a.st.k, 0); assert.ok(a.trace.every((s) => s.k >= 0)); assert.ok(a.trace.every((s, i) => i === 0 || s.k <= a.trace[i - 1].k + 1e-12));
  assert.ok(a.trace.length < 4 * 60, "settles within a few seconds");
  // FLAT -> small R and back: no overshoot past the target
  a = run(C.initialCurve(C.curveTarget(C.NOTCH.flat)), C.curveTarget(0));
  assert.equal(a.st.k, 1 / CURVE.rMin); assert.ok(a.trace.every((s) => s.k <= 1 / CURVE.rMin + 1e-12), "no overshoot");
  // FLAT -> ORTHO: s climbs 0 -> exactly 1, monotone, within [0, 1]
  a = run(C.initialCurve(C.curveTarget(C.NOTCH.flat)), C.curveTarget(C.NOTCH.ortho));
  assert.equal(a.st.s, 1); assert.ok(a.trace.every((s, i) => s.s >= 0 && s.s <= 1 && (i === 0 || s.s >= a.trace[i - 1].s - 1e-12)));
  // ORTHO -> FLAT returns to exactly the perspective camera
  a = run(C.initialCurve(C.curveTarget(C.NOTCH.ortho)), C.curveTarget(C.NOTCH.flat));
  assert.equal(a.st.s, 0);
  // the result is the same at any frame length (the spring is closed form)
  const t60 = run(C.initialCurve({ k: 0.1, s: 0 }), { k: 0.02, s: 0 }, 0.5, 1 / 60).st.k;
  let st = C.initialCurve({ k: 0.1, s: 0 }); for (let i = 0; i < 30; i++) st = C.stepCurve(st, { k: 0.02, s: 0 }, 1 / 60);
  let st2 = C.initialCurve({ k: 0.1, s: 0 }); for (let i = 0; i < 6; i++) st2 = C.stepCurve(st2, { k: 0.02, s: 0 }, 5 / 60);
  near(st.k, st2.k, 1e-9, "frame-length independent");
  void t60;
}

// ---------------------------------------------------------------- the dolly zoom
const ASPECT = 16 / 9, D0 = 14.5;
const ABOVE = (h = 0.6) => ({ eye: [3 + 9 * Math.sin(0.4) * Math.cos(1.0), h + 9 * Math.sin(1.0), 4 + 9 * Math.cos(0.4) * Math.cos(1.0)], target: [3, 0.6, 4], up: [0, 1, 0] });
const camAt = (d0, s, base = ABOVE()) => {
  const params = C.dollyParams(d0, CURVE.fovy, s);
  const dir = base.eye.map((v, i) => v - base.target[i]); const L = Math.hypot(...dir);
  const eye = base.target.map((t, i) => t + (dir[i] / L) * params.dist);
  const eye0 = base.target.map((t, i) => t + (dir[i] / L) * d0);       // the original camera's eye: the view the page keeps
  return { eye, eye0, target: base.target, up: base.up, aspect: ASPECT, params };
};
{
  // blend 0 IS the perspective camera that was there, exactly
  const p0 = C.dollyParams(D0, CURVE.fovy, 0);
  assert.deepEqual([p0.dist, p0.fovy, p0.near, p0.far, p0.ortho], [D0, 0.62, 0.1, 300, false]);
  // the half-height of the plane through the target is the same at every blend (the dolly keeps the framing)
  const H = D0 * Math.tan(0.31);
  for (const s of [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]) {
    const p = C.dollyParams(D0, CURVE.fovy, s);
    near(p.halfHeight, H, 1e-12);
    if (!p.ortho) near(p.dist * Math.tan(p.fovy / 2), H, 1e-9, `framing at s=${s}`);
    assert.ok(p.near > 0 && p.far > p.near);
    near(p.far - p.near, 299.9, 1e-9, "the depth range keeps its width");
  }
  assert.equal(C.dollyParams(D0, CURVE.fovy, 1).ortho, true); assert.equal(C.dollyParams(D0, CURVE.fovy, 0.999).ortho, false);
  // the distance only grows, the field only narrows
  let prev = C.dollyParams(D0, CURVE.fovy, 0);
  for (let i = 1; i <= 100; i++) { const p = C.dollyParams(D0, CURVE.fovy, i / 100); assert.ok(p.dist > prev.dist); if (!p.ortho) assert.ok(p.fovy < prev.fovy); prev = p; }

  // the matrix agrees with the analytic camera at every blend, and its depth is inside [-1, 1] for what the scene holds
  const pts = [[3, 0, 4], [-2, 0, 9], [8, 0, -3], [3, 0.6, 4], [5, 3, 6], [1, 1.5, 2], [10, 0, 14]];
  const toClip = (cam, M, p) => {
    const ax = C.lookAxes(cam.eye0, cam.target, cam.up), r = p.map((v, i) => v - cam.eye0[i]);       // view-space coordinates of the ORIGINAL camera
    const v = [ax.x, ax.y, ax.z].map((a) => a[0] * r[0] + a[1] * r[1] + a[2] * r[2]);
    const h = [0, 1, 2, 3].map((row) => M[row] * v[0] + M[4 + row] * v[1] + M[8 + row] * v[2] + M[12 + row]);
    return h.map((x) => x / h[3]);
  };
  for (const s of [0, 0.3, 0.7, 0.95, 1]) {
    const cam = camAt(D0, s), M = C.projectionMatrix(cam.params, ASPECT);
    for (const p of pts) {
      const n = toClip(cam, M, p), a = C.projectPoint(cam, p);
      near(n[0], a[0], 1e-9, `x s=${s}`); near(n[1], a[1], 1e-9, `y s=${s}`);
      assert.ok(n[2] > -1 && n[2] < 1, `depth in range s=${s}: ${n[2]}`);
    }
  }
  // THE LIMIT: one step short of orthographic, the perspective projection differs from the orthographic one by less than a pixel
  // on a 2000 x 1100 screen, anywhere the scene is (heights 0..6): no pop at the end of the dolly
  const ortho = camAt(D0, 1), last = camAt(D0, 1 - 1e-9);
  assert.equal(last.params.ortho, false);
  const px = (a, b) => Math.hypot((a[0] - b[0]) * 1000, (a[1] - b[1]) * 550);
  let worst = 0;
  for (let ix = -10; ix <= 10; ix++) for (let iz = -10; iz <= 10; iz++) for (const y of [0, 2, 6]) {
    const p = [3 + ix, y, 4 + iz];
    worst = Math.max(worst, px(C.projectPoint(last, p), C.projectPoint(ortho, p)));
  }
  assert.ok(worst < 0.5, `the end of the dolly is within a pixel of the orthographic camera (worst ${worst.toFixed(3)} px)`);
  // ... and a perspective camera a quarter of the way is not (the dolly is a real transition)
  const quarter = camAt(D0, 0.25); let seen = 0;
  for (const p of pts) seen = Math.max(seen, px(C.projectPoint(quarter, p), C.projectPoint(ortho, p)));
  assert.ok(seen > 5);
  // the orthographic matrix is the flat view's: x scale 1 / (hh aspect), y scale 1 / hh, no perspective divide
  const Mo = C.projectionMatrix(ortho.params, ASPECT);
  near(Mo[5], 1 / H, 1e-12); near(Mo[0], 1 / (H * ASPECT), 1e-12); assert.equal(Mo[11], 0); assert.equal(Mo[15], 1);
}

// ---------------------------------------------------------------- particle sizing: radius * focal * (H / 2) / clip.w
{
  const r = 0.05, Hpx = 1000;      // a grain of world radius 0.05 on a screen 1000 px high
  for (const s of [0, 0.5, 1]) {
    const cam = camAt(D0, s), M = C.projectionMatrix(cam.params, ASPECT);
    const ax = C.lookAxes(cam.eye, cam.target, cam.up);
    // a grain at the target, its edge one radius to the camera's right
    const edge = cam.target.map((t, i) => t + ax.x[i] * r);
    const measured = (C.projectPoint(cam, edge)[0] - C.projectPoint(cam, cam.target)[0]) * ASPECT * Hpx / 2;     // ndc x spans aspect * H / 2 px per unit
    const ze = -D0;                                                                                                    // the target's eye-space z in the original view
    const depth = C.sizeDepth(M, ze);
    const formula = r * M[5] * Hpx / 2 / depth;
    near(formula, measured, 1e-6 * Math.max(1, formula), `grain radius px, s=${s}`);
    if (s === 0) near(depth, D0, 1e-9, "a perspective grain divides by its eye depth");
    else near(depth, 1, 1e-9, "after the dolly the target plane has clip.w 1");
  }
  // the old divisor (|eye depth|) would be wrong under ortho by the distance
  const ortho = camAt(D0, 1), Mo = C.projectionMatrix(ortho.params, ASPECT);
  assert.equal(C.sizeDepth(Mo, -D0), 1); assert.equal(C.sizeDepth(Mo, -D0 - 7), 1, "an orthographic grain is the same size at any depth");
  assert.ok(C.sizeDepth(C.projectionMatrix(camAt(D0, 0).params, ASPECT), -D0 - 7) > D0, "a perspective grain shrinks with depth");
  // the matrix stays well conditioned however far the dolly goes (float32-safe products)
  for (const e of C.projectionMatrix(camAt(D0, 0.999).params, ASPECT)) assert.ok(Math.abs(e) < 10, `entry ${e}`);
}

// ---------------------------------------------------------------- picking: the ray through a pixel, in every projection
{
  const bases = [ABOVE(), { eye: [3, 12, 4.001], target: [3, 0.6, 4], up: [0, 0, -1] }, { eye: [-6, 5, 16], target: [4, 0, 5], up: [0, 1, 0] }];
  for (const base of bases) for (const s of [0, 0.2, 0.6, 0.97, 1]) {
    const cam = camAt(D0, s, base);
    for (const h of [0, 0.6, 2.5]) for (const q of [[2.5, h, 3.5], [3, h, 4], [7, h, 1], [-1, h, 9]]) {
      const ndc = C.projectPoint(cam, q);
      if (Math.abs(ndc[0]) > 1.5 || Math.abs(ndc[1]) > 1.5) continue;
      const hit = C.pickPlane(cam, ndc, h);
      assert.ok(hit, `a hit at s=${s}`);
      near(hit[0], q[0], 1e-6, `pick x s=${s} h=${h}`); near(hit[1], q[2], 1e-6, `pick z s=${s} h=${h}`);
    }
  }
  // under ortho the rays are parallel (one direction), their origins on the camera's plane; under perspective they all leave the eye
  const o = camAt(D0, 1), p = camAt(D0, 0);
  const r1 = C.pickRay(o, [-0.8, 0.5]), r2 = C.pickRay(o, [0.7, -0.3]);
  for (let i = 0; i < 3; i++) near(r1.dir[i], r2.dir[i], 1e-12);
  assert.ok(Math.hypot(r1.origin[0] - r2.origin[0], r1.origin[1] - r2.origin[1], r1.origin[2] - r2.origin[2]) > 1);
  const q1 = C.pickRay(p, [-0.8, 0.5]), q2 = C.pickRay(p, [0.7, -0.3]);
  assert.deepEqual(q1.origin, q2.origin); assert.ok(Math.abs(q1.dir[0] - q2.dir[0]) > 0.1);
  // an orthographic pointer at the screen centre hits the target; a point ABOVE the plane is picked at its ground shadow along the view axis
  const c = C.pickPlane(o, [0, 0], 0.6); near(c[0], 3, 1e-9); near(c[1], 4, 1e-9);
  const ax = C.lookAxes(o.eye, o.target, o.up), up3 = [5, 3, 6], n3 = C.projectPoint(o, up3);
  const ground = C.pickPlane(o, n3, 0), t = -up3[1] / -ax.z[1];
  near(ground[0], up3[0] + t * -ax.z[0], 1e-9); near(ground[1], up3[2] + t * -ax.z[2], 1e-9);
  // the flat view's own ortho (camera() flat branch) is the same ray model
  const flat = { eye: [3, 60, 4], target: [3, 0, 4], up: [0, 0, -1], aspect: ASPECT, params: { ortho: true, halfHeight: 7 } };
  const f = C.pickPlane(flat, [0.5, 0.5], 0.0); near(f[0], 3 + 0.5 * 7 * ASPECT, 1e-9); near(f[1], 4 - 0.5 * 7, 1e-9);
  // a plane behind the ray, or parallel to it, is no hit
  assert.equal(C.pickPlane(o, [0, 0], 1e6), null);
  assert.equal(C.pickPlane({ ...flat, eye: [3, 60, 4], target: [60, 60, 4], up: [0, 1, 0] }, [0, 0], 5), null);
}

// ---------------------------------------------------------------- the control in a share link
{
  const src = fs.readFileSync(path.join(web, "app.js"), "utf8");
  function cut(start) {
    let depth = 0, i = start, quote = null, seen = false;
    for (; i < src.length; i++) {
      const ch = src[i];
      if (quote) { if (ch === "\\") i++; else if (ch === quote) quote = null; continue; }
      if (ch === '"' || ch === "'" || ch === "`") { quote = ch; continue; }
      if (ch === "/" && src[i + 1] === "/") { i = src.indexOf("\n", i); continue; }
      if ("([{".includes(ch)) { depth++; seen = true; } else if (")]}".includes(ch)) depth--;
      if (seen && depth === 0 && (ch === ";" || ch === "}")) { if (ch === "}" && src.startsWith("function", start)) return src.slice(start, i + 1); if (ch === ";") return src.slice(start, i + 1); }
    }
    throw new Error("unbalanced");
  }
  const decls = [...src.matchAll(/^const ([A-Z_]+SPECS?) = /gm)].map((m) => cut(m.index));
  const fn = cut(src.indexOf("function pageSpecs(specs)"));
  const ctx = vm.createContext({ ...P, rl, state: { game: null } });
  vm.runInContext(decls.map((d) => d.replace(/^const /, "var ")).join("\n") + "\n" + fn + "\nthis.pageSpecs = pageSpecs;", ctx);
  const terrain = { game: "scroller", scene: { terrain: true } }, plain = { game: "boardwalk", scene: { floor: true } };
  // a terrain game with NO server controls (the bullet hell) still has the control; a game without a curve has none
  ctx.state.game = terrain;
  const bare = ctx.pageSpecs(null);
  assert.equal(JSON.stringify(bare.map((s) => s.id)), '["worldcurve"]'); assert.ok(S.sharable(bare[0]) && bare[0].local);
  ctx.state.game = plain; assert.equal(ctx.pageSpecs(null), null);
  ctx.state.game = terrain;
  const withServer = ctx.pageSpecs([{ id: "autopilot", kind: "toggle", label: "auto", action: "autopilot", field: "on", value: false }]);
  assert.ok(withServer.some((s) => s.id === "worldcurve") && withServer.some((s) => s.id === "autopilot"));
  assert.equal(withServer.filter((s) => s.id === "worldcurve").length, 1);
  // serializeSpecs carries the notch in `l`, parseView clamps it through the spec, the link survives encode / decode
  for (const [notch, label] of [[C.radiusToU(36), "R 36.0"], [C.NOTCH.flat, "flat"], [C.NOTCH.ortho, "orthographic flat"], [0, "R 6.0"]]) {
    const out = S.serializeSpecs(bare, () => notch);
    assert.deepEqual(out.l, { worldcurve: notch }); assert.deepEqual(out.c, {});
    const view = { v: 1, g: "scroller", ...out, ls: { "meowdoku.worldcurve.scroller": String(notch) } };
    const back = await S.decodeView(await S.encodeView(view));
    const parsed = S.parseView(back, bare);
    assert.equal(parsed.l.worldcurve, notch); assert.equal(C.curveLabel(parsed.l.worldcurve), label);
    assert.equal(parsed.ls["meowdoku.worldcurve.scroller"], String(notch));
  }
  assert.equal(S.parseView({ g: "scroller", l: { worldcurve: 1e30 } }, bare).l.worldcurve, 1);
  assert.equal(S.parseView({ g: "scroller", l: { worldcurve: -3 } }, bare).l.worldcurve, 0);
  assert.equal(S.parseView({ g: "scroller", l: { worldcurve: "flat" } }, bare).l.worldcurve, undefined);
  assert.ok(S.isSharedKey("meowdoku.worldcurve.scroller") && S.isSharedKey("meowdoku.worldcurve.worldwalk"));
}
console.log("world_curve.js ok");
