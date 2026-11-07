// Node check of web/share.js over the page's ACTUAL specs: app.js's own `const *_SPEC(S)` declarations and its own
// `function pageSpecs` are cut out of the source and evaluated (app.js itself needs a DOM), then fed the server's controls_ui
// (argv[2] = a JSON file the python wrapper wrote from the fountain games' controls_ui).
import fs from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../examples/meowdoku/web");
const S = await import(pathToFileURL(path.join(web, "share.js")).href);
const P = await import(pathToFileURL(path.join(web, "particle_styles.js")).href);
const RL = await import(pathToFileURL(path.join(web, "replay_layers.js")).href);
const src = fs.readFileSync(path.join(web, "app.js"), "utf8");

// cut a balanced declaration / function starting at `start` (string and comment aware)
function cut(start) {
  let depth = 0, i = start, quote = null, seen = false;
  for (; i < src.length; i++) {
    const ch = src[i];
    if (quote) { if (ch === "\\") i++; else if (ch === quote) quote = null; continue; }
    if (ch === '"' || ch === "'" || ch === "`") { quote = ch; continue; }
    if (ch === "/" && src[i + 1] === "/") { i = src.indexOf("\n", i); continue; }
    if ("([{".includes(ch)) { depth++; seen = true; } else if (")]}".includes(ch)) depth--;
    if (seen && depth === 0 && (ch === ";" || ch === "}")) { if (ch === "}" && src.startsWith("function", start)) return src.slice(start, i + 1); if (ch === ";") return src.slice(start, i + 1); }
    if (seen && depth === 0 && ch === ";") return src.slice(start, i + 1);
  }
  throw new Error("unbalanced");
}
const decls = [...src.matchAll(/^const ([A-Z_]+SPECS?) = /gm)].map((m) => cut(m.index));
const fn = cut(src.indexOf("function pageSpecs(specs)"));
const ctx = vm.createContext({ ...P, rl: RL, state: { game: { game: "fountain3d" } } });          // (particle_styles.js exports: COLOR_SOURCES, PALETTE_IDS, ...)
vm.runInContext(decls.map((d) => d.replace(/^const /, "var ")).join("\n") + "\n" + fn + "\nthis.pageSpecs = pageSpecs;", ctx);
const pageSpecs = ctx.pageSpecs;

const served = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));      // { game: controls_ui }
let total = 0;
for (const [game, controls] of Object.entries(served)) {
  ctx.state.game = { game };
  const specs = pageSpecs(controls);
  const locals = specs.filter((s) => s.local && S.sharable(s)), servers = specs.filter((s) => !s.local && S.sharable(s));
  assert.ok(locals.length >= 10 && servers.length >= 10, `${game}: ${locals.length} local, ${servers.length} server specs`);

  // a "page" whose local values are non-default everywhere
  const store = {};
  const pick = (s) => s.kind === "toggle" ? true : s.kind === "color" ? "#12ab34"
    : s.kind === "select" ? (Array.isArray(s.options[0]) ? s.options[s.options.length - 1][0] : (s.options[0] ?? "x"))
    : (Number.isFinite(s.max) ? s.max : 1) * 0.75 + (Number.isFinite(s.min) ? s.min : 0) * 0.25;
  for (const s of locals) store[s.id] = pick(s);
  const withServer = specs.map((s) => (!s.local && S.sharable(s) ? { ...s, value: pick(s) } : s));

  // serialize -> encode -> decode -> parse round-trips, and every shared spec is in it (no hand-kept list)
  const out = S.serializeSpecs(withServer, (s) => store[s.id]);
  assert.equal(Object.keys(out.l).length, locals.length);
  assert.ok(out.l.wheelblur === true, `${game}: the wheel blur toggle is in a share link`);
  assert.equal(Object.keys(out.c).length, servers.length);
  const view = { v: 1, g: game, ...out, ls: { "meowdoku.pstyle": "heat", "meowdoku.noisefx": '{"on":true,"strength":0.5}' }, cam: { yaw: 1.2, pitch: -0.4, lz: 0.5, px: 0, pz: 2, auto: true }, rt: { gravity: [0, -1, 0] } };
  const enc = await S.encodeView(view);
  const back = await S.decodeView(enc);
  assert.deepEqual(back, view);
  const parsed = S.parseView(back, withServer);
  assert.deepEqual(parsed.c, out.c);
  assert.deepEqual(parsed.l, out.l);
  assert.equal(parsed.ls["meowdoku.pstyle"], "heat");
  assert.deepEqual(parsed.cam, { yaw: 1.2, pitch: -0.4, lz: 0.5, px: 0, pz: 2, auto: true });
  assert.deepEqual(parsed.rt, { gravity: [0, -1, 0] });
  console.log(`${game}: ${locals.length} local + ${servers.length} server specs round-trip; link ${enc.length} chars`);
  total += enc.length;

  // clamping through each spec's own limits; unknown ids / wrong-side ids / junk ignored
  const ranged = specs.find((s) => s.kind === "range" && Number.isFinite(s.min) && Number.isFinite(s.max));
  const sel = specs.find((s) => s.kind === "select" && s.options.length && Array.isArray(s.options[0]));
  const evil = { g: game, c: {}, l: {}, ls: {}, nope: 1 };
  (ranged.local ? evil.l : evil.c)[ranged.id] = 1e30;
  (sel.local ? evil.l : evil.c)[sel.id] = "not-an-option";
  evil.c["no_such_control"] = 5; evil.l["no_such_control"] = 5;
  evil.c["__proto__"] = { polluted: true }; evil.l["constructor"] = 1;
  evil.ls = { "meowdoku.lichess_token": "secret", "meowdoku_player": "bob", "meowdoku_geo": "{}", "other.key": "x", "meowdoku.noisefx": '{"__proto__":{"x":1},"strength":9e99,"mode":"mean"}', "meowdoku.n": "5", "meowdoku.m": 5 };
  const got = S.parseView(evil, withServer);
  assert.equal((ranged.local ? got.l : got.c)[ranged.id], ranged.max);
  assert.equal((sel.local ? got.l : got.c)[sel.id], undefined);
  assert.equal(got.c.no_such_control, undefined); assert.equal(got.l.no_such_control, undefined);
  assert.equal(Object.getPrototypeOf(got.c), Object.prototype); assert.equal(({}).polluted, undefined);
  assert.deepEqual(Object.keys(got.ls).sort(), ["meowdoku.n", "meowdoku.noisefx"]);
  assert.ok(!("__proto__" in JSON.parse(got.ls["meowdoku.noisefx"]) && Object.keys(JSON.parse(got.ls["meowdoku.noisefx"])).includes("__proto__")));
  assert.equal(JSON.parse(got.ls["meowdoku.noisefx"]).strength, 1e6);
  // a local value offered as a server control (and the reverse) is dropped
  const flip = { c: { [locals[0].id]: out.l[locals[0].id] }, l: { [servers[0].id]: out.c[servers[0].id] } };
  assert.deepEqual(S.parseView(flip, withServer).c, {}); assert.deepEqual(S.parseView(flip, withServer).l, {});
}

// credential-ish spec ids and storage keys are never serialized
assert.ok(!S.sharable({ id: "lichess_token", kind: "toggle" }));
assert.ok(!S.isSharedKey("meowdoku.lichess_token") && !S.isSharedKey("meowdoku_player") && !S.isSharedKey("meowdoku_geo") && S.isSharedKey("meowdoku.gscale"));
assert.deepEqual(S.serializeStorage(() => [["meowdoku.gscale", "0.4"], ["meowdoku_player", "x"], ["other", "y"], ["meowdoku.auth", "z"]]), { "meowdoku.gscale": "0.4" });
// malformed links throw rather than evaluate
for (const bad of ["", "x", "qAAAA", "z!!!", "j" + Buffer.from("{not json").toString("base64url")]) await assert.rejects(() => S.decodeView(bad));
// the plain (no CompressionStream) form decodes too
assert.deepEqual(await S.decodeView("j" + Buffer.from('{"g":"fountain"}').toString("base64url")), { g: "fountain" });
console.log("share.js ok; mean link", Math.round(total / Object.keys(served).length), "chars");
