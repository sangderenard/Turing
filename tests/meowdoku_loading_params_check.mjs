// node side of tests/test_meowdoku_loading_library.py: the recorder's parameter handling (web/loading_params.js) and the stream at each bit depth.
//   node tests/meowdoku_loading_params_check.mjs params          -> JSON of normalize / fromQuery / toQuery / attempts results
//   node tests/meowdoku_loading_params_check.mjs bits <out_dir>  -> <out_dir>/b<bits>.bin (raw MLA1 streams) and info.json
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const web = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")), "../examples/meowdoku/web");
const P = await import(pathToFileURL(path.join(web, "loading_params.js")).href);
const [mode, dir] = process.argv.slice(2);

if (mode === "params") {
  const d = P.normalize({});
  const wild = P.normalize({ chamber: "nope", size: "20M", mix: 7, grainScale: -3, grains: 1e9, fps: 0, bits: 13, settle: "x", turns: 5, flipSeconds: 99, speed: 3, cap: 1, colour: 9, fit: 0, junk: 1 });
  const q = P.toQuery({ ...d, chamber: "sdf_circle", bits: 10, mix: 0.25, turns: 6, fit: false });
  const back = P.fromQuery(q);
  console.log(JSON.stringify({
    defaults: d, wild, sizes: P.SIZES, sizeIds: P.SIZE_IDS, bare: P.fromQuery("?record_loading=1"), query: q, back,
    bad: P.fromQuery("?record_loading=1&rp=%7Bnot-json"), off: P.fromQuery("?x=1"),
    legacy: P.attempts(P.DEFAULTS), fixed: P.attempts({ ...P.DEFAULTS, fit: false, fps: 8, grains: 300, bits: 16 }),
    tight: P.attempts({ ...P.DEFAULTS, fit: true, fps: 8, grains: 300, bits: 11 }),
    meta: P.metadata({ ...d, mix: null }, { capped: true }),
  }));
} else if (mode === "visual") {
  const { VisualTape, filterTape, encodeStream } = await import(pathToFileURL(path.join(web, "loading_anim.js")).href);
  const tape = new VisualTape({ sourceSnap: 6, stride: 3, sourceCount: 256 });
  for (let k = 1; k <= 30; k += 3) {
    const frame = new Float32Array(256 * 5);
    for (let i = 0; i < 256; i++) { frame[5 * i] = i / 256; frame[5 * i + 1] = k / 100; frame[5 * i + 4] = 120; }
    tape.push(k, frame);
  }
  tape.inputs.push({ step: 6, values: { grav_x: 0, grav_y: -1 } });
  const anim = filterTape(tape, { from: tape.lo, to: tape.hi, fps: 20, bits: 12, dt: 1 / 360 });
  console.log(JSON.stringify({ frames: tape.hi, sampled: tape.get(1).length / 5, step: tape.stepOf(2), grains: anim.grains, raw: encodeStream(anim).length }));
} else if (mode === "bits") {
  const { Recording } = await import(pathToFileURL(path.join(web, "recording.js")).href);
  const { filterTape, encodeStream } = await import(pathToFileURL(path.join(web, "loading_anim.js")).href);
  const grains = 40, frames = 90, tape = new Recording({ snapEvery: 6, frameBytes: grains * 20, capBytes: grains * 20 * 400 });
  for (let k = 1; k <= frames; k++) {
    const d = new Float32Array(5 * grains);
    for (let i = 0; i < grains; i++) { d[5 * i] = 0.3 + 0.4 * ((i * 37) % 100) / 100 + 0.01 * Math.sin(k / 7 + i); d[5 * i + 1] = 0.2 + 0.6 * ((i * 53) % 100) / 100 - 0.002 * k; d[5 * i + 4] = 120; }
    tape.push(k, d);
  }
  tape.setInputs(0, { grav_x: 0, grav_y: -1 });
  fs.mkdirSync(dir, { recursive: true });
  const info = {};
  for (const bits of P.BITS) {
    const a = filterTape(tape, { from: 1, to: frames, maxGrains: grains, fps: 30, bits, dt: 1 / 360, outline: [[[0, 0], [1, 0], [1, 1.2]]] });
    const raw = encodeStream(a);
    fs.writeFileSync(path.join(dir, `b${bits}.bin`), raw);
    const lowMask = (1 << (16 - bits)) - 1;
    info[bits] = { bytes: raw.length, frames: a.x.length, grains: a.grains, lowBitsZero: a.x.every((r) => r.every((v) => (v & lowMask) === 0)) && a.y.every((r) => r.every((v) => (v & lowMask) === 0)),
                   maxX: Math.max(...a.x.map((r) => Math.max(...r))), x0: Array.from(a.x[0]).slice(0, 6) };
  }
  fs.writeFileSync(path.join(dir, "info.json"), JSON.stringify(info));
} else {
  throw new Error("unknown mode");
}
