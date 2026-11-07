// node twin of tests/test_meowdoku_loading_anim.py: decode what Python packed, encode what Python decodes, the loop seam.
//   node tests/meowdoku_loading_anim_check.mjs decode <packet.mla> <out.json>
//   node tests/meowdoku_loading_anim_check.mjs encode <spec.json> <out.bin>
//   node tests/meowdoku_loading_anim_check.mjs seam <frames> <overlap> <count>
import fs from "node:fs";
import { pathToFileURL } from "node:url";
import path from "node:path";

const web = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")), "../examples/meowdoku/web");
const { decodePacket, encodeStream, seam } = await import(pathToFileURL(path.join(web, "loading_anim.js")).href);
const [mode, a, b, c] = process.argv.slice(2);

if (mode === "decode") {
  const anim = decodePacket(new Uint8Array(fs.readFileSync(a)));
  fs.writeFileSync(b, JSON.stringify({
    fps: anim.fps, seam: anim.seam, bbox: anim.bbox, radius: Array.from(anim.radius), angle: Array.from(anim.angle),
    outline: anim.outline.map((p) => Array.from(p)), x: anim.x.map((r) => Array.from(r)), y: anim.y.map((r) => Array.from(r)),
  }));
} else if (mode === "encode") {
  const s = JSON.parse(fs.readFileSync(a, "utf8"));
  const raw = encodeStream({
    fps: s.fps, seam: s.seam, bbox: s.bbox, radius: Uint16Array.from(s.radius), angle: Uint16Array.from(s.angle),
    outline: s.outline.map((p) => Uint16Array.from(p)), x: s.x.map((r) => Uint16Array.from(r)), y: s.y.map((r) => Uint16Array.from(r)),
  });
  fs.writeFileSync(b, raw);
} else if (mode === "seam") {
  const out = [];
  for (let p = 0; p < Number(c); p++) out.push(seam(p, Number(a), Number(b)));
  console.log(JSON.stringify(out));
} else if (mode === "filter") {
  // a fake state memory: 3 grains (one never alive), 120 frames of 6 steps; gravity flips by pi at step 240 and again at 480
  const { Recording } = await import(pathToFileURL(path.join(web, "recording.js")).href);
  const { filterTape } = await import(pathToFileURL(path.join(web, "loading_anim.js")).href);
  const tape = new Recording({ snapEvery: 6, frameBytes: 60, capBytes: 60 * 400 });
  for (let k = 1; k <= 120; k++) {
    const d = new Float32Array(15);
    for (let i = 0; i < 3; i++) { d[5 * i] = i + 0.01 * k; d[5 * i + 1] = 2 - 0.005 * k * (i + 1); d[5 * i + 4] = i === 2 ? 0 : 100 + 20; }
    tape.push(k, d);
  }
  tape.setInputs(0, { grav_x: 0, grav_y: -1 });
  tape.setInputs(240, { grav_x: 0, grav_y: 1 });
  tape.setInputs(480, { grav_x: 0, grav_y: -1 });
  const a = filterTape(tape, { from: 1, to: 120, maxGrains: 8, fps: 30, dt: 1 / 360, outline: [[[0, 0], [3, 0], [3, 3]]], turnSeconds: 0.2 });
  console.log(JSON.stringify({ grains: a.grains, frames: a.x.length, fps: a.fps, seam: a.seam, bbox: a.bbox, angle: Array.from(a.angle), radius: Array.from(a.radius) }));
} else {
  throw new Error("unknown mode");
}
