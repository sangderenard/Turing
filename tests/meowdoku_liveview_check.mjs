// [LIVEVIEW] the fountain position keyframe (MFK1) round trip, with the stream the octet codec packs: node tests/meowdoku_liveview_check.mjs
import { encodeKey, decodeKey, KEY_HEADER } from "../examples/meowdoku/web/share.js";

const total = 4096, stride = 1, cols = { x: new Float32Array(total), y: new Float32Array(total), vx: new Float32Array(total), vy: new Float32Array(total) };
for (let i = 0; i < total; i++) { cols.x[i] = -3.3 + (i % 64) * 0.103; cols.y[i] = 0.005 + Math.floor(i / 64) * 0.0045; cols.vx[i] = Math.sin(i) * 0.4; cols.vy[i] = Math.cos(i * 3) * 0.9; }
const raw = encodeKey({ steps: 123456, total, stride, cols });
if (raw.length !== KEY_HEADER + 8 * total) throw new Error("size " + raw.length);
const k = decodeKey(raw);
if (k.steps !== 123456 || k.total !== total || k.stride !== stride || k.count !== total) throw new Error("header");
const err = { x: 0, y: 0, vx: 0, vy: 0 };
for (const n of Object.keys(err)) for (let i = 0; i < total; i++) err[n] = Math.max(err[n], Math.abs(k[n][i] - cols[n][i]));
// 12 bits over the column's range: half a step of 6.6 / 4095 = 0.0008; 10 bits of 2 * 0.9: 0.0018
if (err.x > 0.0009 || err.y > 0.0001 || err.vx > 0.0006 || err.vy > 0.0011) throw new Error("quantization error " + JSON.stringify(err));
// a stride over more ids than the cap
const s = encodeKey({ steps: 1, total: 10000, stride: 3, cols: { x: new Float32Array(10000), y: new Float32Array(10000), vx: new Float32Array(10000), vy: new Float32Array(10000) } });
const t = decodeKey(s);
if (t.count !== 3334 || t.stride !== 3) throw new Error("stride");
let bad = 0; for (const mutate of [(b) => b.slice(0, 40), (b) => { const c = b.slice(); c[0] = 0; return c; }, (b) => b.slice(0, b.length - 1)]) { try { decodeKey(mutate(raw)); } catch { bad++; } }
if (bad !== 3) throw new Error("malformed keyframes must be refused");
console.log("ok", JSON.stringify(err), "raw bytes", raw.length);
