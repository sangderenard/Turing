// node --experimental-default-type=module tests/meowdoku_mathlab_page_check.mjs [PRODUCT_DIR]
// 1. mathlab.js parses and imports as an ES module (no document: the page's main() does not run).
// 2. timeWasm against a stand-in program with the TuringGraphProgram surface: row count, finite ns/row, parity against a reference block.
// 3. With PRODUCT_DIR (a math_lab compile directory holding law.wasm, law.abi.json, profile.server.json), the same function against the
//    real product through the real TuringGraphProgram.  Skipped without the argument.
import {readFileSync} from 'node:fs';
import {fileURLToPath, pathToFileURL} from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const {timeWasm, shortUserAgent, mulberry32} = await import(pathToFileURL(path.join(here, '..', 'examples', 'meowdoku', 'web', 'mathlab.js')).href);
const fail = (message) => { console.error('FAIL ' + message); process.exit(1); };

if (!(mulberry32(1)() >= 0)) fail('prng');
if (!shortUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130.0 Safari/537.36').startsWith('Chrome 130')) fail('user agent');

const batch = 256;
const buffers = {x: new Float64Array(batch), y: new Float64Array(batch), r: new Float64Array(batch)};
const fake = {
  abi: {inputs: {x: 1, y: 2}},
  inputNames: () => ['x', 'y'], outputNames: () => ['r'],
  view: () => ({length: batch}),
  setInput: (name, values) => buffers[name].set(values),
  run: () => { for (let i = 0; i < batch; i++) buffers.r[i] = Math.hypot(buffers.x[i], buffers.y[i]); },
  output: (name) => buffers[name].slice(),
};
const ref = {rows: 64, inputs: {x: Array(64).fill(3), y: Array(64).fill(4)}, outputs: {r: Array(64).fill(5)}};
const t = timeWasm(fake, ref, {runs: 10, warmup: 2});
if (t.rows !== batch || !Number.isFinite(t.ns_per_row) || t.ns_per_row < 0) fail('timing ' + JSON.stringify(t));
if (!(t.max_abs_err < 1e-12)) fail('parity ' + t.max_abs_err);
console.log('ok stand-in program', JSON.stringify(t));

const dir = process.argv[2];
if (!dir) { console.log('skipped: no PRODUCT_DIR'); process.exit(0); }
const {TuringGraphProgram} = await import(pathToFileURL(path.join(here, '..', 'examples', 'mobile_reference_app', 'web', 'turing-wasm-shell.js')).href);
const abi = JSON.parse(readFileSync(path.join(dir, 'law.abi.json'), 'utf8'));
const program = await TuringGraphProgram.fromBytes(readFileSync(path.join(dir, 'law.wasm')), abi);
const profile = JSON.parse(readFileSync(path.join(dir, 'profile.server.json'), 'utf8'));
const real = timeWasm(program, profile.reference, {runs: 20});
if (!(real.max_abs_err < 1e-9)) fail('real parity ' + real.max_abs_err);
console.log('ok real wasm product', JSON.stringify(real));
