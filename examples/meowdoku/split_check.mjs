// node examples/meowdoku/split_check.mjs <split dir> <shell js> <expected.json>
// Runs the split program (JS host + wasm regions) and the all-JS host on the
// cases in expected.json, compares each against the Python result recorded
// there, and times both.
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const [dir, shellPath, expectedPath] = process.argv.slice(2);
globalThis.TURING_SHELL = await import(pathToFileURL(path.resolve(shellPath)).href);
const { TuringWasmProgram } = globalThis.TURING_SHELL;
const split = JSON.parse(await readFile(path.join(dir, 'split.json'), 'utf8'));
const expected = JSON.parse(await readFile(expectedPath, 'utf8'));
const host = await import(pathToFileURL(path.resolve(dir, 'host.mjs')).href);
const pure = await import(pathToFileURL(path.resolve(dir, 'host.pure.mjs')).href);

const programs = new Map();
for (const name of split.wasm_regions) {
  const bytes = await readFile(path.join(dir, 'regions', `${name}.wasm`));
  const contract = JSON.parse(await readFile(path.join(dir, 'regions', `${name}.abi.json`), 'utf8'));
  programs.set(name, await TuringWasmProgram.fromBytes(bytes, contract));
}
const calls = {};
const runtime = { call(descriptor, args) {
  calls[descriptor.callee] = (calls[descriptor.callee] || 0) + 1;
  const feeds = {};
  descriptor.feedIds.forEach((id, k) => { feeds[`value_${id}`] = args[k]; });
  const out = programs.get(descriptor.callee).invoke(feeds);
  return descriptor.outputIds.map((id) => out[`value_${id}`]);
} };

const buffers = (c) => split.inputs.map((name) => c[name]);
let worst = { split: 0, pure: 0 };
for (const c of expected.cases) {
  const a = await host.default.run(buffers(c), [], runtime);
  const b = await pure.default.run(buffers(c), [], null);
  for (let k = 0; k < c.result.length; k++) {
    worst.split = Math.max(worst.split, Math.abs(Number(a[k]) - c.result[k]));
    worst.pure = Math.max(worst.pure, Math.abs(Number(b[k]) - c.result[k]));
  }
}
console.log('cases', expected.cases.length, 'max |diff| vs Python  split:', worst.split, ' all-JS:', worst.pure);
console.log('wasm calls', calls);

async function time(label, fn) {
  const c = expected.cases[0];
  for (let i = 0; i < 200; i++) await fn(buffers(c));
  const n = 4000, t0 = performance.now();
  for (let i = 0; i < n; i++) await fn(buffers(c));
  const us = (performance.now() - t0) * 1000 / n;
  console.log(`${label}: ${us.toFixed(2)} us per frame (${c.substeps} substeps)`);
}
await time('split JS+wasm', (b) => host.default.run(b, [], runtime));
await time('all-JS       ', (b) => pure.default.run(b, [], null));
