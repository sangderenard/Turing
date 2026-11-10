// Math lab page: submit math, poll the one server worker, show the products and time them where they run.
// The server rows (sympy, NumPy product, native) are measured by the server (profile.server.json).  The browser rows are
// measured HERE, on this device: law.wasm through TuringGraphProgram (turing-wasm-shell.js), 64 reference rows checked against
// the server's own sympy outputs.  WGSL is not timed: see WGSL_GAP.
const WGSL_GAP = 'not timed in the browser: the WebGPU runtime in compute.js (GpuProduct) is private to the fountain manifest ' +
  '(state commit, pack pass, display stride) and no standalone-law runner is exported; no second runner was written.';

// ---- the timing function (pure: takes an instantiated program, so a node test can call it without a page) ----------------
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const median = (values) => { const s = [...values].sort((x, y) => x - y); return s[s.length >> 1]; };

// program: TuringGraphProgram.  reference: profile.server.json's `reference` block (or null).  Returns
// {rows, ns_per_row, max_abs_err | null, runs}.  Inputs are seeded random Float64 columns in [0.5, 2) (the server's domain).
export function timeWasm(program, reference, {runs = 50, warmup = 5, seed = 1, now = () => performance.now()} = {}) {
  const names = program.inputNames();
  const batch = program.view(program.abi.inputs[names[0]]).length;
  const random = mulberry32(seed);
  const columns = {};
  for (const name of names) {
    columns[name] = Float64Array.from({length: batch}, () => 0.5 + 1.5 * random());
  }
  let maxAbs = null;
  if (reference && reference.rows > 0 && reference.inputs) {
    const take = Math.min(reference.rows, batch);
    const checked = {};
    for (const name of names) {
      const column = Float64Array.from(columns[name]);
      column.set(reference.inputs[name].slice(0, take));
      checked[name] = column;
      program.setInput(name, column);
    }
    program.run();
    maxAbs = 0;
    for (const out of program.outputNames()) {
      const got = program.output(out), want = reference.outputs[out];
      if (!want) continue;
      for (let i = 0; i < take; i++) maxAbs = Math.max(maxAbs, Math.abs(got[i] - want[i]));
    }
  }
  for (const name of names) program.setInput(name, columns[name]);
  for (let i = 0; i < warmup; i++) program.run();
  const times = [];
  for (let i = 0; i < runs; i++) {
    const t0 = now();
    program.run();
    times.push(now() - t0);
  }
  return {rows: batch, runs, ns_per_row: median(times) * 1e6 / batch, max_abs_err: maxAbs};
}

export function shortUserAgent(ua = (typeof navigator !== 'undefined' ? navigator.userAgent : '')) {
  const pick = (re, label) => { const m = ua.match(re); return m ? `${label} ${m[1]}` : null; };
  const browser = pick(/Edg\/(\d+)/, 'Edge') || pick(/Firefox\/(\d+)/, 'Firefox') || pick(/Chrome\/(\d+)/, 'Chrome') ||
    pick(/Version\/(\d+).*Safari/, 'Safari') || 'browser';
  const os = /Windows/.test(ua) ? 'Windows' : /Android/.test(ua) ? 'Android' : /iPhone|iPad/.test(ua) ? 'iOS' :
    /Mac OS X/.test(ua) ? 'macOS' : /Linux/.test(ua) ? 'Linux' : '';
  return [browser, os].filter(Boolean).join(' / ');
}

async function gpuLabel() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return 'no WebGPU';
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return 'WebGPU: no adapter';
    const info = adapter.info || {};
    return 'WebGPU: ' + ([info.vendor, info.architecture, info.device].filter(Boolean).join('/') || 'adapter (info withheld)');
  } catch (error) {
    return 'WebGPU: ' + error.message;
  }
}

// ---- the page ---------------------------------------------------------------------------------------------------------------
const TERMINAL = new Set(['complete', 'rejected', 'timeout', 'error']);
const ALL_TARGETS = ['numpy', 'wasm', 'wgsl', 'native'];
const NOTE = {numpy: 'compiled NumPy source (downloadable)', wasm: 'WebAssembly, runs in this browser', wgsl: 'WebGPU shader (float32)',
  native: 'batched native library on the server (never downloadable)'};

function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'class') node.className = value; else node.setAttribute(key, value);
  }
  for (const kid of kids.flat()) node.append(kid instanceof Node ? kid : String(kid));
  return node;
}
const fmt = (value, digits = 3) => (value === null || value === undefined || !Number.isFinite(value)) ? '-' :
  (Math.abs(value) >= 1000 ? value.toFixed(0) : Math.abs(value) >= 1 ? value.toFixed(digits - 1) : value.toPrecision(digits));
const sci = (value) => (value === null || value === undefined || !Number.isFinite(value)) ? '-' : value.toExponential(2);

async function api(path, options) {
  const response = await fetch(path, {credentials: 'same-origin', ...options});
  let body = null;
  try { body = await response.json(); } catch (error) { /* not json */ }
  return {ok: response.ok, status: response.status, body};
}

function renderResult(job, extra) {
  const $ = (id) => document.getElementById(id);
  $('result').hidden = false;
  const products = job.products || {};
  const law = $('law');
  law.replaceChildren(
    el('p', {}, 'inputs: ', el('code', {}, (products.inputs || []).join(', ') || '-'), '   outputs: ', el('code', {}, (products.outputs || []).join(', ') || '-')),
    el('pre', {}, (products.equations || []).join('\n')));
  const rows = $('targettable').tBodies[0];
  rows.replaceChildren();
  for (const [name, info] of Object.entries(products.targets || {})) {
    const state = info.state === 'complete' ? el('span', {class: 'ok'}, 'complete') : el('span', {class: 'bad'}, info.state);
    const detail = info.state === 'complete' ? NOTE[name] : `${info.stage ? info.stage + ': ' : ''}${info.error_type ? info.error_type + ': ' : ''}${info.error || ''}`;
    rows.append(el('tr', {}, el('td', {}, name), el('td', {}, state), el('td', {class: 'num'}, fmt(info.seconds)), el('td', {}, detail)));
  }
  const ran = $('ran').tBodies[0];
  ran.replaceChildren();
  const profile = job.profile;
  if (profile) {
    const labels = {sympy_lambdify: 'SymPy lambdify (reference)', numpy_product: 'NumPy product (law.numpy.py)', native_product: 'native product (batched LLVM)'};
    for (const [count, entry] of Object.entries(profile.rows)) {
      for (const [label, item] of Object.entries(entry)) {
        ran.append(el('tr', {}, el('td', {}, 'server'), el('td', {}, labels[label] || label), el('td', {class: 'num'}, count),
          el('td', {class: 'num'}, fmt(item.ns_per_row)), el('td', {class: 'num'}, sci(item.max_abs_err)),
          el('td', {}, label === 'sympy_lambdify' ? 'itself' : 'vs SymPy, same rows')));
      }
    }
    if (profile.native_kernel_only_ns_per_row) {
      ran.append(el('tr', {}, el('td', {}, 'server'), el('td', {}, 'native kernel only'), el('td', {class: 'num'}, profile.batch),
        el('td', {class: 'num'}, fmt(profile.native_kernel_only_ns_per_row)), el('td', {}, '-'), el('td', {}, 'buffers already bound')));
    }
  }
  const stored = (job.browser_profile && job.browser_profile.results) || [];
  const mine = extra.measurements.length ? extra.measurements : stored;
  for (const m of mine) {
    ran.append(el('tr', {}, el('td', {}, `this browser (${m.user_agent_hint || extra.machine})`), el('td', {}, `${m.target} product (law.${m.target === 'wasm' ? 'wasm' : 'wgsl'})`),
      el('td', {class: 'num'}, m.rows), el('td', {class: 'num'}, fmt(m.ns_per_row)), el('td', {class: 'num'}, sci(m.max_abs_err)),
      el('td', {}, m.max_abs_err === null || m.max_abs_err === undefined ? 'no reference' : '64 server reference rows')));
  }
  $('ranfoot').textContent = `This browser: ${extra.machine}; ${extra.gpu}. WebGPU row: ${WGSL_GAP}` +
    (extra.note ? ' ' + extra.note : '');
  const id = job.job_id;
  const files = ['law.numpy.py', 'law.at.py'].filter((name) => (name === 'law.numpy.py' ? products.targets && products.targets.numpy && products.targets.numpy.state === 'complete'
    : products.symbolic && products.symbolic.state === 'complete'));
  $('downloads').replaceChildren(...files.map((name) => el('a', {href: `/api/mathlab/jobs/${id}/file/${name}`, download: name}, name)).flatMap((a, i) => i ? [' | ', a] : [a]));
  if (!files.length) $('downloads').textContent = 'nothing to download';
}

async function measureWasm(job, extra) {
  const wasm = job.products && job.products.targets && job.products.targets.wasm;
  if (!wasm || wasm.state !== 'complete') return;
  try {
    const {TuringGraphProgram} = await import('./turing-wasm-shell.js');
    const base = `/api/mathlab/jobs/${job.job_id}/file/`;
    const program = await TuringGraphProgram.load(base + 'law.wasm', base + 'law.abi.json');
    const reference = job.profile ? job.profile.reference : null;
    const timing = timeWasm(program, reference);
    const entry = {target: 'wasm', rows: timing.rows, ns_per_row: timing.ns_per_row, max_abs_err: timing.max_abs_err, user_agent_hint: extra.machine};
    extra.measurements.push(entry);
    const posted = await api(`/api/mathlab/jobs/${job.job_id}/browser_profile`, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(entry)});
    if (!posted.ok) extra.note = `The server did not store the browser timing (${posted.status}).`;
  } catch (error) {
    extra.note = `wasm could not be timed in this browser: ${error.message}`;
  }
}

async function main() {
  const $ = (id) => document.getElementById(id);
  const who = (await api('/api/whoami')).body || {};
  const mayCompile = (who.capabilities || []).includes('compile_submit');
  if (!mayCompile) {
    $('gate').hidden = false;
    $('gate').textContent = who.role === 'guest' || !who.role
      ? 'Sign in to submit: the math lab needs a member account. Anonymous and guest visitors cannot queue compiles.'
      : 'This account may not submit compiles (no compile_submit capability).';
    $('submit').disabled = true;
  }
  for (const name of ALL_TARGETS) {
    const box = el('input', {type: 'checkbox', value: name});
    box.checked = true;
    $('targets').append(el('label', {}, box, ' ' + name));
  }
  const extra = {machine: shortUserAgent(), gpu: await gpuLabel(), measurements: [], note: ''};
  let polling = 0;
  $('form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const targets = [...$('targets').querySelectorAll('input:checked')].map((box) => box.value);
    $('submit').disabled = true;
    $('status').textContent = 'submitting...';
    const posted = await api('/api/mathlab/jobs', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: $('text').value, targets})});
    $('submit').disabled = !mayCompile;
    if (!posted.body || !posted.body.job_id) {
      $('status').textContent = (posted.body && posted.body.error) || `refused (${posted.status})`;
      return;
    }
    if (posted.body.state === 'rejected') {
      const e = posted.body.error;
      $('status').textContent = `rejected: ${e.reason}${e.line ? ` (line ${e.line}, col ${e.col}, ${e.construct})` : ''}`;
      return;
    }
    const id = posted.body.job_id, mine = ++polling;
    extra.measurements = [];
    extra.note = '';
    $('result').hidden = true;
    while (mine === polling) {
      const got = await api(`/api/mathlab/jobs/${id}`);
      const job = got.body;
      if (!job || !job.state) { $('status').textContent = `lost the job (${got.status})`; return; }
      if (!TERMINAL.has(job.state)) {
        $('status').textContent = job.state === 'queued' ? `queued, position ${job.queue_position}` : 'compiling...';
        await new Promise((resolve) => setTimeout(resolve, 1000));
        continue;
      }
      if (job.state !== 'complete') {
        $('status').textContent = `${job.state}: ${typeof job.error === 'object' && job.error ? job.error.reason : job.error || ''}`;
        return;
      }
      $('status').textContent = 'complete; timing the wasm product in this browser...';
      await measureWasm(job, extra);
      const final = (await api(`/api/mathlab/jobs/${id}`)).body || job;
      renderResult(final, extra);
      $('status').textContent = 'complete';
      return;
    }
  });
}

if (typeof document !== 'undefined') main();
