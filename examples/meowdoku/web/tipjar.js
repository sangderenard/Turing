// TIP JAR page layer (server: tipjar.py).  A watch-only Bitcoin piggybank drawn by the 3-D fountain: one grain per incoming transaction.
// This file is the PANEL (balance, address, QR, legend), the grain PICKER (hover or tap a grain: amount, date, status) and a state poll.
// The grains themselves are the law's, drawn by the shared particle pass; nothing here simulates.  Pure helpers are exported for the Node test.

// ---------------------------------------------------------------- QR code (byte mode, error correction L, versions 1..5; pure JS, no CDN)
const EC_L = { 1: [19, 7], 2: [34, 10], 3: [55, 15], 4: [80, 20], 5: [108, 26] };   // version -> [data codewords, ecc codewords]; one block each
const ALIGN = { 2: 18, 3: 22, 4: 26, 5: 30 };
const GF_EXP = new Uint8Array(512), GF_LOG = new Uint8Array(256);
for (let i = 0, x = 1; i < 255; i++) { GF_EXP[i] = x; GF_LOG[x] = i; x <<= 1; if (x & 0x100) x ^= 0x11d; }
for (let i = 255; i < 512; i++) GF_EXP[i] = GF_EXP[i - 255];
const gfMul = (a, b) => (a && b ? GF_EXP[GF_LOG[a] + GF_LOG[b]] : 0);

export function rsEcc(data, degree) {
  let gen = [1];
  for (let i = 0; i < degree; i++) {
    const next = new Array(gen.length + 1).fill(0);
    gen.forEach((c, j) => { next[j] ^= c; next[j + 1] ^= gfMul(c, GF_EXP[i]); });
    gen = next;
  }
  const rem = new Array(degree).fill(0);
  for (const byte of data) {
    const factor = byte ^ rem.shift(); rem.push(0);
    for (let j = 0; j < degree; j++) rem[j] ^= gfMul(gen[j + 1], factor);
  }
  return rem;
}

export function formatBits(mask) {            // ECC level L (01) and the mask, BCH(15,5), xor 0x5412
  const data = (1 << 3) | mask;
  let rem = data;
  for (let i = 0; i < 10; i++) rem = (rem << 1) ^ ((rem >>> 9) * 0x537);
  return ((data << 10) | rem) ^ 0x5412;
}

const MASKS = [(x, y) => (x + y) % 2 === 0, (x, y) => y % 2 === 0, (x, y) => x % 3 === 0, (x, y) => (x + y) % 3 === 0,
  (x, y) => (Math.floor(x / 3) + Math.floor(y / 2)) % 2 === 0, (x, y) => (x * y) % 2 + (x * y) % 3 === 0,
  (x, y) => ((x * y) % 2 + (x * y) % 3) % 2 === 0, (x, y) => ((x + y) % 2 + (x * y) % 3) % 2 === 0];

function penalty(m, size) {
  let score = 0;
  const lines = [];
  for (let a = 0; a < size; a++) { lines.push(m[a].slice()); lines.push(m.map((row) => row[a])); }
  for (const line of lines) {
    let run = 1;
    for (let i = 1; i <= size; i++) {
      if (i < size && line[i] === line[i - 1]) run++;
      else { if (run >= 5) score += 3 + run - 5; run = 1; }
    }
    const s = line.map((v) => (v ? "1" : "0")).join("");
    for (const pat of ["10111010000", "00001011101"]) { let at = -1; while ((at = s.indexOf(pat, at + 1)) >= 0) score += 40; }
  }
  for (let y = 0; y < size - 1; y++) for (let x = 0; x < size - 1; x++) if (m[y][x] === m[y][x + 1] && m[y][x] === m[y + 1][x] && m[y][x] === m[y + 1][x + 1]) score += 3;
  const dark = m.reduce((n, row) => n + row.filter(Boolean).length, 0);
  return score + 10 * Math.floor(Math.abs(dark * 20 - size * size * 10) / (size * size));
}

// the module matrix (array of rows of booleans, true = dark) of `text` (UTF-8 bytes); throws when it does not fit version 5
export function qrMatrix(text) {
  const bytes = Array.from(new TextEncoder().encode(text));
  const version = [1, 2, 3, 4, 5].find((v) => bytes.length <= Math.floor((EC_L[v][0] * 8 - 12) / 8));
  if (!version) throw new Error("text too long for the QR code");
  const [dataWords, eccWords] = EC_L[version], size = 17 + 4 * version;
  const bits = [];
  const put = (value, count) => { for (let i = count - 1; i >= 0; i--) bits.push((value >>> i) & 1); };
  put(4, 4); put(bytes.length, 8); bytes.forEach((b) => put(b, 8));
  put(0, Math.min(4, dataWords * 8 - bits.length));
  while (bits.length % 8) bits.push(0);
  for (let pad = 0xec; bits.length < dataWords * 8; pad ^= 0xec ^ 0x11) put(pad, 8);
  const data = []; for (let i = 0; i < bits.length; i += 8) data.push(parseInt(bits.slice(i, i + 8).join(""), 2));
  const words = data.concat(rsEcc(data, eccWords)), stream = [];
  words.forEach((w) => put2(stream, w));
  function put2(arr, w) { for (let i = 7; i >= 0; i--) arr.push((w >>> i) & 1); }
  const modules = Array.from({ length: size }, () => new Array(size).fill(false)), isFn = Array.from({ length: size }, () => new Array(size).fill(false));
  const setFn = (x, y, dark) => { if (x >= 0 && y >= 0 && x < size && y < size) { modules[y][x] = dark; isFn[y][x] = true; } };
  const finder = (cx, cy) => { for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) { const d = Math.max(Math.abs(dx), Math.abs(dy)); setFn(cx + dx, cy + dy, d !== 4 && d !== 2); } };
  finder(3, 3); finder(size - 4, 3); finder(3, size - 4);
  for (let i = 0; i < size; i++) { setFn(6, i, i % 2 === 0); setFn(i, 6, i % 2 === 0); }
  if (ALIGN[version]) { const p = ALIGN[version]; for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) setFn(p + dx, p + dy, Math.max(Math.abs(dx), Math.abs(dy)) !== 1); }
  const drawFormat = (mask) => {
    const f = formatBits(mask), bit = (i) => ((f >>> i) & 1) !== 0;
    for (let i = 0; i <= 5; i++) setFn(8, i, bit(i));
    setFn(8, 7, bit(6)); setFn(8, 8, bit(7)); setFn(7, 8, bit(8));
    for (let i = 9; i < 15; i++) setFn(14 - i, 8, bit(i));
    for (let i = 0; i < 8; i++) setFn(size - 1 - i, 8, bit(i));
    for (let i = 8; i < 15; i++) setFn(8, size - 15 + i, bit(i));
    setFn(8, size - 8, true);
  };
  drawFormat(0);
  let i = 0;
  for (let right = size - 1; right >= 1; right -= 2) {
    if (right === 6) right = 5;
    for (let vert = 0; vert < size; vert++) for (let j = 0; j < 2; j++) {
      const x = right - j, y = ((right + 1) & 2) === 0 ? size - 1 - vert : vert;
      if (!isFn[y][x] && i < stream.length) modules[y][x] = stream[i++] === 1;
    }
  }
  let best = null, bestScore = Infinity;
  for (let mask = 0; mask < 8; mask++) {
    const trial = modules.map((row) => row.slice());
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) if (!isFn[y][x] && MASKS[mask](x, y)) trial[y][x] = !trial[y][x];
    const keep = modules.map((row) => row.slice()); modules.forEach((row, y) => row.forEach((_, x) => { modules[y][x] = trial[y][x]; }));
    drawFormat(mask);
    const score = penalty(modules, size);
    if (score < bestScore) { bestScore = score; best = modules.map((row) => row.slice()); }
    keep.forEach((row, y) => row.forEach((v, x) => { modules[y][x] = v; }));
  }
  return best;
}

export function drawQr(canvas, text, px = 168) {
  const m = qrMatrix(text), n = m.length, quiet = 4, cell = Math.max(1, Math.floor(px / (n + 2 * quiet)));
  canvas.width = canvas.height = cell * (n + 2 * quiet);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#000";
  m.forEach((row, y) => row.forEach((dark, x) => { if (dark) ctx.fillRect((x + quiet) * cell, (y + quiet) * cell, cell, cell); }));
}

// ---------------------------------------------------------------- decoding a grain's colour (tipjar.py grain_rgb inverse) -- the legend's promise
export function hsl(h, s, l) {
  const a = s * Math.min(l, 1 - l), f = (n) => { const k = (n + h / 30) % 12; return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
  return [f(0), f(8), f(4)].map((v) => Math.round(v * 255));
}
export const btc = (sats) => (sats / 1e8).toFixed(8);
export const withCommas = (n) => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
export function describeGrain(g, kinds) {
  const label = (kinds.find((k) => k[0] === g.kind) || [g.kind, g.kind])[1];
  const date = g.time ? new Date(g.time * 1000).toISOString().slice(0, 10) : "not yet in a block";
  return { amount: `${btc(g.sats)} BTC (${withCommas(g.sats)} sats)`, date, status: g.confirmed ? "confirmed" : "unconfirmed (in the mempool)", from: label };
}

// ---------------------------------------------------------------- the panel, the picker, the poll
export function initTipJar(hooks) {
  // hooks: { canvas, getState() -> game state, getCompute() -> compute runtime, getView() -> { viewProj, proj, toWorld(g) }, applyState(game) }
  const css = document.createElement("style");
  css.textContent = `
  #tipjar { position: fixed; z-index: 3; right: 10px; bottom: 44px; width: 290px; max-height: calc(100% - 120px); overflow: auto; box-sizing: border-box;
            padding: 10px 12px; border-radius: 10px; border: 1px solid var(--line); background: color-mix(in srgb, var(--panel) 88%, transparent);
            backdrop-filter: blur(6px); color: var(--fg); font: 13px/1.35 system-ui, sans-serif; display: none; }
  #tipjar h3 { margin: 0 0 4px; font-size: 14px; display: flex; gap: 8px; align-items: center; }
  #tipjar .badge { font-size: 11px; padding: 1px 7px; border-radius: 9px; background: #8a5a1a; color: #fff; }
  #tipjar.min > :not(h3):not(.big):not(.sub) { display: none; }
  #tipjar h3 button { margin-left: auto; padding: 0 8px; }
  #tipjar .big { font: 600 20px/1.2 ui-monospace, monospace; font-variant-numeric: tabular-nums; }
  #tipjar .sub, #tipjar .note { opacity: .75; font-size: 12px; }
  #tipjar .addr { font: 11px/1.3 ui-monospace, monospace; word-break: break-all; margin: 6px 0 4px; user-select: all; }
  #tipjar button, #tipjar a.btn { background: var(--ctl); color: var(--fg); border: 1px solid var(--line); border-radius: 6px; padding: 4px 9px; font: inherit; cursor: pointer; text-decoration: none; display: inline-block; }
  #tipjar canvas { position: static; display: block; width: 168px; height: 168px; margin: 6px 0; border-radius: 4px; image-rendering: pixelated; touch-action: auto; }
  #tipjar .legend { margin-top: 6px; border-top: 1px solid var(--line); padding-top: 6px; }
  #tipjar .legend div { display: flex; gap: 6px; align-items: center; margin: 2px 0; font-size: 12px; }
  #tipjar .sw { width: 12px; height: 12px; border-radius: 50%; flex: none; }
  #tipjar .ramp { width: 70px; height: 10px; border-radius: 5px; flex: none; }
  #tipjar-tip { position: fixed; z-index: 4; pointer-events: none; display: none; padding: 6px 9px; border-radius: 8px; background: rgba(12,12,16,.92); color: #f4f4f6;
                border: 1px solid #55555e; font: 12px/1.35 system-ui, sans-serif; max-width: 260px; }
  @media (max-width: 700px) { #tipjar { left: 10px; right: 10px; width: auto; bottom: 36px; max-height: 38%; } }`;
  document.head.appendChild(css);
  const panel = document.createElement("div"); panel.id = "tipjar";
  const tip = document.createElement("div"); tip.id = "tipjar-tip";
  document.body.append(panel, tip);
  let current = null, qrFor = null, pinned = 0;
  if (innerWidth < 700 || innerHeight < 700) panel.classList.add("min");       // a small screen starts with the balance only; the header button opens the rest

  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  function render(t) {
    panel.replaceChildren();
    const head = el("h3", null, "Tip jar"); if (t.demo) head.append(el("span", "badge", "DEMO"));
    panel.append(head);
    if (t.demo) panel.append(el("div", "note", t.refused ? `tip jar not configured (${t.refused}). Sample transactions, no real address.` : "tip jar not configured. These are sample transactions, no real address."));
    panel.append(el("div", "big", `${t.balance_btc} BTC`));
    const fiat = t.fiat && t.fiat.USD != null ? ` · about $${withCommas(t.fiat.USD)}` : "";
    panel.append(el("div", "sub", `${withCommas(t.balance_sats)} sats${fiat} · ${t.count} donation${t.count === 1 ? "" : "s"}`));
    if (t.error) panel.append(el("div", "note", "the last poll failed; showing the last list"));
    if (t.address) {
      panel.append(el("div", "addr", t.address));
      const row = el("div"); const copy = el("button", null, "Copy");
      copy.onclick = async () => { try { await navigator.clipboard.writeText(t.address); copy.textContent = "Copied"; } catch (e) { copy.textContent = "Select the address"; } setTimeout(() => { copy.textContent = "Copy"; }, 1500); };
      const open = el("a", "btn", "Open in wallet"); open.href = t.uri; open.style.marginLeft = "6px";
      row.append(copy, open); panel.append(row);
      const q = document.createElement("canvas");
      try { drawQr(q, t.uri); panel.append(q); } catch (e) { panel.append(el("div", "note", "QR unavailable: " + e.message)); }
    }
    const lg = el("div", "legend");
    lg.append(el("div", null, "How to read a grain"));
    lg.append(el("div", null, "size: the amount (" + (t.size_method === "log" ? "log-scaled, the amounts span more than the grains can" : t.size_method === "cbrt" ? "volume proportional to sats" : "one size") + ")"));
    for (const [kind, label, hue] of t.kinds) {
      if (kind === "unknown") continue;
      const row = el("div"), sw = el("span", "sw"), [r, g, b] = hsl(hue, 0.85, 0.5); sw.style.background = `rgb(${r},${g},${b})`;
      row.append(sw, el("span", null, "from " + label)); lg.append(row);
    }
    const [lo, hi, un] = t.legend.light, ramp = el("div"), bar = el("span", "ramp");
    const a = hsl(150, 0.85, lo), b2 = hsl(150, 0.85, hi);
    bar.style.background = `linear-gradient(90deg, rgb(${a}), rgb(${b2}))`; ramp.append(bar, el("span", null, "older to newer"));
    const pend = el("div"), sw2 = el("span", "sw"); sw2.style.background = `rgb(${hsl(150, 0.85, un)})`; sw2.style.boxShadow = "0 0 6px #fff"; pend.append(sw2, el("span", null, "nearly white: unconfirmed"));
    lg.append(ramp, pend, el("div", "note", "Tap or hover a grain for its amount, date and status."));
    panel.append(lg);
  }
  function update(game) {
    const t = game && game.tipjar;
    if (!t) { panel.style.display = "none"; current = null; return; }
    current = t;
    const key = JSON.stringify([t.balance_sats, t.count, t.error, t.address, t.fiat, t.demo, t.size_method]);
    if (key !== qrFor) { qrFor = key; render(t); }
    panel.style.display = "block";
  }

  // --- picking: project the live grains through the shared camera and take the nearest one under the pointer
  const rect = () => hooks.canvas.getBoundingClientRect();
  function pick(clientX, clientY) {
    if (!current) return null;
    const compute = hooks.getCompute(), data = compute && compute.points(), game = hooks.getState();
    if (!data || !game || !game.compute || !compute.rec) return null;
    const view = hooks.getView(), toWorld = view.toWorld(game), f32 = new Float32Array(data.buffer, data.byteOffset, data.byteLength >> 2);
    const stride = compute.rec.display_stride || 1, r = rect(), W = r.width, H = r.height;
    const a = toWorld(0, 0, 0), b = toWorld(1, 0, 0), k = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    const byId = new Map(current.grains.map((g) => [g.id, g]));
    const m = view.viewProj; let best = null, bestD = Infinity;
    for (let row = 0; row * 5 + 4 < f32.length; row++) {
      const id = row * stride, g = byId.get(id), meta = f32[row * 5 + 4];
      if (!g || !(meta >= 50)) continue;
      const w = toWorld(f32[row * 5], f32[row * 5 + 1], f32[row * 5 + 2]);
      const cx = m[0] * w[0] + m[4] * w[1] + m[8] * w[2] + m[12], cy = m[1] * w[0] + m[5] * w[1] + m[9] * w[2] + m[13], cw = m[3] * w[0] + m[7] * w[1] + m[11] * w[2] + m[15];
      if (cw <= 0) continue;
      const sx = (cx / cw * 0.5 + 0.5) * W + r.left, sy = (0.5 - cy / cw * 0.5) * H + r.top;
      const radius = (meta - Math.floor(meta / 100 + 0.001) * 100) * 0.001, rpx = radius * k * view.proj[5] * 0.5 * H / cw;
      const d = Math.hypot(clientX - sx, clientY - sy) - Math.max(rpx, 6);
      if (d < bestD && d < 8) { bestD = d; best = g; }
    }
    return best;
  }
  function show(g, x, y) {
    if (!g) { tip.style.display = "none"; return; }
    const d = describeGrain(g, current.kinds);
    tip.replaceChildren(el("div", null, d.amount), el("div", null, d.date + " · " + d.status), el("div", null, "from " + d.from));
    if (g.txid) { const a = el("a", null, "txid " + g.txid.slice(0, 12) + "…"); a.href = "https://mempool.space/tx/" + g.txid; a.target = "_blank"; a.rel = "noopener"; a.style.pointerEvents = "auto"; a.style.color = "#9bd"; tip.append(a); }
    tip.style.display = "block";
    tip.style.left = Math.min(innerWidth - 270, x + 14) + "px"; tip.style.top = Math.max(8, y - 10) + "px";
  }
  hooks.canvas.addEventListener("pointermove", (e) => {
    if (!current || e.pointerType !== "mouse" || performance.now() < pinned) return;
    show(pick(e.clientX, e.clientY), e.clientX, e.clientY);
  });
  let down = null;
  hooks.canvas.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
  hooks.canvas.addEventListener("pointerup", (e) => {
    if (!current || !down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 8 || performance.now() - down.t > 500) return;
    const g = pick(e.clientX, e.clientY); show(g, e.clientX, e.clientY);
    pinned = g ? performance.now() + 4000 : 0;
    if (g) setTimeout(() => { if (performance.now() >= pinned) tip.style.display = "none"; }, 4100);
  });
  hooks.canvas.addEventListener("pointerleave", () => { if (performance.now() >= pinned) tip.style.display = "none"; });

  // --- the state poll: the server keeps one cached upstream poll; this only re-reads the server's state
  setInterval(async () => {
    const game = hooks.getState();
    if (!game || !game.tipjar || document.hidden) return;
    try {
      const fresh = await (await fetch("/api/state", { cache: "no-store" })).json();
      if (!fresh || !fresh.tipjar) return;
      if (fresh.compute && game.compute && fresh.compute.epoch !== game.compute.epoch) hooks.applyState(fresh);     // a new donation: the jar starts again
      else { game.tipjar = fresh.tipjar; update(game); }
    } catch (e) { /* offline: keep the last */ }
  }, 20000);
  return { update, pick, describeGrain };
}
