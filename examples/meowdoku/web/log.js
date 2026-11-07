// [LOG] The log page: the server's text ring (/api/log) drawn as GLYPH MESHES in a small instanced scene -- the exported letter/digit
// token meshes of assets.json (the ones the main page's Phong scene draws), lit by a Phong light.  Only the lines in view, and only the
// characters on them, become instances; the text itself lives in plain JS strings.
import { SOURCES, visibleRange, clampScroll, tailScroll, atTail, glyphFor, lineText, matches, searchTerms } from "./log_view.js";

const $ = (id) => document.getElementById(id);
const COLORS = { "client-error": [1.0, 0.48, 0.38], "fountain-diag": [0.42, 0.85, 1.0], "server": [0.62, 0.95, 0.55] };
const MAX_ENTRIES = 20000, POLL_MS = 1500;

const S = { entries: [], lines: [], sources: new Set(SOURCES), terms: [], last: 0, follow: true, fontPx: 18,
            scroll: 0, target: 0, scrollX: 0, maxCols: 0, dirty: true, subst: 0, visibleText: [], error: "" };
window.__log = S;                                   // for tests and the curious

// ---------------------------------------------------------------- the glyph scene
const canvas = $("c"), gl = canvas.getContext("webgl2", { antialias: true });
const VS = `#version 300 es
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aNrm; layout(location=2) in vec3 aInst; layout(location=3) in vec3 aCol;
uniform vec2 uHalf; uniform float uUnit; out vec3 vN; out vec3 vCol;
const mat3 R = mat3(0.9553, 0.0, -0.2955,  0.0586, 0.9801, 0.1877,  0.2896, -0.1987, 0.9363);   // a slight tilt so the extrusion shows
void main() {
  vec3 p = R * aPos, n = R * aNrm;
  p.y = -p.y; n.y = -n.y;                                   // the mesh is y-up, the page is y-down
  vec2 px = aInst.xy + p.xy * uUnit * aInst.z;
  gl_Position = vec4(px.x / uHalf.x - 1.0, 1.0 - px.y / uHalf.y, -p.z * 0.5, 1.0);
  vN = n; vCol = aCol;
}`;
const FS = `#version 300 es
precision mediump float; in vec3 vN; in vec3 vCol; out vec4 o;
void main() {
  vec3 n = normalize(vN), L = normalize(vec3(-0.45, 0.55, 0.7)), V = vec3(0.0, 0.0, 1.0);
  float d = max(dot(n, L), 0.0), s = pow(max(dot(normalize(L + V), n), 0.0), 40.0);
  o = vec4(vCol * (0.35 + 0.75 * d) + vec3(0.25) * s, 1.0);
}`;
function program() {
  const mk = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const p = gl.createProgram(); gl.attachShader(p, mk(gl.VERTEX_SHADER, VS)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
let prog, meshes = new Map(), have = new Set(), ext = 0.1;

function uploadGlyph(name, mesh) {
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.positions), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  const nb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, nb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.normals), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  const ib = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, ib);                       // per instance: x, y, scale, r, g, b
  for (const [loc, size, off] of [[2, 3, 0], [3, 3, 12]]) { gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 24, off); gl.vertexAttribDivisor(loc, 1); }
  gl.bindVertexArray(null);
  meshes.set(name, { vao, ib, count: mesh.positions.length / 3 });
}

// ---------------------------------------------------------------- data
function rebuildLines() {
  S.lines = []; S.maxCols = 0;
  for (const e of S.entries) addLine(e);
}
function addLine(e) {
  if (!matches(e, S.sources, S.terms)) return;
  const text = lineText(e);
  S.lines.push({ text, src: e.src }); if (text.length > S.maxCols) S.maxCols = text.length;
}
function ingest(list) {
  for (const e of list) { S.entries.push(e); addLine(e); S.last = Math.max(S.last, e.seq); }
  if (S.entries.length > MAX_ENTRIES) { S.entries.splice(0, S.entries.length - MAX_ENTRIES); rebuildLines(); }
  S.dirty = true;
}
async function poll() {
  try {
    for (let i = 0; i < 20; i++) {
      const r = await fetch(`/api/log?since=${S.last}&limit=1000`, { cache: "no-store" });
      if (r.status === 403) { S.error = "The log is for local (LAN) or dev-tier visitors only."; break; }
      const j = await r.json();
      S.error = "";
      if (j.last < S.last) { S.entries = []; S.last = 0; rebuildLines(); continue; }        // the server restarted without its ring
      ingest(j.entries);
      $("stats").textContent = `${S.lines.length} lines shown / ${S.entries.length} held  (server ring ${(j.bytes / 1024).toFixed(0)} of ${(j.cap / 1024).toFixed(0)} KiB)`;
      if (j.entries.length < 1000) break;
    }
  } catch (err) { S.error = "log fetch failed: " + err; }
  $("msg").style.display = S.error ? "block" : "none"; $("msg").textContent = S.error; S.dirty = true;
  setTimeout(poll, POLL_MS);
}

// ---------------------------------------------------------------- layout + draw
const cell = () => ({ w: Math.round(S.fontPx * 0.72), h: Math.round(S.fontPx * 1.4) });
function size() {
  const wrap = $("wrap"), dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(wrap.clientWidth * dpr)), h = Math.max(1, Math.round(wrap.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; S.dirty = true; }
  return dpr;
}
function frame() {
  requestAnimationFrame(frame);
  const dpr = size(), c = cell(), viewH = canvas.height / dpr, viewW = canvas.width / dpr;
  if (S.follow) S.target = tailScroll(viewH, c.h, S.lines.length);
  S.target = clampScroll(S.target, viewH, c.h, S.lines.length);
  if (Math.abs(S.target - S.scroll) > 0.3) { S.scroll += (S.target - S.scroll) * 0.3; S.dirty = true; } else if (S.scroll !== S.target) { S.scroll = S.target; S.dirty = true; }
  S.scrollX = clampScroll(S.scrollX, viewW, c.w, S.maxCols + 2);
  if (!S.dirty || !prog) return;
  S.dirty = false;
  const rows = visibleRange(S.scroll, viewH, c.h, S.lines.length, 1), cols = visibleRange(S.scrollX, viewW, c.w, S.maxCols, 1);
  const unit = (c.w * 0.92) / (2 * ext), byGlyph = new Map(); let subst = 0, n = 0;
  S.visibleText = [];
  for (let r = rows.first; r < rows.last; r++) {
    const line = S.lines[r], col = COLORS[line.src] || [1, 1, 1], y = r * c.h - S.scroll + c.h / 2;
    const inView = y > -c.h && y < viewH + c.h;
    if (r >= Math.floor(S.scroll / c.h) && r < Math.ceil((S.scroll + viewH) / c.h)) S.visibleText.push(line.text);
    if (!inView) continue;
    const end = Math.min(line.text.length, cols.last);
    for (let k = cols.first; k < end; k++) {
      const g = glyphFor(line.text[k], have); if (!g) continue;
      if (g.substituted) subst++;
      const dim = k < 12 ? 0.55 : 1;                                         // the time and tag prefix is quieter
      let a = byGlyph.get(g.name); if (!a) byGlyph.set(g.name, a = []);
      a.push(k * c.w - S.scrollX + c.w / 2, y, g.scale, col[0] * dim, col[1] * dim, col[2] * dim); n++;
    }
  }
  S.subst = subst; S.drawn = n;
  gl.viewport(0, 0, canvas.width, canvas.height); gl.clearColor(0.047, 0.055, 0.075, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST); gl.useProgram(prog);
  gl.uniform2f(gl.getUniformLocation(prog, "uHalf"), canvas.width / dpr / 2, canvas.height / dpr / 2);
  gl.uniform1f(gl.getUniformLocation(prog, "uUnit"), unit);
  for (const [name, arr] of byGlyph) {
    const m = meshes.get(name); gl.bindVertexArray(m.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.ib); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.DYNAMIC_DRAW);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, m.count, arr.length / 6);
  }
  gl.bindVertexArray(null);
  const thumb = $("thumb"), total = S.lines.length * c.h, frac = total > viewH ? viewH / total : 1;
  thumb.style.display = frac < 1 ? "block" : "none";
  thumb.style.height = Math.max(20, frac * viewH) + "px"; thumb.style.top = (total > viewH ? (S.scroll / (total - viewH)) * (viewH - Math.max(20, frac * viewH)) : 0) + "px";
  $("note").textContent = `Glyph set: A-Z 0-9 + (the exported token meshes). Lowercase is drawn as capitals; ${subst} other character${subst === 1 ? "" : "s"} in view drawn as a small "+". Copy visible / Copy all give the exact text.`;
}

// ---------------------------------------------------------------- input
function setFollow(on) { S.follow = on; $("follow").style.background = on ? "#2a5a3a" : ""; S.dirty = true; }
canvas.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const c = cell();
  if (ev.shiftKey) S.scrollX += ev.deltaY + ev.deltaX; else { S.follow = false; setFollow(false); S.target += ev.deltaY * (ev.deltaMode === 1 ? c.h : 1); }
  const dpr = window.devicePixelRatio || 1;
  if (atTail(clampScroll(S.target, canvas.height / dpr, c.h, S.lines.length), canvas.height / dpr, c.h, S.lines.length) && ev.deltaY > 0) setFollow(true);
  S.dirty = true;
}, { passive: false });
let drag = null;
canvas.addEventListener("pointerdown", (ev) => { drag = { x: ev.clientX, y: ev.clientY, sx: S.scrollX, st: S.target }; canvas.setPointerCapture(ev.pointerId); canvas.style.cursor = "grabbing"; });
canvas.addEventListener("pointermove", (ev) => { if (!drag) return; setFollow(false); S.target = drag.st - (ev.clientY - drag.y); S.scrollX = drag.sx - (ev.clientX - drag.x); S.scroll = S.target; S.dirty = true; });
canvas.addEventListener("pointerup", () => { drag = null; canvas.style.cursor = "grab"; });
window.addEventListener("keydown", (ev) => {
  if (ev.target.id === "q") return;
  const c = cell(), page = canvas.height / (window.devicePixelRatio || 1) - c.h;
  const k = { PageUp: -page, PageDown: page, ArrowUp: -c.h, ArrowDown: c.h, Home: -1e12, End: 1e12 }[ev.key];
  if (k === undefined) return;
  ev.preventDefault(); setFollow(ev.key === "End"); S.target += k; S.dirty = true;
});
$("follow").onclick = () => setFollow(!S.follow);
$("zoomin").onclick = () => { S.fontPx = Math.min(48, S.fontPx + 2); S.dirty = true; };
$("zoomout").onclick = () => { S.fontPx = Math.max(8, S.fontPx - 2); S.dirty = true; };
$("q").oninput = (ev) => { S.terms = searchTerms(ev.target.value); rebuildLines(); S.dirty = true; };
async function copy(text, btn) {
  try { await navigator.clipboard.writeText(text); } catch (_) {
    const t = document.createElement("textarea"); t.value = text; document.body.appendChild(t); t.select(); document.execCommand("copy"); t.remove();
  }
  const old = btn.textContent; btn.textContent = "Copied"; setTimeout(() => { btn.textContent = old; }, 900);
}
$("copyvis").onclick = (ev) => copy(S.visibleText.join("\n"), ev.target);
$("copyall").onclick = (ev) => copy(S.lines.map((l) => l.text).join("\n"), ev.target);
for (const src of SOURCES) {
  const col = COLORS[src].map((v) => Math.round(v * 255)).join(","), l = document.createElement("label");
  l.className = "on"; l.innerHTML = `<input type="checkbox" checked><span class="sw" style="background:rgb(${col})"></span>${src}`;
  l.querySelector("input").onchange = (ev) => { ev.target.checked ? S.sources.add(src) : S.sources.delete(src); l.className = ev.target.checked ? "on" : ""; rebuildLines(); S.dirty = true; };
  $("srcs").appendChild(l);
}

// The game tab persists its console ring in localStorage. Read it here without
// installing a second console capture that could overwrite that tab's ring.
function localEntries() {
  try {
    const rows = JSON.parse(localStorage.getItem("meowdoku.locallog") || "[]");
    return Array.isArray(rows) ? rows : [];
  } catch (_) { return []; }
}
let localShown = [];
function drawLocal() {
  const terms = searchTerms($("local-q").value);
  localShown = localEntries().filter((e) => !terms.length || terms.every((w) => String(e.text).toLowerCase().includes(w)))
    .map(lineText);
  const pane = $("local-lines"), atEnd = pane.scrollTop + pane.clientHeight >= pane.scrollHeight - 24;
  pane.textContent = localShown.join("\n");
  if (atEnd) pane.scrollTop = pane.scrollHeight;
  $("local-stats").textContent = `${localShown.length} lines`;
}
$("local-q").oninput = drawLocal;
$("local-copy").onclick = (ev) => copy(localShown.join("\n"), ev.target);
$("local-clear").onclick = () => { localStorage.removeItem("meowdoku.locallog"); drawLocal(); };
window.addEventListener("storage", (ev) => { if (ev.key === "meowdoku.locallog") drawLocal(); });
setInterval(drawLocal, POLL_MS);
drawLocal();

// ---------------------------------------------------------------- boot
(async () => {
  try {
    if (!gl) throw new Error("WebGL2 is not available");
    prog = program();
    const assets = await (await fetch("assets.json")).json();
    for (const [name, mesh] of Object.entries(assets.meshes)) if (name.length === 1) { uploadGlyph(name, mesh); have.add(name); ext = Math.max(ext === 0.1 ? 0 : ext, mesh.extent); }
  } catch (err) { S.error = "glyph scene failed: " + err; $("msg").style.display = "block"; $("msg").textContent = S.error; }
  setFollow(true);
  poll(); frame();
})();
