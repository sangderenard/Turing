// [LOADING] The loading panel: a full work-area overlay drawn by a tiny standalone WebGL2 program -- no engine, no law, no WebGPU.
//
// It plays the loading ANIMATION (loading_anim.js: a reduced, position-only filter of the fountain's state memory, fetched through the octet
// codec) in a loop; until the server has one it draws a procedural hourglass.  Its tint says what the site is doing:
//     compiling     (amber)   the server is building products this page needs (/api/ready names WHAT: "compiling page law graph... 2/5")
//     downloading   (blue)    the page is fetching assets and products ("downloading 3.1/8 MB": bytes counted by the fetch wrapper below)
//     initializing  (green)   GPU programs, shaders and uploads
// It also owns the page's readiness client: `gate(game)` holds a game until ITS products are built (the panel shows meanwhile) and
// `whenReady(names)` waits silently for a product, using the polling pattern of lifecycle.js.  The panel never takes input.
import { decodePacket, seam, angleOf } from "./loading_anim.js";

// ------------------------------------------------------------------ pure helpers (tested in node)
export const PHASES = {
  compiling: { tint: [1.0, 0.62, 0.16], title: "compiling" },
  downloading: { tint: [0.28, 0.66, 1.0], title: "downloading" },
  initializing: { tint: [0.42, 0.92, 0.55], title: "initializing" },
};

// the products a game's page needs (server: startup.py GAME_NEEDS), for a browser with / without WebGPU
export function needsOf(data, game, webgpu) {
  const need = data && data.needs && data.needs[game];
  if (!need) return [];
  return Array.isArray(need) ? need : (webgpu ? need.gpu : need.cpu) || [];
}
const SETTLED = new Set(["ready", "failed", "skipped"]);
export const settled = (data, names) => names.every((n) => !data.products[n] || SETTLED.has(data.products[n].state));

export function megabytes(bytes) { return (bytes / 1048576).toFixed(bytes < 10485760 ? 1 : 0); }

// the caption for a phase: what the site is doing, in words
export function captionOf(phase, { data, names = [], dl = { loaded: 0, total: 0 }, detail = "" } = {}) {
  if (phase === "compiling") {
    const waiting = data ? (data.compiling || []).filter((c) => names.includes(c.name)) : [];
    const total = names.length, done = total - waiting.length;
    const first = waiting.find((c) => c.state === "building") || waiting[0];
    return first ? `compiling ${first.label}… ${Math.min(done + 1, total)}/${total}` : "compiling…";
  }
  if (phase === "downloading") {
    return dl.total ? `downloading ${megabytes(Math.min(dl.loaded, dl.total))}/${megabytes(dl.total)} MB` : `downloading ${megabytes(dl.loaded)} MB`;
  }
  return `initializing ${detail || "GPU"}`;
}

const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);

// ------------------------------------------------------------------ the page
if (typeof document !== "undefined" && typeof window !== "undefined") start();

function start() {
  const S = {
    holds: new Map(),            // id -> { kind: "boot" | "compiling", names }
    init: null,                  // the initializing phase's detail while the page sets up the GPU
    data: null, fails: 0, timer: null, waiters: new Set(),
    shownAt: 0, visible: false, showTimer: null, raf: 0, gl: null, anim: null, animUrl: null, animTried: false,
    tint: PHASES.downloading.tint.slice(), phase: "downloading", t0: performance.now(),
    forced: new URLSearchParams(location.search).get("loading_demo"),
  };
  const dl = { loaded: 0, total: 0, inflight: 0 };
  const webgpu = !!navigator.gpu;

  // ---- bytes: same-origin GET fetches (not /api/) are counted as their bodies stream through
  const nativeFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    let abs; const method = String((init && init.method) || (input && input.method) || "GET").toUpperCase();
    try { abs = new URL(typeof input === "string" ? input : input.url, location.href); } catch { return nativeFetch(input, init); }
    if (method !== "GET" || abs.origin !== location.origin || abs.pathname.startsWith("/api/")) return nativeFetch(input, init);
    dl.inflight++;
    let res;
    try { res = await nativeFetch(input, init); } catch (error) { dl.inflight--; throw error; }
    const len = Number(res.headers.get("content-length")) || 0;
    if (!res.body || [204, 205, 304].includes(res.status)) { dl.inflight--; return res; }
    dl.total += len;
    const reader = res.body.getReader();
    let counted = false;
    const finish = () => { if (!counted) { counted = true; dl.inflight--; } };
    const body = new ReadableStream({
      async pull(controller) {
        try {
          const { done, value } = await reader.read();
          if (done) { finish(); controller.close(); return; }
          dl.loaded += value.length; if (!len) dl.total = Math.max(dl.total, dl.loaded);
          controller.enqueue(value);
        } catch (error) { finish(); controller.error(error); }
      },
      cancel(reason) { finish(); return reader.cancel(reason); },
    });
    return new Response(body, { status: res.status, statusText: res.statusText, headers: res.headers });
  };

  // ---- readiness client (the lifecycle.js pattern: one poll, backing off when the server is unreachable)
  const fetchReady = async () => {
    try {
      const r = await nativeFetch("/api/ready", { cache: "no-store" });
      if (!r.ok) throw new Error(String(r.status));
      S.data = await r.json(); S.fails = 0;
    } catch { S.fails++; }
    return S.data;
  };
  const wanted = () => S.waiters.size > 0 || [...S.holds.values()].some((h) => h.kind === "compiling");
  const tick = async () => {
    S.timer = null;
    await fetchReady();
    for (const w of [...S.waiters]) if (S.fails >= 3 || (S.data && settled(S.data, w.names))) { S.waiters.delete(w); w.resolve(); }
    if (wanted()) S.timer = setTimeout(tick, S.fails ? Math.min(8000, 1000 * 2 ** S.fails) : 1000);
    refresh();
  };
  const poll = () => { if (!S.timer) S.timer = setTimeout(tick, 0); };
  const wait = (names) => new Promise((resolve) => { S.waiters.add({ names, resolve }); poll(); });

  // ---- the panel element
  const style = document.createElement("style");
  style.textContent = `
    #loading-panel { position: fixed; left: 0; top: var(--bar-h, 0px); width: var(--vw100, 100vw); height: calc(var(--vh100, 100vh) - var(--bar-h, 0px));
      z-index: 1; pointer-events: none; opacity: 0; transition: opacity .45s ease; background: var(--scene-bg, #16161a); display: none; }
    #loading-panel[data-on="1"] { opacity: 1; }
    #loading-panel canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; pointer-events: none; }
    #loading-panel .lp-caption { position: absolute; left: 0; right: 0; bottom: 9%; text-align: center; font: 600 14px/1.4 system-ui, sans-serif;
      color: var(--lp-fg, #e8e8ea); letter-spacing: .02em; pointer-events: none; }
    #loading-panel .lp-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; vertical-align: 0; background: var(--lp-tint, #fff); box-shadow: 0 0 8px var(--lp-tint, #fff); }`;
  document.head.appendChild(style);
  const root = document.createElement("div");
  root.id = "loading-panel"; root.setAttribute("aria-hidden", "true");
  const canvas = document.createElement("canvas");
  const caption = document.createElement("div"); caption.className = "lp-caption";
  const dot = document.createElement("span"); dot.className = "lp-dot";
  const text = document.createElement("span");
  caption.append(dot, text); root.append(canvas, caption);
  document.body.appendChild(root);

  // ---- colours: the page's own (CSS custom properties), parsed once per frame
  const probe = document.createElement("span"); probe.style.display = "none"; root.appendChild(probe);
  const cssColor = (name, fallback) => {
    probe.style.color = ""; probe.style.color = `var(${name})`;
    const m = getComputedStyle(probe).color.match(/[\d.]+/g);
    if (!m || m.length < 3) return fallback;
    return m[0] <= 1 && getComputedStyle(probe).color.startsWith("color(") ? [+m[0], +m[1], +m[2]] : [m[0] / 255, m[1] / 255, m[2] / 255];
  };

  // ---- GL
  const VS_POINTS = `#version 300 es
    in vec2 aPos; in float aRad;
    uniform vec2 uCenter, uRes; uniform float uScale, uAngle, uDpr;
    out float vHash;
    void main() {
      vec2 p = aPos - uCenter; float c = cos(uAngle), s = sin(uAngle);
      p = vec2(c * p.x - s * p.y, s * p.x + c * p.y) * uScale;
      gl_Position = vec4(p / (uRes * 0.5), 0.0, 1.0);
      gl_PointSize = max(3.0, 3.4 * aRad * uScale);
      vHash = fract(sin(float(gl_VertexID) * 12.9898) * 43758.5453);
    }`;
  const FS_POINTS = `#version 300 es
    precision mediump float;
    in float vHash; uniform vec3 uTint, uFg; uniform float uAlpha;
    out vec4 o;
    void main() {
      vec2 q = gl_PointCoord * 2.0 - 1.0; float r = dot(q, q);
      if (r > 1.0) discard;
      vec3 base = mix(uFg * 0.75, uTint, 0.55 + 0.4 * vHash);
      float lit = 0.75 + 0.25 * (1.0 - r) + 0.12 * (q.x * -0.4 + q.y * 0.5);
      o = vec4(base * lit, uAlpha * smoothstep(1.0, 0.82, r));
    }`;
  const VS_LINE = `#version 300 es
    in vec2 aPos; uniform vec2 uCenter, uRes; uniform float uScale, uAngle;
    void main() { vec2 p = aPos - uCenter; float c = cos(uAngle), s = sin(uAngle);
      p = vec2(c * p.x - s * p.y, s * p.x + c * p.y) * uScale; gl_Position = vec4(p / (uRes * 0.5), 0.0, 1.0); }`;
  const FS_LINE = `#version 300 es
    precision mediump float; uniform vec3 uColor; uniform float uAlpha; out vec4 o; void main() { o = vec4(uColor, uAlpha); }`;
  const VS_FULL = `#version 300 es
    void main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;
  const FS_BG = `#version 300 es
    precision mediump float; uniform vec2 uRes; uniform vec3 uBg, uTint; out vec4 o;
    void main() { vec2 q = gl_FragCoord.xy / uRes - 0.5; float v = smoothstep(0.85, 0.0, length(q * vec2(1.0, uRes.y / uRes.x * 1.0)));
      o = vec4(mix(uBg * 0.82, uBg * 1.12, v) + uTint * 0.035 * v, 1.0); }`;
  // the procedural fallback: an hourglass turning over every few seconds, sand falling through the throat
  const FS_PROC = `#version 300 es
    precision highp float; uniform vec2 uRes; uniform float uTime; uniform vec3 uBg, uTint, uFg; out vec4 o;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    void main() {
      vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / (0.5 * min(uRes.x, uRes.y));
      float period = 7.0, ph = mod(uTime, period), turns = floor(uTime / period);
      float flip = smoothstep(period - 0.9, period, ph);                       // the swift turn at the end of each round
      float a = 3.14159265 * (turns + flip) + 0.0;
      float c = cos(a), s = sin(a); vec2 p = vec2(c * uv.x + s * uv.y, -s * uv.x + c * uv.y);
      float H = 0.78, W = 0.5, T = 0.05;
      float w = T + (W - T) * smoothstep(0.0, H, abs(p.y));                   // glass half width
      float inside = step(abs(p.x), w) * step(abs(p.y), H);
      float edge = smoothstep(0.012, 0.0, abs(abs(p.x) - w)) * step(abs(p.y), H) + smoothstep(0.012, 0.0, abs(abs(p.y) - H)) * step(abs(p.x), W);
      float fall = clamp(ph / (period - 1.0), 0.0, 1.0);                       // 0: all above, 1: all below (in the turned frame, "above" is y > 0)
      float topFill = H * (1.0 - fall) * 0.8, botFill = H * fall * 0.8;
      float grain = 0.8 + 0.2 * hash(floor(p * 90.0));
      float sandTop = inside * step(0.0, p.y) * step(p.y, topFill);
      float pile = botFill * (1.0 - 0.7 * smoothstep(0.0, W, abs(p.x)));
      float sandBot = inside * step(p.y, 0.0) * step(p.y, -H * 0.98 + pile);
      float stream = step(abs(p.x), 0.012) * step(p.y, 0.0) * step(-H * 0.98 + botFill, p.y) * step(fall, 0.97) * (0.55 + 0.45 * hash(vec2(floor(p.y * 60.0 + uTime * 30.0), 3.0)));
      float sand = clamp(sandTop + sandBot + stream, 0.0, 1.0);
      vec2 q = gl_FragCoord.xy / uRes - 0.5;
      vec3 bg = mix(uBg * 0.82, uBg * 1.12, smoothstep(0.85, 0.0, length(q))) + uTint * 0.03;
      vec3 col = bg + inside * uFg * 0.05;
      col = mix(col, mix(uFg * 0.75, uTint, 0.7) * grain, sand);
      col = mix(col, uFg * 0.9, edge * 0.55);
      o = vec4(col, 1.0);
    }`;

  const compile = (gl, type, src) => { const sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh)); return sh; };
  const program = (gl, vs, fs) => { const p = gl.createProgram(); gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs)); gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, "aPos"); gl.bindAttribLocation(p, 1, "aRad"); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const name = gl.getActiveUniform(p, i).name; u[name] = gl.getUniformLocation(p, name); }
    return { p, u }; };

  function initGL() {
    if (S.gl !== null) return S.gl;
    try {
      const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
      if (!gl) throw new Error("no WebGL 2");
      S.gl = { gl, points: program(gl, VS_POINTS, FS_POINTS), line: program(gl, VS_LINE, FS_LINE), bg: program(gl, VS_FULL, FS_BG), proc: program(gl, VS_FULL, FS_PROC),
               vao: gl.createVertexArray(), posBuf: gl.createBuffer(), radBuf: gl.createBuffer(), lineBuf: gl.createBuffer() };
    } catch (error) { console.warn("[loading] panel shader unavailable:", error.message || error); S.gl = false; }
    return S.gl;
  }

  async function loadAnimation() {
    if (S.animTried || !S.data) return;
    const info = S.data.loading_anim;
    if (!info || !info.present) return;
    S.animTried = true;
    try {
      const r = await nativeFetch(info.url);
      if (!r.ok) throw new Error(String(r.status));
      const a = decodePacket(new Uint8Array(await r.arrayBuffer()));
      const n = a.radius.length, span = Math.max(a.bbox[1] - a.bbox[0], a.bbox[3] - a.bbox[2]);
      a.n = n; a.dx = (a.bbox[1] - a.bbox[0]) / span; a.dy = (a.bbox[3] - a.bbox[2]) / span;
      a.rad = Float32Array.from(a.radius, (v) => v / 65535);
      a.pos = new Float32Array(2 * n);
      a.lines = a.outline.map((poly) => Float32Array.from(poly, (v) => v / 65535));
      S.anim = a; S.animT0 = performance.now();
    } catch (error) { console.warn("[loading] animation unavailable, using the procedural one:", error.message || error); }
  }

  function phaseNow() {
    if (S.forced && PHASES[S.forced]) return S.forced;
    if ([...S.holds.values()].some((h) => h.kind === "compiling")) return "compiling";
    if (S.init) return "initializing";
    return "downloading";
  }

  function frame(now) {
    S.raf = 0;
    if (!S.visible) return;
    const phase = phaseNow(); S.phase = phase;
    S.tint = mix(S.tint, PHASES[phase].tint, 0.08);
    const names = [...S.holds.values()].filter((h) => h.kind === "compiling").flatMap((h) => h.names);
    text.textContent = captionOf(phase, { data: S.data, names, dl, detail: S.init || "" });
    root.style.setProperty("--lp-tint", `rgb(${S.tint.map((v) => Math.round(v * 255)).join(",")})`);
    const g = initGL();
    if (g) draw(g, now);
    S.raf = requestAnimationFrame(frame);
  }

  function draw({ gl, points, line, bg, proc, vao, posBuf, radBuf, lineBuf }, now) {
    const dpr = Math.min(2, window.devicePixelRatio || 1), w = Math.max(2, Math.round(canvas.clientWidth * dpr)), h = Math.max(2, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h);
    const bgc = cssColor("--scene-bg", [0.09, 0.09, 0.1]);
    const light = 0.2126 * bgc[0] + 0.7152 * bgc[1] + 0.0722 * bgc[2] > 0.5;          // a light sky: the ink turns dark so it stays readable
    const fg = light ? [0.1, 0.12, 0.16] : cssColor("--fg", [0.9, 0.9, 0.92]);
    root.style.setProperty("--lp-fg", `rgb(${fg.map((v) => Math.round(v * 255)).join(",")})`);
    gl.disable(gl.BLEND);
    const a = S.anim;
    if (!a) {
      gl.useProgram(proc.p);
      gl.uniform2f(proc.u.uRes, w, h); gl.uniform1f(proc.u.uTime, (now - S.t0) / 1000);
      gl.uniform3fv(proc.u.uBg, bgc); gl.uniform3fv(proc.u.uTint, S.tint); gl.uniform3fv(proc.u.uFg, fg);
      gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, 3);
      return;
    }
    gl.useProgram(bg.p);
    gl.uniform2f(bg.u.uRes, w, h); gl.uniform3fv(bg.u.uBg, bgc); gl.uniform3fv(bg.u.uTint, S.tint);
    gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, 3);
    const R = 0.5 * Math.hypot(a.dx, a.dy), scale = (0.46 * Math.min(w, h)) / R;
    const p = Math.floor(((now - S.animT0) / 1000) * a.fps);
    const [fa, fb, wb] = seam(p, a.x.length, a.seam);
    const uniforms = (prog, angle) => {
      gl.uniform2f(prog.u.uCenter, a.dx / 2, a.dy / 2); gl.uniform2f(prog.u.uRes, w, h); gl.uniform1f(prog.u.uScale, scale); gl.uniform1f(prog.u.uAngle, angle);
    };
    const angle = (f) => angleOf(a.angle[f]);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(line.p); uniforms(line, angle(fa));
    gl.uniform3fv(line.u.uColor, fg); gl.uniform1f(line.u.uAlpha, 0.55);
    gl.bindVertexArray(vao); gl.disableVertexAttribArray(1);
    for (const poly of a.lines) {
      gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf); gl.bufferData(gl.ARRAY_BUFFER, poly, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINE_STRIP, 0, poly.length / 2);
    }
    gl.useProgram(points.p);
    gl.uniform3fv(points.u.uTint, S.tint); gl.uniform3fv(points.u.uFg, fg);
    gl.bindBuffer(gl.ARRAY_BUFFER, radBuf); gl.bufferData(gl.ARRAY_BUFFER, a.rad, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);
    const tFrame = (((now - S.animT0) / 1000) * a.fps) % 1;                           // between two recorded frames: interpolate (the grains keep their ids)
    const set = (f, alpha) => {
      const g = Math.min(f + 1, a.x.length - 1), t = f === fa || f === fb ? tFrame : 0;
      for (let i = 0; i < a.n; i++) {
        a.pos[2 * i] = (a.x[f][i] + (a.x[g][i] - a.x[f][i]) * t) / 65535; a.pos[2 * i + 1] = (a.y[f][i] + (a.y[g][i] - a.y[f][i]) * t) / 65535;
      }
      uniforms(points, angle(f)); gl.uniform1f(points.u.uAlpha, alpha);
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferData(gl.ARRAY_BUFFER, a.pos, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.POINTS, 0, a.n);
    };
    if (wb > 0) { set(fa, 1 - wb); set(fb, wb); } else set(fa, 1);
    gl.disable(gl.BLEND);
  }

  // ---- show / hide
  function show() {
    S.showTimer = null;
    if (!S.holds.size && !S.forced) return;
    S.visible = true; S.shownAt = performance.now();
    root.style.display = "block";
    requestAnimationFrame(() => { root.dataset.on = "1"; });
    fetchReady().then(loadAnimation);
    if (!S.raf) S.raf = requestAnimationFrame(frame);
  }
  function hide() {
    if (!S.visible) return;
    const left = Math.max(0, 700 - (performance.now() - S.shownAt));                 // once shown, stay long enough to read
    setTimeout(() => {
      if (S.holds.size || S.forced) return;
      root.dataset.on = "0"; S.visible = false;
      setTimeout(() => { if (!S.visible) root.style.display = "none"; }, 500);
    }, left);
  }
  function refresh() {
    if (S.holds.size || S.forced) {
      if (!S.visible && !S.showTimer) S.showTimer = setTimeout(show, S.visible ? 0 : 250);       // a wait under a quarter second shows nothing
    } else {
      if (S.showTimer) { clearTimeout(S.showTimer); S.showTimer = null; }
      hide();
    }
  }
  const hold = (id, kind, names = []) => { S.holds.set(id, { kind, names }); if (kind === "compiling") poll(); refresh(); };
  const release = (id) => { if (id === "boot" && S.holds.has(id)) S.bootMs = Math.round(performance.now()); S.holds.delete(id); refresh(); };

  let gateCount = 0;
  const api = {
    // The game's page waits here until its products are built (it shows the panel meanwhile).  A server that cannot be reached lets it through.
    // `loaded` maps a product name to a promise for the page's own loading of it (the predictor cores): awaited after the product is built.
    async gate(game, loaded = {}) {
      const data = S.data && Date.now() - (S.stamp || 0) < 1500 ? S.data : await fetchReady();
      S.stamp = Date.now();
      if (!data) return;
      const names = needsOf(data, game, webgpu);
      if (!settled(data, names)) {
        const id = `gate:${game}:${gateCount++}`;
        hold(id, "compiling", names);
        try { await wait(names); } finally { release(id); }
      }
      await Promise.all(names.map((n) => loaded[n]).filter(Boolean));
    },
    // silent: resolves when the products are settled (ready, failed or skipped)
    async whenReady(names) {
      const data = await fetchReady();
      if (!data || settled(data, names)) return;
      await wait(names);
    },
    release,                                                         // release("boot"): the first game is up
    initializing(detail) { S.init = detail || "GPU"; refresh(); },
    done() { S.init = null; },
    state: () => ({ bootMs: S.bootMs, phase: S.phase, holds: [...S.holds.keys()], dl: { ...dl }, visible: S.visible, anim: !!S.anim }),
  };
  window.meowdokuLoading = api;
  hold("boot", "boot");                                              // from the first byte until the first game is drawn
  window.addEventListener("error", () => release("boot"), { once: true });
  if (S.forced) refresh();
  fetchReady().then(loadAnimation);
}
