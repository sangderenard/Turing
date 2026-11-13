// /nodus.html: Living Data graph picture and read-only UI feeds from Nodus.
const $ = (id) => document.getElementById(id);
const state = { instance: null, revision: null, graphKey: null, tick: null, graph: null, map: null, tools: [], activity: null, abstract: null, busy: false };
let feedQuads = [];

function makeFrameQuad(canvas, status) {
  const gl = canvas.getContext("webgl2", { alpha: false, antialias: false });
  if (!gl) { status.textContent = "WebGL 2 is unavailable for this console."; return null; }
  const shader = (kind, source) => {
    const object = gl.createShader(kind);
    gl.shaderSource(object, source); gl.compileShader(object);
    if (!gl.getShaderParameter(object, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(object));
    return object;
  };
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, `#version 300 es
    out vec2 uv;
    void main() {
      vec2 p = vec2((gl_VertexID & 1) * 2 - 1, (gl_VertexID & 2) - 1);
      uv = vec2((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
      gl_Position = vec4(p, 0.0, 1.0);
    }`));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `#version 300 es
    precision mediump float;
    in vec2 uv; uniform sampler2D frame; out vec4 color;
    void main() { color = texture(frame, uv); }`));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.useProgram(program);
  gl.uniform1i(gl.getUniformLocation(program, "frame"), 0);
  const quad = { url: "", busy: false, etag: null, abort: null, stopped: false,
    stop() { this.stopped = true; this.abort?.abort(); },
    setUrl(url) { this.abort?.abort(); this.url = url; this.etag = null; },
    async update() {
      if (this.stopped || this.busy || !this.url) return;
      this.busy = true;
      const abort = new AbortController(); this.abort = abort;
      try {
        const headers = this.etag ? { "If-None-Match": this.etag } : {};
        const response = await fetch(this.url, { credentials: "same-origin", cache: "no-store", headers, signal: abort.signal });
        if (response.status === 304) return;
        if (!response.ok) throw new Error(`Frame unavailable (${response.status})`);
        const bitmap = await createImageBitmap(await response.blob());
        if (this.stopped || abort.signal.aborted) { bitmap.close(); return; }
        canvas.width = bitmap.width; canvas.height = bitmap.height;
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        bitmap.close();
        this.etag = response.headers.get("ETag"); status.textContent = "Live Nodus frame";
      } catch (error) { if (!abort.signal.aborted) status.textContent = error.message; }
      finally { this.busy = false; if (this.abort === abort) this.abort = null; }
    },
  };
  return quad;
}

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (cls) node.className = cls;
  return node;
}

async function get(path) {
  try {
    const response = await fetch(path, { credentials: "same-origin", cache: "no-store" });
    let body = null;
    try { body = await response.json(); } catch (_) { /* not JSON */ }
    return { status: response.status, body };
  } catch (_) {
    return { status: 0, body: null };
  }
}

const q = (path) => path + "?instance=" + encodeURIComponent(state.instance || "");
const show = (v) => (v === null || v === undefined ? "" : v);

export function digestLine(d) {
  if (!d) return "";
  const sha = String(d.sha256 || "").slice(0, 12);
  return "rev " + show(d.revision) + "  sha " + sha + "  modules " + show(d.modules) + "  rows " + show(d.rows) +
    "  edges " + show(d.edges) + "  contract ports " + show(d.contract_ports) + "  tick " + show(d.tick) + "  rate " + show(d.rate);
}

export function countByKind(nodes) {
  const counts = {};
  for (const node of nodes || []) counts[node.kind] = (counts[node.kind] || 0) + 1;
  return counts;
}

function fill(tableId, rows, cells) {
  const body = $(tableId).tBodies[0];
  body.replaceChildren();
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const value of cells(row)) tr.append(el("td", value));
    body.append(tr);
  }
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = el("td", "none", "mut");
    td.colSpan = $(tableId).tHead.rows[0].cells.length;
    tr.append(td);
    body.append(tr);
  }
}

function setStatus(text, bad) {
  const node = $("status");
  node.textContent = text;
  node.className = bad ? "bad" : "mut";
}

function moduleName(g, index) {
  const m = (g.modules || []).find((x) => x.index === index) || (g.modules || [])[index];
  return m ? m.name || "module " + index : "module " + index;
}

function drawGraph() {
  const g = state.graph;
  if (!g) return;
  fill("modules", g.modules || [], (m) => [m.name, (m.rows || []).map((r) => r.tool).join(", "), m.exec_mode, m.place]);
  const fills = new Map(((state.activity && state.activity.edges) || []).map((e) => [e.index, e]));
  fill("edges", g.edges || [], (e) => {
    const f = fills.get(e.index);
    const end = (x) => (x ? moduleName(g, x.module) + " #" + x.contact : "");
    return [end(e.from), end(e.to), e.token_bytes, e.slots, f ? f.fill : ""];
  });
  fill("ports", g.contract_ports || [], (p) => [p.prefix, p.capacity, p.listen_port, p.loan_ticks, p.edge]);
}

function drawActivity() {
  const a = state.activity;
  fill("hosts", (a && a.compile_hosts) || [], (h) => [h.prefix, h.host_id, h.accepted, h.completed, h.refused, h.acknowledged, h.in_flight]);
  drawGraph();
}

function drawTools() {
  const needle = $("filter").value.trim().toLowerCase();
  const all = state.tools || [];
  const rows = needle ? all.filter((t) => [t.name, t.plugin_id, t.source, t.description].some((v) => String(v || "").toLowerCase().includes(needle))) : all;
  fill("tools", rows, (t) => [t.name, t.plugin_id, t.source, t.description]);
  $("toolcount").textContent = rows.length + " of " + all.length;
}

function drawGeneric() {
  const nodes = (state.abstract && state.abstract.nodes) || [];
  const counts = Object.entries(countByKind(nodes)).map(([k, n]) => n + " " + k).join(", ");
  $("generic").textContent = nodes.length + " nodes (" + counts + "); " +
    nodes.reduce((n, x) => n + (x.dependencies || []).length, 0) + " dependencies";
}

function drawMap() {
  const map = state.map;
  $("maproot").textContent = map ? `${map.name} [${map.identity}]` : "Map unavailable.";
  const nodes = $("mapregions");
  const wall = $("console-wall");
  for (const quad of feedQuads) quad.stop();
  nodes.replaceChildren(); wall.replaceChildren(); feedQuads = [];
  if (!map) return;
  const feeds = (map.tracks || []).filter((t) => t.relationship === "feeds");
  for (const region of map.regions || []) {
    if (region.kind === "console" || region.kind === "console-wall") continue;
    const card = el("div", region.name, "map-node");
    card.title = region.identity;
    const outgoing = feeds.filter((t) => t.source === region.identity).length;
    if (outgoing) card.append(el("small", ` → ${outgoing} feed${outgoing === 1 ? "" : "s"}`, "mut"));
    nodes.append(card);
  }
  const consoles = (map.regions || []).filter((r) => r.kind === "console" && r.view?.full);
  if (!consoles.length) { wall.append(el("p", "No Nodus UI feed is configured for this graph.", "mut")); return; }
  for (const consoleNode of consoles) {
    const card = el("article", null, "console-device card");
    card.dataset.identity = consoleNode.identity;
    card.append(el("strong", consoleNode.name));
    const canvas = el("canvas"); canvas.width = 800; canvas.height = 600;
    canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", consoleNode.name + " UI feed");
    card.append(canvas);
    const status = el("small", "Connecting to Nodus UI feed…", "mut"); card.append(status);
    const quad = makeFrameQuad(canvas, status);
    if (quad) { quad.setUrl(consoleNode.view.full); feedQuads.push(quad); quad.update(); }
    const full = el("a", "Open full view"); full.href = consoleNode.view.full; full.target = "_blank"; full.rel = "noopener";
    card.append(full);
    const form = el("form");
    const fields = {};
    for (const [key, value] of Object.entries({ x: 0, y: 0, width: 320, height: 240 })) {
      const label = el("label", key + " ");
      const input = el("input"); input.type = "number"; input.min = key === "width" || key === "height" ? "1" : "0";
      input.value = String(value); input.required = true;
      label.append(input); form.append(label); fields[key] = input;
    }
    const select = el("button", "Show region"); select.type = "submit"; form.append(select);
    const reset = el("button", "Show whole UI"); reset.type = "button";
    reset.addEventListener("click", () => { if (quad) { quad.setUrl(consoleNode.view.full); quad.update(); } });
    form.append(reset);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const values = Object.fromEntries(Object.entries(fields).map(([key, input]) => [key, Number(input.value)]));
      if (Object.values(values).some((value) => !Number.isInteger(value)) || values.width < 1 || values.height < 1) return;
      if (quad) {
        quad.setUrl(consoleNode.view.region.replace(/\{(x|y|width|height)\}/g, (_, key) => String(values[key])));
        quad.update();
      }
    });
    card.append(form); wall.append(card);
  }
}

function explain(res, what) {
  if (res.status === 501) return what + ": not supported by this host";
  if (res.status === 503) return (res.body && res.body.error) || "the console is unreachable";
  if (res.status === 403) return "this page needs the dev_ui capability";
  return what + ": " + ((res.body && res.body.error) || "failed (" + res.status + ")");
}

async function refresh() {
  if (state.busy || !state.instance) return;
  state.busy = true;
  try {
    const d = await get(q("/api/nodus/digest"));
    if (d.status !== 200) { setStatus(explain(d, "digest"), true); return; }
    setStatus("Connected to " + d.body.instance + ".", false);
    $("digest").textContent = digestLine(d.body);
    const key = state.instance + "@" + d.body.revision;
    if (key !== state.graphKey) {
      const [g, t, a, m] = await Promise.all([get(q("/api/nodus/graph")), get(q("/api/nodus/tools")), get(q("/api/nodus/abstract")), get(q("/api/nodus/map"))]);
      if (g.status === 200) { state.graph = g.body; state.graphKey = key; }
      else setStatus(explain(g, "graph"), true);
      if (t.status === 200) { state.tools = t.body.tools || []; if (t.body.complete === false) setStatus("Tool list is incomplete: " + show(t.body.missing), true); }
      if (a.status === 200) state.abstract = a.body;
      if (m.status === 200) state.map = m.body.map;
      drawTools();
      drawGeneric();
      drawMap();
    }
    const bucket = d.body.tick;
    if (bucket !== state.tick || !state.activity) {
      const act = await get(q("/api/nodus/activity"));
      if (act.status === 200) { state.activity = act.body; state.tick = bucket; }
    }
    drawActivity();
  } finally {
    state.busy = false;
  }
}

async function loadInstances() {
  const res = await get("/api/nodus/instances");
  if (res.status !== 200) { setStatus(explain(res, "instances"), true); return; }
  const list = (res.body.instances || []);
  const select = $("instance");
  select.replaceChildren();
  for (const i of list) {
    const o = el("option", i.name + " (" + i.state + (i.console ? "" : ", no console") + ")");
    o.value = i.name;
    select.append(o);
  }
  const requested = new URLSearchParams(location.search).get("instance");
  const ready = list.find((i) => i.name === requested) || list.find((i) => i.console) || list[0];
  if (!ready) { setStatus("No nodus instance is recorded.", false); return; }
  select.value = ready.name;
  state.instance = ready.name;
  await refresh();
}

function resetFor(name) {
  Object.assign(state, { instance: name, revision: null, graphKey: null, tick: null, graph: null, map: null, tools: [], activity: null, abstract: null });
  drawMap();
}

async function whoamiHasMessage() {
  const res = await get("/api/whoami");
  return res.status === 200 && res.body && (res.body.capabilities || []).includes("message");
}

async function share(event) {
  event.preventDefault();
  const out = $("shared");
  out.textContent = "Sending...";
  try {
    const response = await fetch("/api/nodus/share", {
      method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instance: state.instance, to: $("to").value.trim(), note: $("note").value }),
    });
    let body = null;
    try { body = await response.json(); } catch (_) { /* not JSON */ }
    out.textContent = response.ok ? "Sent." : (body && body.error) || "Failed (" + response.status + ").";
  } catch (_) {
    out.textContent = "Failed.";
  }
}

// ---- Walk: the account's Living Data graph cursor over the canvas (/api/nodus/map, /api/living-data/sessions...).  Graph position only.
const walk = { id: null, revision: null, map: null };
const rid = (prefix) => prefix + "-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);

async function send(path, body) {
  try {
    const response = await fetch(path, {
      method: "POST", credentials: "same-origin", cache: "no-store",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    let data = null;
    try { data = await response.json(); } catch (_) { /* not JSON */ }
    return { status: response.status, body: data };
  } catch (_) {
    return { status: 0, body: null };
  }
}

export function exitLabel(number, item, names) {
  return number + ". " + item.relationship + " " + item.direction + " -> " + ((names && names[item.to]) || item.to) + " [" + item.to + "]";
}

function walkNames() {
  const names = {};
  const m = walk.map;
  if (m) { names[m.identity] = m.name || m.identity; for (const r of m.regions || []) names[r.identity] = r.name || r.identity; }
  return names;
}

function walkSay(text) { $("walkstatus").textContent = text; }

function drawWalk(cursor) {
  const inside = Boolean(cursor);
  $("walkenter").hidden = inside;
  $("walkleave").hidden = !inside;
  $("walkhere").hidden = !inside;
  const box = $("walkexits");
  box.replaceChildren();
  if (!inside) return;
  walk.id = cursor.session_id;
  walk.revision = cursor.revision;
  $("walkhere").textContent = cursor.text;
  if (cursor.at?.kind === "console") {
    const consoleCard = [...$("console-wall").children].find((card) => card.dataset.identity === cursor.at.identity);
    consoleCard?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  const names = walkNames();
  (cursor.exits || []).forEach((item, i) => {
    const b = el("button", exitLabel(i + 1, item, names));
    b.type = "button";
    b.addEventListener("click", () => walkGo(item));
    box.append(b);
  });
}

async function walkEnter() {
  walkSay("Entering...");
  const found = await get(q("/api/nodus/map"));
  if (found.status !== 200) { walkSay(explain(found, "map")); return; }
  const res = await send("/api/living-data/sessions", { map: found.body.map, at: found.body.at, request_id: rid("walk") });
  if (res.status !== 201) {
    walkSay(res.status === 409 ? "This account already has a cursor from another session; it cannot be closed from here." : explain(res, "enter"));
    return;
  }
  walk.map = found.body.map;
  walkSay("");
  drawWalk(res.body);
}

async function walkGo(item) {
  const res = await send("/api/living-data/sessions/" + walk.id + "/move",
    { to: item.to, relationship: item.relationship, direction: item.direction, expected_revision: walk.revision, request_id: rid("move") });
  if (res.status === 200) { walkSay(""); drawWalk(res.body); return; }
  walkSay((res.body && res.body.error) || "Move failed (" + res.status + ").");
  const now = await get("/api/living-data/sessions/" + walk.id);
  if (now.status === 200) drawWalk(now.body); else drawWalk(null);
}

async function walkLeave() {
  const res = await send("/api/living-data/sessions/" + walk.id + "/close", { expected_revision: walk.revision });
  walkSay(res.status === 200 || res.status === 404 ? "Left." : (res.body && res.body.error) || "Failed (" + res.status + ").");
  if (res.status === 200 || res.status === 404) { walk.id = null; drawWalk(null); }
}

async function setupWalk() {
  const res = await get("/api/whoami");
  const who = res.status === 200 && res.body ? res.body : null;
  if (!who || !(who.capabilities || []).includes("dev_ui")) return;
  $("walk").hidden = false;
  if (!who.signed_in) { $("walknote").hidden = false; return; }
  $("walkbar").hidden = false;
  $("walkenter").addEventListener("click", walkEnter);
  $("walkleave").addEventListener("click", walkLeave);
}

async function copyJson() {
  const text = JSON.stringify((state.abstract && state.abstract.nodes) || [], null, 1);
  try { await navigator.clipboard.writeText(text); $("copied").textContent = "Copied."; }
  catch (_) { $("copied").textContent = "Copy was refused by the browser."; }
}

async function main() {
  if (new URLSearchParams(location.search).get("picture") === "1") document.body.classList.add("picture-mode");
  $("instance").addEventListener("change", () => { resetFor($("instance").value); refresh(); });
  $("filter").addEventListener("input", drawTools);
  $("copy").addEventListener("click", copyJson);
  $("shareform").addEventListener("submit", share);
  if (await whoamiHasMessage()) $("share").hidden = false;
  await setupWalk();
  await loadInstances();
  setInterval(() => { if (document.visibilityState === "visible") refresh(); }, 2000);
  setInterval(() => { if (document.visibilityState === "visible") for (const quad of feedQuads) quad.update(); }, 500);
}

if (typeof document !== "undefined" && document.getElementById("instance")) main();
