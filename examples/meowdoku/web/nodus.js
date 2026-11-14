// /nodus.html: Living Data graph picture and opt-in actions on Nodus UI feeds.
const $ = (id) => document.getElementById(id);
const state = { instance: null, revision: null, graphKey: null, tick: null, graph: null, map: null, tools: [], activity: null, abstract: null, busy: false, canEdit: false, forwarding: null };
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
  const quad = { url: "", busy: false, etag: null, abort: null, stopped: false, hasFrame: false, fullView: true,
    stop() { this.stopped = true; this.abort?.abort(); },
    setUrl(url, fullView = true) { this.abort?.abort(); this.url = url; this.etag = null; this.hasFrame = false; this.fullView = fullView; },
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
        this.etag = response.headers.get("ETag"); this.hasFrame = true;
        if (!canvas.classList.contains("forwarding")) status.textContent = "Live Nodus frame";
      } catch (error) { if (!abort.signal.aborted) { this.hasFrame = false; status.textContent = error.message; } }
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

function attachForwarding(card, canvas, status, quad, identity) {
  if (!state.canEdit) return;
  const label = el("label", null, "forward-toggle");
  const toggle = el("input"); toggle.type = "checkbox";
  toggle.checked = state.forwarding === identity;
  label.append(toggle, document.createTextNode("Forward my mouse and keyboard actions to this canvas"));
  card.append(label);
  let revision = state.revision, draining = false, lastPoint = null;
  const queue = [];
  const instance = state.instance;
  // Every button and key the canvas was told is down, with the action that lets it go. A release is never dropped: when
  // forwarding stops for any reason (switched off, refused, another canvas, another instance, focus lost) whatever is still
  // held is released, so no drag or key stays down on the canvas.
  const held = new Map();
  const releaseOf = (action) => action.op === "mouse" ? { op: "mouse", kind: "up", x: action.x, y: action.y, dx: 0, dy: 0, button: action.button }
    : { op: "key", key: action.key, scancode: 0, action: 0, mods: action.mods };
  const heldKey = (action) => action.op === "mouse" ? `b${action.button}` : `k${action.key}`;
  const isPress = (action) => (action.op === "mouse" && action.kind === "down") || (action.op === "key" && action.action === 1);
  const isRelease = (action) => (action.op === "mouse" && action.kind === "up") || (action.op === "key" && action.action === 0);
  const active = () => state.forwarding === identity && state.instance === instance && card.isConnected;
  // Releases go straight to the canvas, outside the queue and whatever the page's state is now; a release refused for a
  // stale revision is tried once more at the current one.
  const release = async (actions) => {
    for (const action of actions) {
      for (let attempt = 0; attempt < 2; attempt++) {
        const result = await send("/api/nodus/action", { instance, action: { ...action, expected_revision: revision } });
        if (result.status === 200 && !result.body?.error) { revision = result.body.revision; if (state.instance === instance) state.revision = revision; break; }
        const found = await get("/api/nodus/digest?instance=" + encodeURIComponent(instance || ""));
        if (found.status !== 200 || found.body.instance !== instance) break;
        revision = found.body.revision;
      }
    }
  };
  const releaseAll = () => {
    const actions = [...held.values()];
    held.clear();
    if (actions.length) release(actions);
  };
  const stop = (message) => {
    if (state.forwarding === identity) state.forwarding = null;
    toggle.checked = false; canvas.classList.remove("forwarding");
    queue.length = 0;
    releaseAll();
    if (message) status.textContent = message;
    state.graphKey = null;
  };
  state.releaseForwarding = () => { if (state.forwarding === identity || held.size) stop(); };
  const drain = async () => {
    if (draining) return;
    draining = true;
    try {
      while (queue.length && active()) {
        const action = queue.shift();
        const result = await send("/api/nodus/action", {
          instance, action: { ...action, expected_revision: revision },
        });
        if (result.status !== 200 || result.body?.error) {
          stop((result.body && result.body.error) || `Action refused (${result.status}); refresh before retrying.`);
          break;
        }
        revision = result.body.revision; state.revision = revision;
        status.textContent = result.body.handled ? "Action applied to running canvas" : "Action reached canvas; no change";
      }
      if (!active() && held.size) stop();       // forwarding ended while actions waited: let go of what is held
    } finally { draining = false; }
  };
  const enqueue = (action) => {
    if (!active()) {                            // no longer forwarding: a release of something held still goes through
      if (isRelease(action) && held.delete(heldKey(action))) release([action]);
      return;
    }
    if (isPress(action)) held.set(heldKey(action), releaseOf(action));
    else if (isRelease(action)) held.delete(heldKey(action));
    if (action.kind === "move" && queue.at(-1)?.kind === "move") queue[queue.length - 1] = action;
    else if (queue.length < 64) queue.push(action);
    else if (isRelease(action)) {               // full: a release takes the place of the moves, never the other way
      const kept = queue.filter((queued) => queued.kind !== "move");
      queue.length = 0; queue.push(...kept, action);
    } else { stop("Action queue full; refresh before retrying."); return; }
    drain();
  };
  const point = (event) => {
    const bounds = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(canvas.width - 1, (event.clientX - bounds.left) * canvas.width / bounds.width));
    const y = Math.max(0, Math.min(canvas.height - 1, (event.clientY - bounds.top) * canvas.height / bounds.height));
    const next = { x, y, dx: lastPoint ? x - lastPoint.x : 0, dy: lastPoint ? y - lastPoint.y : 0 };
    lastPoint = { x, y };
    return next;
  };
  toggle.addEventListener("change", async () => {
    if (!toggle.checked) { stop("Action forwarding off"); return; }
    if (!quad || !quad.hasFrame || !quad.fullView) {
      stop("Show the live whole UI before enabling actions."); return;
    }
    const found = await get(q("/api/nodus/digest"));
    if (!toggle.checked || !card.isConnected || state.instance !== instance || found.status !== 200 || found.body.instance !== instance) {
      stop("Could not verify the current canvas revision."); return;
    }
    revision = found.body.revision;
    const viewport = await send("/api/nodus/action", { instance,
      action: { op: "viewport", expected_revision: revision, width: canvas.width, height: canvas.height } });
    if (!toggle.checked || !card.isConnected || state.instance !== instance || viewport.status !== 200 || viewport.body?.error) {
      stop((viewport.body && viewport.body.error) || "Could not synchronize the canvas viewport."); return;
    }
    revision = viewport.body.revision; state.revision = revision;
    for (const other of $("console-wall").querySelectorAll(".console-device")) {
      if (other === card) continue;
      const previous = other.querySelector(".forward-toggle input");
      if (previous?.checked) previous.click();
    }
    state.forwarding = identity; canvas.classList.add("forwarding"); canvas.tabIndex = 0;
    status.textContent = "Action forwarding on; focus the frame for keyboard controls.";
  });
  canvas.addEventListener("pointerdown", (event) => {
    if (!active()) return;
    event.preventDefault(); canvas.focus(); canvas.setPointerCapture(event.pointerId);
    enqueue({ op: "mouse", kind: "down", ...point(event), button: event.button });
  });
  canvas.addEventListener("pointermove", (event) => {
    if (active() && event.buttons) { event.preventDefault(); enqueue({ op: "mouse", kind: "move", ...point(event), button: event.button < 0 ? 0 : event.button }); }
  });
  canvas.addEventListener("pointerup", (event) => {
    if (!active() && !held.has(`b${event.button}`)) return;
    event.preventDefault(); enqueue({ op: "mouse", kind: "up", ...point(event), button: event.button });
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  });
  for (const name of ["pointercancel", "lostpointercapture"]) canvas.addEventListener(name, () => {
    for (const [key, action] of [...held]) if (key.startsWith("b")) enqueue(action);   // the pointer went away mid-press
  });
  canvas.addEventListener("blur", () => {
    for (const [key, action] of [...held]) if (key.startsWith("k")) enqueue(action);   // focus left with a key down
  });
  canvas.addEventListener("click", (event) => {
    if (active()) { event.preventDefault(); enqueue({ op: "mouse", kind: "click", ...point(event), button: event.button }); }
  });
  canvas.addEventListener("wheel", (event) => {
    if (!active()) return;
    event.preventDefault(); enqueue({ op: "mouse", kind: "scroll", ...point(event), scroll: Math.max(-10, Math.min(10, -event.deltaY / 100)) });
  }, { passive: false });
  canvas.addEventListener("contextmenu", (event) => { if (active()) event.preventDefault(); });
  for (const [name, action] of [["keydown", 1], ["keyup", 0]]) canvas.addEventListener(name, (event) => {
    if (!event.keyCode || (!active() && !(action === 0 && held.has(`k${event.keyCode}`)))) return;
    event.preventDefault();
    const mods = (event.shiftKey ? 1 : 0) | (event.ctrlKey ? 2 : 0) | (event.altKey ? 4 : 0) | (event.metaKey ? 8 : 0);
    enqueue({ op: "key", key: event.keyCode, scancode: 0, action, mods });
  });
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
    attachForwarding(card, canvas, status, quad, consoleNode.identity);
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
        if (state.forwarding === consoleNode.identity) card.querySelector(".forward-toggle input").click();
        quad.setUrl(consoleNode.view.region.replace(/\{(x|y|width|height)\}/g, (_, key) => String(values[key])), false);
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
      if (!state.forwarding) drawMap();
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
  state.releaseForwarding?.();                  // another instance: let the old canvas go of anything still held
  Object.assign(state, { instance: name, revision: null, graphKey: null, tick: null, graph: null, map: null, tools: [], activity: null, abstract: null, forwarding: null });
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
  state.canEdit = !!(who && (who.capabilities || []).includes("nodus_edit"));
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
