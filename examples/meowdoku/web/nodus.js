// /nodus.html: READ-ONLY view of the graph of a running Nodus canvas host (/api/nodus/*).  Structure only; no canvas is drawn and nothing here
// can change one.  Everything is set with textContent / createElement; no innerHTML.
const $ = (id) => document.getElementById(id);
const state = { instance: null, revision: null, graphKey: null, tick: null, graph: null, tools: [], activity: null, abstract: null, busy: false };

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
      const [g, t, a] = await Promise.all([get(q("/api/nodus/graph")), get(q("/api/nodus/tools")), get(q("/api/nodus/abstract"))]);
      if (g.status === 200) { state.graph = g.body; state.graphKey = key; }
      else setStatus(explain(g, "graph"), true);
      if (t.status === 200) { state.tools = t.body.tools || []; if (t.body.complete === false) setStatus("Tool list is incomplete: " + show(t.body.missing), true); }
      if (a.status === 200) state.abstract = a.body;
      drawTools();
      drawGeneric();
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
  const ready = list.find((i) => i.console) || list[0];
  if (!ready) { setStatus("No nodus instance is recorded.", false); return; }
  select.value = ready.name;
  state.instance = ready.name;
  await refresh();
}

function resetFor(name) {
  Object.assign(state, { instance: name, revision: null, graphKey: null, tick: null, graph: null, tools: [], activity: null, abstract: null });
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

async function copyJson() {
  const text = JSON.stringify((state.abstract && state.abstract.nodes) || [], null, 1);
  try { await navigator.clipboard.writeText(text); $("copied").textContent = "Copied."; }
  catch (_) { $("copied").textContent = "Copy was refused by the browser."; }
}

async function main() {
  $("instance").addEventListener("change", () => { resetFor($("instance").value); refresh(); });
  $("filter").addEventListener("input", drawTools);
  $("copy").addEventListener("click", copyJson);
  $("shareform").addEventListener("submit", share);
  if (await whoamiHasMessage()) $("share").hidden = false;
  await loadInstances();
  setInterval(() => { if (document.visibilityState === "visible") refresh(); }, 2000);
}

if (typeof document !== "undefined" && document.getElementById("instance")) main();
