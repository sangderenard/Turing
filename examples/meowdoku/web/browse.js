// /browse/<root>/<path...> page: fetches ?format=json (schema turing.listing.v1) and renders it.  Everything is set with
// textContent / createElement; no innerHTML, no raw HTML from the narration or from file previews.
const $ = (id) => document.getElementById(id);
const PREVIEW_LIMIT = 512 * 1024;
const state = { doc: null, rows: [], sel: 0, sortKey: "name", sortDir: 1, filter: "" };

function el(tag, props = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) { if (k === "text") node.textContent = v; else node.setAttribute(k, v); }
  for (const kid of kids) node.append(kid);
  return node;
}

// A tiny safe markdown subset: #/##/### headings, paragraphs, - / * / 1. lists, ``` fences, `code` spans.  Nothing else is interpreted.
export function renderMarkdown(text, into) {
  into.replaceChildren();
  const lines = String(text).split(/\r?\n/);
  let para = [], list = null, fence = null;
  const spans = (parent, s) => {
    s.split(/(`[^`]+`)/).forEach((part) => {
      if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) parent.append(el("code", { text: part.slice(1, -1) }));
      else if (part) parent.append(document.createTextNode(part));
    });
  };
  const flushPara = () => { if (para.length) { const p = el("p"); spans(p, para.join(" ")); into.append(p); para = []; } };
  const flushList = () => { list = null; };
  for (const raw of lines) {
    if (fence) { if (raw.trim().startsWith("```")) { into.append(el("pre", { text: fence.join("\n") })); fence = null; } else fence.push(raw); continue; }
    const line = raw.trimEnd();
    if (line.trim().startsWith("```")) { flushPara(); flushList(); fence = []; continue; }
    const h = /^(#{1,3})\s+(.*)$/.exec(line);
    if (h) { flushPara(); flushList(); const n = el("h" + h[1].length); spans(n, h[2]); into.append(n); continue; }
    const li = /^\s*(?:[-*]|\d+[.)])\s+(.*)$/.exec(line);
    if (li) { flushPara(); if (!list) { list = el("ul"); into.append(list); } const item = el("li"); spans(item, li[1]); list.append(item); continue; }
    if (!line.trim()) { flushPara(); flushList(); continue; }
    flushList(); para.push(line.trim());
  }
  if (fence) into.append(el("pre", { text: fence.join("\n") }));
  flushPara();
}

const fmtBytes = (n) => (n == null ? "" : n < 1024 ? `${n}` : n < 1048576 ? `${(n / 1024).toFixed(1)} K` : `${(n / 1048576).toFixed(1)} M`);

function visibleRows() {
  const f = state.filter.toLowerCase();
  const rows = state.doc.entries.filter((e) => !f || [e.name, e.kind, e.content_type || ""].some((s) => s.toLowerCase().includes(f)));
  const k = state.sortKey, d = state.sortDir;
  rows.sort((a, b) => {
    if (a.kind !== b.kind) return a.kind === "dir" ? -1 : 1;
    const x = a[k], y = b[k];
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return (typeof x === "number" ? x - y : String(x).localeCompare(String(y))) * d;
  });
  return rows;
}

function renderTable() {
  state.rows = visibleRows();
  state.sel = Math.min(state.sel, Math.max(0, state.rows.length - 1));
  const body = document.querySelector("#table tbody");
  body.replaceChildren();
  state.rows.forEach((e, i) => {
    const link = el("a", { href: e.href, text: e.kind === "dir" ? e.name + "/" : e.name });
    const tr = el("tr", i === state.sel ? { class: "sel" } : {},
      el("td", {}, link), el("td", { text: e.kind }), el("td", { class: "num", text: fmtBytes(e.bytes) }),
      el("td", { text: e.mtime || "" }), el("td", { text: e.content_type || "" }));
    tr.addEventListener("click", (ev) => { if (ev.target !== link) { ev.preventDefault(); state.sel = i; renderTable(); } });
    body.append(tr);
  });
  renderDetail();
}

async function preview(e, into) {
  const type = e.content_type || "";
  if (e.kind !== "file" || e.bytes == null || e.bytes > PREVIEW_LIMIT) { into.append(el("p", { class: "mut", text: "No inline preview." })); return; }
  if (type.startsWith("image/") && type !== "image/svg+xml") { into.append(el("img", { class: "preview", src: e.href, alt: e.name })); return; }
  if (!(type.startsWith("text/") || type === "application/json")) { into.append(el("p", { class: "mut", text: "No inline preview." })); return; }
  try {
    const r = await fetch(e.href, { credentials: "same-origin" });
    let text = await r.text();
    if (type === "application/json") { try { text = JSON.stringify(JSON.parse(text), null, 2); } catch { /* shown as is */ } }
    if (type === "text/markdown") { const box = el("div"); renderMarkdown(text, box); into.append(box); }
    else into.append(el("pre", { text }));
  } catch (err) { into.append(el("p", { class: "bad", text: String(err) })); }
}

function renderDetail() {
  const box = $("detail");
  const e = state.rows[state.sel];
  box.replaceChildren();
  if (!e) { box.append(el("span", { class: "mut", text: "Nothing selected." })); return; }
  box.append(el("h2", { text: e.name }));
  if (e.sha256) {
    const copy = el("button", { text: "Copy" });
    copy.addEventListener("click", () => navigator.clipboard && navigator.clipboard.writeText(e.sha256));
    box.append(el("p", {}, document.createTextNode("sha256 "), el("code", { text: e.sha256 }), document.createTextNode(" "), copy));
  } else if (e.hash_deferred) box.append(el("p", { class: "mut", text: "hash deferred (request hash budget)" }));
  if (e.kind === "file") box.append(el("p", {}, el("a", { href: e.href, download: e.name, text: "Download" })));
  if (e.describe) box.append(el("h3", { text: "Describes" }), el("pre", { text: JSON.stringify(e.describe, null, 2) }));
  const holder = el("div"); box.append(holder);
  preview(e, holder);
}

function renderCrumbs(doc) {
  const nav = $("crumbs"); nav.replaceChildren();
  doc.parents.forEach((p, i) => { if (i) nav.append(el("span", { class: "sep", text: "/" })); nav.append(el("a", { href: p.href, text: p.name })); });
  const here = doc.path ? doc.path.split("/").pop() : doc.root;
  if (here) { nav.append(el("span", { class: "sep", text: "/" }), el("strong", { text: here })); }
}

function agentPrompt() {
  const url = new URL(location.pathname + "?format=text", location.origin).href;
  return `${url}\nRead this plain-text listing (tab-separated: kind, bytes, sha256 prefix, name, description) and follow the "next:" URLs to walk subdirectories.`;
}

async function load() {
  const url = location.pathname + "?format=json";
  let doc;
  try {
    const r = await fetch(url, { headers: { Accept: "application/json" }, credentials: "same-origin" });
    doc = await r.json();
    if (!r.ok) throw new Error(doc.error || r.statusText);
  } catch (err) { $("status").textContent = String(err.message || err); $("status").className = "bad"; return; }
  state.doc = doc;
  $("status").textContent = `${doc.entries.length} entries` + (doc.truncated ? " (truncated)" : "") + (doc.hash_deferred ? ", some hashes deferred" : "");
  document.title = "Browse " + ([doc.root, doc.path].filter(Boolean).join("/") || "/");
  renderCrumbs(doc);
  if (doc.narration) {
    renderMarkdown(doc.narration.text, $("narration"));
    $("narration").hidden = false;
    $("narration").append(el("p", { class: "mut", text: "source: " + doc.narration.source }));
    if ("speechSynthesis" in window && doc.narration.speak) $("narrate").hidden = false;
  }
  renderTable();
}

$("filter").addEventListener("input", (ev) => { state.filter = ev.target.value; state.sel = 0; renderTable(); });
document.querySelectorAll("th[data-k]").forEach((th) => th.addEventListener("click", () => {
  const k = th.dataset.k; state.sortDir = state.sortKey === k ? -state.sortDir : 1; state.sortKey = k; renderTable();
}));
$("agent").addEventListener("click", () => { if (navigator.clipboard) navigator.clipboard.writeText(agentPrompt()); $("agent").textContent = "Copied"; setTimeout(() => ($("agent").textContent = "Copy as agent prompt"), 1500); });
$("narrate").addEventListener("click", () => {
  if (window.speechSynthesis.speaking) { window.speechSynthesis.cancel(); $("narrate").textContent = "Narrate"; return; }
  const u = new SpeechSynthesisUtterance(state.doc.narration.speak);
  u.onend = () => ($("narrate").textContent = "Narrate");
  $("narrate").textContent = "Stop";
  window.speechSynthesis.speak(u);
});
document.addEventListener("keydown", (ev) => {
  if (ev.target instanceof HTMLInputElement || ev.ctrlKey || ev.metaKey || ev.altKey || !state.doc) return;
  if (ev.key === "j") { state.sel = Math.min(state.rows.length - 1, state.sel + 1); renderTable(); }
  else if (ev.key === "k") { state.sel = Math.max(0, state.sel - 1); renderTable(); }
  else if (ev.key === "Enter") { const e = state.rows[state.sel]; if (e) location.href = e.href; }
  else if (ev.key === "Backspace") { const p = state.doc.parents; if (p.length) location.href = p[p.length - 1].href; }
  else return;
  ev.preventDefault();
});
load();
