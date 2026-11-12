// /profile.html: the signed-in account's card and, for admin/operator, a READ-ONLY dev-server lease dashboard (/api/profile/*).
// Everything is set with textContent / createElement; no innerHTML.  There are no action buttons: start/stop/sync do not exist here yet.
const $ = (id) => document.getElementById(id);
const state = { selected: null };

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (cls) node.className = cls;
  return node;
}

async function get(path) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store" });
  let body = null;
  try { body = await response.json(); } catch (_) { /* not JSON */ }
  return { status: response.status, body };
}

const when = (t) => (t ? new Date(t * 1000).toLocaleString() : "");

export function gitText(g) {
  if (!g) return "";
  if (g.error) return "git unavailable: " + g.error;
  let text = (g.branch || "?") + " @ " + (g.commit || "?");
  if (g.dirty) text += ", dirty";
  if (g.upstream_known) text += ", +" + g.ahead + "/-" + g.behind;
  return text;
}

function fill(tableId, rows, cells, onRow) {
  const body = $(tableId).tBodies[0];
  body.replaceChildren();
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const value of cells(row)) tr.append(el("td", value));
    if (onRow) tr.addEventListener("click", () => onRow(row, tr));
    body.append(tr);
  }
  if (!rows.length) { const tr = document.createElement("tr"); const td = el("td", "none", "mut"); td.colSpan = 7; tr.append(td); body.append(tr); }
}

function renderMe(me) {
  const card = $("me");
  card.replaceChildren(
    el("div", me.display_name + (me.display_name !== me.username ? " (" + me.username + ")" : "")),
    el("div", "Role: " + (me.roles || []).join(", "), "mut"),
    el("div", "Preferred site: " + (me.site_preference || "not set"), "mut"),
    el("div", "Capabilities: " + (me.capabilities || []).join(", "), "mut"),
  );
}

async function showDiagnostics(name) {
  state.selected = name;
  $("diag-name").textContent = "(" + name + ")";
  const { status, body } = await get("/api/profile/diagnostics?name=" + encodeURIComponent(name) + "&lines=80");
  if (state.selected !== name) return;
  if (status !== 200) { $("diag").textContent = (body && body.error) || "HTTP " + status; return; }
  $("diag").textContent = body.log ? body.log + "\n\n" + body.lines.join("\n") : "No log file for this instance yet.";
}

async function refresh() {
  const mine = await get("/api/profile/me");
  if (mine.status !== 200) {
    $("status").textContent = "Sign in to see your profile.";
    $("status").className = "bad";
    $("me").replaceChildren(el("a", "Sign in")); $("me").firstChild.href = "/login.html";
    return;
  }
  renderMe(mine.body);
  const inst = await get("/api/profile/instances");
  if (inst.status !== 200) { $("status").textContent = "Signed in."; $("admin").hidden = true; return; }
  $("admin").hidden = false;
  fill("instances", inst.body.instances, (r) => [
    r.name + (r.enabled === false ? " (disabled)" : ""), r.kind, r.state,
    (r.port ? ":" + r.port : "") + (r.route ? " route " + r.route : ""),
    gitText(r.git), (r.pid || "-") + " / " + r.restarts, when(r.started),
  ], (r, tr) => {
    for (const other of $("instances").tBodies[0].rows) other.classList.remove("sel");
    tr.classList.add("sel");
    showDiagnostics(r.name);
  });
  const leases = await get("/api/profile/leases");
  if (leases.status === 200) {
    fill("leases", leases.body.leases, (r) => [r.owner, r.table, r.purpose || "", when(r.taken), when(r.expires), r.state]);
  }
  $("status").textContent = "Updated " + new Date().toLocaleTimeString();
  $("status").className = "mut";
}

$("refresh").addEventListener("click", refresh);
setInterval(() => { if (document.visibilityState === "visible") refresh(); }, 15000);
refresh();
