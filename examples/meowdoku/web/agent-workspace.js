const $ = id => document.getElementById(id);
const kinds = ["orientation", "prompt", "personality", "guidance", "trouble", "scenario"];
let current = null, parent = null, mode = "orientation", editing = null;
async function api(path, body) {
  const response = await fetch(path, {method: body ? "POST" : "GET", credentials: "same-origin", cache: "no-store",
    headers: body ? {"Content-Type": "application/json", "X-Meowdoku-Workspace": "1"} : {}, body: body ? JSON.stringify(body) : undefined});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || String(response.status));
  return value;
}
function notice(message) { $("status").textContent = message; }
function itemButton(label, action) { const button = document.createElement("button"); button.type = "button"; button.textContent = label; button.onclick = action; return button; }
async function branch(id = null) {
  parent = id;
  const query = new URLSearchParams(); if (id) query.set("parent", id); if (!id) query.set("kind", mode);
  const result = await api("/api/agent-workspace/children?" + query);
  $("tree").replaceChildren();
  if (id) $("tree").append(itemButton("← Parent branch", async () => { const row = await api("/api/agent-workspace/document/" + id); await branch(row.parent_id); }));
  for (const row of result.documents) {
    const wrap = document.createElement("div");
    wrap.append(itemButton(row.title + " · " + row.kind, () => open(row.id)), itemButton("Expand", () => branch(row.id)),
      itemButton("Copy ID", () => navigator.clipboard.writeText(row.id).then(() => notice("Copied document ID"))));
    $("tree").append(wrap);
  }
  if (!result.documents.length) $("tree").append("No documents in this branch.");
}
async function open(id) {
  const row = await api("/api/agent-workspace/document/" + id);
  current = row; editing = id; parent = row.parent_id;
  $("kind").value = row.kind; $("title").value = row.title; $("body").value = row.body;
  $("contexts").value = row.contexts.join("\n"); $("scenario").hidden = row.kind !== "scenario";
  $("identity").textContent = "Document " + row.id + " · author account " + row.owner_id;
  notice("Opened " + row.title);
}
function fresh() {
  editing = null; current = null; $("kind").value = mode; $("title").value = ""; $("body").value = "";
  $("contexts").value = ""; $("scenario").hidden = mode !== "scenario"; $("identity").textContent = "New document under " + (parent || "root"); notice("New document");
}
async function save() {
  const data = {action:"save", kind:$("kind").value, title:$("title").value, body:$("body").value,
    parent_id:current ? current.parent_id : parent, contexts:$("contexts").value.split(/\r?\n/).map(s => s.trim()).filter(Boolean)};
  if (editing) data.id = editing;
  const row = await api("/api/agent-workspace/documents", data); await branch(row.parent_id); await open(row.id); notice("Saved");
}
async function remove() {
  if (!editing || !confirm("Delete this document?")) return;
  await api("/api/agent-workspace/documents", {action:"delete", id:editing}); const p = current.parent_id; fresh(); await branch(p); notice("Deleted");
}
async function scratch() {
  const result = await api("/api/scratch"); $("scratch-list").replaceChildren();
  $("quota").textContent = result.files.length + "/" + (result.quota.files || "cookie") + " files";
  for (const file of result.files) $("scratch-list").append(itemButton(file.name + " (" + file.bytes + " bytes)", async () => {
    const value = await api("/api/scratch/file?name=" + encodeURIComponent(file.name));
    $("scratch-name").value = value.name; $("scratch-body").value = value.body;
  }));
}
async function scratchChange(action) {
  await api("/api/scratch", {action, name:$("scratch-name").value, body:$("scratch-body").value});
  if (action === "delete") { $("scratch-name").value = ""; $("scratch-body").value = ""; }
  await scratch(); notice("Scratch " + action + " complete");
}
function guard(fn) { return async () => { try { await fn(); } catch (error) { notice(error.message); } }; }
for (const kind of kinds) {
  $("kind").append(new Option(kind, kind));
  const button = itemButton(kind[0].toUpperCase() + kind.slice(1), guard(async () => {
    mode = kind; for (const other of $("modes").children) other.removeAttribute("aria-current"); button.setAttribute("aria-current", "page");
    await branch(); fresh();
  }));
  if (kind === mode) button.setAttribute("aria-current", "page"); $("modes").append(button);
}
$("kind").onchange = () => $("scenario").hidden = $("kind").value !== "scenario";
$("new").onclick = fresh; $("save").onclick = guard(save); $("delete").onclick = guard(remove);
$("new-child").onclick = () => { if (!editing) return notice("Open a parent document first"); const child = editing; fresh(); parent = child; $("identity").textContent = "New child under " + child; };
$("parent").onclick = guard(async () => { if (current?.parent_id) await open(current.parent_id); });
$("full").onclick = () => { const full = $("workspace").classList.toggle("full"); $("full").textContent = full ? "Restore panes" : "Full screen"; };
$("scratch-save").onclick = guard(() => scratchChange("write")); $("scratch-delete").onclick = guard(() => scratchChange("delete"));
guard(async () => { const result = await api("/api/agent-workspace"); $("orientation").textContent = result.orientation; await branch(); fresh(); await scratch(); })();
