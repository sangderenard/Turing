const $ = id => document.getElementById(id);
const kinds = ["orientation", "prompt", "personality", "guidance", "trouble", "scenario"];
let current = null, parent = null, mode = "orientation", editing = null;
let activeProfile = null, activeJob = null, activeInstance = null;
const ids = value => value.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
function showProfile() { $("profile-editor").hidden = false; $("document-editor").hidden = true; $("decision-editor").hidden = true; }
function showDocument() { $("profile-editor").hidden = true; $("document-editor").hidden = false; $("decision-editor").hidden = true; }
function showDecisions() { $("profile-editor").hidden = true; $("document-editor").hidden = true; $("decision-editor").hidden = false; }
async function api(path, body) {
  const response = await fetch(path, {method: body ? "POST" : "GET", credentials: "same-origin", cache: "no-store",
    headers: body ? {"Content-Type": "application/json", "X-Meowdoku-Workspace": "1"} : {}, body: body ? JSON.stringify(body) : undefined})
    .catch(() => { throw new Error("could not reach the server"); });
  // A stopped stable answers with Apache's HTML 503 page: report its status, not a JSON parse error (ticket dd8efbc1).
  const text = await response.text(); let value = null;
  try { value = text ? JSON.parse(text) : {}; } catch { value = null; }
  if (!response.ok) throw new Error(response.status + " " + (response.statusText || "error") + (value?.error ? ": " + value.error : ""));
  if (value === null) throw new Error("the server answered " + response.status + " with a page that is not JSON");
  return value;
}
function notice(message, error = false) { $("status").textContent = message; $("status").classList.toggle("error", error); }
// The editor session: fresh(), fill() and restoreDraft() each put a different document in the editor and count it.
// A save's or delete's answer may change editing/current/parent, the draft or the editor only while the session it
// started in is still the one shown (reviews 962c9145 and 53b3ca65, comments 94-97 and 102-104).
let session = 0;
// Documents deleted from this page, so a later Save under one of them says plainly why it failed (comment 104).
const deleted = new Set();
// The document being typed is kept in this browser until the server confirms a save or delete of exactly that text,
// so a failed save or a reload while the server is down loses nothing (ticket dd8efbc1). Only Save sends it anywhere.
const DRAFT = "agent-workspace-draft";
function editorState() { return JSON.stringify([$("kind").value, $("title").value, $("body").value, $("contexts").value]); }
function keepDraft() {
  try { localStorage.setItem(DRAFT, JSON.stringify({editing, parent, kind:$("kind").value, title:$("title").value,
    body:$("body").value, contexts:$("contexts").value})); } catch {}
}
function dropDraft() { try { localStorage.removeItem(DRAFT); } catch {} }
function restoreDraft() {
  let draft = null; try { draft = JSON.parse(localStorage.getItem(DRAFT) || "null"); } catch {}
  if (!draft || !(draft.title || draft.body)) return false;
  session++; showDocument(); editing = draft.editing || null; current = null; parent = draft.parent || null;
  $("kind").value = draft.kind; $("title").value = draft.title; $("body").value = draft.body;
  $("contexts").value = draft.contexts || ""; $("scenario").hidden = draft.kind !== "scenario";
  $("identity").textContent = editing ? "Unsaved changes to document " + editing : "New document under " + (parent || "root");
  notice("Restored an unsaved draft from this browser; press Save to keep it."); return true;
}
function itemButton(label, action) { const button = document.createElement("button"); button.type = "button"; button.textContent = label; button.onclick = action; return button; }
// branch(id) shows a branch and makes it where "New here" puts documents. branch(id, false) only redraws the tree and
// leaves `parent` alone (the refresh after a save or delete). Only the latest request draws, so a slow answer cannot
// replace the branch the visitor moved to while it was pending.
// `shown` is the branch the tree last drew (null for the root list of `mode`; undefined before the first draw), and
// `inFlight` is the latest tree request while it is still pending (null when none is). `parent` and `shown` change
// together, only when the latest request draws: a rejected or superseded navigation leaves both where they were, so
// "New here" always targets the branch on screen (review 41bcd53c, comments 118-123). A navigation does not take
// `parent` back from fill(), restoreDraft() or "New child" if one of them set it while the request was pending.
// `stale` holds every branch a Save committed into while a tree request was pending (the root branch is null, a valid
// member). That pending listing may predate the Save, whatever branch it is going to, so when the latest request
// settles, succeeded or rejected, and the branch on screen is in `stale`, the tree is fetched once more; then `stale`
// is cleared (reviews e2b02b66 and 6198d83c, comments 127-132 and 140-144). Clearing the other marks is safe: moving
// to one of those branches later fetches a fresh listing. A superseded request leaves `stale` to the newer one.
let treeRequest = 0, shown, inFlight = null;
const stale = new Set();
async function branch(id = null, navigate = true) {
  const request = ++treeRequest, before = parent; inFlight = request;
  const query = new URLSearchParams(); if (id) query.set("parent", id); if (!id) query.set("kind", mode);
  let result, failure = null;
  try { result = await api("/api/agent-workspace/children?" + query); }
  catch (error) { failure = error; }
  finally { if (inFlight === request) inFlight = null; }
  if (request !== treeRequest) { if (failure) throw failure; return; }
  if (!failure) {
    shown = id; if (navigate && parent === before) parent = id;
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
  const again = shown !== undefined && stale.has(shown); stale.clear();
  if (again) try { await branch(shown, false); } catch (error) { notice("The tree did not refresh after the save: " + error.message, true); }
  if (failure) throw failure;
}
async function open(id) {
  showDocument();
  const row = await api("/api/agent-workspace/document/" + id);
  fill(row); notice("Opened " + row.title);
}
function fill(row) {                                   // synchronous: nothing can be typed between these writes
  session++; showDocument(); current = row; editing = row.id; parent = row.parent_id;
  $("kind").value = row.kind; $("title").value = row.title; $("body").value = row.body;
  $("contexts").value = row.contexts.join("\n"); $("scenario").hidden = row.kind !== "scenario";
  $("identity").textContent = "Document " + row.id + " · author account " + row.owner_id;
}
function fresh() {
  session++; showDocument();
  editing = null; current = null; $("kind").value = mode; $("title").value = ""; $("body").value = "";
  $("contexts").value = ""; $("scenario").hidden = mode !== "scenario"; $("identity").textContent = "New document under " + (parent || "root"); notice("New document");
}
// Typing goes on while Save waits for the server and the tree refresh (review ced45d8b, comments 80 and 82), and the
// visitor may open or start another document meanwhile (review 962c9145, comments 94-97). So:
// - the answer touches editing/current/parent, the draft and the editor only while `session` is the one Save started in;
// - the editor is replaced by the saved document, and the draft dropped, only while it still holds exactly the submitted
//   text; newer edits stay in the editor and the draft, attached to the saved id, so the next Save updates that record;
// - the tree refresh after a save never assigns `parent`.
let saving = false;
async function save() {
  if (saving) return notice("A save is already waiting for the server.");
  const submitted = editorState(), started = session;
  const data = {action:"save", kind:$("kind").value, title:$("title").value, body:$("body").value,
    parent_id:current ? current.parent_id : parent, contexts:$("contexts").value.split(/\r?\n/).map(s => s.trim()).filter(Boolean)};
  if (editing) data.id = editing;
  if (deleted.has(data.parent_id)) {                   // the parent was deleted from this page (comment 104, step e)
    keepDraft();
    throw new Error("Not saved: its parent document " + data.parent_id + " was deleted. Your text is still here and in this " +
      "browser's draft. Open another branch in the tree (Expand, or ← Parent branch), then press Save to put it there.");
  }
  let row; saving = true; $("save").disabled = true;
  try { row = await api("/api/agent-workspace/documents", data); }
  catch (error) {
    if (session !== started)
      throw new Error("Not saved: \"" + data.title + "\" (" + error.message + "). You had already moved to another document, so its text is no longer in the editor.");
    keepDraft();
    throw new Error("Not saved: " + error.message + ". Your title and body are still here and kept in this browser's draft; press Save again.");
  }
  finally { saving = false; $("save").disabled = false; }
  if (session !== started) {                           // the editor has switched (review 3817e271, comments 112-115)
    // Leave editing, current, parent and the other document's draft alone; only the tree may change. If a tree request
    // is pending, its listing may predate this Save wherever it is going (comment 140): mark the saved document's
    // branch in `stale`, and branch() fetches it once more if that is where the tree ends up. Otherwise refresh now,
    // only when the branch on screen is where the saved document lives.
    const home = row.parent_id || null; let refreshed = "";
    if (home || row.kind === mode) {
      if (inFlight !== null) stale.add(home);
      else if (shown === home) try { await branch(home, false); } catch (error) { refreshed = " (the tree did not refresh: " + error.message + ")"; }
    }
    return notice("Saved " + row.title + refreshed + ". The document you moved to was left as it is.");
  }
  current = row; editing = row.id; parent = row.parent_id;
  if (editorState() === submitted) dropDraft(); else keepDraft();
  if (!row.parent_id && row.kind !== mode) {     // a root document is listed under its own kind
    mode = row.kind; for (const other of $("modes").children) other.removeAttribute("aria-current");
    $("modes").children[kinds.indexOf(mode)]?.setAttribute("aria-current", "page");
  }
  let refreshed = "";
  try { await branch(row.parent_id, false); } catch (error) { refreshed = " (the tree did not refresh: " + error.message + ")"; }
  if (session !== started) return notice("Saved " + row.title + refreshed + ". The document you moved to was left as it is.");
  if (editorState() === submitted) { dropDraft(); fill(row); notice("Saved " + row.title + refreshed); return; }
  keepDraft(); $("identity").textContent = "Unsaved changes to document " + row.id;
  notice("Saved " + row.title + refreshed + ". What you typed since is still here and in this browser's draft; press Save to keep it.");
}
// The same rule for Delete (review 53b3ca65, comments 102-104): if the visitor opened or started another document while
// the delete was pending, its answer leaves that editor, its draft and `parent` alone, and only redraws the tree.
let removing = false;
async function remove() {
  if (removing) return notice("A delete is already waiting for the server.");
  if (!editing || !confirm("Delete this document?")) return;
  const id = editing, title = $("title").value, started = session, p = current ? current.parent_id : parent;
  removing = true; $("delete").disabled = true;
  try { await api("/api/agent-workspace/documents", {action:"delete", id}); }
  finally { removing = false; $("delete").disabled = false; }
  deleted.add(id);
  if (session === started) { dropDraft(); parent = p; fresh(); await branch(p); notice("Deleted " + title); return; }
  let note = "Deleted " + title + ". The document you moved to was left as it is.";
  if (editing === id) {                                // A was opened again while its delete was pending
    editing = null; current = null; keepDraft(); $("identity").textContent = "Document " + id + " was deleted; Save makes a new one";
    note = "Deleted " + title + ". Its text is still in the editor; Save makes a new document under its old parent.";
  } else if (parent === id && !current) {              // B is a new draft under the deleted document (step e)
    $("identity").textContent = "New document under " + id + " (deleted)";
    note += " Its parent was the deleted document: open another branch in the tree, then press Save to put it there.";
  }
  try { await branch(parent === id ? p : parent, false); } catch (error) { note += " (the tree did not refresh: " + error.message + ")"; }
  notice(note);
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
async function memories() {
  const result = await api("/api/memory"); $("memory-list").replaceChildren();
  for (const file of result.files) $("memory-list").append(itemButton(file.title + " · " + file.id, guard(async () => {
    const row = await api("/api/memory/file/" + file.id);
    $("memory-id").value = row.id; $("memory-title").value = row.title; $("memory-body").value = row.body;
  })));
}
async function memoryChange(action) {
  const data = {action, id:$("memory-id").value || undefined};
  if (action === "write") { data.title = $("memory-title").value; data.body = $("memory-body").value; data.create = true; }
  if (action === "transfer") data.to = $("memory-to").value;
  if (action !== "write" && !data.id) throw new Error("Choose a memory ID first");
  const row = await api("/api/memory", data);
  if (action === "write") $("memory-id").value = row.id;
  else { $("memory-id").value = ""; $("memory-title").value = ""; $("memory-body").value = ""; }
  await memories(); notice("Memory " + action + " complete");
}
async function profiles() {
  const result = await api("/api/agent-workspace/profiles"); $("profile-list").replaceChildren();
  for (const row of result.profiles) $("profile-list").append(itemButton(row.title + " · " + row.individual, guard(() => openProfile(row.id))));
}
async function openProfile(id) {
  const row = await api("/api/agent-workspace/profile/" + id); activeProfile = row; activeJob = null; activeInstance = null; showProfile();
  $("profile-title").value = row.title; $("profile-individual").value = row.individual;
  $("profile-documents").value = row.document_ids.join("\n"); $("profile-resolved").value = "";
  showCapacity(row.capacity);
  $("job-list").replaceChildren();
  for (const job of row.jobs) $("job-list").append(itemButton(job.title + " · " + job.job_key, guard(() => openJob(job.id))));
  freshJob(); freshInstance(); notice("Opened character " + row.title);
}
function freshProfile() {
  activeProfile = null; activeJob = null; activeInstance = null; showProfile();
  $("profile-title").value = ""; $("profile-individual").value = "";
  $("profile-documents").value = ""; $("profile-resolved").value = "";
  showCapacity(null);
  $("job-list").replaceChildren(); freshJob(); freshInstance(); notice("New character");
}
function showCapacity(declared) {                       // [CAPACITY] stateful or stateless, as declared
  $("capacity-state").value = declared ? declared.state : ""; $("capacity-tokens").value = declared ? declared.context_tokens : "";
  $("capacity-tools").value = declared ? declared.tools : "own"; $("capacity-session").value = declared ? declared.session : "none";
  $("capacity-note").value = declared ? declared.note : "";
  $("capacity-declared").textContent = declared ? "Declared " + new Date(declared.declared_at * 1000).toLocaleString() : "Not declared";
}
// [DECISIONS] the decision graphs: places with acts, and routes labelled by their conditions
let graph = null, activePlace = null;
async function openGraph() {
  showDecisions();
  graph = await api("/api/agent-workspace/decisions?graph=" + encodeURIComponent($("graph-name").value.trim() || "line"));
  $("place-list").replaceChildren();
  for (const place of graph.places) $("place-list").append(itemButton((place.start ? "▶ " : "") + place.title + " · " + place.key, () => openPlace(place.key)));
  if (activePlace && graph.places.some(p => p.key === activePlace)) openPlace(activePlace); else freshPlace();
  notice("Graph " + graph.graph + ": " + graph.places.length + " places, " + graph.routes.length + " routes");
}
function freshPlace() {
  activePlace = null; for (const id of ["place-key", "place-title", "place-body", "place-acts", "place-routes"]) $(id).value = "";
  $("place-start").checked = false;
}
function openPlace(key) {
  const place = graph.places.find(p => p.key === key); activePlace = key;
  $("place-key").value = place.key; $("place-title").value = place.title; $("place-body").value = place.body;
  $("place-acts").value = place.acts.map(a => a.tool + ": " + a.for).join("\n"); $("place-start").checked = place.start;
  $("place-routes").value = graph.routes.filter(r => r.source === key).map(r => r.condition + " -> " + r.target).join("\n");
}
async function savePlace() {
  const acts = $("place-acts").value.split("\n").map(line => line.trim()).filter(Boolean).map(line => {
    const at = line.indexOf(":"); return at < 0 ? {tool: line, for: ""} : {tool: line.slice(0, at).trim(), for: line.slice(at + 1).trim()};
  });
  await api("/api/agent-workspace/decisions", {action: "place", graph: graph.graph, key: $("place-key").value.trim(),
    title: $("place-title").value, body: $("place-body").value, acts, start: $("place-start").checked});
  activePlace = $("place-key").value.trim(); await openGraph(); notice("Saved place " + activePlace);
}
async function deletePlace() {
  if (!activePlace || !confirm("Delete this place and every route to or from it?")) return;
  await api("/api/agent-workspace/decisions", {action: "delete_place", graph: graph.graph, key: activePlace});
  activePlace = null; await openGraph();
}
async function saveRoutes() {
  if (!activePlace) return notice("Open or save a place first");
  const wanted = $("place-routes").value.split("\n").map(line => line.trim()).filter(Boolean).map(line => {
    const at = line.lastIndexOf("->"); if (at < 0) throw new Error("Each route is: condition -> place key");
    return {condition: line.slice(0, at).trim(), target: line.slice(at + 2).trim()};
  });
  for (const old of graph.routes.filter(r => r.source === activePlace && !wanted.some(w => w.condition === r.condition)))
    await api("/api/agent-workspace/decisions", {action: "delete_route", graph: graph.graph, source: activePlace, condition: old.condition});
  for (const route of wanted)
    await api("/api/agent-workspace/decisions", {action: "route", graph: graph.graph, source: activePlace, ...route});
  await openGraph(); notice("Saved " + wanted.length + " routes from " + activePlace);
}
async function assignGraph() {
  if (!activeProfile) return notice("Open a character first");
  const name = $("profile-graph").value.trim();
  const result = await api("/api/agent-workspace/decisions", {action: "assign", profile_id: activeProfile.id, graph: name || null});
  notice(activeProfile.title + " walks the decision graph " + result.graph);
}
async function saveCapacity() {
  if (!activeProfile) return notice("Open or save a character first");
  if (!$("capacity-state").value) return notice("Choose stateful or stateless");
  const row = await api("/api/agent-workspace/profiles", {action:"capacity", id:activeProfile.id, state:$("capacity-state").value,
    context_tokens:Number($("capacity-tokens").value), tools:$("capacity-tools").value, session:$("capacity-session").value,
    note:$("capacity-note").value});
  activeProfile = row; showCapacity(row.capacity); notice("Saved how " + row.title + " is run");
}
async function saveProfile() {
  const data = {action:"save", title:$("profile-title").value, individual:$("profile-individual").value,
    document_ids:ids($("profile-documents").value)};
  if (activeProfile) data.id = activeProfile.id;
  const row = await api("/api/agent-workspace/profiles", data); await profiles(); await openProfile(row.id); notice("Saved profile");
}
async function deleteProfile() {
  if (!activeProfile || !confirm("Delete this profile and its job changes?")) return;
  await api("/api/agent-workspace/profiles", {action:"delete", id:activeProfile.id}); await profiles(); freshProfile(); notice("Deleted profile");
}
function freshJob() {
  activeJob = null; for (const key of ["job-key", "job-title", "job-replacements", "job-additions"]) $(key).value = "";
}
async function openJob(id) {
  const row = await api("/api/agent-workspace/job/" + id); activeJob = row;
  $("job-key").value = row.job_key; $("job-title").value = row.title;
  $("job-replacements").value = Object.entries(row.replacements).map(([a,b]) => a + " = " + b).join("\n");
  $("job-additions").value = row.additions.join("\n"); notice("Opened job " + row.title);
}
function replacements() {
  const result = {};
  for (const line of ids($("job-replacements").value)) {
    const match = /^([a-f0-9]{32})\s*=\s*([a-f0-9]{32})$/.exec(line);
    if (!match) throw new Error("Each replacement must be old ID = new ID");
    result[match[1]] = match[2];
  }
  return result;
}
async function saveJob() {
  if (!activeProfile) throw new Error("Save an individual profile first");
  const data = {action:"save", profile_id:activeProfile.id, job_key:$("job-key").value,
    title:$("job-title").value, replacements:replacements(), additions:ids($("job-additions").value)};
  if (activeJob) data.id = activeJob.id;
  const row = await api("/api/agent-workspace/jobs", data); await openProfile(activeProfile.id); await openJob(row.id); notice("Saved job changes");
}
async function deleteJob() {
  if (!activeJob || !confirm("Delete these job changes?")) return;
  await api("/api/agent-workspace/jobs", {action:"delete", id:activeJob.id}); await openProfile(activeProfile.id); notice("Deleted job changes");
}
async function previewProfile(job = false) {
  if (!activeProfile) throw new Error("Save the profile first");
  const query = new URLSearchParams({profile:activeProfile.id}); if (job) query.set("job", $("job-key").value);
  const row = await api("/api/agent-workspace/resolve?" + query); $("profile-resolved").value = row.preamble;
  notice("Resolved " + row.documents.length + " documents for " + row.profile.individual);
}
async function instances() {
  const result = await api("/api/agent-workspace/instances"); $("instance-list").replaceChildren();
  for (const row of result.instances) $("instance-list").append(itemButton(row.title + " · " + row.character,
    guard(() => openInstance(row.id))));
}
function freshInstance() {
  activeInstance = null; $("instance-title").value = ""; $("instance-memory").value = "";
  $("instance-id").textContent = "New instance"; $("spawn-command").textContent = "";
  $("llama-queue").hidden = activeProfile?.individual?.toLowerCase() !== "llama";
}
async function openInstance(id) {
  const row = await api("/api/agent-workspace/instance/" + id);
  if (!activeProfile || activeProfile.id !== row.profile_id) await openProfile(row.profile_id);
  activeInstance = row; $("instance-title").value = row.title; $("instance-memory").value = row.memory_id;
  $("instance-id").textContent = "Instance " + row.id;
  if (["claude", "codex"].includes(row.individual.toLowerCase()))
    $("spawn-command").textContent = "Spawn from the operator terminal: Start-AgentDay.ps1 -RunOne " +
      row.individual + " -InstanceId " + row.id + (activeJob ? " -JobKey " + activeJob.job_key : "");
  notice("Opened instance " + row.title);
}
async function saveInstance() {
  if (!activeProfile) throw new Error("Save a character first");
  const data = {action:"save", profile_id:activeProfile.id, title:$("instance-title").value,
    memory_id:$("instance-memory").value || undefined};
  if (activeInstance) data.id = activeInstance.id;
  const row = await api("/api/agent-workspace/instances", data); await instances(); await openInstance(row.id);
  notice("Saved instance and attached memory " + row.memory_id);
}
async function deleteInstance() {
  if (!activeInstance || !confirm("Remove this instance from the catalog? Its memory file remains.")) return;
  await api("/api/agent-workspace/instances", {action:"delete", id:activeInstance.id});
  freshInstance(); await instances(); notice("Removed instance");
}
async function transferInstance() {
  if (!activeInstance) throw new Error("Open an instance first");
  const target = $("instance-target").value.trim();
  if (!target) throw new Error("Enter a destination character ID");
  const row = await api("/api/agent-workspace/instances", {action:"transfer", id:activeInstance.id,
    target_profile_id:target});
  await instances(); await openInstance(row.id); $("instance-target").value = "";
  notice("Transferred instance and attached memory to " + row.individual);
}
async function runs() {
  const result = await api("/api/agent-runs"); $("run-list").replaceChildren();
  for (const row of result.runs) $("run-list").append(itemButton(row.status + " · " + row.id,
    guard(async () => { const run = await api("/api/agent-runs/" + row.id);
      $("profile-resolved").value = run.response || run.error || "Waiting for Llama";
      showProfile(); notice("Run " + row.id + ": " + run.status); })));
}
async function submitRun() {
  if (!activeInstance || activeInstance.individual.toLowerCase() !== "llama") throw new Error("Open a Llama instance first");
  const data = {instance_id:activeInstance.id, prompt:$("run-prompt").value};
  if (activeJob) data.job_key = activeJob.job_key;
  const row = await api("/api/agent-runs", data); await runs(); notice("Queued Llama run " + row.id);
}
function guard(fn) { return async () => { try { await fn(); } catch (error) { notice(error.message, true); } }; }
for (const kind of kinds) {
  $("kind").append(new Option(kind, kind));
  const button = itemButton(kind[0].toUpperCase() + kind.slice(1), guard(async () => {
    mode = kind; for (const other of $("modes").children) other.removeAttribute("aria-current"); button.setAttribute("aria-current", "page");
    await branch(); fresh();
  }));
  if (kind === mode) button.setAttribute("aria-current", "page"); $("modes").append(button);
}
$("kind").onchange = () => { $("scenario").hidden = $("kind").value !== "scenario"; keepDraft(); };
for (const id of ["title", "body", "contexts"]) $(id).addEventListener("input", keepDraft);
$("new").onclick = fresh; $("save").onclick = guard(save); $("delete").onclick = guard(remove);
$("new-child").onclick = () => { if (!editing) return notice("Open a parent document first"); const child = editing; fresh(); parent = child; $("identity").textContent = "New child under " + child; };
$("parent").onclick = guard(async () => { if (current?.parent_id) await open(current.parent_id); });
$("full").onclick = () => { const full = $("workspace").classList.toggle("full"); $("full").textContent = full ? "Restore panes" : "Full screen"; };
$("scratch-save").onclick = guard(() => scratchChange("write")); $("scratch-delete").onclick = guard(() => scratchChange("delete"));
$("memory-save").onclick = guard(() => memoryChange("write")); $("memory-delete").onclick = guard(() => memoryChange("delete"));
$("memory-transfer").onclick = guard(() => memoryChange("transfer"));
$("profile-new").onclick = freshProfile; $("profile-save").onclick = guard(saveProfile);
$("profile-delete").onclick = guard(deleteProfile); $("profile-preview").onclick = guard(() => previewProfile());
$("capacity-save").onclick = guard(saveCapacity);
$("decisions-open").onclick = guard(openGraph); $("graph-open").onclick = guard(openGraph); $("place-new").onclick = freshPlace;
$("place-save").onclick = guard(savePlace); $("place-delete").onclick = guard(deletePlace); $("routes-save").onclick = guard(saveRoutes);
$("profile-graph-save").onclick = guard(assignGraph);
$("job-new").onclick = freshJob; $("job-save").onclick = guard(saveJob); $("job-delete").onclick = guard(deleteJob);
$("job-preview").onclick = guard(() => previewProfile(true));
$("instance-new").onclick = freshInstance; $("instance-save").onclick = guard(saveInstance);
$("instance-delete").onclick = guard(deleteInstance); $("run-submit").onclick = guard(submitRun);
$("instance-transfer").onclick = guard(transferInstance);
const restored = restoreDraft();   // first, so a draft shows even when the server is down
guard(async () => { const result = await api("/api/agent-workspace"); $("orientation").textContent = result.orientation; await branch(); if (!restored || !restoreDraft()) fresh(); await profiles(); await instances(); await runs(); await memories(); await scratch(); })();
