const $ = id => document.getElementById(id);
const kinds = ["orientation", "prompt", "personality", "guidance", "trouble", "scenario"];
let current = null, parent = null, mode = "orientation", editing = null;
let activeProfile = null, activeJob = null, activeInstance = null;
const ids = value => value.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
function showProfile() { $("profile-editor").hidden = false; $("document-editor").hidden = true; }
function showDocument() { $("profile-editor").hidden = true; $("document-editor").hidden = false; }
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
  showDocument();
  const row = await api("/api/agent-workspace/document/" + id);
  current = row; editing = id; parent = row.parent_id;
  $("kind").value = row.kind; $("title").value = row.title; $("body").value = row.body;
  $("contexts").value = row.contexts.join("\n"); $("scenario").hidden = row.kind !== "scenario";
  $("identity").textContent = "Document " + row.id + " · author account " + row.owner_id;
  notice("Opened " + row.title);
}
function fresh() {
  showDocument();
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
$("memory-save").onclick = guard(() => memoryChange("write")); $("memory-delete").onclick = guard(() => memoryChange("delete"));
$("memory-transfer").onclick = guard(() => memoryChange("transfer"));
$("profile-new").onclick = freshProfile; $("profile-save").onclick = guard(saveProfile);
$("profile-delete").onclick = guard(deleteProfile); $("profile-preview").onclick = guard(() => previewProfile());
$("capacity-save").onclick = guard(saveCapacity);
$("job-new").onclick = freshJob; $("job-save").onclick = guard(saveJob); $("job-delete").onclick = guard(deleteJob);
$("job-preview").onclick = guard(() => previewProfile(true));
$("instance-new").onclick = freshInstance; $("instance-save").onclick = guard(saveInstance);
$("instance-delete").onclick = guard(deleteInstance); $("run-submit").onclick = guard(submitRun);
$("instance-transfer").onclick = guard(transferInstance);
guard(async () => { const result = await api("/api/agent-workspace"); $("orientation").textContent = result.orientation; await branch(); fresh(); await profiles(); await instances(); await runs(); await memories(); await scratch(); })();
