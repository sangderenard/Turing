const byId = id => document.getElementById(id);
const el = (tag, text) => Object.assign(document.createElement(tag), { textContent: text });
let selected = null, nextOffset = null, listGeneration = 0, detailGeneration = 0;
let pendingSubmission = null, sending = false;
const params = new URLSearchParams(location.search);
const source = params.get("source");
const sourcePath = source && source.startsWith("/") && !/[?#\r\n]/.test(source) ? source.slice(0, 2000) : location.pathname;

async function request(path, body) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store",
    ...(body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json", "X-Meowdoku-Issue": "1" }, body: JSON.stringify(body) }) });
  const result = await response.json();
  if (!response.ok) {
    if (response.status === 401) { byId("signin").hidden = false; byId("account-content").hidden = true; }
    throw new Error(result.error || `HTTP ${response.status}`);
  }
  return result;
}
function status(message) { byId("message").textContent = message; }
function openForm() { byId("create-panel").hidden = false; byId("title").focus(); }

async function load(append = false) {
  const generation = ++listGeneration;
  byId("more").disabled = true;
  try {
    const query = new URLSearchParams({ scope: byId("scope").value, status: byId("filter").value, offset: String(append ? nextOffset || 0 : 0) });
    const result = await request(`/api/issues?${query}`);
    if (generation !== listGeneration) return;
    byId("all-option").hidden = !result.can_review_all;
    if (!append) byId("list").replaceChildren();
    for (const ticket of result.tickets) {
      const row = el("button", ticket.title); row.type = "button"; row.className = "row";
      row.append(el("small", `${ticket.status} · ${new Date(ticket.updated_at * 1000).toLocaleString()}`));
      row.onclick = () => show(ticket.id);
      byId("list").append(row);
    }
    if (!append && !result.tickets.length) byId("list").append(el("p", "No matching tickets."));
    nextOffset = result.next_offset; byId("more").hidden = nextOffset === null;
    status("Tickets are up to date.");
  } catch (error) { if (generation === listGeneration) status(error.message); }
  finally { if (generation === listGeneration) byId("more").disabled = false; }
}

async function show(id) {
  selected = id;
  const generation = ++detailGeneration;
  try {
    const { ticket } = await request(`/api/issues/${encodeURIComponent(id)}`);
    if (generation !== detailGeneration) return;
    const heading = el("h2", ticket.title), description = el("p", ticket.description);
    const meta = el("p", `${ticket.status} · Ticket ${ticket.id} · version ${ticket.version}`); meta.className = "muted";
    const details = document.createElement("details"); details.append(el("summary", "Submitted diagnostics"), el("pre", JSON.stringify(ticket.diagnostics, null, 2)));
    const form = document.createElement("form"), label = el("label", "Status"), select = document.createElement("select");
    label.htmlFor = "ticket-status"; select.id = "ticket-status";
    for (const value of ["open", "investigating", "resolved"]) { const option = el("option", value); option.value = value; select.append(option); }
    select.value = ticket.status;
    const noteLabel = el("label", "Investigation or resolution note"), note = document.createElement("textarea");
    noteLabel.htmlFor = "update-note"; note.id = "update-note"; note.maxLength = 4000; note.rows = 4; note.required = true;
    const save = el("button", "Save status and note"); save.type = "submit";
    const feedback = el("p", "Resolving preserves the report and its history; it does not delete the ticket."); feedback.setAttribute("role", "status");
    form.append(label, select, noteLabel, note, save, feedback);
    let updating = false;
    form.onsubmit = async event => {
      event.preventDefault(); if (updating || !note.value.trim()) return;
      updating = true; save.disabled = true; select.disabled = true; note.disabled = true;
      try {
        await request(`/api/issues/${id}/status`, { status: select.value, note: note.value, version: ticket.version });
        if (selected === id) await show(id); await load();
      } catch (error) { feedback.textContent = `${error.message}. Your note is kept. Refresh this ticket before retrying if the outcome is uncertain.`; }
      finally { updating = false; save.disabled = false; select.disabled = false; note.disabled = false; }
    };
    const history = document.createElement("div"), entries = document.createElement("div"), older = el("button", "Load older history");
    older.type = "button"; older.hidden = ticket.next_before_version === null;
    let before = ticket.next_before_version;
    const historyEntry = event => { const entry = el("div", `${new Date(event.created_at * 1000).toLocaleString()} · ${event.status} · account ${event.actor_id}\n${event.note}`); entry.className = "history"; return entry; };
    history.append(el("h3", "History"), older, entries);
    for (const event of ticket.events) entries.append(historyEntry(event));
    older.onclick = async () => {
      older.disabled = true;
      try {
        const previous = (await request(`/api/issues/${id}?before_version=${before}`)).ticket;
        for (const event of [...previous.events].reverse()) entries.prepend(historyEntry(event));
        before = previous.next_before_version; older.hidden = before === null;
      } catch (error) { feedback.textContent = error.message; }
      finally { older.disabled = false; }
    };
    byId("detail").replaceChildren(heading, meta, description, details, form, history);
  } catch (error) { if (generation === detailGeneration) { byId("detail").replaceChildren(el("p", error.message)); status(error.message); } }
}

byId("create-form").onsubmit = async event => {
  event.preventDefault(); if (sending) return;
  const content = { title: byId("title").value, description: byId("description").value,
    diagnostics: byId("attach-details").checked ? { url: sourcePath, ua: navigator.userAgent } : {} };
  const fingerprint = JSON.stringify(content);
  if (!pendingSubmission || pendingSubmission.fingerprint !== fingerprint) pendingSubmission = { fingerprint, key: crypto.randomUUID().replaceAll("-", "") };
  sending = true;
  for (const id of ["send", "cancel", "title", "description", "attach-details"]) byId(id).disabled = true;
  byId("create-status").textContent = "Sending…";
  try {
    const result = await request("/api/issues", { ...content, client_key: pendingSubmission.key });
    pendingSubmission = null; byId("create-form").reset(); byId("create-panel").hidden = true;
    byId("create-status").textContent = "";
    await load(); await show(result.id); status(`Ticket ${result.id} submitted.`);
  } catch (error) { byId("create-status").textContent = `${error.message}. Your draft is kept. Retrying unchanged content uses the same submission key.`; }
  finally { sending = false; for (const id of ["send", "cancel", "title", "description", "attach-details"]) byId(id).disabled = false; }
};
byId("new-ticket").onclick = openForm;
byId("cancel").onclick = () => { if (!sending) { byId("create-panel").hidden = true; byId("new-ticket").focus(); } };
byId("refresh").onclick = async () => { await load(); if (selected) await show(selected); };
byId("scope").onchange = byId("filter").onchange = () => load();
byId("more").onclick = () => load(true);
try {
  const who = await request("/api/whoami");
  if (!who.signed_in) { byId("signin").hidden = false; status("Sign in to use issue tickets."); }
  else { byId("account-content").hidden = false; await load(); if (params.get("new") === "1") openForm(); const id = params.get("id"); if (id && /^[a-f0-9]{32}$/.test(id)) await show(id); }
} catch (error) { status(error.message); }
