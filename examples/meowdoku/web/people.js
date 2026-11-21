// People: every account, its authority, and the SpeakToMe tools its groups give it. HAND-AUTHORED. createElement and
// textContent only; the server decides every permission and checks every grant, this page only decides what to offer.
//   GET  /api/account/admin/people                 accounts with their authority (account_admin.py)
//   GET  /api/account/admin/tools?username=NAME    that account's capabilities and tools
//   POST /api/account/authority/grant              {username, group: dev|operator|nodus|message|execution, enabled}
//   POST /api/account/admin/grant | revoke         {username}
//   POST /api/account/admin/purge                  {username, password}: an ordinary account and its private rows; posts stay
const byId = id => document.getElementById(id);
const el = (tag, text, cls) => { const n = document.createElement(tag); if (text != null) n.textContent = text; if (cls) n.className = cls; return n; };
let people = [], selected = null;

async function request(path, body) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store",
    ...(body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }) });
  let result = {};
  try { result = await response.json(); } catch (error) { /* no body */ }
  if (!response.ok) {
    if (response.status === 401) { byId("signin").hidden = false; byId("content").hidden = true; }
    throw new Error(result.error || `HTTP ${response.status}`);
  }
  return result;
}
const status = message => { byId("message").textContent = message; };
const when = seconds => seconds == null ? "—" : new Date(seconds * 1000).toLocaleDateString();

async function load() {
  try {
    ({ people } = await request("/api/account/admin/people"));
    byId("content").hidden = false;
    const list = byId("people");
    list.replaceChildren();
    for (const person of people) {
      const a = person.authority || {};
      const row = el("button", person.username, "row"); row.type = "button";
      row.append(el("small", `${a.highest || "member"} · groups ${(a.groups || []).join(", ") || "none"} · since ${when(person.created_at)}`));
      row.setAttribute("aria-current", String(person.username === selected));
      row.onclick = () => open(person.username);
      list.append(row);
    }
    status(`${people.length} account${people.length === 1 ? "" : "s"}.`);
    const activity = await request("/api/account/admin/access");
    byId("access-log-path").textContent = `Full Apache access log on server: ${activity.access_log}`;
    const access = byId("access-entries"); access.replaceChildren();
    for (const entry of activity.entries.slice(-100).reverse())
      access.append(el("p", `${new Date(entry.ts * 1000).toLocaleString()} · ${entry.event} · ${entry.username} · ${entry.ip}`));
    if (selected) await open(selected);
  } catch (error) {
    status(error.message.includes("account_manage") ? "This page is for admins (it needs account_manage)." : error.message);
  }
}

async function grant(person, group, enabled) {
  const verb = enabled ? "Grant" : "Remove";
  if (!confirm(`${verb} ${group} ${enabled ? "to" : "from"} ${person.username}? Their tools change with it.`)) return;
  try {
    if (group === "admin") await request(`/api/account/admin/${enabled ? "grant" : "revoke"}`, { username: person.username });
    else await request("/api/account/authority/grant", { username: person.username, group, enabled });
    status(`${verb === "Grant" ? "Granted" : "Removed"} ${group} ${enabled ? "to" : "from"} ${person.username}.`);
    await load();
  } catch (error) { status(`Not changed: ${error.message}`); }
}

async function open(username) {
  selected = username;
  for (const row of byId("people").children) row.setAttribute?.("aria-current", String(row.firstChild?.textContent === username));
  const person = people.find(p => p.username === username);
  const panel = byId("person");
  if (!person) { panel.replaceChildren(el("p", "That account is gone.")); return; }
  const a = person.authority || {};
  panel.replaceChildren(el("h2", person.username), el("p", `Highest group: ${a.highest || "member"}. Role row: ${person.role_row || "none (member)"}` +
    (person.role_granted_by ? `, granted by ${person.role_granted_by}` : "") + `. Account since ${when(person.created_at)}.`, "muted"));

  // groups: what it holds, and a grant or a removal for each one an admin may change
  const groups = el("div", null, "actions");
  groups.append(el("span", "Groups:"));
  for (const g of a.groups || []) groups.append(el("span", g, "badge on"));
  panel.append(groups);
  if (a.highest === "intern") {
    const enabled = Boolean(a.modifiers?.message);
    const button = el("button", `${enabled ? "Remove" : "Grant"} message`); button.type = "button";
    button.onclick = () => grant(person, "message", !enabled);
    panel.append(button, el("p", "An intern account may receive an account-specific message modifier for direct messages.", "muted"));
  }
  else if (a.service) panel.append(el("p", "A service account is not on the people ladder; its role decides.", "muted"));
  else {
    if (person.server_agent) {
      const enabled = (a.groups || []).includes("execution");
      const button = el("button", `${enabled ? "Remove" : "Grant"} exec and Python`); button.type = "button";
      button.onclick = () => grant(person, "execution", !enabled);
      panel.append(button, el("p", "Separate account permission for commands and Python run as this agent's Windows account. Llama is excluded.", "muted"));
    }
    const buttons = el("div", null, "actions");
    const direct = new Set(a.direct || []);
    for (const group of ["dev", "operator", "nodus", "admin"]) {
      const has = direct.has(group);
      const b = el("button", `${has ? "Remove" : "Grant"} ${group}`); b.type = "button";
      b.onclick = () => grant(person, group, !has);
      buttons.append(b);
    }
    panel.append(buttons, el("p", "A role row (operator, admin) set by authz.py is a direct grant too; removing a group here removes only a grant made here.", "muted"));
    if (!person.role_row && !(a.direct || []).length) {   // [PURGE] ordinary accounts only; the server refuses any other
      const purge = el("button", "Purge account"); purge.type = "button";
      purge.onclick = async () => {
        if (prompt(`Purging removes ${person.username}'s private rows and sign-in for good; its public posts stay under a sealed name. Type ${person.username} to confirm:`) !== person.username) return;
        const password = prompt("Enter your admin account password to confirm:");
        if (!password) return;
        purge.disabled = true;
        try {
          const response = await fetch("/api/account/admin/purge", { method: "POST", credentials: "same-origin", cache: "no-store",
            headers: { "Content-Type": "application/json", "X-Meowdoku-Account-Purge": "1" },
            body: JSON.stringify({ username: person.username, password }) });
          const result = await response.json();
          if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
          status(`Purged ${person.username}.`); await load();
        } catch (error) { status(error.message); purge.disabled = false; }
      };
      panel.append(purge);
    }
  }

  // tools: what the speaktome MCP offers this account, and what it does not, with what each needs
  const tools = el("div"); tools.append(el("h3", "SpeakToMe tools"), el("p", "Working them out…", "muted"));
  panel.append(tools);
  try {
    const t = await request(`/api/account/admin/tools?username=${encodeURIComponent(username)}`);
    const changing = t.tools.filter(x => x.changes).length;
    tools.replaceChildren(el("h3", `SpeakToMe tools: ${t.tools.length} offered (${changing} change server state), ${t.not_offered.length} not`),
      el("p", `As if ${t.username} connected from where you are (origin ${t.origin}, tier ${t.tier}). Role ${t.role}; ` +
        `capabilities: ${t.capabilities.join(", ") || "none"}. ${t.note}.`, "muted"));
    tools.append(table(t.tools, "Offered", false), table(t.not_offered, "Not offered (and what each needs)", true));
  } catch (error) { tools.replaceChildren(el("h3", "SpeakToMe tools"), el("p", `Could not work them out: ${error.message}`)); }
}

function table(rows, title, showNeeds) {
  const box = el("details"); box.append(el("summary", `${title}: ${rows.length}`));
  const t = el("table"), head = el("tr");
  for (const h of ["Tool", "Group", "What it does", showNeeds ? "Needs" : "Changes"]) head.append(el("th", h));
  t.append(head);
  for (const r of rows) {
    const tr = el("tr"), name = el("td"); name.append(el("code", r.name));
    tr.append(name, el("td", r.group), el("td", r.summary), el("td", showNeeds ? r.needs : (r.changes ? "yes" : "read-only")));
    t.append(tr);
  }
  box.append(t);
  return box;
}

byId("refresh").onclick = load;
load();
