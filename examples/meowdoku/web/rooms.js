// Rooms: the chat server's rooms (chat_rooms.py) as site objects. HAND-AUTHORED. createElement/textContent only; the server
// decides every permission, this page only decides what to offer.
const byId = id => document.getElementById(id);
const el = (tag, text, cls) => { const n = document.createElement(tag); if (text != null) n.textContent = text; if (cls) n.className = cls; return n; };
let who = null, rooms = [], selected = null;

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
const when = seconds => seconds == null ? "—" : new Date(seconds * 1000).toLocaleString();
const field = (label, input) => { const box = el("div"), l = el("label", label); input.id ||= `f-${Math.random().toString(36).slice(2)}`; l.htmlFor = input.id; box.append(l, input); return box; };
const input = (value = "", type = "text") => { const i = document.createElement("input"); i.type = type; i.value = value; return i; };
const choice = (options, value) => { const s = document.createElement("select"); for (const o of options) { const opt = el("option", o); opt.value = o; s.append(opt); } s.value = value; return s; };

async function load() {
  try {
    ({ rooms } = await request("/api/chat/rooms"));
    const list = byId("rooms");
    list.replaceChildren();
    for (const room of rooms) {
      const row = el("button", room.title, "row"); row.type = "button";
      if (!room.system) row.append(el("span", room.stage === "dev" ? "dev" : "released", "badge " + room.stage));
      row.append(el("small", room.system ? `system room · ${room.room}` : `${room.room} · by ${room.owner}`));
      row.setAttribute("aria-current", String(room.room === selected));
      row.onclick = () => open(room.room);
      list.append(row);
    }
    if (!rooms.length) list.append(el("p", "No room is open to you yet.", "muted"));
    status(`${rooms.length} room${rooms.length === 1 ? "" : "s"} open to you.`);
    if (selected) await open(selected, { quiet: true });
  } catch (error) { status(error.message); }
}

async function open(name, { quiet = false } = {}) {
  selected = name;
  const panel = byId("room");
  const typing = panel.contains(document.activeElement) && ["TEXTAREA", "INPUT"].includes(document.activeElement.tagName);
  if (quiet && typing) return;
  try {
    const tail = encodeURIComponent(byId("tail").value);
    const [{ settings }, { messages }, set] = await Promise.all([request(`/api/chat/rooms/${name}/settings`),
      request(`/api/chat/rooms/${name}?tail=${tail}`), request(`/api/chat/rooms/${name}/tools`)]);
    panel.replaceChildren(el("h2", settings.title), el("p", settings.about || "", "muted"));
    // the chat
    const list = el("ul", null, "msgs");
    for (const m of messages) {
      const li = el("li");
      li.append(el("span", m.username, "who"), document.createTextNode(m.removed ? "(removed by an admin)" : m.body), el("span", when(m.at), "at"));
      list.append(li);
    }
    if (!messages.length) list.append(el("li", "No message yet.", "muted"));
    const compose = el("div", null, "compose"), text = document.createElement("textarea"), send = el("button", "Send");
    text.rows = 2; text.maxLength = 4000; text.setAttribute("aria-label", `Message to ${settings.title}`); send.type = "button";
    send.onclick = async () => { if (!text.value.trim()) return;
      try { await request(`/api/chat/rooms/${name}`, { body: text.value }); text.value = ""; await open(name); } catch (error) { status(`Not sent: ${error.message}`); } };
    compose.append(text, send);
    panel.append(list, compose);
    if (!settings.system) panel.append(roomSet(name, set), ...(isOwner(settings) ? [ownerSettings(name, settings)] : []), forwarding(name, set));
  } catch (error) { panel.replaceChildren(el("p", `The room could not be read: ${error.message}`, "muted")); }
}

const isOwner = settings => who && (settings.owner_name === who.username || who.role === "admin");

function roomSet(name, set) {
  const box = el("section", null, "part");
  box.append(el("h3", "The room's set"), el("p", set.say, "muted"));
  box.append(el("p", set.objects.length ? "Site objects: " + set.objects.map(o => `${o.kind} ${o.ref}`).join(", ") : "No site objects.", null),
             el("p", set.bridges.length ? "Bridges: " + set.bridges.join(", ") : "No bridges.", null));
  for (const grant of set.forwarded) {
    const row = el("div", null, "part");
    row.append(el("p", `${grant.owner} forwards ${grant.provider}: ${grant.tools.join(", ")} (until ${when(grant.expires)})`));
    const tool = choice(grant.tools, grant.tools[0]), args = document.createElement("textarea"), call = el("button", "Call");
    args.rows = 2; args.value = "{}"; call.type = "button";
    const out = el("pre", null, "muted");
    call.onclick = async () => {
      let parsed;
      try { parsed = JSON.parse(args.value || "{}"); } catch (error) { status("The arguments must be a JSON object."); return; }
      try { const { result } = await request(`/api/chat/rooms/${name}/forward/${grant.grant}/call`, { name: tool.value, arguments: parsed });
        out.textContent = JSON.stringify(result, null, 2); } catch (error) { out.textContent = error.message; }
    };
    row.append(field("Tool", tool), field("Arguments (JSON)", args), call, out);
    if (grant.owner === who.username) {
      const back = el("button", "Take it back"); back.type = "button";
      back.onclick = async () => { try { await request(`/api/chat/rooms/${name}/forward/${grant.grant}/revoke`, {}); await open(name); } catch (error) { status(error.message); } };
      row.append(back);
    }
    box.append(row);
  }
  return box;
}

function ownerSettings(name, settings) {
  const box = el("section", null, "part");
  const title = input(settings.title), about = input(settings.about), stage = choice(["dev", "release"], settings.stage);
  const write = choice(["members", "owner"], settings.write), objects = document.createElement("textarea"), bridges = input(settings.bridges.join(", "));
  objects.rows = 3; objects.value = JSON.stringify(settings.objects);
  const save = el("button", "Save settings"); save.type = "button";
  save.onclick = async () => {
    let parsed;
    try { parsed = JSON.parse(objects.value || "[]"); } catch (error) { status("Site objects must be a JSON list of {kind, ref}."); return; }
    try {
      await request(`/api/chat/rooms/${name}/settings`, { title: title.value, about: about.value, stage: stage.value, write: write.value,
        objects: parsed, bridges: bridges.value.split(",").map(t => t.trim()).filter(Boolean) });
      status("Settings saved."); await load();
    } catch (error) { status(`Not saved: ${error.message}`); }
  };
  const sharing = settings.sharing || { visibility: "personal", revision: 0 };
  const visibility = choice(["personal", "group", "public"], sharing.visibility), group = input(sharing.group_id || "");
  const share = el("button", "Share"); share.type = "button";
  share.onclick = async () => {
    if (visibility.value === "public" && !confirm("Public means everyone the stage allows may enter. Sign off on that?")) return;
    try {
      await request(`/api/chat/rooms/${name}/access`, { visibility: visibility.value, group_id: visibility.value === "group" ? group.value : null,
        expected_revision: sharing.revision, sign_off: visibility.value === "public" });
      status(`Now ${visibility.value}.`); await load();
    } catch (error) { status(`Not shared: ${error.message}`); }
  };
  box.append(el("h3", "Settings (owner)"), field("Title", title), field("About", about),
    field("Stage: dev is the development site and needs dev permission; release is public facing", stage),
    field("Who writes", write), field("Site objects (JSON list of {kind, ref}; kinds: page, render, mesh, shader, media, room; at most 8)", objects),
    field("Bridges (tool names, comma separated; at most 8)", bridges), save,
    el("h3", "Who may enter, beyond the stage"), field("Sharing", visibility), field("Group id (for group)", group), share);
  return box;
}

function forwarding(name, set) {
  const box = el("section", null, "part");
  const tools = document.createElement("select"); tools.multiple = true; tools.size = 4;
  const hours = input("24", "number"), lend = el("button", "Forward to this room"); lend.type = "button";
  const loadTools = el("button", "Load my Suno tools"); loadTools.type = "button";
  loadTools.onclick = async () => {
    try { const result = await request("/api/connections/suno/tools");
      tools.replaceChildren(...result.tools.map(t => { const o = el("option", t.name); o.value = t.name; return o; }));
      if (!result.tools.length) status("Your Suno connection offers no tools."); }
    catch (error) { status(`No tools to lend: ${error.message}`); }
  };
  lend.onclick = async () => {
    const chosen = [...tools.selectedOptions].map(o => o.value);
    if (!chosen.length) { status("Choose the tools to lend."); return; }
    if (!confirm(`Lend ${chosen.join(", ")} to ${name}? Its members will call them under your connection until you take them back.`)) return;
    try { await request(`/api/chat/rooms/${name}/forward`, { provider: "suno", tools: chosen, hours: Number(hours.value) || 24 });
      status("Forwarded."); await open(name); } catch (error) { status(`Not forwarded: ${error.message}`); }
  };
  box.append(el("h3", "Forward your own connection"), el("p", "Lend some tools of a connection you secured (Connected services) to " +
    "this room. Calls run under your connection, are limited per member and recorded; take them back at any time.", "muted"),
    loadTools, field("Tools", tools), field("For how many hours (at most 720)", hours), lend);
  return box;
}

async function main() {
  try { who = await request("/api/whoami"); } catch (error) { status(error.message); return; }
  if (!who.signed_in) { byId("signin").hidden = false; status("Sign in to see your rooms."); return; }
  byId("content").hidden = false;
  byId("make").hidden = !(who.capabilities || []).includes("dev_ui");
  byId("make").onsubmit = async event => {
    event.preventDefault();
    try {
      const { settings } = await request("/api/chat/rooms", { room: byId("make-room").value, title: byId("make-title").value, about: byId("make-about").value });
      byId("make").reset(); selected = settings.room; await load(); status(`Room ${settings.room} made: personal, on the development site.`);
    } catch (error) { status(`Not made: ${error.message}`); }
  };
  byId("refresh").onclick = load;
  byId("tail").onchange = () => selected && open(selected);
  await load();
  setInterval(() => { if (!document.hidden) load(); }, 20000);
}
main();
