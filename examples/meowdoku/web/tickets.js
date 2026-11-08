const list = document.getElementById("list"), detail = document.getElementById("detail"), message = document.getElementById("message");
let selected = null, canEdit = false;
const el = (tag, text, className) => { const node = document.createElement(tag); node.textContent = text; if (className) node.className = className; return node; };
async function request(url, options) { const response = await fetch(url, { cache: "no-store", ...options }); const body = await response.json(); if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`); return body; }
async function load() {
  try {
    const [who, result] = await Promise.all([request("/api/whoami"), request("/api/tickets")]);
    canEdit = !!who.edit; list.replaceChildren();
    for (const ticket of result.tickets) {
      const row = el("button", `${ticket.status.toUpperCase()} · ${ticket.id}\n${(ticket.note || "(no note)").slice(0, 85)}`, "row");
      row.append(el("small", new Date(ticket.created * 1000).toLocaleString() + ` · ${ticket.peer}`));
      row.onclick = () => show(ticket.id); list.append(row);
    }
    if (!result.tickets.length) list.append(el("p", "No tickets yet."));
    message.textContent = `${result.tickets.length} tickets`;
    if (selected) await show(selected);
  } catch (error) { message.textContent = error.message; }
}
async function show(id) {
  try {
    selected = id; const { ticket } = await request(`/api/tickets?id=${encodeURIComponent(id)}`);
    const heading = el("h2", `Ticket ${ticket.id}`), meta = el("p", `${ticket.status} · ${new Date(ticket.created * 1000).toLocaleString()} · ${ticket.peer} · log sequence ${ticket.log_seq}`);
    const logLink = el("a", "Open server log near this report"); logLink.href = `log.html?since=${Math.max(0, ticket.log_seq - 100)}`; logLink.style.color = "#9bd";
    const note = el("textarea", ticket.handling_note || ""); note.placeholder = "Investigation notes";
    const status = document.createElement("select"); for (const value of ["open", "investigating", "resolved"]) { const opt = el("option", value); opt.value = value; status.append(opt); } status.value = ticket.status;
    const save = el("button", "Save status and note"); save.disabled = !canEdit;
    save.onclick = async () => { try { await request("/api/tickets/status", { method: "POST", headers: { "Content-Type": "application/json", "X-Meowdoku-Ticket": "1" }, body: JSON.stringify({ id, status: status.value, note: note.value }) }); message.textContent = "Saved"; await load(); } catch (error) { message.textContent = error.message; } };
    const actions = el("div", "", "actions"); actions.id = "actions"; actions.append(status, save);
    const r = ticket.report || {}, c = r.compute || {}, p = r.particles || {}, phone = r.phone || {}, d = c.diag || {};
    const facts = el("div", "", "facts");
    for (const [name, value] of [
      ["Report", r.note || "(no description)"], ["Game", `${r.game?.game || ticket.server_state?.game || "unknown"} · ${c.manifest || "no compute"}`],
      ["Compute", `${c.ready ? "ready" : "not ready"} · step ${d.step ?? c.steps ?? "?"} · ${c.rec?.n ?? p.n ?? "?"} ids`],
      ["Particles", `${p.live ?? "?"} live · ${p.nonfinite ?? "?"} nonfinite · ${p.sample?.length ?? 0} sampled`],
      ["Phone", `${phone.state || "unknown"} · ${phone.rawLast?.source || "no sensor event"}`],
      ["Law gravity", JSON.stringify(c.gravity ?? (c.lawInputs ? [c.lawInputs.grav_x, c.lawInputs.grav_y, c.lawInputs.grav_z] : null))],
      ["Energy / speed", `${d.ke ?? "?"} / ${d.vmax2 == null ? "?" : Math.sqrt(d.vmax2).toFixed(3)}`],
      ["Client log", `${r.local_log?.length ?? 0} lines · server seq ${ticket.log_seq}`],
    ]) { const box = el("div", ""); box.append(el("strong", name), el("span", value)); facts.append(box); }
    const raw = el("pre", JSON.stringify({ report: ticket.report, server_state: ticket.server_state }, null, 2));
    const rawDetails = document.createElement("details"); rawDetails.append(el("summary", "Full state and logs"), raw);
    detail.replaceChildren(heading, meta, facts, logLink, actions, note, rawDetails);
  } catch (error) { message.textContent = error.message; }
}
document.getElementById("refresh").onclick = load;
load();
