import * as fileCrypto from "/file-crypto.js";

const $ = id => document.getElementById(id);
const el = (tag, value) => { const node = document.createElement(tag); if (value != null) node.textContent = value; return node; };
const mode = document.body.dataset.mode;
const room = mode;
let current = new URLSearchParams(location.search).get("case");
let state = null, who = null, briefShown = null, loading = false;
const status = value => { $("message").textContent = value; };
const when = value => new Date(value * 1000).toLocaleString();

async function request(path, body) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store",
    ...(body === undefined ? {} : { method: "POST",
      headers: { "Content-Type": "application/json", "X-Meowdoku-Research": "1" },
      body: JSON.stringify(body) }) });
  let data = {};
  try { data = await response.json(); } catch {}
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
async function change(action, body) {
  await request(`/api/research/cases/${current}/${action}`, body);
  await loadCase();
}
async function loadCases() {
  const { cases } = await request("/api/research/cases");
  const visible = cases.filter(item => item.phase === mode);
  const box = $("cases"); box.replaceChildren();
  for (const item of visible) {
    const button = el("button", `${item.title} · ${item.author}`);
    button.type = "button";
    button.onclick = () => { current = item.id; history.replaceState(null, "", `/${mode}.html?case=${item.id}`); loadCase(); };
    box.append(button);
  }
  if (!visible.length) box.append(el("p", mode === "research" ? "No Research cases yet." : "No cases promoted to Development yet."));
  if (current) await loadCase();
  else status(`${visible.length} ${mode} case${visible.length === 1 ? "" : "s"}.`);
}
const person = id => state?.participants.find(item => item.id === id);
function renderTabs() {
  const brief = $("brief-tabs"); brief.replaceChildren();
  const ids = state.brief_user_ids;
  if (!ids.includes(briefShown)) briefShown = ids[0] ?? null;
  for (const id of ids) {
    const p = person(id); if (!p?.document_title) continue;
    const button = el("button", p.document_title); button.type = "button";
    button.setAttribute("role", "tab"); button.setAttribute("aria-selected", String(id === briefShown));
    button.onclick = () => { briefShown = id; renderTabs(); };
    brief.append(button);
  }
  $("brief-body").textContent = person(briefShown)?.document_body || "No design document tabs have been selected.";
  const docs = $("document-tabs"); docs.replaceChildren();
  for (const p of state.participants) {
    const button = el("button", p.username + (p.document_title ? ` · ${p.document_title}` : " · no document"));
    button.type = "button"; button.disabled = !p.document_title || p.id === state.case.active_user_id;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(p.id === state.case.active_user_id));
    button.title = p.id === state.case.active_user_id ? "Current shared tab" : "Propose switching the shared tab";
    button.onclick = async () => {
      if (!confirm(`Hold a vote to switch the shared document to ${p.username}?`)) return;
      try { await change("propose", { target_user_id: p.id, revision: state.case.revision }); }
      catch (error) { status(error.message); }
    };
    docs.append(button);
  }
  $("document-body").textContent = person(state.case.active_user_id)?.document_body || "No participant has set a document yet.";
  const choices = $("brief-choices"); choices.replaceChildren();
  for (const p of state.participants.filter(item => item.document_title)) {
    const label = el("label"), input = document.createElement("input");
    input.type = "checkbox"; input.value = String(p.id); input.checked = ids.includes(p.id);
    label.append(input, document.createTextNode(` ${p.username}: ${p.document_title}`)); choices.append(label);
  }
  $("brief-form").hidden = !(state.case.phase === "research" || state.can_edit_brief);
}
function renderVote() {
  const box = $("vote"); box.replaceChildren();
  if (!state.proposal) return;
  const p = person(state.proposal.target_user_id);
  box.append(el("h3", `Vote: switch to ${p?.username || "the proposed"} document`),
             el("p", `New ${state.proposal.new} · Current ${state.proposal.current} · closes ${when(state.proposal.deadline)}. Ties switch to the new document.`));
  if (!state.proposal.eligible) { box.append(el("p", "This vote began before your account joined.")); return; }
  for (const [value,labelText] of [["new","Switch to new"],["current","Keep current"]]) {
    const label = el("label"), radio = document.createElement("input");
    radio.type = "radio"; radio.name = "tab-vote"; radio.value = value;
    radio.checked = state.proposal.my_vote === value;
    radio.onchange = async () => { try { await change("vote", { choice: value }); } catch (error) { status(error.message); } };
    label.append(radio, document.createTextNode(labelText)); box.append(label);
  }
}
function renderSurvey(container, axes) {
  const box = $(container); box.replaceChildren();
  for (const axis of axes) {
    const field = document.createElement("fieldset"); field.className = "survey";
    field.append(el("legend", axis[0].toUpperCase() + axis.slice(1)));
    const record = state.surveys[axis];
    for (let value=1; value<=5; value++) {
      const label = el("label"), radio = document.createElement("input");
      radio.type = "radio"; radio.name = axis; radio.value = String(value);
      radio.checked = record.mine === value;
      radio.onchange = async () => { try { await change("survey", { axis, value }); } catch (error) { status(error.message); } };
      label.append(radio, document.createTextNode(String(value))); field.append(label);
    }
    const total = Object.entries(record.counts).reduce((n,[_,count]) => n+count,0);
    const score = Object.entries(record.counts).reduce((n,[value,count]) => n+Number(value)*count,0);
    field.append(el("span", total ? ` Average ${(score/total).toFixed(1)} · ${total} responses` : " No responses yet"));
    box.append(field);
  }
}
function renderEvents() {
  const list = $("events"); list.replaceChildren();
  for (const event of state.events) {
    const item = el("li"), old = event.before_json ? JSON.parse(event.before_json) : null;
    const next = event.after_json ? JSON.parse(event.after_json) : null;
    let delta = "";
    if (event.kind.startsWith("survey:")) delta = `${old ?? "—"} → ${next}`;
    else if (event.kind === "invariants") delta = `${old?.length ?? 0} → ${next?.length ?? 0} invariants`;
    else if (event.kind === "document") delta = next?.title || "";
    else if (event.kind === "vote_closed") delta = next?.state || "";
    else if (event.kind === "affirmation") delta = "phrase changed";
    else if (event.kind === "brief_tabs") delta = `${old?.length ?? 0} → ${next?.length ?? 0} tabs`;
    item.textContent = `${when(event.at)} · ${event.actor || "system"} · ${event.kind}${delta ? " · " + delta : ""}`;
    list.append(item);
  }
}
function render() {
  const c = state.case;
  $("case").hidden = false; $("case-heading").textContent = c.title;
  $("case-phase").textContent = c.phase;
  $("case-goal-text").textContent = c.goal;
  $("case-tickets-text").textContent = c.ticket_ids.length ? "Related tickets: " + c.ticket_ids.join(", ") : "Original goal only";
  $("other-stage").href = `/${c.phase}.html?case=${c.id}`;
  $("other-stage").textContent = c.phase === mode ? "" : `Open in ${c.phase}`;
  $("other-stage").hidden = c.phase === mode;
  $("affirmation-gate").hidden = !state.requires_affirmation;
  $("workspace").hidden = state.requires_affirmation;
  if (state.requires_affirmation) { status("Enter your affirmation phrase to join this case."); return; }
  const list = $("invariants"); list.replaceChildren();
  for (const item of c.invariants) list.append(el("li", item));
  if (document.activeElement !== $("invariant-input")) $("invariant-input").value = c.invariants.join("\n");
  $("invariant-form").hidden = !(c.phase === "research" || state.can_edit_brief);
  const me = person(state.my_id);
  if (document.activeElement !== $("affirmation-input")) $("affirmation-input").value = me?.phrase || "";
  const affirmations = $("affirmations"); affirmations.replaceChildren();
  for (const p of state.participants.filter(item => item.phrase)) {
    affirmations.append(el("li", `${p.username}: ${p.phrase}`));
  }
  $("promote").hidden = c.phase !== "research" || !(c.owner_id === state.my_id || who?.role === "admin");
  renderTabs(); renderVote();
  renderSurvey("progress-survey", ["clarity","confidence","momentum"]);
  renderSurvey("social-survey", ["trust","inclusion","coordination"]);
  renderEvents();
  status(`${c.title} · updated ${when(c.updated_at)}`);
}
async function loadCase() {
  if (!current || loading) return;
  loading = true;
  try { state = await request(`/api/research/cases/${current}`); render(); await Promise.all([loadChat(),loadThreads(),loadArena()]); }
  catch (error) { status(error.message); }
  finally { loading = false; }
}
async function loadArena() {
  const box = $("arena-attempts"); box.replaceChildren();
  const refs = mode === "development" ? [{kind:"research",id:state.case.id}] :
    state.case.ticket_ids.map(id => ({kind:"ticket",id}));
  if (!refs.length) { box.append(el("p","No related ticket references.")); return; }
  let count = 0;
  for (const ref of refs) {
    try {
      const query = new URLSearchParams(ref);
      const data = await request(`/api/agent-training-arena/by-source?${query}`);
      for (const run of data.runs) {
        for (const team of run.result.teams || []) {
          count++;
          const item = el("details"), title = el("summary",`${team.name} · ${ref.kind} ${ref.id} · ${Object.values(team.scores).reduce((a,b)=>a+b,0)}/20 · ${when(run.finished)}`);
          const body = el("pre",`Goal: ${run.specification.goal}\n\nRuntime: ${JSON.stringify(run.specification.runtime)}\n\nPrompt: ${team.system_prompt}\n\nFirst message: ${team.first_message}\n\nResult: ${team.response}\n\nScores: ${JSON.stringify(team.scores)}\nEach piece: ${JSON.stringify(team.pieces || team.reasons)}\nTests: ${JSON.stringify(team.test_evidence)}`);
          item.append(title,body); box.append(item);
        }
      }
    } catch(error) { box.append(el("p",`Could not read arena attempts for ${ref.id}: ${error.message}`)); }
  }
  if (!count) box.append(el("p","No completed arena attempts for these references."));
}
async function loadChat() {
  if (!state || state.requires_affirmation) return;
  try {
    const { messages } = await request(`/api/chat/rooms/${room}?tail=40`);
    const box = $("chat-messages"); box.replaceChildren();
    for (const m of messages) box.append(el("li", `${m.username} · ${when(m.at)}\n${m.removed ? "(removed)" : m.body}`));
    if (!messages.length) box.append(el("li", "No messages yet."));
  } catch (error) { status(error.message); }
}
async function loadThreads() {
  if (!state || state.requires_affirmation) return;
  try {
    const { comments } = await request(`/api/comments?kind=research_case&ref=${current}&tail=100`);
    const box = $("threads"); box.replaceChildren();
    const nodes = new Map(), roots = [];
    for (const comment of comments) nodes.set(comment.id, { comment, children: [] });
    for (const node of nodes.values()) {
      const parent = nodes.get(node.comment.reply_to);
      if (parent) parent.children.push(node); else roots.push(node);
    }
    const draw = (nodes, parent) => {
      for (const node of nodes) {
        const li = el("li"), label = el("div", `${node.comment.username} · ${when(node.comment.at)}`);
        label.className = "entry";
        li.append(label, el("div", node.comment.removed ? "(removed)" : node.comment.body));
        const reply = el("button", "Reply"); reply.type = "button";
        reply.onclick = () => {
          if (li.querySelector("form")) return;
          const form = document.createElement("form"), text = document.createElement("textarea");
          text.maxLength = 4000; text.rows = 2;
          const send = el("button", "Post reply"); send.type = "submit";
          form.append(text, send);
          form.onsubmit = async event => {
            event.preventDefault(); if (!text.value.trim()) return;
            try { await request("/api/comments", { kind:"research_case", ref:current,
              body:text.value, reply_to:node.comment.id }); await loadThreads(); }
            catch (error) { status(error.message); }
          };
          li.append(form); text.focus();
        };
        li.append(reply);
        if (node.children.length) { const ul = el("ul"); draw(node.children,ul); li.append(ul); }
        parent.append(li);
      }
    };
    draw(roots,box);
    if (!roots.length) box.append(el("li", "No threads yet."));
  } catch (error) { status(error.message); }
}
$("create-form").onsubmit = async event => {
  event.preventDefault();
  try {
    const ids = $("case-tickets").value.split(",").map(x=>x.trim()).filter(Boolean);
    const created = await request("/api/research/cases", {
      title:$("case-title").value, goal:$("case-goal").value, ticket_ids:ids });
    current = created.id; history.replaceState(null,"",`/research.html?case=${current}`);
    await loadCases();
  } catch (error) { status(error.message); }
};
for (const [formId,inputId] of [["affirmation-gate-form","affirmation-gate-input"],["affirmation-form","affirmation-input"]]) {
  $(formId).onsubmit = async event => {
    event.preventDefault();
    try { await change("affirmation",{phrase:$(inputId).value}); }
    catch (error) { status(error.message); }
  };
}
$("invariant-form").onsubmit = async event => {
  event.preventDefault();
  try { await change("invariants",{invariants:$("invariant-input").value.split("\n").map(x=>x.trim()).filter(Boolean),
    revision:state.case.revision}); }
  catch (error) { status(error.message); }
};
$("brief-form").onsubmit = async event => {
  event.preventDefault();
  try { await change("brief",{user_ids:[...$("brief-choices").querySelectorAll("input:checked")].map(x=>Number(x.value)),
    revision:state.case.revision}); }
  catch (error) { status(error.message); }
};
$("promote").onclick = async () => {
  if (!confirm("Promote this Research case and its brief to Development?")) return;
  try { await change("promote",{revision:state.case.revision}); location.assign(`/development.html?case=${current}`); }
  catch (error) { status(error.message); }
};
$("document-source").onchange = async () => {
  const kind = $("document-source").value;
  $("inventory-label").hidden = kind === "text";
  $("document-text-label").hidden = kind !== "text";
  $("copy-warning").hidden = kind !== "file_copy";
  $("file-unlock").hidden = kind !== "file_copy" || fileCrypto.unlocked();
  const picker = $("document-inventory"); picker.replaceChildren();
  if (kind === "page") {
    try {
      const { pages } = await request("/api/pages?mine=1");
      for (const page of pages) picker.append(new Option(page.title, page.id));
    } catch (error) { status(error.message); }
  } else if (kind === "file_copy") {
    if (!fileCrypto.unlocked()) { status("Unlock your file key on this page to choose an encrypted file."); return; }
    try {
      const { files } = await request("/api/user-files");
      for (const file of files) {
        const title = await fileCrypto.decryptName(file.envelope);
        picker.append(new Option(title, file.id));
      }
    } catch (error) { status(error.message); }
  }
};
$("unlock-files").onclick = async () => {
  try { await fileCrypto.unlock($("file-passphrase").value); $("file-passphrase").value = "";
    $("file-unlock").hidden = true; await $("document-source").onchange(); }
  catch (error) { status(error.message); }
};
$("document-form").onsubmit = async event => {
  event.preventDefault();
  try {
    const kind = $("document-source").value, ref = kind === "text" ? null : $("document-inventory").value;
    if (kind !== "text" && !ref) throw new Error("Choose a document from your inventory.");
    let body = $("document-text").value;
    if (kind === "file_copy") {
      if (!confirm("Create a readable Research copy of this private file for all signed-in participants?")) return;
      const file = await fileCrypto.openFile(ref);
      body = new TextDecoder("utf-8",{fatal:true}).decode(file.data);
    }
    await change("document",{title:$("document-title").value,body,source_kind:kind,source_ref:ref});
  } catch (error) { status(error.message); }
};
$("chat-form").onsubmit = async event => {
  event.preventDefault();
  const body = $("chat-text").value.trim(); if (!body) return;
  try { await request(`/api/chat/rooms/${room}`,{body}); $("chat-text").value=""; await loadChat(); }
  catch (error) { status(error.message); }
};
$("thread-form").onsubmit = async event => {
  event.preventDefault();
  const body = $("thread-text").value.trim(); if (!body) return;
  try { await request("/api/comments",{kind:"research_case",ref:current,body}); $("thread-text").value=""; await loadThreads(); }
  catch (error) { status(error.message); }
};
try {
  who = await request("/api/whoami");
  if (!who.signed_in) { $("signin").hidden=false; status("Sign in to participate."); }
  else { await loadCases(); setInterval(() => { if (current && state && !state.requires_affirmation) {
    loadChat(); loadThreads(); if (!["TEXTAREA","INPUT"].includes(document.activeElement?.tagName)) loadCase();
  } },15000); }
} catch (error) { status(error.message); }
