// Change reviews: the rows of /api/reviews (branch_review.py), each change's own commits and diff read through the
// core's Git over REST (core_git.py: log, diff), and the admin's decision on exactly the commit shown.
// HAND-AUTHORED. createElement/textContent only. The server enforces every capability; this only decides what to show.
const byId = id => document.getElementById(id);
const el = (tag, text, cls) => { const n = document.createElement(tag); if (text != null) n.textContent = text; if (cls) n.className = cls; return n; };
const OPEN = new Set(["waiting", "objected", "accepted", "revoked", "held", "consolidating"]);
const DIFF_LINES = 6000;
let who = null, caps = new Set(), reviews = [], selected = null, detailGeneration = 0;
let landing = { queue: [], updater: false, apply: true, schedule: null, clockOffset: 0 };

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
function status(message) { byId("message").textContent = message; }
function when(seconds) { return seconds == null ? "—" : new Date(seconds * 1000).toLocaleString(); }
function until(seconds) {
  const left = seconds - Date.now() / 1000;
  if (left <= 0) return "passed";
  const hours = Math.floor(left / 3600), minutes = Math.floor((left % 3600) / 60);
  return `in ${hours} h ${minutes} min`;
}
function badge(standing) { return el("span", standing, "badge " + standing); }
const moderator = () => who && (who.role === "admin" || who.role === "operator");
const tail = () => byId("tail").value;
const drafts = new Map();          // unsent text per box, kept across the periodic refresh
let roomOpen = null;               // may this account enter the reviewers' room (asked once)

function messageList(messages) {
  const list = el("ul", null, "msgs");
  for (const m of messages) {
    const item = el("li");
    item.append(el("span", m.username, "who"), document.createTextNode(m.removed ? "(removed by an admin)" : m.body), el("span", when(m.at), "at"));
    list.append(item);
  }
  return list;
}

function composer(key, label, send) {
  const box = el("div", null, "compose"), text = document.createElement("textarea"), button = el("button", "Send");
  text.rows = 2; text.maxLength = 4000; text.setAttribute("aria-label", label); text.value = drafts.get(key) || "";
  text.oninput = () => drafts.set(key, text.value);
  button.type = "button";
  button.onclick = async () => {
    if (!text.value.trim()) return;
    button.disabled = true;
    try { await send(text.value); drafts.delete(key); text.value = ""; }
    catch (error) { status(`Not sent: ${error.message}`); }
    button.disabled = false;
  };
  box.append(text, button);
  return box;
}

// Threads: one per change, read top to bottom; a reply tells the change's author, approver and revoker.
async function loadThreads() {
  const holder = byId("threads");
  if (holder.contains(document.activeElement) && document.activeElement.tagName === "TEXTAREA") return;
  try {
    const { threads } = await request(`/api/reviews/threads?tail=${encodeURIComponent(tail())}`);
    holder.replaceChildren();
    if (!threads.length) holder.append(el("p", "No thread yet. Open a change and write in its thread, or revoke an approval.", "muted"));
    for (const review of threads) {
      const box = el("article", null, "thread"), title = el("h3");
      const open = el("button", `${review.repository} · ${review.branch}`, "row"); open.type = "button"; open.style.marginTop = "0";
      open.onclick = () => show(review.id);
      open.append(badge(review.standing));
      title.append(open);
      box.append(title, el("small", `${review.head.slice(0, 12)} · by ${review.owner_name}` +
        (review.reviewer_name ? ` · decided by ${review.reviewer_name}` : "")));
      if (review.standing === "revoked")
        box.append(el("p", `Approval revoked by ${review.revoked_by} ${when(review.revoked_at)}: ${review.revoke_reason}`, "revoked"));
      box.append(messageList(review.comments));
      box.append(composer(`thread:${review.id}`, `Reply in the thread of ${review.branch}`, async body => {
        await request("/api/comments", { kind: "review", ref: review.id, body }); await loadThreads(); }));
      holder.append(box);
    }
  } catch (error) { holder.replaceChildren(el("p", `Threads could not be read: ${error.message}`, "muted")); }
}

// The reviewers' chat: one persistent room of the operators and admins.
async function loadRoom() {
  const chat = byId("chat");
  if (roomOpen === null) {                  // the chat server says which rooms this account may enter
    try { roomOpen = (await request("/api/chat/rooms")).rooms.some(r => r.room === "reviewers"); } catch (error) { roomOpen = false; }
  }
  byId("chat-compose").hidden = !roomOpen;
  if (!roomOpen) { byId("chat-note").textContent = "The reviewers' chat is for operators and admins; a change's thread is open to its author."; chat.replaceChildren(); return; }
  if (document.activeElement === byId("chat-text") && byId("chat-text").value) return;
  try {
    const { messages: room } = await request(`/api/chat/rooms/reviewers?tail=${encodeURIComponent(tail())}`);
    chat.replaceChildren(...messageList(room).children);
    if (!room.length) chat.append(el("li", "No message yet.", "muted"));
  } catch (error) { chat.replaceChildren(el("li", `The chat could not be read: ${error.message}`, "muted")); }
}

async function sendRoom() {
  const text = byId("chat-text");
  if (!text.value.trim()) return;
  try { await request("/api/chat/rooms/reviewers", { body: text.value }); text.value = ""; await loadRoom(); }
  catch (error) { status(`Not sent: ${error.message}`); }
}

async function revokeApproval(review, button) {
  const reason = prompt(`Revoke the approval of ${review.repository} ${review.branch}? It stops landing and opens its thread; ` +
    "its author and approver are told. Why?");
  if (!reason || !reason.trim()) return;
  button.disabled = true;
  try { await request("/api/reviews/revoke", { id: review.id, reason }); status(`Approval of ${review.branch} revoked.`); }
  catch (error) { status(`Not revoked: ${error.message}`); }
  button.disabled = false;
  await load();
}

function render() {
  const list = byId("list");
  list.replaceChildren();
  const shown = byId("settled").checked ? reviews : reviews.filter(r => OPEN.has(r.standing));
  for (const review of shown) {
    const row = el("button", null, "row"); row.type = "button";
    const title = el("strong", `${review.repository} · ${review.branch}`);
    row.append(title, badge(review.standing),
      el("small", `${review.owner_name} · commit ${review.head.slice(0, 12)} · review ${review.id.slice(0, 12)} · submitted ${when(review.created_at)}`
        + (review.together?.length ? ` · with ${review.together.map(t => `${t.repository} ${t.branch}`).join(", ")}` : "")));
    row.setAttribute("aria-current", String(review.id === selected));
    row.onclick = () => show(review.id);
    list.append(row);
  }
  if (!shown.length) list.append(el("p", byId("settled").checked ? "No changes yet." : "No change is waiting.", "muted"));
}

async function load({quiet = false} = {}) {
  try {
    // Revoked changes that were marked read or revoked over a week ago are left out unless settled changes are shown.
    const result = await request(byId("settled").checked ? "/api/reviews?all=1" : "/api/reviews");
    const priorStanding = reviews.find(r => r.id === selected)?.standing;
    reviews = result.reviews;
    landing = { queue: result.queue || [], updater: Boolean(result.updater), apply: result.apply !== false,
      schedule: result.automatic_landing || null,
      clockOffset: Number.isFinite(result.server_now) ? result.server_now - Date.now() / 1000 : 0, lander: result.lander || null };
    renderLander();
    renderAwaiting();
    renderQueue();
    const open = reviews.filter(r => OPEN.has(r.standing)).length;
    status(`${open} change${open === 1 ? "" : "s"} in play. Lock: ${result.lock}.` +
      (result.hidden ? ` ${result.hidden} revoked change${result.hidden === 1 ? " is" : "s are"} hidden (marked read or revoked over a week ago); show settled changes to see them.` : "") +
      (who.role === "admin" ? "" : " Only an admin account can approve or reject; you can agree or object."));
    const approvable = reviews.filter(r => r.standing === "waiting" || r.standing === "accepted");
    byId("approve-all").hidden = who.role !== "admin" || !approvable.length;
    byId("approve-all").textContent = `Approve all ${approvable.length} waiting`;
    render();
    loadBundles();
    loadThreads();
    loadRoom();
    if (selected && reviews.some(r => r.id === selected) && (!quiet || priorStanding !== reviews.find(r => r.id === selected).standing)) show(selected);
  } catch (error) { status(error.message); }
}

// The approval feed, above the landing queue: every change still waiting for a decision (or held by an objection), newest
// first, refreshed with the page. Opening one shows its commits and diff, where the decision is made on exactly that commit.
function renderAwaiting() {
  const list = byId("awaiting-list");
  list.replaceChildren();
  const awaiting = reviews.filter(r => r.standing === "waiting" || r.standing === "objected" || r.standing === "held")
    .sort((a, b) => b.created_at - a.created_at);
  byId("awaiting-note").textContent = !awaiting.length ? "No change is waiting for approval." :
    `${awaiting.length} change${awaiting.length === 1 ? "" : "s"} waiting for a decision. ` +
    (who.role === "admin" ? "Open one to read its diff and approve or reject it." : "Open one to read its diff, agree or object.");
  for (const review of awaiting) {
    const row = el("div", null, "queue-row"), text = el("div");
    const title = el("button", `${review.repository} · ${review.branch}`, "row"); title.type = "button";
    title.style.marginTop = "0";
    title.onclick = async () => { await show(review.id); byId("detail").scrollIntoView({ block: "start" }); };
    title.append(badge(review.standing));
    text.append(title, el("small", `commit ${review.head.slice(0, 12)} · review ${review.id.slice(0, 12)} · by ${review.owner_name} · submitted ${when(review.created_at)}` +
      (review.standing === "waiting" ? ` · accepted by silence ${until(review.accept_after)}` : "")));
    if (review.note) text.append(el("small", review.note));
    row.append(text);
    list.append(row);
  }
}

// The landing queue: what a review accepted and has not landed yet, with the updater's last word on it. "Land now" asks the
// updater for one reviews.land task at once instead of waiting for its schedule; the result shows here on the next refresh.
function renderQueue() {
  const list = byId("queue-list"), admin = who.role === "admin";
  list.replaceChildren();
  const { queue, updater } = landing;
  updateQueueTimer();
  byId("queue-note").textContent = !queue.length ? "Nothing is waiting to land." :
    `${queue.length} accepted change${queue.length === 1 ? "" : "s"} not landed yet. A change that is not a fast-forward of its base stays held until its branch takes the base again.` +
    (!landing.apply ? " The policy's apply is off: the updater only says what it would do." : "") +
    (admin && !updater ? " This server cannot reach the updater, so landing now is not offered (they land on the updater's schedule)." : "");
  const landAll = byId("land-all");
  landAll.hidden = !admin || !updater || !queue.length;
  landAll.textContent = `Land all ${queue.length}`;
  for (const review of queue) {
    const row = el("div", null, "queue-row"), text = el("div");
    const title = el("button", `${review.repository} · ${review.branch}`, "row"); title.type = "button";
    title.style.marginTop = "0"; title.onclick = () => show(review.id);
    title.append(badge(review.standing));
    text.append(title, el("small", `${review.head.slice(0, 12)} · by ${review.owner_name}` +
      (review.reviewer_name ? ` · approved by ${review.reviewer_name} ${when(review.reviewed_at)}` : " · accepted by its time-out")));
    text.append(review.land_tried_at == null ? el("small", "Not tried yet.")
      : el("small", `Last tried ${when(review.land_tried_at)}: ${review.land_result}`, "held"));
    row.append(text);
    if (moderator() && review.standing === "approved") {
      const revoke = el("button", "Revoke approval", "reject"); revoke.type = "button";
      revoke.onclick = () => revokeApproval(review, revoke);
      row.append(revoke);
    }
    if (admin && updater) {
      const button = el("button", "Land now", "approve"); button.type = "button";
      button.onclick = () => landNow({ ids: [review.id] }, `${review.repository} ${review.branch}`, button);
      row.append(button);
    }
    list.append(row);
  }
}

// The lander's own state: its last run, a run in progress, and a warning when one has run too long or failed.
function renderLander() {
  const line = byId("lander"), lander = landing.lander;
  line.classList.remove("lander-warn");
  if (!lander) { line.textContent = ""; return; }
  const parts = [];
  const { running, last } = lander;
  if (running) {
    const minutes = Math.floor(running.seconds / 60);
    parts.push(running.stuck
      ? `The lander has been running since ${when(running.started)} (${minutes} min, ${running.trigger}): it is probably stuck, for example on a Git sign-in nobody sees. Check the task worker's log.`
      : `The lander is running (${running.trigger}, since ${when(running.started)}).`);
    if (running.stuck) line.classList.add("lander-warn");
  }
  if (last) {
    const took = Math.max(0, Math.round(last.finished - last.started));
    parts.push(last.error
      ? `Its last run (${last.trigger}, ${when(last.finished)}) failed: ${last.error}`
      : `Last landing run: ${when(last.finished)} (${last.trigger}, ${took} s): ${last.summary}.`);
    if (last.error) line.classList.add("lander-warn");
  }
  line.textContent = parts.join(" ") || "The lander has not run since this record began.";
}

function updateQueueTimer() {
  const timer = byId("queue-timer");
  if (!landing.queue.length) { timer.textContent = "No accepted review is waiting to land."; return; }
  if (!landing.apply) { timer.textContent = "Automatic landing is in report-only mode."; return; }
  if (!landing.schedule || !landing.updater) {
    timer.textContent = "The automatic landing time is unavailable; refresh for the updater's latest result.";
    return;
  }
  const seconds = Math.max(0, Math.ceil(landing.schedule.next_at - Date.now() / 1000 - landing.clockOffset));
  const minutes = Math.floor(seconds / 60), remainder = seconds % 60;
  timer.textContent = seconds ? `Next automatic landing attempt in ${minutes}:${String(remainder).padStart(2, "0")} ` +
    `(${when(landing.schedule.next_at)}).` : "Scheduled landing time reached; waiting for the updater's result.";
}

async function landNow(body, what, button) {
  if (!confirm(`Ask the updater to land ${what} now? It lands only a fast-forward of the base and says why it holds anything else.`)) return;
  button.disabled = true;
  let message;
  try {
    const result = await request("/api/reviews/land", body);
    const said = result.outcome.outcome === "done" ? "the updater has run" :
      result.outcome.outcome === "timeout" ? "the updater is still working; refresh in a minute" : `the updater answered: ${result.outcome.outcome}`;
    message = `Asked to land ${result.asked.length} change${result.asked.length === 1 ? "" : "s"}: ${said}.`;
  } catch (error) { message = `Not asked: ${error.message}`; }
  button.disabled = false;
  await load();
  status(message);
}

// Approve every change that is waiting or already accepted by its time-out, each at exactly the commit listed. A change
// with a standing objection is left for a decision of its own, and the confirmation names every branch and commit.
async function approveAll() {
  const approvable = reviews.filter(r => r.standing === "waiting" || r.standing === "accepted");
  const objected = reviews.filter(r => r.standing === "objected").length;
  if (!approvable.length) return;
  const listing = approvable.map(r => `  ${r.repository} ${r.branch} at ${r.head.slice(0, 12)}`).join("\n");
  if (!confirm(`Approve these ${approvable.length} changes at exactly these commits?\n\n${listing}` +
               (objected ? `\n\n${objected} with an objection are left for you to decide one by one.` : ""))) return;
  const button = byId("approve-all");
  button.disabled = true;
  const failed = [];
  for (const review of approvable) {
    try { await request("/api/reviews/decide", { id: review.id, head: review.head, decision: "approved", note: "approved with approve all" }); }
    catch (error) { failed.push(`${review.branch}: ${error.message}`); }
  }
  button.disabled = false;
  await load();
  status(`Approved ${approvable.length - failed.length} of ${approvable.length}.` +
         (failed.length ? " Not approved: " + failed.join("; ") : "") + " The admin tool lands them; a branch behind its base is held.");
}

function logLines(output) {
  return output.split("\n").filter(Boolean).map(line => {
    const [hash, time, author, ...subject] = line.split("\t");
    return { hash, time: Number(time), author, subject: subject.join("\t") };
  });
}

// The change's own commits are the head's history that the base branch does not hold; the newest commit the base
// does hold is where the change departs from it, so the diff is read from there and shows the change alone.
async function changeOf(review) {
  const [head, base] = await Promise.all([
    request("/api/git/log", { repository: review.repository, ref: review.head, count: 80 }),
    request("/api/git/log", { repository: review.repository, ref: review.base, count: 200 }),
  ]);
  const inBase = new Set(logLines(base.output).map(c => c.hash));
  const commits = logLines(head.output);
  const own = commits.filter(c => !inBase.has(c.hash));
  const departure = commits.find(c => inBase.has(c.hash)) || null;
  const diff = await request("/api/git/diff", { repository: review.repository, base: departure ? departure.hash : review.base, head: review.head });
  return { own, departure, diff };
}

function diffView(text) {
  const pre = el("div", null, "diff");
  const lines = text.split("\n");
  for (const line of lines.slice(0, DIFF_LINES)) {
    let cls = "";
    if (line.startsWith("diff --git")) cls = "file";
    else if (line.startsWith("@@")) cls = "hunk";
    else if (line.startsWith("+") && !line.startsWith("+++")) cls = "add";
    else if (line.startsWith("-") && !line.startsWith("---")) cls = "del";
    pre.append(el("span", line + "\n", cls));
  }
  if (lines.length > DIFF_LINES) pre.append(el("span", `… ${lines.length - DIFF_LINES} more lines not shown\n`, "hunk"));
  return pre;
}

async function post(path, body, feedback) {
  try {
    const { review } = await request(path, body);
    await load();
    status(`Saved: ${review.repository} ${review.branch} is now ${review.standing}. ` + byId("message").textContent);
  }
  catch (error) { feedback.textContent = error.message; }
}

function controls(review, detail) {
  const settled = !OPEN.has(review.standing);
  if (settled) return;
  const box = el("section", null, "decide");
  const feedback = el("p", null, "muted"); feedback.setAttribute("role", "status");
  const noteLabel = el("label", "Note or hold reason"), note = document.createElement("textarea");
  note.id = "review-note"; noteLabel.htmlFor = note.id; note.rows = 3; note.maxLength = 2000;
  const buttons = el("div", null, "actions");
  if (who.role === "admin" && review.standing === "held") {
    box.append(el("h3", "Held for review"), el("p", "Release gives this review a fresh 30-minute window. Decide it separately after release.", "muted"));
    const release = el("button", "Release hold"); release.type = "button";
    release.onclick = () => { if (confirm(`Release the hold on ${review.repository} ${review.branch}?`))
      post("/api/reviews/release", { id: review.id, note: note.value }, feedback); };
    buttons.append(release);
  } else if (who.role === "admin") {
    box.append(el("h3", "Decide"), el("p", `Approving lands ${review.branch} at ${review.head.slice(0, 12)} on ${review.base} when the admin tool next runs. A newer commit on the branch is a different change and needs its own review.`, "muted"));
    const approve = el("button", "Approve this commit", "approve"); approve.type = "button";
    const reject = el("button", "Reject", "reject"); reject.type = "button";
    const hold = el("button", "Hold for review"); hold.type = "button";
    approve.onclick = () => { if (confirm(`Approve ${review.repository} ${review.branch} at ${review.head.slice(0, 12)}?`))
      post("/api/reviews/decide", { id: review.id, head: review.head, decision: "approved", note: note.value }, feedback); };
    reject.onclick = () => { if (confirm(`Reject ${review.repository} ${review.branch}?`))
      post("/api/reviews/decide", { id: review.id, head: review.head, decision: "rejected", note: note.value }, feedback); };
    hold.onclick = () => { if (!note.value.trim()) { feedback.textContent = "Enter a reason for the hold."; return; }
      post("/api/reviews/hold", { id: review.id, reason: note.value }, feedback); };
    buttons.append(approve, reject, hold);
  } else box.append(el("h3", "Remark"));
  const mine = String(review.owner_name).toLowerCase() === String(who.username).toLowerCase();
  if (review.standing === "revoked" && !review.hidden && (mine || who.role === "admin")) {
    const read = el("button", "Mark read"); read.type = "button";
    read.title = "Hide this revoked change from the default list. Nothing is deleted.";
    read.onclick = () => post("/api/reviews/read", { id: review.id }, feedback);
    buttons.append(read);
  }
  if (mine) {
    if (review.standing !== "held") {
      const withdraw = el("button", "Withdraw my change"); withdraw.type = "button";
      withdraw.onclick = () => { if (confirm("Withdraw this change?")) post("/api/reviews/withdraw", { id: review.id }, feedback); };
      buttons.append(withdraw);
    }
  } else {
    for (const verdict of ["agree", "object"]) {
      const b = el("button", verdict === "agree" ? "Agree" : "Object"); b.type = "button";
      b.onclick = () => post("/api/reviews/remark", { id: review.id, verdict, note: note.value }, feedback);
      buttons.append(b);
    }
  }
  box.append(noteLabel, note, buttons, feedback);
  detail.append(box);
  if (moderator() && review.standing !== "redelegated")
    detail.append(redelegateBox("review", review.id, `${review.repository} ${review.branch}`));
}

// The consolidation and redelegation queues (review_bundles.py): only flagged work is held there.
let bundleState = { bundles: [], checked_out: [], redelegations: [], may_flag: false, repositories: [] };
const picked = new Set();          // "repository branch" keys ticked for a new bundle, kept across refreshes

// "title: detail", one separate path per line.
function pathsFrom(text) {
  return text.split("\n").map(line => line.trim()).filter(Boolean).map(line => {
    const at = line.indexOf(":");
    return at > 0 ? { title: line.slice(0, at).trim(), detail: line.slice(at + 1).trim() } : { title: line, detail: "" };
  });
}

function redelegateBox(kind, id, label) {
  const box = el("div", null, "decide");
  const area = document.createElement("textarea"); area.rows = 4; area.maxLength = 30000;
  area.id = `redelegate-${kind}-${id}`;
  const areaLabel = el("label", "Separate paths, one per line (title: detail)"); areaLabel.htmlFor = area.id;
  const why = document.createElement("textarea"); why.rows = 2; why.maxLength = 2000; why.id = `${area.id}-why`;
  const whyLabel = el("label", "Why (optional)"); whyLabel.htmlFor = why.id;
  const feedback = el("p", null, "muted"); feedback.setAttribute("role", "status");
  const go = el("button", `Redelegate ${label} into a ticket`); go.type = "button";
  go.onclick = async () => {
    const paths = pathsFrom(area.value);
    if (paths.length < 2) { feedback.textContent = "Name at least two separate paths."; return; }
    if (!confirm(`Redelegate ${label}? It will not land; one issue ticket names its ${paths.length} paths.`)) return;
    try {
      const { redelegation } = await request("/api/reviews/redelegate", { kind, id, paths, note: why.value });
      status(`Redelegated into ticket ${redelegation.ticket_id}.`);
      await load();
    } catch (error) { feedback.textContent = error.message; }
  };
  const actions = el("div", null, "actions");
  actions.append(go);
  box.append(el("h3", "Redelegate"), areaLabel, area, whyLabel, why, actions, feedback);
  return box;
}

function renderFlagForm(rows, repositories) {
  const select = byId("bundle-repo"), chosen = select.value || rows[0]?.repository || repositories[0] || "";
  select.replaceChildren(...repositories.map(name => { const option = el("option", name); option.value = name; return option; }));
  select.value = chosen;
  select.onchange = renderBundles;
  const picks = byId("bundle-picks");
  picks.replaceChildren();
  for (const row of rows.filter(r => r.repository === select.value)) {
    const key = `${row.repository} ${row.branch}`;
    const label = el("label"), box = document.createElement("input");
    box.type = "checkbox"; box.checked = picked.has(key); box.disabled = Boolean(row.bundle);
    box.onchange = () => { if (box.checked) picked.add(key); else picked.delete(key); };
    const text = `${row.branch} — ${row.owner || "unregistered"}` + (row.checked_out ? "" : " (registered, not checked out)")
      + (row.review ? ` · review ${row.review.id.slice(0, 8)} ${row.review.standing}` : "")
      + (row.bundle ? ` · in bundle ${row.bundle.slice(0, 8)}` : "");
    label.append(box, el("span", text));
    picks.append(label);
  }
  if (!picks.children.length) picks.append(el("p", "No branch of this repository is checked out.", "muted"));
  byId("bundle-make").onclick = async () => {
    const prefix = select.value + " ";
    const branches = [...picked].filter(key => key.startsWith(prefix)).map(key => key.slice(prefix.length));
    const feedback = byId("bundle-feedback");
    if (branches.length < 2) { feedback.textContent = "Tick at least two branches."; return; }
    if (!byId("bundle-reason").value.trim()) { feedback.textContent = "Say why they are to be merged together."; return; }
    try {
      const { bundle, tell } = await request("/api/reviews/bundles",
        { repository: select.value, branches, reason: byId("bundle-reason").value });
      branches.forEach(branch => picked.delete(prefix + branch));
      byId("bundle-reason").value = "";
      feedback.textContent = `Bundle ${bundle.id.slice(0, 8)} made; told ${tell.join(", ") || "nobody else"}. `
        + "Copy its notice below into each conversation working on these branches.";
      await load();
    } catch (error) { feedback.textContent = error.message; }
  };
}

// Close one branch of a bundle, with its reason (review_bundles.close): its owner, an operator or an admin. Removing
// it is carried out by the cleanup account at its next run; archive keeps the branch's final commit as archive/<branch>.
function closeMemberBox(bundle, member) {
  const box = el("details", null, "close-member");
  box.append(el("summary", "Close"));
  const key = `${bundle.id}-${member.branch}`.replace(/[^A-Za-z0-9_-]/g, "-");
  const reason = document.createElement("input"); reason.id = `close-reason-${key}`; reason.size = 40;
  reason.placeholder = "why: abandoned on purpose, superseded, needs a continuation report…";
  const reasonLabel = el("label", "Reason"); reasonLabel.htmlFor = reason.id;
  const what = document.createElement("select"); what.id = `close-what-${key}`;
  for (const [value, text] of [["keep", "close, keep the branch"], ["archive", "close and remove, archive its work"],
                               ["discard", "close and remove, discard its work"]]) {
    const option = el("option", text); option.value = value; what.append(option);
  }
  const whatLabel = el("label", "Branch"); whatLabel.htmlFor = what.id;
  const go = el("button", "Close this branch"); go.type = "button";
  const feedback = el("small", null, "muted"); feedback.setAttribute("role", "status");
  go.onclick = async () => {
    if (!reason.value.trim()) { feedback.textContent = "Say why it is closed."; return; }
    const remove = what.value !== "keep";
    if (remove && !confirm(`Remove ${member.branch}? The cleanup account removes its worktree, checkout and branch at `
      + `its next run${what.value === "archive" ? ", keeping its final commit as archive/" + member.branch : ""}.`)) return;
    try {
      await request("/api/reviews/bundles/close", { bundle: bundle.id, branch: member.branch, reason: reason.value,
        remove, ...(remove ? { work: what.value } : {}) });
      status(`${member.branch} closed.`);
      await load();
    } catch (error) { feedback.textContent = error.message; }
  };
  box.append(reasonLabel, reason, whatLabel, what, go, feedback);
  return box;
}

function bundleItem(bundle, mayFlag) {
  const item = el("div", null, "bundle");
  const head = el("div");
  head.append(el("strong", `${bundle.repository} · bundle ${bundle.id.slice(0, 8)}`), badge(bundle.status));
  item.append(head, el("small", `by ${bundle.created_by} ${when(bundle.created_at)} · ${bundle.reason}`
    + (bundle.settle_note ? ` · ${bundle.settle_note}` : ""), "muted"));
  const members = el("ul", null, "commits");
  for (const member of bundle.members) {
    const li = el("li", `${member.branch} — ${member.owner || "unregistered"}`);
    if (member.review) li.append(" · review ", el("code", member.review.id.slice(0, 8)), badge(member.review.standing));
    if (member.closed) {
      const closed = member.closed;
      li.append(badge("closed"), el("small", ` by ${closed.closed_by} ${when(closed.closed_at)}: ${closed.reason}`
        + (closed.remove ? ` · to be removed, work ${closed.work === "archive" ? "archived" : "discarded"}`
          + (closed.carried_at ? ` · done ${when(closed.carried_at)}: ${closed.carried}` : " · at the cleanup's next run")
          : " · branch kept"), "muted"));
    } else if (bundle.status === "consolidating") {
      li.append(closeMemberBox(bundle, member));
    }
    members.append(li);
  }
  const copy = el("button", "Copy notice"); copy.type = "button";
  copy.onclick = async () => {
    try { await navigator.clipboard.writeText(bundle.notice); status("Notice copied."); }
    catch (error) { status("Select the notice and copy it."); }
  };
  const copyRow = el("div", null, "actions");
  copyRow.append(copy);
  item.append(members, el("div", bundle.notice, "notice-text"), copyRow);
  if (bundle.status !== "consolidating") return item;
  const feedback = el("p", null, "muted"); feedback.setAttribute("role", "status");
  const meta = document.createElement("input"); meta.id = `meta-${bundle.id}`; meta.placeholder = "meta review id"; meta.size = 34;
  const metaLabel = el("label", "Meta review (its commit must contain every branch above that is not closed)");
  metaLabel.htmlFor = meta.id;
  const resolve = el("button", "Resolve with this meta review", "approve"); resolve.type = "button";
  resolve.onclick = async () => {
    try {
      await request("/api/reviews/bundles/consolidate", { bundle: bundle.id, id: meta.value.trim() });
      status(`Bundle ${bundle.id.slice(0, 8)} resolved.`);
      await load();
    } catch (error) { feedback.textContent = error.message; }
  };
  const actions = el("div", null, "actions");
  actions.append(resolve);
  if (mayFlag) {
    const dissolve = el("button", "Dissolve"); dissolve.type = "button";
    dissolve.onclick = async () => {
      const reason = prompt("Why is this bundle dissolved?");
      if (!reason) return;
      try { await request("/api/reviews/bundles/dissolve", { bundle: bundle.id, reason }); await load(); }
      catch (error) { feedback.textContent = error.message; }
    };
    actions.append(dissolve);
  }
  item.append(metaLabel, meta, actions, feedback);
  if (mayFlag) item.append(redelegateBox("bundle", bundle.id, `bundle ${bundle.id.slice(0, 8)}`));
  return item;
}

function renderBundles() {
  const { bundles, checked_out: rows, redelegations, may_flag: mayFlag, repositories } = bundleState;
  byId("bundle-flag").hidden = !mayFlag;
  if (mayFlag) renderFlagForm(rows, repositories);
  const list = byId("bundle-list");
  list.replaceChildren(...bundles.map(bundle => bundleItem(bundle, mayFlag)));
  if (!bundles.length) list.append(el("p", "No bundle. Nothing is waiting for consolidation.", "muted"));
  const done = byId("redelegation-list");
  done.replaceChildren();
  for (const made of redelegations) {
    const item = el("div", null, "bundle");
    item.append(el("strong", `${made.repository} · ${made.source_kind} ${made.source_id.slice(0, 8)} → ticket ${made.ticket_id.slice(0, 8)}`),
      el("small", ` by ${made.created_by} ${when(made.created_at)}` + (made.note ? ` · ${made.note}` : ""), "muted"));
    const paths = el("ol");
    for (const path of made.paths) paths.append(el("li", path.detail ? `${path.title} — ${path.detail}` : path.title));
    const link = el("a", "Open the issue tickets"); link.href = "/issues.html";
    item.append(paths, link);
    done.append(item);
  }
  if (!redelegations.length) done.append(el("p", "No redelegation yet.", "muted"));
}

async function loadBundles() {
  try { bundleState = await request("/api/reviews/bundles"); renderBundles(); }
  catch (error) { byId("bundle-list").replaceChildren(el("p", `Could not read the bundles: ${error.message}`, "muted")); }
}

async function show(id) {
  selected = id;
  render();
  const review = reviews.find(r => r.id === id);
  if (!review) return;
  const generation = ++detailGeneration;
  const detail = byId("detail");
  const heading = el("h2", `${review.repository} · ${review.branch}`); heading.append(badge(review.standing));
  const facts = document.createElement("dl");
  for (const [term, value] of [
    ["Lands on", review.base], ["Review ID", review.id], ["Commit", review.head], ["Author", review.owner_name], ["Submitted", when(review.created_at)],
    ["Accepted after", review.standing === "held" ? "Paused for further review" : `${when(review.accept_after)} (${until(review.accept_after)})`],
    ["Branch after landing", review.delete_branch ? "deleted" : "kept"],
    ...(review.reviewer_name ? [["Decided by", `${review.reviewer_name}, ${when(review.reviewed_at)}`]] : []),
    ...(review.note ? [["Note", review.note]] : []),
    ...(review.revoked_at ? [["Approval revoked", `by ${review.revoked_by}, ${when(review.revoked_at)}: ${review.revoke_reason}`]] : []),
    ...(review.held_at ? [["Held for review", `by ${review.held_by}, ${when(review.held_at)}: ${review.held_reason}`]] : []),
    ...(review.landed_at ? [["Landed", `${when(review.landed_at)} ${review.landed_detail || ""}`]] : []),
    ...(review.together?.length ? [["Lands together with", review.together.map(t => `${t.repository} ${t.branch} (${t.standing})`).join("; ")
      + (review.together_ready ? "" : " — waits until every one is accepted or approved")
      + ". The lander lands a ready set in one run; a push by hand can still move one alone."]] : []),
  ]) { facts.append(el("dt", term)); const dd = el("dd"); dd.append(el(term === "Commit" || term === "Review ID" ? "code" : "span", value)); facts.append(dd); }
  detail.replaceChildren(heading, facts);
  detail.append(el("p", "Reviewer remarks are formal agree or object votes. Discussion comments are in Threads.", "muted"));
  if (review.remarks.length) {
    detail.append(el("h3", "Reviewer remarks"));
    for (const r of review.remarks) detail.append(el("p", `${r.username}: ${r.verdict}${r.note ? " — " + r.note : ""} (${when(r.at)})`));
  }
  if (review.hold_events?.length) {
    detail.append(el("h3", "Hold history"));
    for (const event of review.hold_events)
      detail.append(el("p", `${event.actor} ${event.action} ${when(event.at)}${event.reason ? ": " + event.reason : ""}`));
  }
  controls(review, detail);
  const thread = el("section", null, "decide");
  thread.append(el("h3", "Thread"), el("p", "Write to this change's author, approver and revoker; they are told, and the " +
    "thread is listed under Threads.", "muted"),
    composer(`thread:${review.id}`, `Write in the thread of ${review.branch}`, async body => {
      await request("/api/comments", { kind: "review", ref: review.id, body }); status(`Written in the thread of ${review.branch}.`);
      await loadThreads(); }));
  detail.append(thread);
  const change = el("section");
  detail.append(change);
  if (!caps.has("core_git")) {
    change.append(el("p", "This account does not hold core_git, so the commits and diff cannot be read here.", "muted"));
    return;
  }
  change.append(el("p", "Reading the change…", "muted"));
  try {
    const { own, departure, diff } = await changeOf(review);
    if (generation !== detailGeneration) return;
    change.replaceChildren(el("h3", `Commits (${own.length})`));
    const list = el("ul", null, "commits");
    for (const c of own) {
      const li = el("li"); li.append(el("code", c.hash.slice(0, 10)), document.createTextNode(` ${c.subject} — ${c.author}, ${when(c.time)}`));
      list.append(li);
    }
    change.append(list, el("h3", "Diff"),
      el("p", departure ? `From ${departure.hash.slice(0, 12)}, where the branch departs from ${review.base}.`
                        : `From the tip of ${review.base}: no shared commit in its last 200, so this may include the base's newer changes reversed.`, "muted"));
    if (diff.truncated) change.append(el("p", `The server cut this diff at ${diff.bytes} bytes.`, "muted"));
    change.append(diffView(diff.output));
  } catch (error) {
    if (generation === detailGeneration) change.replaceChildren(el("p", `Could not read the change: ${error.message}`, "muted"));
  }
}

async function main() {
  try { who = await request("/api/whoami"); } catch (error) { status(error.message); return; }
  caps = new Set(who.capabilities || []);
  if (!who.signed_in) { byId("signin").hidden = false; status("Sign in to read the reviews."); return; }
  if (!caps.has("dev_ui")) { status("Reviews need the dev_ui capability."); return; }
  byId("content").hidden = false;
  byId("refresh").onclick = load;
  byId("approve-all").onclick = approveAll;
  byId("land-all").onclick = () => landNow({ all: true }, `all ${landing.queue.length} queued changes`, byId("land-all"));
  byId("settled").onchange = () => load();
  byId("tail").onchange = () => { loadThreads(); loadRoom(); };
  byId("chat-send").onclick = sendRoom;
  await load();
  setInterval(updateQueueTimer, 1000);
  setInterval(() => { if (!document.hidden) load({quiet: true}); }, 20000);
}
main();
