// Change reviews: the rows of /api/reviews (branch_review.py), each change's own commits and diff read through the
// core's Git over REST (core_git.py: log, diff), and the admin's decision on exactly the commit shown.
// HAND-AUTHORED. createElement/textContent only. The server enforces every capability; this only decides what to show.
const byId = id => document.getElementById(id);
const el = (tag, text, cls) => { const n = document.createElement(tag); if (text != null) n.textContent = text; if (cls) n.className = cls; return n; };
const OPEN = new Set(["waiting", "objected", "accepted", "revoked"]);
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

function messageList(messages) {
  const list = el("ul", null, "msgs");
  for (const m of messages) {
    const item = el("li");
    item.append(el("span", m.username, "who"), document.createTextNode(m.body), el("span", when(m.at), "at"));
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
        await request("/api/reviews/comment", { id: review.id, body }); await loadThreads(); }));
      holder.append(box);
    }
  } catch (error) { holder.replaceChildren(el("p", `Threads could not be read: ${error.message}`, "muted")); }
}

// The reviewers' chat: one persistent room of the operators and admins.
async function loadRoom() {
  const chat = byId("chat");
  byId("chat-compose").hidden = !moderator();
  if (!moderator()) { byId("chat-note").textContent = "The reviewers' chat is for operators and admins."; chat.replaceChildren(); return; }
  if (document.activeElement === byId("chat-text") && byId("chat-text").value) return;
  try {
    const { room } = await request(`/api/reviews/room?tail=${encodeURIComponent(tail())}`);
    chat.replaceChildren(...messageList(room).children);
    if (!room.length) chat.append(el("li", "No message yet.", "muted"));
  } catch (error) { chat.replaceChildren(el("li", `The chat could not be read: ${error.message}`, "muted")); }
}

async function sendRoom() {
  const text = byId("chat-text");
  if (!text.value.trim()) return;
  try { await request("/api/reviews/room", { body: text.value }); text.value = ""; await loadRoom(); }
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
      el("small", `${review.owner_name} · ${review.head.slice(0, 12)} · submitted ${when(review.created_at)}`));
    row.setAttribute("aria-current", String(review.id === selected));
    row.onclick = () => show(review.id);
    list.append(row);
  }
  if (!shown.length) list.append(el("p", byId("settled").checked ? "No changes yet." : "No change is waiting.", "muted"));
}

async function load({quiet = false} = {}) {
  try {
    const result = await request("/api/reviews");
    const priorStanding = reviews.find(r => r.id === selected)?.standing;
    reviews = result.reviews;
    landing = { queue: result.queue || [], updater: Boolean(result.updater), apply: result.apply !== false,
      schedule: result.automatic_landing || null,
      clockOffset: Number.isFinite(result.server_now) ? result.server_now - Date.now() / 1000 : 0 };
    renderQueue();
    const open = reviews.filter(r => OPEN.has(r.standing)).length;
    status(`${open} change${open === 1 ? "" : "s"} in play. Lock: ${result.lock}.` +
      (who.role === "admin" ? "" : " Only an admin account can approve or reject; you can agree or object."));
    const approvable = reviews.filter(r => r.standing === "waiting" || r.standing === "accepted");
    byId("approve-all").hidden = who.role !== "admin" || !approvable.length;
    byId("approve-all").textContent = `Approve all ${approvable.length} waiting`;
    render();
    loadThreads();
    loadRoom();
    if (selected && reviews.some(r => r.id === selected) && (!quiet || priorStanding !== reviews.find(r => r.id === selected).standing)) show(selected);
  } catch (error) { status(error.message); }
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
  const noteLabel = el("label", "Note (kept with the decision)"), note = document.createElement("textarea");
  note.id = "review-note"; noteLabel.htmlFor = note.id; note.rows = 3; note.maxLength = 2000;
  const buttons = el("div", null, "actions");
  if (who.role === "admin") {
    box.append(el("h3", "Decide"), el("p", `Approving lands ${review.branch} at ${review.head.slice(0, 12)} on ${review.base} when the admin tool next runs. A newer commit on the branch is a different change and needs its own review.`, "muted"));
    const approve = el("button", "Approve this commit", "approve"); approve.type = "button";
    const reject = el("button", "Reject", "reject"); reject.type = "button";
    approve.onclick = () => { if (confirm(`Approve ${review.repository} ${review.branch} at ${review.head.slice(0, 12)}?`))
      post("/api/reviews/decide", { id: review.id, head: review.head, decision: "approved", note: note.value }, feedback); };
    reject.onclick = () => { if (confirm(`Reject ${review.repository} ${review.branch}?`))
      post("/api/reviews/decide", { id: review.id, head: review.head, decision: "rejected", note: note.value }, feedback); };
    buttons.append(approve, reject);
  } else box.append(el("h3", "Remark"));
  if (String(review.owner_name).toLowerCase() === String(who.username).toLowerCase()) {
    const withdraw = el("button", "Withdraw my change"); withdraw.type = "button";
    withdraw.onclick = () => { if (confirm("Withdraw this change?")) post("/api/reviews/withdraw", { id: review.id }, feedback); };
    buttons.append(withdraw);
  } else {
    for (const verdict of ["agree", "object"]) {
      const b = el("button", verdict === "agree" ? "Agree" : "Object"); b.type = "button";
      b.onclick = () => post("/api/reviews/remark", { id: review.id, verdict, note: note.value }, feedback);
      buttons.append(b);
    }
  }
  box.append(noteLabel, note, buttons, feedback);
  detail.append(box);
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
    ["Lands on", review.base], ["Commit", review.head], ["Author", review.owner_name], ["Submitted", when(review.created_at)],
    ["Accepted after", `${when(review.accept_after)} (${until(review.accept_after)})`],
    ["Branch after landing", review.delete_branch ? "deleted" : "kept"],
    ...(review.reviewer_name ? [["Decided by", `${review.reviewer_name}, ${when(review.reviewed_at)}`]] : []),
    ...(review.note ? [["Note", review.note]] : []),
    ...(review.revoked_at ? [["Approval revoked", `by ${review.revoked_by}, ${when(review.revoked_at)}: ${review.revoke_reason}`]] : []),
    ...(review.landed_at ? [["Landed", `${when(review.landed_at)} ${review.landed_detail || ""}`]] : []),
  ]) { facts.append(el("dt", term)); const dd = el("dd"); dd.append(el(term === "Commit" ? "code" : "span", value)); facts.append(dd); }
  detail.replaceChildren(heading, facts);
  if (review.remarks.length) {
    detail.append(el("h3", "Remarks"));
    for (const r of review.remarks) detail.append(el("p", `${r.username}: ${r.verdict}${r.note ? " — " + r.note : ""} (${when(r.at)})`));
  }
  controls(review, detail);
  const thread = el("section", null, "decide");
  thread.append(el("h3", "Thread"), el("p", "Write to this change's author, approver and revoker; they are told, and the " +
    "thread is listed under Threads.", "muted"),
    composer(`thread:${review.id}`, `Write in the thread of ${review.branch}`, async body => {
      await request("/api/reviews/comment", { id: review.id, body }); status(`Written in the thread of ${review.branch}.`);
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
  byId("settled").onchange = render;
  byId("tail").onchange = () => { loadThreads(); loadRoom(); };
  byId("chat-send").onclick = sendRoom;
  await load();
  setInterval(updateQueueTimer, 1000);
  setInterval(() => { if (!document.hidden) load({quiet: true}); }, 20000);
}
main();
