// Change reviews: the rows of /api/reviews (branch_review.py), each change's own commits and diff read through the
// core's Git over REST (core_git.py: log, diff), and the admin's decision on exactly the commit shown.
// HAND-AUTHORED. createElement/textContent only. The server enforces every capability; this only decides what to show.
const byId = id => document.getElementById(id);
const el = (tag, text, cls) => { const n = document.createElement(tag); if (text != null) n.textContent = text; if (cls) n.className = cls; return n; };
const OPEN = new Set(["waiting", "objected", "accepted"]);
const DIFF_LINES = 6000;
let who = null, caps = new Set(), reviews = [], selected = null, detailGeneration = 0;

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

async function load() {
  try {
    const result = await request("/api/reviews");
    reviews = result.reviews;
    const open = reviews.filter(r => OPEN.has(r.standing)).length;
    status(`${open} change${open === 1 ? "" : "s"} in play. Lock: ${result.lock}.` +
      (who.role === "admin" ? "" : " Only an admin account can approve or reject; you can agree or object."));
    render();
    if (selected && reviews.some(r => r.id === selected)) show(selected);
  } catch (error) { status(error.message); }
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
    ...(review.landed_at ? [["Landed", `${when(review.landed_at)} ${review.landed_detail || ""}`]] : []),
  ]) { facts.append(el("dt", term)); const dd = el("dd"); dd.append(el(term === "Commit" ? "code" : "span", value)); facts.append(dd); }
  detail.replaceChildren(heading, facts);
  if (review.remarks.length) {
    detail.append(el("h3", "Remarks"));
    for (const r of review.remarks) detail.append(el("p", `${r.username}: ${r.verdict}${r.note ? " — " + r.note : ""} (${when(r.at)})`));
  }
  controls(review, detail);
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
  byId("settled").onchange = render;
  await load();
}
main();
