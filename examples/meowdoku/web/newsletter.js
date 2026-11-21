// Newsletter: the server's articles (newsletter.py). HAND-AUTHORED. createElement and textContent only: an article's
// Markdown is rendered into elements, never into HTML, so no article can put markup or script on the page.
const byId = id => document.getElementById(id);
const el = (tag, text, cls) => { const n = document.createElement(tag); if (text != null) n.textContent = text; if (cls) n.className = cls; return n; };
let who = { signed_in: false }, lists = { published: [] }, selected = new URLSearchParams(location.search).get("a");

async function request(path, body) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store",
    ...(body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }) });
  let result = {};
  try { result = await response.json(); } catch (error) { /* no body */ }
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}
const status = message => { byId("message").textContent = message; };
const when = s => s == null ? "" : new Date(s * 1000).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
const operator = () => (who.capabilities || []).includes("account_manage") || who.role === "operator" || who.role === "admin";

// ---- Markdown, as elements: headings, paragraphs, lists, quotes, code blocks, inline code, bold, italic, http(s) links.
function inline(parent, text) {
  const pattern = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*|_[^_]+_)|(\[[^\]]+\]\((https?:\/\/[^\s)]+)\))|(https?:\/\/[^\s<]+)/g;
  let last = 0, m;
  while ((m = pattern.exec(text))) {
    if (m.index > last) parent.append(document.createTextNode(text.slice(last, m.index)));
    if (m[1]) parent.append(el("code", m[1].slice(1, -1)));
    else if (m[2]) parent.append(el("strong", m[2].slice(2, -2)));
    else if (m[3]) parent.append(el("em", m[3].slice(1, -1)));
    else if (m[4]) { const a = el("a", m[4].slice(1, m[4].indexOf("]("))); a.href = m[5]; a.rel = "noopener noreferrer"; parent.append(a); }
    else { const a = el("a", m[6]); a.href = m[6]; a.rel = "noopener noreferrer"; parent.append(a); }
    last = pattern.lastIndex;
  }
  if (last < text.length) parent.append(document.createTextNode(text.slice(last)));
  return parent;
}
function markdown(source) {
  const out = el("div", null, "article"), lines = source.replace(/\r\n/g, "\n").split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      const code = []; i++;
      while (i < lines.length && !/^```/.test(lines[i])) code.push(lines[i++]);
      i++; const pre = el("pre"); pre.append(el("code", code.join("\n"))); out.append(pre); continue;
    }
    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) { out.append(inline(el("h" + Math.min(6, heading[1].length + 2)), heading[2])); i++; continue; }
    if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line), list = el(ordered ? "ol" : "ul");
      while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) list.append(inline(el("li"), lines[i++].replace(/^\s*([-*]|\d+\.)\s+/, "")));
      out.append(list); continue;
    }
    if (/^>\s?/.test(line)) {
      const quote = []; while (i < lines.length && /^>\s?/.test(lines[i])) quote.push(lines[i++].replace(/^>\s?/, ""));
      out.append(inline(el("blockquote"), quote.join(" "))); continue;
    }
    if (!line.trim()) { i++; continue; }
    const para = []; while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|```|>|\s*([-*]|\d+\.)\s)/.test(lines[i])) para.push(lines[i++]);
    out.append(inline(el("p"), para.join(" ")));
  }
  return out;
}

function row(item, list) {
  const b = el("button", item.title, "row"); b.type = "button"; b.dataset.id = item.id;
  if (item.status !== "published" || item.revision_of) b.append(el("span", item.revision_of ? "new version" : item.status, "badge " + item.status));
  b.append(el("small", `${item.author_name} · ${when(item.published_at || item.updated)}`));
  b.setAttribute("aria-current", String(item.id === selected));
  b.onclick = () => open(item.id);
  list.append(b);
}

async function load() {
  try {
    who = await request("/api/whoami").catch(() => ({ signed_in: false }));
    lists = await request("/api/newsletter");
    for (const [name, box] of [["published", null], ["waiting", "waiting-box"], ["mine", "mine-box"]]) {
      const list = byId(name); list.replaceChildren();
      for (const item of lists[name] || []) row(item, list);
      if (box) byId(box).hidden = !(lists[name] || []).length;
    }
    if (!lists.published.length) byId("published").append(el("p", "Nothing is published yet.", "muted"));
    byId("new").hidden = !who.signed_in; byId("signin").hidden = !!who.signed_in;
    status(`${lists.published.length} published article${lists.published.length === 1 ? "" : "s"}` +
      (lists.waiting && lists.waiting.length ? `; ${lists.waiting.length} waiting for review.` : "."));
    if (selected) await open(selected, true);
  } catch (error) { status(error.message); }
}

function keyField() { const k = el("input"); k.type = "password"; k.autocomplete = "off"; k.placeholder = "the key you were given"; return k; }

async function open(id, quiet) {
  selected = id;
  const view = byId("view");
  try {
    const { article } = await request(`/api/newsletter/${id}`);
    history.replaceState(null, "", `?a=${id}`);
    view.replaceChildren(el("h2", article.title), el("p", `${article.author_name} · ${article.status === "published" ? "published " +
      when(article.published_at) : article.status}${article.decided_by ? " · reviewed by " + article.decided_by : ""}`, "muted"));
    if (article.note) view.append(el("p", `Reviewer's note: ${article.note}`, "notice"));
    if (article.waiting_version) view.append(el("p", "A new version of this article is waiting for review.", "notice"));
    view.append(markdown(article.body));
    if (operator() && article.status === "pending") view.append(reviewBox(article));
    if (who.signed_in && !article.revision_of && ["pending", "published"].includes(article.status)) view.append(authorBox(article));
    const thread = el("p", null, "part"); const a = el("a", "Discussion of this article"); a.href = `#thread`;
    a.onclick = event => { event.preventDefault(); showThread(article, thread); }; thread.append(a); view.append(thread);
  } catch (error) { if (!quiet) status(error.message); view.replaceChildren(el("p", error.message, "muted")); }
  for (const b of document.querySelectorAll(".row")) b.setAttribute("aria-current", String(b.dataset.id === id));
}

function reviewBox(article) {
  const box = el("div", null, "part"); box.append(el("h3", article.revision_of ? "Review this new version" : "Review this article"));
  const note = el("input"); note.type = "text"; note.placeholder = "a note for the author (optional)";
  const act = decision => async () => {
    try { await request(`/api/newsletter/${article.id}/decide`, { decision, note: note.value }); status(`${decision === "published" ? "Published" : "Declined"}: ${article.title}`); await load(); }
    catch (error) { status(`Not done: ${error.message}`); }
  };
  const yes = el("button", "Publish"), no = el("button", "Decline"); yes.onclick = act("published"); no.onclick = act("declined");
  const actions = el("div", null, "actions"); actions.append(yes, no); box.append(note, actions);
  return box;
}

function authorBox(article) {
  const box = el("details", null, "part"); box.append(el("summary", "I wrote this: edit, prove it, or take it back"));
  const key = keyField(), title = el("input"), body = el("textarea");
  title.type = "text"; title.value = article.title; body.value = article.body;
  const send = (path, extra, done) => async () => {
    try { const answer = await request(`/api/newsletter/${article.id}/${path}`, { key: key.value, ...extra() }); status(done(answer)); await load(); }
    catch (error) { status(`Not done: ${error.message}`); }
  };
  const edit = el("button", "Send the new version"), check = el("button", "Check the key"), back = el("button", "Take it back");
  edit.onclick = send("edit", () => ({ title: title.value, body: body.value }),
    a => a.article.status === "pending" && a.article.revision_of ? "The new version waits for review; the published one stays until then." : "Changed.");
  check.onclick = send("verify", () => ({}), a => a.original_poster ? "That key is this article's: you are its original poster." : "That is not this article's key.");
  back.onclick = send("withdraw", () => ({}), () => "Taken back.");
  const actions = el("div", null, "actions"); actions.append(edit, check, back);
  box.append(field("Your key", key), field("Title", title), field("Article (Markdown)", body), actions);
  return box;
}

function field(label, input) { const box = el("div"), l = el("label", label); input.id ||= `f-${Math.random().toString(36).slice(2)}`; l.htmlFor = input.id; box.append(l, input); return box; }

async function showThread(article, place) {
  try {
    const { comments } = await request(`/api/comments?kind=article&ref=${article.revision_of || article.id}&tail=all`);
    const list = el("ul"); for (const c of comments) list.append(el("li", `${c.username}: ${c.removed ? "(removed)" : c.body}`));
    if (!comments.length) list.append(el("li", "No comments yet.", "muted"));
    const text = el("textarea"); text.style.minHeight = "80px"; const post = el("button", "Comment");
    post.onclick = async () => { try { await request("/api/comments", { kind: "article", ref: article.revision_of || article.id, body: text.value }); showThread(article, place); } catch (error) { status(error.message); } };
    place.replaceChildren(el("h3", "Discussion"), list, ...(who.signed_in ? [field("Your comment (@name tells someone)", text), post] : []));
  } catch (error) { place.replaceChildren(el("p", `The discussion is not open to you: ${error.message}`, "muted")); }
}

function writeForm() {
  const view = byId("view"), title = el("input"), body = el("textarea"), send = el("button", operator() ? "Publish" : "Submit for review");
  title.type = "text";
  send.onclick = async () => {
    try {
      const answer = await request("/api/newsletter", { title: title.value, body: body.value });
      const key = el("p", null, "key"); key.append(el("strong", "Your key: "), el("code", answer.key));
      view.replaceChildren(el("h2", answer.article.status === "published" ? "Published" : "Submitted for review"), key,
        el("p", answer.note, "muted"), el("p", "Copy the key somewhere safe now.", "notice"));
      selected = answer.article.id; await load();
    } catch (error) { status(`Not sent: ${error.message}`); }
  };
  view.replaceChildren(el("h2", "Write an article"), el("p", operator() ? "As an operator, it is published at once." :
    "It is published once an operator accepts it.", "muted"), field("Title", title), field("Article (Markdown)", body), send);
}

byId("new").onclick = writeForm;
load();
