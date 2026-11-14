const $ = (id) => document.getElementById(id);
const note = (value) => { $("notice").textContent = value; };

async function request(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store", ...options });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
const post = (path, body) => request(path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});
const element = (tag, text) => { const out = document.createElement(tag); out.textContent = text; return out; };
let signedIn = false;

// Two views of the feed: icons (a grid of tiles) or a thread (one column, each page with its comments). A per-viewer
// convenience kept in this browser; the server keeps nothing of it.
function viewMode() { try { return localStorage.getItem("feed-view") === "thread" ? "thread" : "icons"; } catch { return "icons"; } }
function setView(mode) {
  try { localStorage.setItem("feed-view", mode); } catch { /* private window: the choice lasts this visit */ }
  $("view-icons").setAttribute("aria-pressed", String(mode === "icons"));
  $("view-thread").setAttribute("aria-pressed", String(mode === "thread"));
  for (const id of ["feed", "dev-feed"]) $(id).classList.toggle("thread", mode === "thread");
  if (mode === "thread") for (const box of document.querySelectorAll("#feed .card, #dev-feed .card")) loadThread(box);
}

// A page's comment thread (comments.py, kind "page"): its last 20 comments and, signed in, a reply box.
async function loadThread(box) {
  const pageId = box.dataset.page;
  if (!pageId || box.querySelector(".card-thread")) return;
  const thread = element("section", ""); thread.className = "card-thread"; thread.setAttribute("aria-label", "Comments");
  box.querySelector(".card-body").append(thread);
  if (!signedIn) { thread.append(element("p", "Sign in to read and write comments.")); return; }
  const list = document.createElement("ul");
  const draw = async () => {
    try {
      const { comments } = await request(`/api/comments?kind=page&ref=${encodeURIComponent(pageId)}&tail=20`);
      list.replaceChildren(...comments.map((c) => {
        const item = document.createElement("li");
        item.append(element("span", c.username), document.createTextNode(c.removed ? "(removed by an admin)" : c.body));
        item.firstChild.className = "who";
        return item;
      }));
      if (!comments.length) list.append(element("li", "No comments yet."));
    } catch (error) { list.replaceChildren(element("li", error.message)); }
  };
  const form = document.createElement("form");
  const text = document.createElement("textarea"); text.rows = 2; text.maxLength = 4000;
  text.setAttribute("aria-label", "Comment on this page");
  const send = element("button", "Comment"); send.type = "submit";
  form.append(text, send);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!text.value.trim()) return;
    try { await post("/api/comments", { kind: "page", ref: pageId, body: text.value }); text.value = ""; await draw(); }
    catch (error) { note(error.message); }
  });
  thread.append(list, form);
  await draw();
}

function card(page, mine = false) {
  const box = element("article", ""); box.className = "card"; box.dataset.page = page.id;
  const href = "/p/" + page.id + "/";
  const poster = element("a", ""); poster.className = "poster"; poster.href = href;
  poster.dataset.kind = page.kind;
  poster.setAttribute("aria-label", "Open " + page.title);
  const contract = page.page_contract;
  const posterPath = contract?.presentation?.poster;
  if (posterPath) {
    poster.style.backgroundImage = `linear-gradient(0deg, #10172288, transparent 60%), url("${href + posterPath.split("/").map(encodeURIComponent).join("/")}")`;
    poster.style.backgroundSize = "cover";
    poster.style.backgroundPosition = "center";
  }
  const mark = element("span", page.title.trim().slice(0, 1).toUpperCase());
  mark.className = "poster-mark"; poster.append(mark);
  const body = element("div", ""); body.className = "card-body";
  const badges = element("div", ""); badges.className = "badges";
  const stage = element("span", page.stage === "dev" ? "In development" : "Released");
  stage.className = "badge " + page.stage;
  const kind = element("span", page.kind === "bundle" ? "Compiled page" : "Uploaded page");
  kind.className = "badge"; badges.append(stage, kind);
  const badge = (label, title) => {
    const item = element("span", label); item.className = "badge";
    if (title) item.title = title;
    badges.append(item);
  };
  if (!contract) badge("Requirements unknown");
  else {
    badge(contract.presentation.surface.toUpperCase());
    const compute = contract.execution.compute;
    if (compute.length) badge("Compute · " + compute[0] + (compute.length > 1 ? ` +${compute.length - 1}` : ""), compute.join(", "));
    const caps = contract.access.required_capabilities;
    if (caps.length) badge("Access · " + caps[0] + (caps.length > 1 ? ` +${caps.length - 1}` : ""), caps.join(", "));
    badge(contract.execution.standalone === "requires-site" ? "Needs Meowdoku" : "Offline unverified");
  }
  const title = element("h3", "");
  const link = element("a", page.title); link.href = href; title.append(link);
  if (page.stage === "dev") {
    const openDev = async (event) => {
      event.preventDefault();
      try { await post("/api/route", { site: "dev" }); location.assign(href); }
      catch (error) { note(error.message); }
    };
    poster.addEventListener("click", openDev);
    link.addEventListener("click", openDev);
  }
  const author = element("a", "@" + page.author);
  author.href = "/?author=" + encodeURIComponent(page.author);
  const meta = element("p", ""); meta.className = "byline";
  meta.append(author, document.createTextNode(" · " + new Date(page.updated_at * 1000).toLocaleDateString()));
  body.append(badges, title, meta);
  if (page.description) { const description = element("p", page.description); description.className = "description"; body.append(description); }
  if (mine) {
    const actions = element("div", ""); actions.className = "row";
    const preview = element("button", page.stage === "dev" ? "Preview on dev" : "Open page");
    preview.type = "button";
    preview.onclick = async () => {
      try {
        if (page.stage === "dev") await post("/api/route", { site: "dev" });
        location.assign("/p/" + page.id + "/");
      } catch (error) { note(error.message); }
    };
    actions.append(preview);
    if (page.stage === "dev") {
      const bake = element("button", "Bake public release"); bake.type = "button";
      bake.onclick = async () => {
        if (!confirm("Create a separate public release of this dev page?")) return;
        try {
          const result = await post("/api/pages/" + page.id + "/bake", {});
          note("Public release baked. Your dev page remains available.");
          await refresh();
          $("mine").querySelector(`a[href="/p/${result.id}/"]`)?.focus();
        } catch (error) { note(error.message); }
      };
      actions.append(bake);
    }
    body.append(actions);
  }
  box.append(poster, body);
  return box;
}

async function refresh() {
  const author = new URLSearchParams(location.search).get("author");
  if (author) {
    $("author-heading").hidden = false;
    $("author-title").textContent = "Pages by @" + author;
  }
  const query = author ? "?author=" + encodeURIComponent(author) : "";
  const feed = await request("/api/pages" + query);
  const released = feed.pages.filter((page) => page.stage === "release");
  $("feed").replaceChildren(...released.map((page) => card(page)));
  $("release-count").textContent = released.length + " shown";
  if (!released.length) { const empty = element("p", "No released pages here yet."); empty.className = "empty"; $("feed").append(empty); }
  const who = await request("/api/whoami");
  signedIn = Boolean(who.signed_in);
  setView(viewMode());
  $("games-link").hidden = !who.signed_in && who.tier === "shader";
  $("files-link").hidden = !who.signed_in;
  if (!who.signed_in) return;
  const route = await request("/api/route");
  if (route.dev_route && route.route === route.dev_route && !(who.capabilities || []).includes("dev_ui")) {
    $("games-link").onclick = async (event) => {
      event.preventDefault();
      try { await post("/api/route", { site: "stable" }); location.assign("/games.html"); }
      catch (error) { note(error.message); }
    };
  }
  $("publish").hidden = false;
  const mine = await request("/api/pages?mine=1");
  $("mine").replaceChildren(...mine.pages.map((page) => card(page, true)));
  if (!mine.pages.length) $("mine").append(element("p", "You have not published a page."));
  const drafts = mine.pages.filter((page) => page.stage === "dev");
  $("dev-section").hidden = !drafts.length;
  $("dev-feed").replaceChildren(...drafts.map((page) => card(page)));
  $("dev-count").textContent = drafts.length + " shown";
  setView(viewMode());
}

$("publish-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("page-file").files[0];
  if (!file || file.size < 1 || file.size > 512 * 1024) { note("Choose a page no larger than 512 KiB."); return; }
  const name = file.name.toLowerCase();
  const contentType = name.endsWith(".html") || name.endsWith(".htm") ? "text/html" :
    name.endsWith(".txt") ? "text/plain" : null;
  if (!contentType) { note("Choose an HTML or text file."); return; }
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    let binary = "";
    for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
    const result = await post("/api/pages", {
      title: $("page-title").value, description: $("page-description").value,
      content_type: contentType, content_base64: btoa(binary), confirmed_public_copy: $("public-copy").checked,
    });
    $("publish-form").reset();
    note("Dev page created. Preview it before release.");
    await refresh();
    $("mine").querySelector(`a[href="/p/${result.id}/"]`)?.focus();
  } catch (error) { note(error.message); }
});

$("view-icons").addEventListener("click", () => setView("icons"));
$("view-thread").addEventListener("click", () => setView("thread"));
refresh().catch((error) => note(error.message));
