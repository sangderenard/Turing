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

function card(page, mine = false) {
  const box = element("article", ""); box.className = "card";
  const href = "/p/" + page.id + "/";
  const poster = element("a", ""); poster.className = "poster"; poster.href = href;
  poster.dataset.kind = page.kind;
  poster.setAttribute("aria-label", "Open " + page.title);
  const mark = element("span", page.title.trim().slice(0, 1).toUpperCase());
  mark.className = "poster-mark"; poster.append(mark);
  const body = element("div", ""); body.className = "card-body";
  const badges = element("div", ""); badges.className = "badges";
  const stage = element("span", page.stage === "dev" ? "In development" : "Released");
  stage.className = "badge " + page.stage;
  const kind = element("span", page.kind === "bundle" ? "Compiled page" : "Uploaded page");
  kind.className = "badge"; badges.append(stage, kind);
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
    const change = element("button", page.stage === "dev" ? "Release" : "Return to dev");
    change.type = "button";
    change.onclick = async () => {
      if (page.stage === "dev" && !confirm("Release this page to the stable server and public feed?")) return;
      try {
        await post("/api/pages/" + page.id + "/stage", { stage: page.stage === "dev" ? "release" : "dev" });
        await refresh();
      } catch (error) { note(error.message); }
    };
    actions.append(preview, change); body.append(actions);
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

refresh().catch((error) => note(error.message));
