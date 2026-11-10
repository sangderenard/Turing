// Meowdoku site navigation chip: ONE self-contained ES module (no imports).  HAND-AUTHORED, not produced by the Turing compiler.
// Opt in: <script type="module" src="/site-nav.js" data-site-nav="allow"></script>.  Without that attribute it does nothing.
// Signed out: a small "Sign in" link to /login.html.  Signed in: "<username> · <server>" opening a menu of the pages this account's
// capabilities reach (GET /api/whoami), the Stable / Development server switch (GET/POST /api/route, dev_ui only) and Sign out
// (POST /api/account/logout).  Built with createElement/textContent only, inside a closed Shadow DOM; the server enforces every
// capability, this only decides what to show.
const VERSION = 1;

function allowed() {
  const here = import.meta.url;
  for (const s of document.querySelectorAll('script[type="module"][src]')) {
    if (s.src === here && s.getAttribute("data-site-nav") === "allow") return true;
  }
  return false;
}

function el(tag, props, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === "text") n.textContent = v;
    else if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const kid of kids) if (kid) n.append(kid);
  return n;
}

async function getJson(path) {
  const r = await fetch(path, { credentials: "same-origin", cache: "no-store" });
  if (!r.ok) throw new Error(path + " " + r.status);
  return r.json();
}

async function postJson(path, body) {
  const r = await fetch(path, { method: "POST", credentials: "same-origin", cache: "no-store",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  let j = {};
  try { j = await r.json(); } catch (e) { /* empty body */ }
  return { status: r.status, body: j };
}

const CSS = `
:host { all: initial; }
.wrap { position: fixed; top: 8px; right: 8px; z-index: 2147483000; font: 13px/1.3 system-ui, sans-serif; color: #e8e8ee; }
.chip { all: unset; cursor: pointer; padding: 5px 10px; border-radius: 14px; background: rgba(28,28,36,.82); border: 1px solid rgba(255,255,255,.18);
        color: #e8e8ee; font: inherit; text-decoration: none; display: inline-block; }
.chip:hover, .chip:focus-visible { background: rgba(48,48,62,.95); outline: 2px solid #7aa2ff; outline-offset: 1px; }
.chip.quiet { opacity: .55; } .chip.quiet:hover, .chip.quiet:focus-visible { opacity: 1; }
.menu { position: absolute; top: 34px; right: 0; min-width: 210px; background: #1c1c24; border: 1px solid rgba(255,255,255,.18);
        border-radius: 8px; padding: 6px; box-shadow: 0 6px 24px rgba(0,0,0,.5); }
.menu a, .menu button { all: unset; display: block; box-sizing: border-box; width: 100%; padding: 6px 10px; border-radius: 5px; cursor: pointer;
        color: #e8e8ee; font: inherit; }
.menu a:hover, .menu button:hover, .menu a:focus-visible, .menu button:focus-visible { background: #2c2c3a; }
.menu a[aria-current="page"] { color: #9fb8ff; }
.menu button[aria-pressed="true"] { color: #9fb8ff; font-weight: 600; }
.menu button:disabled { opacity: .45; cursor: default; }
.head { padding: 4px 10px 6px; color: #a8a8b8; font-size: 12px; }
.sep { height: 1px; margin: 5px 4px; background: rgba(255,255,255,.12); }
.note { padding: 4px 10px; color: #ffb3a8; font-size: 12px; }
`;

async function main() {
  if (!allowed()) { console.info("[site-nav] not loaded: this page's script tag lacks data-site-nav=\"allow\""); return; }
  let who;
  try { who = await getJson("/api/whoami"); } catch (e) { return; }
  const caps = new Set(who.capabilities || []);
  const host = document.createElement("div");
  host.style.cssText = "all:initial !important;display:block !important";
  const root = host.attachShadow({ mode: "closed" });
  root.append(el("style", { text: CSS }));
  const wrap = el("div", { class: "wrap" });
  root.append(wrap);
  document.body.append(host);

  if (!who.signed_in) {
    wrap.append(el("a", { class: "chip" + (who.shader ? " quiet" : ""), href: "/login.html", text: "Sign in" }));
    return;
  }

  let route = {};
  if (caps.has("dev_ui")) { try { route = await getJson("/api/route"); } catch (e) { route = {}; } }
  const onDev = !!(route.dev_route && route.route === route.dev_route);
  const serverName = caps.has("dev_ui") ? (onDev ? "development" : "stable") : "";
  const chip = el("button", { class: "chip", "aria-haspopup": "true", "aria-expanded": "false",
    text: who.username + (serverName ? " · " + serverName : "") });
  const menu = el("div", { class: "menu", role: "menu", hidden: "" });
  const note = el("div", { class: "note", hidden: "" });

  const here = location.pathname === "/index.html" ? "/" : location.pathname;
  function link(label, href, show) {
    if (show === false) return null;
    const a = el("a", { href, role: "menuitem", text: label });
    if (href === here) a.setAttribute("aria-current", "page");
    return a;
  }
  async function choose(site) {
    const res = await postJson("/api/route", { site });
    if (res.status === 200) { location.reload(); return; }
    note.textContent = (res.body && res.body.error) || "could not switch server";
    note.hidden = false;
  }
  function serverButton(label, site, pressed, disabled) {
    const b = el("button", { role: "menuitemradio", "aria-pressed": String(pressed), text: label, onclick: () => choose(site) });
    if (disabled) b.disabled = true;
    return b;
  }

  menu.append(el("div", { class: "head", text: "Signed in as " + who.username + " (" + (who.role || "member") + ")" }));
  for (const item of [
    link("Home", "/"),
    link("Profile and servers", "/profile.html"),
    link("Browse the site", "/browse"),
    link("Math lab", "/mathlab.html", caps.has("compile_submit")),
    link("Source divider", "/divider.html", caps.has("compile_submit")),
    link("Marble studio", "/marble_studio.html", caps.has("dev_ui")),
    link("Creature studio", "/mechanical_creature_studio.html", caps.has("dev_ui")),
    link("Account and password", "/login.html"),
  ]) if (item) menu.append(item);
  if (caps.has("dev_ui")) {
    menu.append(el("div", { class: "sep" }), el("div", { class: "head", text: "Server: " + (who.instance || serverName) }),
      serverButton("Stable server", "stable", !onDev, false),
      serverButton("Development server", "dev", onDev, route.dev_available === false));
  }
  menu.append(note, el("div", { class: "sep" }),
    el("button", { role: "menuitem", text: "Sign out", onclick: async () => { await postJson("/api/account/logout", {}); location.reload(); } }));

  function setOpen(open) {
    menu.hidden = !open;
    chip.setAttribute("aria-expanded", String(open));
  }
  chip.addEventListener("click", () => setOpen(menu.hidden));
  wrap.addEventListener("keydown", (e) => { if (e.key === "Escape") { setOpen(false); chip.focus(); } });
  document.addEventListener("pointerdown", (e) => { if (!menu.hidden && e.target !== host) setOpen(false); });
  wrap.append(chip, menu);
}

window.MeowdokuSiteNav = { version: VERSION };
main();
