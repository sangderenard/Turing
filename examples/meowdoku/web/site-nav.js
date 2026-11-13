// Meowdoku site controls: ONE self-contained ES module (no imports).  HAND-AUTHORED, not produced by the Turing compiler.
// Opt in: <script type="module" src="/site-nav.js" data-site-nav="allow"></script>.  Without that attribute it does nothing.
// Signed out: a small "Sign in" link to /login.html.  Signed in: "<username> · <server>" opening a menu of the pages this account's
// capabilities reach (GET /api/whoami), the Stable / Development server switch (GET/POST /api/route, dev_ui only) and Sign out
// (POST /api/account/logout).  Built with createElement/textContent only, inside a closed Shadow DOM; the server enforces every
// capability, this only decides what to show.
const VERSION = 7;

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
.wrap { position: fixed; top: max(8px, env(safe-area-inset-top)); right: max(8px, env(safe-area-inset-right)); z-index: 2147483000; font: 13px/1.3 system-ui, sans-serif; color: #e8e8ee; display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 6px; max-width: calc(100vw - 32px); padding: 6px; border: 1px solid #454556; border-radius: 10px; background: #14141eee; }
.actions { display: contents; }
.chip.toggle { min-height: 32px; box-sizing: border-box; }
.label { font-size: 11px; color: #b6b6ca; margin: 0 4px; }
.chip { all: unset; cursor: pointer; padding: 5px 10px; border-radius: 14px; background: rgba(28,28,36,.82); border: 1px solid rgba(255,255,255,.18);
        color: #e8e8ee; font: inherit; text-decoration: none; display: inline-block; }
.chip:hover, .chip:focus-visible { background: rgba(48,48,62,.95); outline: 2px solid #7aa2ff; outline-offset: 1px; }
.chip.quiet { opacity: .55; } .chip.quiet:hover, .chip.quiet:focus-visible { opacity: 1; }
.menu { position: absolute; top: calc(100% + 6px); right: 0; min-width: 210px; max-width: calc(100vw - 48px); max-height: 65vh; overflow: auto; background: #1c1c24; border: 1px solid rgba(255,255,255,.18);
        border-radius: 8px; padding: 6px; box-shadow: 0 6px 24px rgba(0,0,0,.5); }
.menu a, .menu button { all: unset; display: block; box-sizing: border-box; width: 100%; padding: 6px 10px; border-radius: 5px; cursor: pointer;
        color: #e8e8ee; font: inherit; }
.menu a:hover, .menu button:hover, .menu a:focus-visible, .menu button:focus-visible { background: #2c2c3a; }
.menu a[aria-current="page"] { color: #9fb8ff; }
.menu button[aria-pressed="true"] { color: #9fb8ff; font-weight: 600; }
.menu button:disabled { opacity: .45; cursor: default; }
.head { padding: 4px 10px 6px; color: #a8a8b8; font-size: 12px; }
.sep { height: 1px; margin: 5px 4px; background: rgba(255,255,255,.12); }
.suite { position: absolute; right: 0; top: calc(100% + 6px); box-sizing: border-box; width: min(420px, calc(100vw - 32px)); max-height: min(70vh, 560px); overflow: auto; padding: 12px; border: 1px solid #454556; border-radius: 10px; background: #14141e; }
.binding { display: grid; gap: 3px; padding: 7px; margin: 4px 0; border: 1px solid #454556; border-radius: 4px; overflow-wrap: anywhere; }
.binding span { font-size: 11px; color: #b6b6ca; }
.binding.active { border-color: #ffd166; background: #80551e; }
.radio-bands { display: flex; gap: 4px; align-items: center; min-height: 30px; }
.radio-bands meter { flex: 1; min-width: 0; width: 40px; height: 24px; }
[hidden] { display: none !important; }
.note { padding: 4px 10px; color: #ffb3a8; font-size: 12px; }
`;

// Pages supply their existing policy and sampler. This is presentation only:
// no key listeners, pointer capture, device permissions or second control engine.
const suites = new Map();
let suiteMount = null;
let dismissHeaderPanel = null;
export function registerControlSuite(spec) {
  if (!spec || typeof spec.identity !== "string" || !spec.identity ||
      !Array.isArray(spec.policy?.bindings)) throw new TypeError("Control suite needs identity and policy.bindings");
  if (spec.sample !== undefined && typeof spec.sample !== "function") throw new TypeError("sample must be a function");
  if (suites.has(spec.identity)) throw new Error("Control suite already registered: " + spec.identity);
  if (spec.policy.bindings.some(binding => typeof binding.action !== "string" || !Array.isArray(binding.inputs) || binding.inputs.some(input => typeof input !== "string"))) throw new TypeError("Invalid control bindings");
  if ((spec.actions || []).some(action => typeof action.run !== "function")) throw new TypeError("Control action needs run callback");
  const record = {spec, dispose: null};
  suites.set(spec.identity, record);
  if (suiteMount) record.dispose = suiteMount(spec);
  return () => {
    if (suites.get(spec.identity) !== record) return;
    record.dispose?.();
    suites.delete(spec.identity);
  };
}

function mountControlSuite(wrap, host, spec) {
  const chip = el("button", {class: "chip", type: "button", text: spec.label || "Controls", "aria-expanded": "false"});
  const panel = el("section", {class: "suite", "aria-label": spec.label || "Controls", hidden: ""});
  const status = el("div", {class: "note", role: "status"});
  const close = el("button", {class: "chip", type: "button", text: "Close controls"});
  panel.append(close);
  const rows = spec.policy.bindings.map(binding => {
    const row = el("div", {class: "binding"});
    row.append(el("strong", {text: binding.action}), el("span", {text: binding.inputs.join(" · ")}));
    panel.append(row);
    return {row, action: binding.action};
  });
  for (const action of spec.actions || []) {
    if (typeof action.run !== "function") throw new TypeError("Control action needs run callback");
    const button = el("button", {class: "chip", type: "button", text: action.label});
    button.addEventListener("click", async () => {
      button.disabled = true;
      try { if (action.close) setOpen(false); await action.run(); status.textContent = ""; }
      catch (error) { status.textContent = String(error.message || error); }
      finally { button.disabled = false; }
    });
    panel.append(button);
  }
  panel.append(status);
  let frame = 0;
  function refresh() {
    if (panel.hidden) return;
    try {
      for (const {row, action} of rows) {
        const value = spec.sample?.(action);
        row.classList.toggle("active", Number.isFinite(value) && value !== 0);
      }
    } catch (error) { status.textContent = "Control telemetry unavailable: " + String(error.message || error); }
    if (spec.sample) frame = requestAnimationFrame(refresh);
  }
  function setOpen(open, focus = true) {
    if (open === !panel.hidden) return;
    try { (open ? spec.onOpen : spec.onClose)?.(); }
    catch (error) { status.textContent = String(error.message || error); return; }
    if (open && dismissHeaderPanel && dismissHeaderPanel !== dismiss) dismissHeaderPanel();
    dismissHeaderPanel = open ? dismiss : (dismissHeaderPanel === dismiss ? null : dismissHeaderPanel);
    panel.hidden = !open;
    chip.setAttribute("aria-expanded", String(open));
    cancelAnimationFrame(frame);
    if (open) { refresh(); if (focus) close.focus(); }
    else if (focus) chip.focus();
  }
  const dismiss = () => setOpen(false, false);
  chip.addEventListener("click", () => setOpen(panel.hidden));
  close.addEventListener("click", () => setOpen(false));
  panel.addEventListener("keydown", event => {
    if (event.key === "Escape") { event.stopPropagation(); setOpen(false); }
  });
  const outside = event => { if (!event.composedPath().includes(host)) setOpen(false, false); };
  document.addEventListener("pointerdown", outside);
  wrap.append(chip, panel);
  return () => {
    setOpen(false, false);
    cancelAnimationFrame(frame);
    document.removeEventListener("pointerdown", outside);
    chip.remove(); panel.remove();
  };
}

// Nodus owns capture/FFT; Python reads its existing UDP relay inbox.
// This widget shows NRAD bands, not a browser FFT or an invented audio stream.
function mountRadio(wrap, host, canControl = false) {
  const chip = el("button", {class: "chip", type: "button", text: "Nodus Radio", "aria-expanded": "false"});
  const panel = el("section", {class: "suite", hidden: "", "aria-label": "Nodus Radio"});
  const close = el("button", {class: "chip", type: "button", text: "Close radio"});
  const title = el("strong", {text: "Nodus Radio"});
  const status = el("p", {role: "status", text: "Waiting for Nodus"});
  const bands = el("div", {class: "radio-bands", "aria-label": "Spectral power bands"});
  panel.append(close, title, status, bands,
    el("p", {text: "Live spectrum from Nodus. This feed carries levels, not playable audio."}));
  const queueStatus = el("p", {role: "status", text: "Jukebox: checking configuration"});
  const songs = el("select", {"aria-label": "Song to request"});
  const request = el("button", {class: "chip", type: "button", text: "Request song"});
  const queue = el("div", {"aria-label": "Radio request lists"});
  const charts = el("div", {"aria-label": "Radio billboard"});
  const rating = el("select", {"aria-label": "Rate the playing song"});
  rating.append(el("option", {value: "", text: "No rating"}));
  for (let n = 0; n <= 5; n++) rating.append(el("option", {value: String(n), text: String(n)}));
  const vote = el("button", {class: "chip", type: "button", text: "Vote 0–5"});
  const adminButtons = [];
  let jukeboxBusy = false, currentPlay = null;
  request.disabled = vote.disabled = true;
  panel.append(queueStatus, songs, request, rating, vote, queue, charts);
  function showJukebox(data) {
    const catalogue = data.catalogue || [];
    const names = new Map(catalogue.map(s => [s.id, s.title]));
    const selected = songs.value;
    songs.replaceChildren(...catalogue.map(s => el("option", {value: s.id, text: s.title})));
    if (names.has(selected)) songs.value = selected;
    currentPlay = data.current?.id || null;
    queueStatus.textContent = data.error || data.runner?.error || (data.paused ? "Jukebox paused" : "Jukebox running") +
      (data.current ? ": " + (names.get(data.current.song_id) || data.current.song_id) : ": no song playing");
    request.disabled = jukeboxBusy || !catalogue.length;
    vote.disabled = jukeboxBusy || !currentPlay;
    for (const button of adminButtons) button.disabled = jukeboxBusy || !!data.error;
    queue.replaceChildren();
    for (const row of (data.requests || [])) {
      const line = el("div", {text: (row.mine ? "Your request: " : "Request: ") + (names.get(row.song_id) || row.song_id)});
      if (row.mine || canControl) {
        const remove = el("button", {type: "button", text: "Remove"});
        remove.disabled = jukeboxBusy;
        remove.addEventListener("click", () => commandJukebox({action: "remove", request_id: row.id}));
        line.append(remove);
      }
      queue.append(line);
    }
    charts.replaceChildren(el("strong", {text: "Billboard · average rating, then vote count"}));
    for (const row of (data.charts || [])) charts.append(el("p", {text:
      (names.get(row.song_id) || row.song_id) + " · " + Number(row.average).toFixed(2) + "/5 · " + row.votes + " votes"}));
  }
  async function commandJukebox(body) {
    if (jukeboxBusy) return;
    jukeboxBusy = true; request.disabled = vote.disabled = true;
    for (const button of adminButtons) button.disabled = true;
    const epoch = generation;
    try {
      const response = await fetch("/api/radio/jukebox", {method: "POST", credentials: "same-origin", cache: "no-store",
        headers: {"Content-Type": "application/json", "X-Meowdoku-Radio": "1"}, body: JSON.stringify(body)});
      const data = await response.json();
      jukeboxBusy = false;
      if (epoch === generation && !panel.hidden) showJukebox(data);
    } catch (error) {jukeboxBusy = false; if (epoch === generation && !panel.hidden) showJukebox({error: "Request failed; refresh status before retrying"});}
  }
  request.addEventListener("click", () => commandJukebox({action: "request", song_id: songs.value}));
  vote.addEventListener("click", () => {
    if (rating.value !== "" && currentPlay) commandJukebox({action: "rate", play_id: currentPlay, rating: Number(rating.value)});
  });
  if (canControl) for (const [action, label] of [["resume", "Start / resume jukebox"], ["pause", "Pause after this song"], ["skip", "Skip current song"]]) {
    const button = el("button", {class: "chip", type: "button", text: label});
    button.disabled = true; button.addEventListener("click", () => commandJukebox({action}));
    adminButtons.push(button); panel.append(button);
  }
  let paTimer = null, paBusy = false;
  const paStatus = el("p", {role: "status", text: "Server speakers: checking configuration"});
  const play = el("button", {class: "chip", type: "button", text: "Play on server speakers"});
  const stop = el("button", {class: "chip", type: "button", text: "Stop my server playback"});
  play.disabled = stop.disabled = true;
  function showPA(data) {
    paStatus.textContent = "Server speakers: " + (data.error || data.status || "unavailable");
    play.disabled = paBusy || !data.configured || !["stopped", "failed"].includes(data.status);
    stop.disabled = paBusy || !data.can_stop;
  }
  async function pollPA(epoch) {
    try {
      const data = await getJson("/api/radio/jukebox");
      if (epoch !== generation || panel.hidden) return;
      if (!jukeboxBusy) showJukebox(data);
    } catch (error) {
      if (epoch !== generation || panel.hidden) return;
      showJukebox({error: "Jukebox unavailable"});
    }
    if (canControl) try {
      const data = await getJson("/api/radio/playback");
      if (epoch !== generation || panel.hidden) return;
      showPA(data);
    } catch (error) {
      if (epoch !== generation || panel.hidden) return;
      showPA({error: "unavailable"});
    }
    if (epoch === generation && !panel.hidden) paTimer = setTimeout(() => pollPA(epoch), 1000);
  }
  async function commandPA(action) {
    if (paBusy) return;
    paBusy = true; play.disabled = stop.disabled = true;
    try {
      const response = await fetch("/api/radio/playback", {method: "POST", credentials: "same-origin", cache: "no-store",
        headers: {"Content-Type": "application/json", "X-Meowdoku-Radio": "1"}, body: JSON.stringify({action})});
      const data = await response.json();
      paBusy = false;
      showPA(data);
    } catch (error) {paBusy = false; showPA({error: "request failed; check status before retrying"});}
  }
  if (canControl) {
    panel.append(el("p", {text: "Admin PA control: plays on the server's selected speakers, not this browser. Continues until Stop or the source ends."}), paStatus, play, stop);
    play.addEventListener("click", () => commandPA("start"));
    stop.addEventListener("click", () => commandPA("stop"));
  }
  let generation = 0, timer = null;
  const dismiss = () => setOpen(false);
  async function poll(epoch) {
    try {
      const data = await getJson("/api/radio");
      if (epoch !== generation || panel.hidden) return;
      window.dispatchEvent(new CustomEvent("nodus-radio-signal", {detail: data}));
      title.textContent = data.title || "Nodus Radio";
      status.textContent = data.status === "live" ? "Live · RMS " + Number(data.rms).toFixed(3)
        : data.status === "stale" ? "Signal stopped · last packet " + Number(data.age_s).toFixed(1) + "s ago"
        : data.status === "waiting" ? "Waiting for Nodus packets" : "Radio is not broadcasting";
      bands.replaceChildren();
      for (const [index, value] of (data.bands || []).slice(0, 128).entries()) {
        const level = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
        const meter = el("meter", {min: "0", max: "1", value: String(data.status === "live" ? level : 0),
          "aria-label": "Band " + (index + 1), title: "Band " + (index + 1) + ": " + level.toFixed(3)});
        bands.append(meter);
      }
    } catch (error) {
      if (epoch !== generation || panel.hidden) return;
      window.dispatchEvent(new CustomEvent("nodus-radio-signal", {detail: {status: "unavailable"}}));
      status.textContent = "Radio unavailable";
      bands.replaceChildren();
    }
    if (epoch === generation && !panel.hidden) timer = setTimeout(() => poll(epoch), 100);
  }
  function setOpen(open) {
    if (open && dismissHeaderPanel && dismissHeaderPanel !== dismiss) dismissHeaderPanel();
    dismissHeaderPanel = open ? dismiss : (dismissHeaderPanel === dismiss ? null : dismissHeaderPanel);
    generation++; clearTimeout(timer); timer = null; clearTimeout(paTimer); paTimer = null;
    if (!open) window.dispatchEvent(new CustomEvent("nodus-radio-signal", {detail: {status: "unavailable"}}));
    panel.hidden = !open; chip.setAttribute("aria-expanded", String(open));
    if (open) {poll(generation); pollPA(generation);}
  }
  chip.addEventListener("click", () => setOpen(panel.hidden));
  close.addEventListener("click", () => {setOpen(false); chip.focus();});
  panel.addEventListener("keydown", event => {
    if (event.key === "Escape") {event.stopPropagation(); setOpen(false); chip.focus();}
  });
  document.addEventListener("pointerdown", event => {if (!event.composedPath().includes(host)) setOpen(false);});
  window.addEventListener("pagehide", () => setOpen(false));
  document.addEventListener("visibilitychange", () => {if (document.hidden) setOpen(false);});
  wrap.append(chip, panel);
}

async function main() {
  if (!allowed()) { console.info("[site-nav] not loaded: this page's script tag lacks data-site-nav=\"allow\""); return; }

  const host = document.createElement("div");
  host.style.cssText = "all:initial !important;display:block !important;flex:0 0 auto !important;grid-column:1 / -1 !important";
  const root = host.attachShadow({ mode: "closed" });
  root.append(el("style", { text: CSS }));
  const wrap = el("nav", { class: "wrap", "aria-label": "Site controls" });
  const actions = el("div", {class: "actions", id: "site-actions"});
  const toggle = el("button", {class: "chip toggle", type: "button",
    text: "Hide menu", "aria-expanded": "true", "aria-controls": "site-actions"});
  actions.append(el("span", { class: "label", text: "Site controls" }));
  wrap.append(actions, toggle);
  root.append(wrap);
  // The fixed banner used to cover the page's first controls. Reserve its
  // measured height in normal flow, including wrapping on narrow screens.
  document.body.prepend(host);
  function reserveSpace() {
    host.style.setProperty("height", Math.ceil(wrap.getBoundingClientRect().bottom + 8) + "px", "important");
  }
  toggle.addEventListener("click", () => {
    const collapse = !actions.hidden;
    if (collapse) dismissHeaderPanel?.();
    actions.hidden = collapse;
    toggle.textContent = collapse ? "Show site menu" : "Hide menu";
    toggle.setAttribute("aria-expanded", String(!collapse));
    reserveSpace();
    toggle.focus();
  });
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(reserveSpace).observe(wrap);
  window.addEventListener("resize", reserveSpace);
  reserveSpace();
  suiteMount = spec => mountControlSuite(actions, host, spec);
  for (const record of suites.values()) record.dispose = suiteMount(record.spec);
  window.dispatchEvent(new Event("site-controls-ready"));
  let who;
  try { who = await getJson("/api/whoami"); } catch (e) { return; }
  const caps = new Set(who.capabilities || []);

  if (!who.signed_in) {
    actions.append(el("a", { class: "chip" + (who.shader ? " quiet" : ""), href: "/login.html", text: "Sign in" }));
    reserveSpace();
    return;
  }

  if (caps.has("view_site")) mountRadio(actions, host, caps.has("account_manage"));

  if (!new URLSearchParams(location.search).has("viewer")) {
    // Tickets is a small menu: the account's issue tickets, and (dev_ui) the change reviews an admin approves.
    const ticketsChip = el("button", { class: "chip", type: "button", "aria-haspopup": "true", "aria-expanded": "false",
      text: "Tickets ▾", title: "Issue tickets and change reviews" });
    const ticketsMenu = el("div", { class: "menu", role: "menu", hidden: "" });
    ticketsMenu.append(el("a", { href: "/issues.html", role: "menuitem", text: "Issue tickets" }));
    if (caps.has("dev_ui")) ticketsMenu.append(el("a", { href: "/reviews.html", role: "menuitem",
      text: who.role === "admin" ? "Change reviews (approve)" : "Change reviews" }));
    const dismissTickets = () => setTicketsOpen(false);
    function setTicketsOpen(open) {
      if (open && dismissHeaderPanel && dismissHeaderPanel !== dismissTickets) dismissHeaderPanel();
      dismissHeaderPanel = open ? dismissTickets : (dismissHeaderPanel === dismissTickets ? null : dismissHeaderPanel);
      ticketsMenu.hidden = !open;
      ticketsChip.setAttribute("aria-expanded", String(open));
    }
    ticketsChip.addEventListener("click", () => setTicketsOpen(ticketsMenu.hidden));
    wrap.addEventListener("keydown", (e) => { if (e.key === "Escape" && !ticketsMenu.hidden) { setTicketsOpen(false); ticketsChip.focus(); } });
    document.addEventListener("pointerdown", (e) => { if (!ticketsMenu.hidden && e.target !== host) setTicketsOpen(false); });
    actions.append(ticketsChip, ticketsMenu);
    actions.append(el("button", { class: "chip", type: "button", text: "Report issue", onclick: () => {
      // Frozen game pages can still own the legacy /api/tickets form. The
      // shared header always opens the account-owned SQL issue interface.
      location.assign("/issues.html?new=1&source=" + encodeURIComponent(location.pathname));
    } }));
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
    link("My files", "/files.html", caps.has("message")),
    link("Paper markets", "/markets.html", caps.has("message")),
    link("Browse the site", "/browse"),
    link("Help: the manual", "/manual/", caps.has("view_site")),
    link("Repository world map", "/worldmap.html", caps.has("dev_ui")),
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

  const dismiss = () => setOpen(false);
  function setOpen(open) {
    if (open && dismissHeaderPanel && dismissHeaderPanel !== dismiss) dismissHeaderPanel();
    dismissHeaderPanel = open ? dismiss : (dismissHeaderPanel === dismiss ? null : dismissHeaderPanel);
    menu.hidden = !open;
    chip.setAttribute("aria-expanded", String(open));
  }
  chip.addEventListener("click", () => setOpen(menu.hidden));
  wrap.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) { setOpen(false); chip.focus(); } });
  document.addEventListener("pointerdown", (e) => { if (!menu.hidden && e.target !== host) setOpen(false); });
  actions.append(chip, menu);
  reserveSpace();
}

window.MeowdokuSiteNav = { version: VERSION, registerControlSuite };
main();
