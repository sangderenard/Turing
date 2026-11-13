// Meowdoku messenger widget: ONE self-contained ES module (no imports).  HAND-AUTHORED, not produced by the Turing compiler;
// "compiled" here only means this single built artifact that a page opts into.
//
// MESSAGING IS SERVER-MEDIATED AND OPERATOR-READABLE: every message passes through and is stored by the server (/api/messages/*),
// there is no peer-to-peer channel, and an administrator can read any thread, remove a message or suspend an account.  A contact
// sees only the other person's username and display name: no presence, typing or read receipts.  Not end-to-end encrypted.
//
// Opt in: <script type="module" src="/messenger-widget.js" data-messenger="allow"></script>
// Without data-messenger="allow" on that script tag the module does nothing (one console line).  When the visitor lacks the
// `message` capability (GET /api/whoami) it renders nothing at all.  Closed: only a small launcher button with an unread badge
// (a light unread check every 60 s).  Open: contact list, add-contact field, incoming requests, one conversation at a time, and a
// long poll that runs only while open.  All DOM is built with createElement/textContent (bodies are plain text: no links,
// embeds or markup, and nothing is fetched or navigated from message content); all styles live in a closed Shadow DOM.
// Global surface: window.MeowdokuMessenger = {open, close, version}.
const VERSION = "3";
const MAX_IMAGE_BYTES = 2097152;
function safeImageURL(image, id) {
  return image && image.locked === false && image.mime === "image/png" &&
    image.url === "/api/messages/image?id=" + id ? image.url : null;
}
async function pngAttachment(file) {
  if (!file || file.size > MAX_IMAGE_BYTES) throw new Error("Choose a PNG up to 2 MiB.");
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (bytes.length > MAX_IMAGE_BYTES || ![137,80,78,71,13,10,26,10].every((v,i) => bytes[i] === v)) throw new Error("Choose a PNG file.");
  let binary = "";
  for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return { data: btoa(binary) };
}

function allowed() {
  const here = import.meta.url;
  for (const s of document.getElementsByTagName("script")) {
    if (s.src === here && s.getAttribute("data-messenger") === "allow") return true;
  }
  return false;
}

const CSS = `
:host { all: initial; }
* { box-sizing: border-box; font: 14px/1.35 system-ui, sans-serif; }
.launch { position: fixed; right: 16px; bottom: 16px; z-index: 2147483000; width: 48px; height: 48px; border-radius: 24px; border: 1px solid #556;
  background: #1d2433; color: #fff; cursor: pointer; font-size: 22px; line-height: 1; }
.launch:focus-visible, button:focus-visible, input:focus-visible { outline: 2px solid #7aa7ff; outline-offset: 2px; }
.badge { position: absolute; top: -4px; right: -4px; min-width: 20px; padding: 1px 5px; border-radius: 10px; background: #d33; color: #fff; font-size: 12px; text-align: center; }
.panel { position: fixed; right: 16px; bottom: 72px; z-index: 2147483000; width: min(380px, calc(100vw - 32px)); height: min(520px, calc(100vh - 96px));
  display: flex; flex-direction: column; background: #12161f; color: #e8ecf4; border: 1px solid #556; border-radius: 10px; overflow: hidden; }
.bar { display: flex; align-items: center; gap: 8px; padding: 8px 10px; background: #1d2433; }
.bar .title { flex: 1; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
button { background: #2b3550; color: #fff; border: 1px solid #556; border-radius: 6px; padding: 4px 10px; cursor: pointer; }
button.small { padding: 2px 8px; font-size: 12px; }
input { background: #0b0e14; color: #e8ecf4; border: 1px solid #556; border-radius: 6px; padding: 6px 8px; min-width: 0; flex: 1; }
.row { display: flex; gap: 6px; padding: 6px 10px; align-items: center; }
.body { flex: 1; overflow-y: auto; padding: 6px 10px; }
.item { display: flex; align-items: center; gap: 8px; width: 100%; text-align: left; margin: 2px 0; }
.item .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.unread { background: #d33; border-radius: 10px; padding: 0 6px; font-size: 12px; }
.note { color: #9aa4b8; font-size: 12px; padding: 4px 10px; }
.msg { max-width: 85%; margin: 4px 0; padding: 5px 8px; border-radius: 8px; white-space: pre-wrap; overflow-wrap: anywhere; background: #232b3d; }
.msg.mine { margin-left: auto; background: #2d4a86; }
.msg.removed { font-style: italic; opacity: .7; }
.attachment { display:block; width:100%; max-width:280px; height:180px; object-fit:contain; }
.when { display: block; color: #9aa4b8; font-size: 11px; }
h3 { margin: 8px 0 2px; font-size: 12px; text-transform: uppercase; color: #9aa4b8; }
`;

function el(tag, props, ...kids) {
  const node = document.createElement(tag);
  if (props) for (const [k, v] of Object.entries(props)) {
    if (k === "text") node.textContent = v;
    else if (k === "on") for (const [ev, fn] of Object.entries(v)) node.addEventListener(ev, fn);
    else if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const kid of kids) if (kid) node.appendChild(kid);
  return node;
}

async function api(path, body) {
  const init = { credentials: "same-origin", cache: "no-store", redirect: "error" };
  if (body !== undefined) {
    init.method = "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  const res = await fetch(path, init);
  let data = null;
  try { data = await res.json(); } catch (e) {
    if (res.ok) throw new Error("Invalid JSON response; message outcome may be uncertain.");
  }
  if (!res.ok) {
    const error = new Error((data && data.error) || ("HTTP " + res.status));
    error.status = res.status;
    throw error;
  }
  if (res.ok && (!data || typeof data !== "object" || Array.isArray(data))) {
    throw new Error("Invalid API response; message outcome may be uncertain.");
  }
  return data;
}

async function start() {
  if (!allowed()) { console.info("[messenger] not loaded: this page's script tag lacks data-messenger=\"allow\""); return; }
  let who;
  try { who = await api("/api/whoami"); } catch (e) { return; }
  if (!who || !Array.isArray(who.capabilities) || who.capabilities.indexOf("message") < 0) return;

  const host = document.createElement("div");
  host.style.cssText = "all:initial !important;display:block !important";   // inline !important beats a host page stylesheet rule
  const root = host.attachShadow({ mode: "closed" });
  root.appendChild(el("style", { text: CSS }));
  document.body.appendChild(host);

  const state = { open: false, contacts: null, peer: null, seen: new Set(), after: -1, abort: null, panel: null, unread: 0, timer: null, view: null };
  const launcher = el("button", { class: "launch", type: "button", "aria-label": "Messages", "aria-expanded": "false", on: { click: () => (state.open ? close() : open()) } });
  launcher.appendChild(document.createTextNode("✉"));
  const badge = el("span", { class: "badge", hidden: "" });
  launcher.appendChild(badge);
  root.appendChild(launcher);

  function setUnread(n) {
    state.unread = n;
    badge.textContent = n > 99 ? "99+" : String(n);
    if (n > 0) badge.removeAttribute("hidden"); else badge.setAttribute("hidden", "");
    launcher.setAttribute("aria-label", n > 0 ? "Messages, " + n + " unread" : "Messages");
  }
  function countUnread(c) { return c.contacts.reduce((a, x) => a + (x.unread || 0), 0) + c.requests.length; }

  async function loadContacts() {
    state.contacts = await api("/api/messages/contacts");
    setUnread(countUnread(state.contacts));
    if (state.open && !state.peer) renderBody();   // an open thread re-renders itself; rebuilding here looped loadThread
  }
  async function lightCheck() { if (!state.open) { try { await loadContacts(); } catch (e) { /* keep the last badge */ } } }
  lightCheck();
  state.timer = setInterval(lightCheck, 60000);

  // ---- rendering ---------------------------------------------------------------------------------------------------------------
  let titleEl, bodyEl, footEl, noteEl;
  function buildPanel() {
    titleEl = el("span", { class: "title", text: "Messages" });
    const back = el("button", { type: "button", class: "small", text: "Back", hidden: "", on: { click: () => { state.peer = null; renderBody(); } } });
    const shut = el("button", { type: "button", class: "small", text: "Close", "aria-label": "Close messages", on: { click: close } });
    bodyEl = el("div", { class: "body", "aria-live": "polite" });
    footEl = el("div");
    noteEl = el("div", { class: "note", role: "status" });
    const panel = el("div", { class: "panel", role: "dialog", "aria-label": "Messages" },
      el("div", { class: "bar" }, back, titleEl, shut), bodyEl, noteEl, footEl);
    panel.addEventListener("keydown", (ev) => { if (ev.key === "Escape") { ev.stopPropagation(); close(); launcher.focus(); } });
    panel.back = back;
    return panel;
  }
  function note(text) { if (noteEl) noteEl.textContent = text || ""; }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function renderBody() {
    if (!state.panel) return;
    state.view = null;
    clear(bodyEl); clear(footEl);
    state.panel.back.toggleAttribute("hidden", !state.peer);
    if (state.peer) return renderThread();
    titleEl.textContent = "Messages";
    const c = state.contacts;
    if (!c) { bodyEl.appendChild(el("div", { class: "note", text: "Loading..." })); return; }
    if (c.requests.length) {
      bodyEl.appendChild(el("h3", { text: "Requests" }));
      for (const r of c.requests) {
        bodyEl.appendChild(el("div", { class: "row" }, el("span", { class: "name", text: label(r) }),
          el("button", { type: "button", class: "small", text: "Accept", on: { click: () => act("accept", r.username) } }),
          el("button", { type: "button", class: "small", text: "Decline", on: { click: () => act("decline", r.username) } })));
      }
    }
    bodyEl.appendChild(el("h3", { text: "Contacts" }));
    if (!c.contacts.length) bodyEl.appendChild(el("div", { class: "note", text: "No contacts yet." }));
    for (const p of c.contacts) {
      const item = el("button", { type: "button", class: "item", on: { click: () => selectPeer(p.username) } }, el("span", { class: "name", text: label(p) }));
      if (p.unread) item.appendChild(el("span", { class: "unread", text: String(p.unread) }));
      bodyEl.appendChild(item);
    }
    if (c.outgoing.length) {
      bodyEl.appendChild(el("h3", { text: "Waiting for a reply" }));
      for (const o of c.outgoing) bodyEl.appendChild(el("div", { class: "note", text: o.username }));
    }
    const input = el("input", { type: "text", placeholder: "Add contact by username", "aria-label": "Add contact by username", maxlength: "200", autocomplete: "off" });
    const add = () => { const v = input.value.trim(); if (v) { input.value = ""; act("request", v, "If that account exists, a request was sent."); } };
    input.addEventListener("keydown", (ev) => { if (ev.key === "Enter") add(); });
    footEl.appendChild(el("div", { class: "row" }, input, el("button", { type: "button", text: "Add", on: { click: add } })));
  }
  function label(p) { return p.display_name && p.display_name !== p.username ? p.display_name + " (" + p.username + ")" : p.username; }

  async function act(action, username, okNote) {
    try { await api("/api/messages/contacts", { username, action }); note(okNote || ""); await loadContacts(); }
    catch (e) { note(e.message); }
  }

  // ---- one conversation ---------------------------------------------------------------------------------------------------------
  const drafts = new Map();
  const pendingSends = new Map(); // Preserve retry identity across Back/Close within this page.
  let threadEl, olderBtn, oldest = null, newest = 0;
  function msgNode(m) {
    const mine = m.from === who.username;
    const node = el("div", { class: "msg" + (mine ? " mine" : "") + (m.removed ? " removed" : "") });
    if (!m.removed && m.subject) {
      if (m.subject.kind === "file" && /^[0-9a-f]{32}$/.test(m.subject.ref)) {
        node.appendChild(el("a", { class: "note", href: "/api/user-files/" + m.subject.ref,
                                   text: "Download shared file" }));
      } else node.appendChild(el("div", { class: "note", text: m.subject.kind + ": " + m.subject.ref }));
    }
    node.appendChild(document.createTextNode(m.removed ? "[removed by an administrator]" : (m.type && m.type !== "note" ? "[" + m.type + "] " : "") + m.body));
    if (!m.removed && m.image) {
      const url = safeImageURL(m.image, m.id);
      node.appendChild(url ? el("img", { class: "attachment", src: url, alt: "Chat image", loading: "lazy", referrerpolicy: "no-referrer" }) :
        el("div", { class: "note", text: "[Image locked. Both people must enable images in this chat to reveal it.]" }));
    }
    node.appendChild(el("span", { class: "when", text: new Date(m.created * 1000).toLocaleString() }));
    // A system announcement of a counted-down restart or shutdown: its subject names the control request, and the
    // button interrupts that request.  The server decides whether this account may; the button only asks.
    if (!mine && !m.removed && m.type === "approval_request" && m.subject && m.subject.kind === "control") {
      const stop = el("button", { class: "small", text: "Interrupt", on: { click: async () => {
        stop.disabled = true;
        try { await api("/api/control/instances/cancel", { id: Number(m.subject.ref) }); stop.textContent = "Interrupted"; }
        catch (e) { stop.textContent = e.message; }
      } } });
      node.appendChild(el("div", { class: "note" }, stop));
    }
    return node;
  }
  async function selectPeer(username) {
    state.peer = username; oldest = null; newest = 0;
    renderBody();
  }
  async function renderThread() {
    const peer = state.peer;
    const view = { nodes: new Map(), imageState: null, refreshing: false, toggling: false };
    state.view = view;
    const active = () => state.open && state.peer === peer && state.view === view;
    const viewNote = text => { if (active()) note(text); };
    const info = state.contacts && state.contacts.contacts.find((x) => x.username === peer);
    titleEl.textContent = info ? label(info) : peer;
    threadEl = el("div");
    olderBtn = el("button", { type: "button", class: "small", text: "Load older", hidden: "", on: { click: () => loadThread(true) } });
    bodyEl.appendChild(olderBtn); bodyEl.appendChild(threadEl);
    const input = el("input", { type: "text", placeholder: "Message (Enter sends)", "aria-label": "Message to " + peer, maxlength: "4000", autocomplete: "off" });
    const draft = drafts.get(peer) || { body: "", image: null };
    drafts.set(peer, draft);
    input.value = draft.body;
    let image = draft.image;
    let choosing = false;
    const imageStatus = el("div", { class: "note", text: "Images: loading chat setting..." });
    const toggle = el("button", { type: "button", text: "Enable images" });
    toggle.disabled = true;
    const chooser = el("input", { type: "file", accept: "image/png", "aria-label": "Attach PNG (up to 2 MiB)" });
    chooser.disabled = true;
    const selected = el("span", { class: "note", text: image ? "PNG attached" : "" });
    const removeImage = el("button", { type: "button", text: "Clear image" });
    const updateControls = () => {
      const setting = view.imageState;
      const busy = !!pendingSends.get(peer) || choosing || view.toggling;
      toggle.disabled = !setting || busy || view.refreshing;
      chooser.disabled = !setting || !setting.enabled || busy;
      removeImage.disabled = busy || !image;
      toggle.textContent = setting && setting.enabled ? "Disable images" : "Enable images";
      selected.textContent = image ? "PNG attached" : "";
      imageStatus.textContent = "Images are server-readable. Both people must opt in. " +
        (!setting ? "Setting unavailable." : setting.available ? "Unlocked for this chat." : setting.enabled ? "Waiting for the other person to enable images." : "Off for you.");
    };
    view.refreshImages = async (refreshMessages = false) => {
      if (!active() || view.refreshing || view.toggling) return;
      view.refreshing = true; updateControls();
      try {
        const setting = await api("/api/messages/images?peer=" + encodeURIComponent(peer));
        if (!active()) return;
        const changed = JSON.stringify(setting) !== JSON.stringify(view.imageState);
        view.imageState = setting;
        updateControls();
        if (changed || refreshMessages) await refreshLoaded(view, peer);
      } catch (e) { viewNote(e.message); }
      finally { view.refreshing = false; if (active()) updateControls(); }
    };
    toggle.addEventListener("click", async () => {
      if (!active() || view.refreshing || view.toggling || !view.imageState || pendingSends.has(peer) || choosing) return;
      const enabled = !view.imageState.enabled;
      view.toggling = true; updateControls();
      try {
        const list = await api("/api/messages/conversations");
        if (!active()) return;
        const relationship = list.contacts.find(x => x.username === peer);
        if (!relationship) throw new Error("You cannot message this user.");
        await api("/api/messages/images", { conversation_handle: relationship.conversation_handle, enabled });
      } catch (e) { viewNote(e.message); }
      finally {
        view.toggling = false;
        if (active()) { await view.refreshImages(true); updateControls(); }
      }
    });
    chooser.addEventListener("change", async () => {
      if (chooser.disabled || !active()) return;
      const file = chooser.files && chooser.files[0];
      if (!file) return;
      choosing = true; updateControls();
      try { const attachment = await pngAttachment(file); if (active()) { image = attachment; draft.image = image; } }
      catch (e) { viewNote(e.message); }
      finally { choosing = false; chooser.value = ""; if (active()) updateControls(); }
    });
    removeImage.addEventListener("click", () => {
      if (removeImage.disabled) return;
      image = null; draft.image = null; updateControls();
    });
    input.addEventListener("input", () => { draft.body = input.value; });
    footEl.appendChild(imageStatus);
    footEl.appendChild(el("div", { class: "row" }, toggle, chooser));
    footEl.appendChild(el("div", { class: "row" }, selected, removeImage));
    let sending = false;
    let pending = pendingSends.get(peer) || null;
    if (pending) { input.value = pending.body; image = pending.image || null; draft.image = image; }
    updateControls();
    const send = async () => {
      const text = input.value;
      if (sending || choosing || view.toggling || !text.trim()) return;
      if (location.protocol !== "https:") { viewNote("Messaging requires HTTPS."); return; }
      // A reopened pane can still hold the operation that completed in the old
      // pane. Clear that stale composer instead of treating it as a new send.
      if (pending && pending.completed && pending.body === text && pending.image === image) { pending = null; input.value = ""; image = null; draft.body = ""; draft.image = null; updateControls(); viewNote("Sent."); return; }
      pending = pendingSends.get(peer) || null;
      if (pending && pending.inFlight) { viewNote("This message is still sending."); return; }
      if (pending && (pending.body !== text || pending.image !== image)) {
        viewNote("The previous message has an uncertain outcome. Retry its original text and image before sending another message.");
        return;
      }
      if (!pending && image && (!view.imageState || !view.imageState.enabled)) { viewNote("Enable images in this chat first."); return; }
      if (!pending) pending = { body: text, image, idempotency_key: crypto.randomUUID(), inFlight: false, completed: false, uncertain: false };
      pendingSends.set(peer, pending);
      pending.inFlight = true;
      const operation = pending;
      sending = true;
      input.disabled = true; updateControls();
      try {
        for (let attempt = 0; attempt < 2; attempt++) {
          const conversations = await api("/api/messages/conversations");
          const relationship = conversations.contacts.find((x) => x.username === peer);
          if (!relationship) throw new Error("You cannot message this user.");
          try {
            const receipt = await api("/api/messages/send", { body: pending.body, idempotency_key: pending.idempotency_key,
              conversation_handle: relationship.conversation_handle, ...(pending.image ? {image: pending.image} : {}) });
            if (!receipt || receipt.ok !== true || !Number.isInteger(receipt.id) || receipt.id <= 0 ||
                typeof receipt.created !== "number" || !Number.isFinite(receipt.created)) {
              throw new Error("Invalid send receipt; message outcome may be uncertain.");
            }
            pending.completed = true;
            if (pendingSends.get(peer) === pending) pendingSends.delete(peer);
            pending = null;
            input.value = "";
            image = null; draft.body = ""; draft.image = null;
            viewNote("");
            break;
          } catch (e) {
            operation.uncertain = operation.uncertain || !e.status || e.status >= 500;
            if (attempt === 0 && (!e.status || e.status === 403 || e.status >= 500)) continue;
            throw e;
          }
        }
      } catch (e) {
        if (!operation.uncertain && e.status >= 400 && e.status < 500) {
          if (pendingSends.get(peer) === operation) pendingSends.delete(peer);
          pending = null;
          viewNote(e.message); // Definitive rejection: the text can be corrected normally.
        } else {
          viewNote(e.message + " Retry unchanged text and image to safely retry this message.");
        }
      }
      finally { operation.inFlight = false; sending = false; input.disabled = false; if (active()) updateControls(); }
    };
    input.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); send(); } });
    footEl.appendChild(el("div", { class: "row" }, input, el("button", { type: "button", text: "Send", on: { click: send } })));
    state.seen = new Set();
    await loadThread(false);
    if (active()) { input.focus(); await view.refreshImages(); }
  }
  async function loadThread(older) {
    const peer = state.peer;
    const view = state.view;
    try {
      const q = "/api/messages/thread?peer=" + encodeURIComponent(peer) + (older && oldest ? "&before=" + oldest : "");
      const r = await api(q);
      if (!state.open || peer !== state.peer || view !== state.view) return;
      const frag = document.createDocumentFragment();
      for (const m of r.messages) { if (!state.seen.has(m.id)) { state.seen.add(m.id); const node = msgNode(m); view.nodes.set(m.id, node); frag.appendChild(node); } }
      if (older) threadEl.insertBefore(frag, threadEl.firstChild); else threadEl.appendChild(frag);
      if (r.messages.length) { oldest = oldest === null ? r.messages[0].id : Math.min(oldest, r.messages[0].id); newest = Math.max(newest, r.messages[r.messages.length - 1].id); }
      olderBtn.toggleAttribute("hidden", !r.more);
      if (!older) bodyEl.scrollTop = bodyEl.scrollHeight;
      if (newest) { await api("/api/messages/read", { peer, up_to: newest }); loadContacts().catch(() => {}); }
    } catch (e) { if (state.view === view && state.open) note(e.message); }
  }
  // Refresh every already-loaded page, including older locked messages, in place.
  async function refreshLoaded(view, peer) {
    const ids = [...view.nodes.keys()];
    if (!ids.length) return;
    const floor = Math.min(...ids);
    let before = null;
    do {
      const r = await api("/api/messages/thread?peer=" + encodeURIComponent(peer) + (before ? "&before=" + before : ""));
      if (!state.open || state.view !== view || state.peer !== peer) return;
      for (const m of r.messages) {
        const old = view.nodes.get(m.id);
        if (old) { const node = msgNode(m); old.replaceWith(node); view.nodes.set(m.id, node); }
      }
      const next = r.messages.length ? r.messages[0].id : null;
      if (!r.more || next === null || next <= floor || (before !== null && next >= before)) break;
      before = next;
    } while (true);
  }
  function incoming(m) {
    const peer = m.from === who.username ? m.to : m.from;
    if (state.peer === peer && threadEl) {
      if (state.seen.has(m.id)) return;
      state.seen.add(m.id);
      const node = msgNode(m); state.view.nodes.set(m.id, node); threadEl.appendChild(node);
      newest = Math.max(newest, m.id);
      bodyEl.scrollTop = bodyEl.scrollHeight;
      if (m.from === peer) api("/api/messages/read", { peer, up_to: m.id }).then(() => loadContacts()).catch(() => {});
    }
  }

  // ---- long poll: runs only while open ------------------------------------------------------------------------------------------
  async function poll(signal) {
    while (!signal.aborted) {
      try {
        const r = await fetch("/api/messages/since?after=" + state.after + "&wait=25", { credentials: "same-origin", cache: "no-store", signal });
        if (!r.ok) throw new Error("HTTP " + r.status);
        const data = await r.json();
        if (signal.aborted) return;
        if (state.view && state.view.refreshImages) state.view.refreshImages();
        if (state.after < 0) { state.after = data.latest || 0; continue; }
        let refresh = false;
        for (const m of data.messages) { state.after = Math.max(state.after, m.id); incoming(m); if (m.to === who.username) refresh = true; }
        if (refresh) loadContacts().catch(() => {});
      } catch (e) {
        if (signal.aborted) return;
        await new Promise((res) => setTimeout(res, 5000));
      }
    }
  }

  function open() {
    if (state.open) return;
    state.open = true;
    launcher.setAttribute("aria-expanded", "true");
    state.panel = buildPanel();
    root.appendChild(state.panel);
    renderBody();
    loadContacts().catch((e) => note(e.message));
    state.abort = new AbortController();
    state.after = -1;
    poll(state.abort.signal);
    const first = state.panel.querySelector("input, button.item");
    if (first) first.focus();
  }
  function close() {
    if (!state.open) return;
    state.open = false;
    launcher.setAttribute("aria-expanded", "false");
    if (state.abort) state.abort.abort();
    state.abort = null;
    if (state.panel) state.panel.remove();
    state.panel = null; state.peer = null; state.view = null; threadEl = null;
  }

  window.MeowdokuMessenger = Object.freeze({ open, close, version: VERSION });
  if (location.pathname === "/login.html" && new URLSearchParams(location.search).get("messages") === "1") open();
}

start();
