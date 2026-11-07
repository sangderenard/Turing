// SERVER LIFECYCLE (server side: lifecycle.py).  Three things on every page, none of which takes layout space:
//   1. a countdown banner while an operator has announced a restart ("server updating in 0:45: your game will be saved");
//   2. a "server down, reconnecting..." chip while the server is unreachable, with backoff retries, resuming on its own;
//   3. the "cookie space" copy: the last restorable dump of this page's session (the server hands it over with each poll when it
//      changed), kept in localStorage under `meowdoku.session_copy`.  If the restarted server has no session for this browser
//      (its file was lost), the copy re-seeds one through POST /api/lifecycle/restore, and the page reloads onto it.
//
// The copy carries the game and its controls and nothing else: no player name, no token, no cookie.  The server strips such keys
// before it hands a dump out; this file strips them again on the way into storage and on the way out of it (validateCopy), and the
// server validates what it receives once more.  The key contains "session", so a shared view (share.js SECRET_KEY) never carries it.
// Local-only games (the fountain's compute runs on this page) keep running while the server is down: nothing here blocks input.

export const COPY_KEY = "meowdoku.session_copy";
export const COPY_MAX = 64 * 1024;
const SECRET_KEY = /token|secret|passw|credential|auth|cookie|session|api[_-]?key|lichess|player|geo/i;
const DUMP_KEYS = new Set(["v", "kind", "mode", "new", "game", "chess_clock"]);
const FORBIDDEN = new Set(["__proto__", "constructor", "prototype"]);

// ---------------------------------------------------------------- pure (tested in node)
export function stripSecrets(value, depth = 0) {
  if (depth > 12) return null;
  if (Array.isArray(value)) return value.map((v) => stripSecrets(v, depth + 1));
  if (value && typeof value === "object") {
    const out = {};
    for (const [k, v] of Object.entries(value)) if (!SECRET_KEY.test(k) && !FORBIDDEN.has(k)) out[k] = stripSecrets(v, depth + 1);
    return out;
  }
  return value;
}

// A stored or received copy as this page will keep or send it: bounded, versioned, scrubbed; null when it is not one.
export function validateCopy(raw) {
  let obj = raw;
  if (typeof raw === "string") {
    if (raw.length > COPY_MAX * 2) return null;
    try { obj = JSON.parse(raw); } catch { return null; }
  }
  if (!obj || typeof obj !== "object" || obj.v !== 1 || !obj.dump || typeof obj.dump !== "object") return null;
  const dump = {};
  for (const [k, v] of Object.entries(stripSecrets(obj.dump))) if (DUMP_KEYS.has(k)) dump[k] = v;
  if (dump.v !== 1 || typeof dump.kind !== "string" || !/^[a-z0-9_]{1,32}$/.test(dump.kind)) return null;
  if (!dump.new || typeof dump.new !== "object" || !dump.game || typeof dump.game !== "object") return null;
  if (JSON.stringify(dump).length > COPY_MAX) return null;
  return { v: 1, rev: typeof obj.rev === "string" ? obj.rev.slice(0, 24) : "", savedAt: Number(obj.savedAt) || 0, dump };
}

export const formatCountdown = (seconds) => {
  const s = Math.max(0, Math.ceil(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

export const backoffMs = (attempt) => Math.min(8000, 500 * 2 ** Math.min(attempt, 5)) + Math.floor(Math.random() * 250);

// ---------------------------------------------------------------- the page
if (typeof document !== "undefined" && typeof window !== "undefined") start();

function start() {
  const viewer = /(^|[#&])watch=/.test(location.hash);          // a view-only link has no session of its own to save or restore
  const life = { boot: null, restartAt: null, down: false, downSince: 0, attempt: 0, rev: "", timer: null, reloading: false };
  const storage = {
    get() { try { return localStorage.getItem(COPY_KEY); } catch { return null; } },
    set(v) { try { localStorage.setItem(COPY_KEY, v); } catch { /* storage may be blocked or full */ } },
  };

  const style = document.createElement("style");
  style.textContent = `
    #lc-banner { position: fixed; top: 8px; left: 50%; transform: translateX(-50%); z-index: 2147483000; max-width: min(92vw, 40rem);
      padding: 6px 14px; border-radius: 999px; font: 600 13px/1.35 system-ui, sans-serif; color: #fff; pointer-events: none;
      box-shadow: 0 2px 10px rgba(0,0,0,.45); display: none; text-align: center; }
    #lc-banner[data-mode="warn"] { display: block; background: rgba(176, 106, 0, .94); }
    #lc-banner[data-mode="down"] { display: block; background: rgba(168, 32, 32, .95); }
    #lc-banner[data-mode="back"] { display: block; background: rgba(30, 120, 60, .95); }
    #lc-banner small { display: block; font-weight: 400; opacity: .9; font-size: 11px; }`;
  document.head.appendChild(style);
  const banner = document.createElement("div");
  banner.id = "lc-banner"; banner.setAttribute("role", "status"); banner.setAttribute("aria-live", "polite");
  document.body.appendChild(banner);

  const render = () => {
    if (life.down) {
      const secs = Math.round((performance.now() - life.downSince) / 1000);
      banner.dataset.mode = "down";
      banner.innerHTML = `server down, reconnecting… <small>${secs}s · attempt ${life.attempt + 1} · your game is saved on this device; local simulations keep running</small>`;
    } else if (life.restartAt != null) {
      const left = (life.restartAt - performance.now()) / 1000;
      banner.dataset.mode = "warn";
      banner.textContent = left > 0 ? `server updating in ${formatCountdown(left)}: your game will be saved` : "server restarting now… your game is saved";
    } else if (!life.flash) banner.dataset.mode = "";
  };
  setInterval(render, 250);

  const remember = (data) => {
    if (viewer || !data || !data.snap) return;
    const copy = validateCopy({ v: 1, rev: data.snap_rev, savedAt: Date.now(), dump: data.snap });
    if (copy) storage.set(JSON.stringify(copy));
  };

  const goDown = () => { if (!life.down) { life.down = true; life.downSince = performance.now(); life.attempt = 0; render(); } };

  async function restoreAfterRestart() {
    // The cookie names the restored session if the server kept its file; ask the server what it holds.
    let held = null;
    try { held = await (await fetch("/api/state", { cache: "no-store" })).json(); } catch { return false; }
    const realtime = held && held.game && held.input === "realtime";
    if (held && held.game && !realtime) return false;                  // restored from its file: the page carries on where it was
    if (held && held.game && realtime) return true;                    // a real-time game restarts fresh on the server: the page reloads onto it
    const copy = validateCopy(storage.get());
    if (!copy) return false;
    try {
      const r = await fetch("/api/lifecycle/restore", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dump: copy.dump }) });
      return r.ok;
    } catch { return false; }
  }

  async function poll() {
    clearTimeout(life.timer);
    let delay = life.restartAt != null ? 1000 : 3000;
    try {
      const response = await fetch(`/api/lifecycle${viewer ? "" : `?snap=1&have=${encodeURIComponent(life.rev)}`}`, { cache: "no-store" });
      if (!response.ok) throw new TypeError(`lifecycle ${response.status}`);
      const data = await response.json();
      const wasDown = life.down, bootChanged = life.boot != null && data.boot !== life.boot;
      life.boot = data.boot;
      life.restartAt = data.seconds == null ? null : performance.now() + data.seconds * 1000;
      if (data.snap_rev !== undefined) life.rev = data.snap_rev;
      remember(data);
      if (wasDown || bootChanged) {
        life.down = false; life.attempt = 0;
        if (!viewer && (await restoreAfterRestart()) && !life.reloading) {
          life.reloading = true;
          banner.dataset.mode = "back"; banner.textContent = "server is back: restoring your game…";
          try { sessionStorage.setItem("meowdoku.resume", "1"); } catch { /* the reload then starts a new game, as before */ }
          setTimeout(() => location.reload(), 400);
          return;
        }
        life.flash = true; banner.dataset.mode = "back"; banner.textContent = "server is back";
        setTimeout(() => { life.flash = false; render(); }, 2500);
      }
      render();
    } catch (error) {
      if (!(error instanceof TypeError)) { life.timer = setTimeout(poll, delay); return; }      // e.g. the view-only guard refusing: not an outage
      goDown();
      delay = backoffMs(life.attempt++);
      render();
    }
    life.timer = setTimeout(poll, delay);
  }

  // An outage is noticed the moment any same-origin call fails to connect, not only at the next poll.
  const realFetch = window.fetch.bind(window);
  window.fetch = (input, init) => realFetch(input, init).catch((error) => {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (error instanceof TypeError && /^(https?:\/\/[^/]+)?\/api\//.test(url) && !url.includes("/api/lifecycle")) { goDown(); clearTimeout(life.timer); life.timer = setTimeout(poll, 500); }
    throw error;
  });
  poll();
}
