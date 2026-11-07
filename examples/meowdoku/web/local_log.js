// [LOG] The page's own (client) log: console.log/info/debug/warn/error and uncaught errors, kept in a byte-capped ring and mirrored to
// localStorage ("meowdoku.locallog") so the log page can show what the game page printed.  Importing this module installs the capture.
export const KEY = "meowdoku.locallog", CAP = 200 * 1024, LINE = 1500;
const cost = (t) => t.length + 48;
const ring = (() => { try { const rows = JSON.parse(localStorage.getItem(KEY) || "[]"); return Array.isArray(rows) ? rows : []; } catch (_) { return []; } })();
let bytes = ring.reduce((n, e) => n + cost(String(e.text)), 0), seq = ring.at(-1)?.seq || 0, dirty = false;
export function add(src, text) {
  text = String(text).slice(0, LINE);
  ring.push({ seq: ++seq, ts: Date.now() / 1000, src, text }); bytes += cost(text);
  while (bytes > CAP && ring.length > 1) bytes -= cost(ring.shift().text);
  dirty = true;
}
export const entries = () => ring;
export function stored() { try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch (_) { return []; } }
const fmt = (a) => { try { return typeof a === "string" ? a : a instanceof Error ? (a.stack || a.message) : JSON.stringify(a); } catch (_) { return String(a); } };
if (!window.__localLogInstalled) {
  window.__localLogInstalled = true;
  for (const [m, src] of [["log", "console-log"], ["info", "console-log"], ["debug", "console-log"], ["warn", "console-warn"], ["error", "console-error"]]) {
    const orig = console[m].bind(console);
    console[m] = (...a) => { try { add(src, a.map(fmt).join(" ")); } catch (_) {} orig(...a); };
  }
  addEventListener("error", (e) => add("uncaught", `${e.message} @ ${(e.filename || "").split("/").pop()}:${e.lineno}`));
  addEventListener("unhandledrejection", (e) => add("uncaught", "unhandled rejection: " + fmt(e.reason)));
}
export function persist() {                         // called on a timer by the page that owns the key (the game page); the log page reads it
  if (!dirty) return; dirty = false;
  try { localStorage.setItem(KEY, JSON.stringify(ring)); } catch (_) {}
}
setInterval(persist, 1000);
addEventListener("pagehide", persist);
