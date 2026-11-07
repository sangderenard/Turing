// [LOG] The pure parts of the log page (log.js): which lines/characters are in view, which glyph draws a character, the filters.
// No DOM, no GL: tests/meowdoku_log_view_check.mjs runs this under node.

export const SOURCES = ["client-error", "fountain-diag", "server"];
export const TAGS = { "client-error": "ERR", "fountain-diag": "DIA", "server": "SRV", "console-log": "LOG", "console-warn": "WRN", "console-error": "CER", "uncaught": "UNC" };
export const LOCAL_SOURCES = ["console-log", "console-warn", "console-error", "uncaught"];

/** The window of items in view along one axis: items [first, last) intersect [scroll, scroll + view); `overscan` extra items each side. */
export function visibleRange(scroll, view, item, count, overscan = 0) {
  if (count <= 0 || view <= 0 || item <= 0) return { first: 0, last: 0 };
  const first = Math.max(0, Math.floor(scroll / item) - overscan);
  const last = Math.min(count, Math.ceil((scroll + view) / item) + overscan);
  const f = Math.min(first, count); return { first: f, last: Math.max(f, last) };
}

/** Scroll is clamped to [0, content - view] (0 when the content fits). */
export function clampScroll(scroll, view, item, count) {
  return Math.max(0, Math.min(scroll, count * item - view));
}

/** The scroll offset that shows the last item flush with the bottom. */
export function tailScroll(view, item, count) { return Math.max(0, count * item - view); }

/** True when the view sits at the end (within `slack` px): scrolling there turns follow-tail back on. */
export function atTail(scroll, view, item, count, slack = 2) { return scroll >= tailScroll(view, item, count) - slack; }

/**
 * The glyph that draws a character.  The exported set is A-Z, 0-9 and "+" (assets.json); lowercase is drawn as its capital, a space draws
 * nothing, and every other character draws as a small "+" (`substituted`: true).  `have` is the set of mesh names.
 */
export function glyphFor(ch, have) {
  if (ch === " " || ch === "\t") return null;
  const up = ch.length === 1 ? ch.toUpperCase() : ch;
  if (up.length === 1 && have.has(up)) return { name: up, scale: 1, substituted: false };
  return have.has("+") ? { name: "+", scale: 0.45, substituted: true } : null;
}

export function formatTime(ts) {
  const d = new Date(ts * 1000), p = (n, w = 2) => String(n).padStart(w, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

/** The text of one log line as shown: time, a three-letter source tag, the text (the tag is also its colour's key). */
export function lineText(e) { return `${formatTime(e.ts)} ${TAGS[e.src] || "???"} ${String(e.text).replace(/\s+/g, " ")}`; }

/** Filter by source set and a case-insensitive search (every whitespace-separated term must appear). */
export function matches(e, sources, terms) {
  if (!sources.has(e.src)) return false;
  if (!terms.length) return true;
  const t = String(e.text).toLowerCase();
  return terms.every((w) => t.includes(w));
}
export function searchTerms(q) { return String(q || "").toLowerCase().split(/\s+/).filter(Boolean); }
