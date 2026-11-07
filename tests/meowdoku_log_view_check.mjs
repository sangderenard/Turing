// Node check of web/log_view.js: the visible-window (virtualization) math, the glyph fallback, the filters.
import { visibleRange, clampScroll, tailScroll, atTail, glyphFor, matches, searchTerms, lineText } from "../examples/meowdoku/web/log_view.js";

const fail = (m) => { console.error("FAIL " + m); process.exit(1); };
const eq = (a, b, m) => { if (JSON.stringify(a) !== JSON.stringify(b)) fail(`${m}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`); };

// 1,000,000 lines of 20 px in a 100 px view: only 5 (6 on a partial line) are built, wherever the scroll is
eq(visibleRange(0, 100, 20, 1e6), { first: 0, last: 5 }, "top");
eq(visibleRange(10, 100, 20, 1e6), { first: 0, last: 6 }, "partial first line");
eq(visibleRange(2000, 100, 20, 1e6), { first: 100, last: 105 }, "aligned");
eq(visibleRange(2005, 100, 20, 1e6, 2), { first: 98, last: 108 }, "overscan");
eq(visibleRange(0, 100, 20, 3), { first: 0, last: 3 }, "fewer lines than the view");
eq(visibleRange(0, 100, 20, 0), { first: 0, last: 0 }, "empty");
eq(visibleRange(1e9, 100, 20, 50), { first: 50, last: 50 }, "scrolled past the end is empty, not negative");
for (let s = 0; s < 5000; s += 7) { const r = visibleRange(s, 100, 20, 1e6); if (r.last - r.first > 6) fail("window too large at " + s); }
// clamp + tail
eq(clampScroll(-5, 100, 20, 50), 0, "clamp low"); eq(clampScroll(9999, 100, 20, 50), 900, "clamp high"); eq(clampScroll(50, 100, 20, 3), 0, "content fits");
eq(tailScroll(100, 20, 50), 900, "tail"); eq(tailScroll(100, 20, 3), 0, "tail when fitting");
eq(atTail(899, 100, 20, 50), true, "at tail within slack"); eq(atTail(800, 100, 20, 50), false, "not at tail");
// the last line is in the window at the tail scroll
const t = visibleRange(tailScroll(100, 20, 50), 100, 20, 50); eq(t, { first: 45, last: 50 }, "tail window");
// glyphs: A-Z 0-9 +, lowercase as capitals, space nothing, the rest a small +
const have = new Set([..."ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+"]);
eq(glyphFor("A", have), { name: "A", scale: 1, substituted: false }, "capital");
eq(glyphFor("q", have), { name: "Q", scale: 1, substituted: false }, "lowercase");
eq(glyphFor(" ", have), null, "space");
eq(glyphFor("[", have), { name: "+", scale: 0.45, substituted: true }, "punctuation falls back");
eq(glyphFor("\u00e9", have).substituted, true, "non-ascii falls back");
eq(glyphFor("[", new Set(["A"])), null, "no fallback glyph: nothing drawn");
// filters
const e = { seq: 1, ts: 0, src: "client-error", text: "TypeError boom at page" };
eq(matches(e, new Set(["client-error"]), searchTerms("boom PAGE")), true, "terms all present");
eq(matches(e, new Set(["server"]), []), false, "source filtered");
eq(matches(e, new Set(["client-error"]), searchTerms("nope")), false, "term absent");
if (!/^\d\d:\d\d:\d\d ERR TypeError/.test(lineText(e))) fail("line text " + lineText(e));
console.log("ok");
