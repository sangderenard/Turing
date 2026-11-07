// Wordle's scene layout: pure helpers (no DOM, no GL), imported by the page and held by tests/meowdoku_wordle_layout_check.mjs.
//
// Three groups of tiles, all centred on the board's middle line (x = cols / 2) and all of one family (obsidian keys):
//   the ENTRY BAR   above the board: [Enter] one backing tile per letter of the word [Backspace]
//   the BOARD       6 x cols (laid out by the page's own board code, from x = 0)
//   the RACK        3 rows of up to 10 keys below the board; every row is centred by its OWN key count
// A tile owns its whole pitch cell (the visible gap is carved out of it), so the cells tile the plane and a tap between two tiles
// still lands on one of them; the gaps are drawn, not dead.

export const RACK_ASPECT = 1.5, RACK_WIDE = 1.5;        // rack keys: 1.5 x the board's width in all, each key RACK_ASPECT tall per wide
export const KEY_GAP = 0.09, ROW_GAP = 0.09;            // per side, as a fraction of the key's width (the board's tiles use 0.03)
export const BAR_LIFT = 0.45, RACK_DROP = 0.45;         // clear space between the bar and the board, and between the board and the rack

// the rack: `rowLens` = the key count of each row (the page passes game.rack.letters[r].length)
export function rackLayout(cols, rows, rackRows, rackCols, rowLens) {
  const span = cols * RACK_WIDE, size = span / rackCols, zsize = size * RACK_ASPECT;
  const lens = rowLens && rowLens.length === rackRows ? rowLens : Array(rackRows).fill(rackCols);
  return { size, zsize, span, origin: [(cols - span) / 2, rows + RACK_DROP], rows: rackRows, cols: rackCols, rowLens: lens,
           gapX: KEY_GAP * size, gapZ: ROW_GAP * size };
}
// x of the left edge of row r's first key (a short row sits centred)
export function rackRowLeft(L, r) { return L.origin[0] + (L.cols - L.rowLens[r]) * L.size / 2; }
export function rackCentre(L, r, c) { return [rackRowLeft(L, r) + (c + 0.5) * L.size, L.origin[1] + (r + 0.5) * L.zsize]; }
// the rack key under (x, z), or null
export function rackHit(L, x, z) {
  const r = Math.floor((z - L.origin[1]) / L.zsize); if (r < 0 || r >= L.rows) return null;
  const c = Math.floor((x - rackRowLeft(L, r)) / L.size); if (c < 0 || c >= L.rowLens[r]) return null;
  return { r, c };
}

// the entry bar: `barCols` tiles (Enter, the word's letters, Backspace); keys as wide as the rack's unless the row would outrun the rack
export function barLayout(cols, barCols, rackCols) {
  const rackSize = cols * RACK_WIDE / rackCols, size = Math.min(rackSize, (cols * RACK_WIDE) / barCols), zsize = size * RACK_ASPECT;
  const span = barCols * size;
  return { size, zsize, span, origin: [(cols - span) / 2, -(zsize + BAR_LIFT)], rows: 1, cols: barCols, gapX: KEY_GAP * size, gapZ: ROW_GAP * size };
}
export function barCentre(B, c) { return [B.origin[0] + (c + 0.5) * B.size, B.origin[1] + 0.5 * B.zsize]; }
export function barHit(B, x, z) {
  const c = Math.floor((x - B.origin[0]) / B.size), r = Math.floor((z - B.origin[1]) / B.zsize);
  return r === 0 && c >= 0 && c < B.cols ? { c } : null;
}
// what a bar tile is: the first is Enter, the last Backspace, the rest are letter slots
export function barKey(B, c) { return c === 0 ? "enter" : c === B.cols - 1 ? "back" : "slot"; }

// the give-up tile (overtime only): beside the board's last row, past its right edge (the "+n" badge sits above it)
export function giveLayout(cols, rows) {
  const size = 0.8;
  return { size, zsize: size, origin: [cols + 0.25, rows - 1.1], rows: 1, cols: 1, gapX: KEY_GAP * size, gapZ: ROW_GAP * size };
}

// the scene's extent for the whole Wordle layout: x span [left, left + width], z from the bar's top edge to the rack's bottom edge
export function wordleExtent(cols, rows, R, B, G) {
  const lo = Math.min(0, R.origin[0], B.origin[0]), hi = Math.max(cols + 1.45, R.origin[0] + R.span, B.origin[0] + B.span);
  const half = Math.max(cols / 2 - lo, hi - cols / 2), left = cols / 2 - half, right = cols / 2 + half;   // symmetric about the board's middle
  return { left, width: right - left, z0: B.origin[1], depth: R.origin[1] + R.rows * R.zsize };
}
