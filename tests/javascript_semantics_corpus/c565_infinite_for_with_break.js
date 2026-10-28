// for (;;) terminates only through its break.
/** @param {number} n @param {number} b */
function root(n, b) { let i = 0; let s = 0; for (;;) { i++; s += b; if (i >= n || i > 20) break; } return s; }
// INPUTS [[0, 1], [5, 2], [10, 3], [NaN, 1], [3, -1], [7, 0.5], [Infinity, 2]]
