// break leaves with the values it had: the rest of the iteration and the
// for update do not run.
/** @param {number} n @param {number} b */
function root(n, b) { let s = 0; let i = 0; for (i = 0; i < n; i++) { s += b; if (s > 4) { break; } s += 1; } return s * 100 + i; }
// INPUTS [[0, 1], [5, 2], [10, 3], [NaN, 1], [3, -1], [7, 0.5], [Infinity, 2]]
