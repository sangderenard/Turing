// Number arithmetic is IEEE double arithmetic: integral literals and
// parameters are doubles, never int64 (1 / 2 is 0.5).
/** @param {number} a @param {number} b */
function root(a, b) { const s = a + b; return s * s - a / b + 1 / 2; }
// INPUTS [[1, 2], [3, 0], [-0, 5], [NaN, 1], [Infinity, -Infinity], [0.1, 0.2], [1e308, 1e308], [7, 2]]
