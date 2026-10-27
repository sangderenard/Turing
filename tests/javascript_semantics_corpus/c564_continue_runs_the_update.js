// continue skips the rest of the body but still runs the for update.
/** @param {number} n @param {number} b */
function root(n, b) { let s = 0; for (let i = 0; i < n && i < 30; i++) { if (i % 3 === 1) continue; s += i * b; } return s; }
// INPUTS [[0, 1], [5, 2], [10, 3], [NaN, 1], [3, -1], [7, 0.5], [Infinity, 2]]
